import os
import sys
import json
import time
import h5py
import random
import pickle as pkl
import argparse

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

from englib.common import utils
from englib.models.naive_psych import RGCNNaivePsychologyLinkPredict
from englib.models.naive_psych import MASLOW_LABEL2IDX, REISS_LABEL2IDX, PLUTCHIK_LABEL2IDX
from englib.models.naive_psych import sample_png_edges


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='train PNG for NaivePsych')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='naive psych config file.')
    parser.add_argument('unsupervised_input_dir', metavar='UNSUPERVISED_INPUT_DIR',
                        help='input directory.')
    parser.add_argument('supervised_input_dir', metavar='SUPERVISED_INPUT_DIR',
                        help='input directory.')
    parser.add_argument('sample_input_dir', metavar='SAMPLE_INPUT_DIR',
                        help='sample input directory.')
    parser.add_argument('task', metavar='TASK', choices=['maslow', 'reiss', 'plutchik'],
                        help='task')
    parser.add_argument('weight_name', metavar='WEIGHT_NAME',
                        choices=['bert-large-cased', 'roberta-large', 'bert-base-cased', 'roberta-base', 'elmo'],
                        help='model name')
    parser.add_argument('mode', metavar='MODE', choices=['train', 'test', 'test-dev', 'test-train'],
                        help='train or test')
    # parser.add_argument('model_class', metavar='MODEL_CLASS',
    #                     choices=['SimpleLM', 'SimpleFFN'],
    #                     help='model class')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    # parser.add_argument("--model_config", default=None, type=str,
    #                     help="model config")

    parser.add_argument("--lr", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup steps.")
    parser.add_argument("--warmup_portion", default=0.0, type=float,
                        help="Linear warmup over warmup portion.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=20,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--train_batch_size', default=1, type=int,
                        help='training batch size')
    parser.add_argument('--eval_batch_size', default=4, type=int,
                        help='evaluating batch size')
    parser.add_argument('--from_checkpoint', default=None, type=str,
                        help='checkpoint directory')
    parser.add_argument('--no_first_eval', action='store_true', default=False,
                        help='no first_evaluation (train only)')
    parser.add_argument('--no_eval', action='store_true', default=False,
                        help='no evaluation (train only)')
    parser.add_argument('--use_pos_weight', action='store_true', default=False,
                        help='use pos weight for criterion')
    parser.add_argument("--edge_sample_rate", default=0.2, type=float,
                        help="proportion of missing edges for each PNG.")
    parser.add_argument('--n_neg_per_pos_edge', default=5, type=int,
                        help='number of negative edges for each positive edge')
    parser.add_argument('--freeze_lm', action='store_true', default=False,
                        help='freeze LM')
    parser.add_argument('--is_semisupervised', action='store_true', default=False,
                        help='is semisupervised')
    parser.add_argument('--multi_gpus', action='store_true', default=False,
                        help='pararell gpus')
    parser.add_argument('--master_addr', type=str, default="localhost",
                        help='master address for distributed training')
    parser.add_argument('--master_port', type=int, default=19191,
                        help='master port for distributed training')

    parser.add_argument('--seed', type=int, default=135,
                        help='seed for random')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='gpu id')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


class PNGPretrainDataset(Dataset):
    def __init__(self, input_f, sample_f, task, is_train=False):
        super(PNGPretrainDataset, self).__init__()
        self.input_f = input_f
        self.sample_f = sample_f
        self.task = task
        self.is_train = is_train
        self.fp = None
        self.fp_sample = None

        # logger.info('loading {}...'.format(self.sample_f))
        fp = h5py.File(sample_f, 'r')
        self.sids = sorted(list(fp.keys()))
        fp.close()

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, idx):
        # a trick to solve picklability issue with torch.multiprocessing
        if self.fp is None:
            self.fp = h5py.File(self.input_f, 'r')
            self.fp_sample = h5py.File(self.sample_f, 'r')

        sid = self.sids[idx]
        input_ids = torch.from_numpy(self.fp[sid][self.task]['input_ids'][:])
        attention_mask = torch.from_numpy(self.fp[sid][self.task]['input_mask'][:])
        token_type_ids = torch.from_numpy(self.fp[sid][self.task]['token_type_ids'][:])

        input_edges = torch.from_numpy(self.fp_sample[sid]['input_edges'][:])
        pos_edges = torch.from_numpy(self.fp_sample[sid]['pos_edges'][:])
        neg_edges = torch.from_numpy(self.fp_sample[sid]['neg_edges'][:])

        if self.is_train:
            edges = torch.cat((input_edges, pos_edges), dim=1)
            return (sid, input_ids, attention_mask, token_type_ids, edges)
        return (sid, input_ids, attention_mask, token_type_ids, input_edges, pos_edges, neg_edges)


def my_dev_collate(samples):
    sid, input_ids, attention_mask, token_type_ids, input_edges, pos_edges, neg_edges = \
        map(list, zip(*samples))

    edge_types = [ie[1] for ie in input_edges]

    batch = {
        'sid': sid,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'input_edges': input_edges,
        'edge_types': edge_types,
        'pos_edges': pos_edges,
        'neg_edges': neg_edges
    }
    return batch


def my_train_collate(samples):
    sid, input_ids, attention_mask, token_type_ids, edges = \
        map(list, zip(*samples))

    batch = {
        'sid': sid,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'edges': edges
    }
    return batch


def to_gpu(batch, gpu_id):
    batch['input_ids'] = [ids.cuda(gpu_id) for ids in batch['input_ids']]
    batch['attention_mask'] = [im.cuda(gpu_id) for im in batch['attention_mask']]
    batch['token_type_ids'] = [tti.cuda(gpu_id) for tti in batch['token_type_ids']]
    batch['edge_norms'] = [en.cuda(gpu_id) for en in batch['edge_norms']]
    batch['edge_types'] = [et.cuda(gpu_id) for et in batch['edge_types']]
    return batch


def calculate_norms(batch):
    input_ids = batch['input_ids']
    input_edges = batch['input_edges']

    batch_norms = []
    for i in range(len(input_edges)):
        n_nodes = input_ids[i].shape[0]

        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.add_edges(input_edges[i][0], input_edges[i][2])

        norms = torch.FloatTensor(input_edges[i].shape[1])
        for j in range(input_edges[i].shape[1]) :
            nid2 = int(input_edges[i][2][j])
            norms[j] = 1.0 / g.in_degree(nid2)
        batch_norms.append(norms)
    batch['edge_norms'] = batch_norms
    return batch


def class_measure(class_cm, n_examples, rels, y_pred, y, n_classes):
    for i in range(n_classes):
        idxs = (rels == i).nonzero().flatten()
        if idxs.nelement() == 0:
            continue
        pred = y_pred[idxs]
        gold = y[idxs]
        cm = confusion_matrix(gold, pred)
        class_cm[i] += cm

        n_pos = gold.sum()
        n_examples[i][1] += n_pos
        n_examples[i][0] += (gold.shape[0] - n_pos)


def evaluate(model, dataloader, gpu, get_prec_recall_f1=False, logger=None):
    model.eval()

    n_output_rels = model.n_rtypes
    class_cm = np.zeros((n_output_rels, 2, 2), dtype=np.int64)
    n_examples = np.zeros((n_output_rels, 2), dtype=np.int64)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = calculate_norms(batch)

            # to GPU
            if gpu != -1:
                batch = to_gpu(batch, gpu)

            y_scores, ys, rtypes = model.predict(**batch)
            y_preds = (y_scores.cpu() >= 0.5).long()
            class_measure(
                class_cm, n_examples,
                rtypes.cpu(), y_preds, ys.cpu(), n_output_rels
            )

    # macro-averaged
    precisions, recalls = [], []
    for i in range(n_output_rels):
        tn, fp, fn, tp = class_cm[i].ravel()
        c_prec = tp / (tp + fp) if tp + fp != 0 else 0.0
        c_recall = tp / (tp + fn) if tp + fn != 0 else 0.0
        c_f1 = (2.0 * c_prec * c_recall) / (c_prec + c_recall) if c_prec + c_recall != 0 else 0.0

        precisions.append(c_prec)
        recalls.append(c_recall)
        if logger:
            logger.info('class={}, #pos={}, #neg={}, prec={}, recall={}, f1={}'.format(
                i, n_examples[i][1], n_examples[i][0], c_prec, c_recall, c_f1))

    prec_macro = sum(precisions) / len(precisions)
    recall_macro = sum(recalls) / len(recalls)
    f1_macro = (2 * prec_macro * recall_macro) / (prec_macro + recall_macro) if \
        (prec_macro + recall_macro) != 0 else 0.0
    if get_prec_recall_f1:
        return prec_macro, recall_macro, f1_marco
    return f1_macro


def get_batch_relations(target_edges):
    rels = [te[:, 1, :].flatten() for te in target_edges]
    rels = torch.cat(rels, dim=0)
    return rels.cpu()


# def test(test_inputs, model_dir):
#     test_dataset = PNGDataset(*test_inputs)
#     test_dataloader = DataLoader(
#         test_dataset,
#         batch_size=args.eval_batch_size, # fix 1 for sampling
#         shuffle=False,
#         collate_fn=my_collate,
#         num_workers=1) # 1 is safe for hdf5

#     model = get_model(model_dir)
#     if args.gpu_id != -1:
#         model = model.cuda(args.gpu_id)

#     test_metric = evaluate(model, test_dataloader, args.gpu_id, dump_predictions=True)
#     logger.info('test metric: {}'.format(test_metric))


def get_model(model_dir, weight_name, freeze_lm):
    if model_dir is not None:
        target_dir = utils.get_target_model_dir(model_dir)
        logger.info('loading model from {}...'.format(target_dir))
        model = RGCNNaivePsychologyLinkPredict.from_pretrained(target_dir)
    else:
        mconfig = {
            'weight_name': weight_name,
            'dropout': 0.2,
            'freeze_lm': freeze_lm,
            'n_hidden_layers': 2,
            'n_rtypes': 8, # relation type
            'reg_param': 0.0,
            'use_gate': False,
            'use_rgcn': True,
            "class_weights": [
                1.0,
                2.1105199717361485,
                18.75494347487699,
                34.22207245155855,
                99.63949821290909,
                74.15020340043807,
                538.5439393939394,
                17.80555799070746
            ]
        }
        model = RGCNNaivePsychologyLinkPredict(**mconfig)
    return model


def batch_sample_truncated_graphs(batch, rtype_distr, args):
    n_pngs = len(batch['sid'])

    all_input_edges, all_pos_edges, all_neg_edges = [], [], []

    for i_png in range(n_pngs):
        n_nodes = batch['input_ids'][i_png].shape[0]
        sampled = sample_png_edges(
            batch['edges'][i_png][0],
            batch['edges'][i_png][2],
            batch['edges'][i_png][1],
            n_nodes,
            args.n_neg_per_pos_edge,
            edge_sample_rate=args.edge_sample_rate,
            rtype_distr=rtype_distr
        )
        if sampled is None:
            return None

        input_edges, pos_edges, neg_edges = sampled
        if neg_edges is None:
            return None

        all_input_edges.append(input_edges)
        all_pos_edges.append(pos_edges)
        all_neg_edges.append(neg_edges)

    all_edge_types = [ie[1] for ie in all_input_edges]
    batch['input_edges'] = all_input_edges
    batch['pos_edges'] = all_pos_edges
    batch['neg_edges'] = all_neg_edges
    batch['edge_types'] = all_edge_types
    return batch


def train(rank, train_inputs, dev_inputs, config, args):
    logger = utils.get_root_logger(args, log_fname='log_rank{}'.format(rank))
    if args.n_gpus > 1:
        local_rank = rank
        args.gpu_id = rank
    else:
        local_rank = -1
    if args.n_gpus > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.n_gpus,
            rank=local_rank
        )
    set_seed(args.gpu_id, args.seed) # in distributed training, this has to be same for all processes

    logger.info('local_rank = {}, n_gpus = {}'.format(local_rank, args.n_gpus))
    logger.info('n_epochs = {}'.format(args.n_epochs))
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    rtype_distr = config['rtype_distr']
    rtype2idx = config['rtype2idx']
    rtype_distr = {rtype2idx[r]: d for r, d in rtype_distr.items()}

    # dev dataset
    if local_rank in [-1, 0]:
        dev_dataset = PNGPretrainDataset(dev_inputs[0], dev_inputs[1], args.task, is_train=False)
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.eval_batch_size, # fix 1 for sampling
            shuffle=False,
            collate_fn=my_dev_collate,
            num_workers=1) # 1 is safe for hdf5

    # train dataset
    train_dataset = PNGPretrainDataset(train_inputs[0], train_inputs[1], args.task, is_train=True)
    if args.n_gpus > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.n_gpus,
            rank=local_rank
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size, # fix 1 for sampling
        shuffle=shuffle,
        collate_fn=my_train_collate,
        num_workers=1,
        sampler=train_sampler
    ) # 1 is safe for hdf5


    # model
    # pos_weight = torch.FloatTensor([args.pos_weight])
    model = get_model(args.from_checkpoint, args.weight_name, args.freeze_lm)
    criterion = model.criterion
    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)
        # pos_weight = pos_weight.cuda(args.gpu_id)
    if args.n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[args.gpu_id],
                                        find_unused_parameters=True)

    optimizer = utils.get_optimizer_adam(
        model, args.weight_decay, args.lr,
        args.adam_epsilon, args.from_checkpoint,
        (args.adam_beta1, args.adam_beta2)
    )
    # optimizer = utils.get_optimizer_adamw(
    #     model, args.weight_decay, args.lr,
    #     args.adam_epsilon, args.from_checkpoint
    # )
    n = len(train_dataset)
    scheduler = utils.get_scheduler(n, optimizer, args.train_batch_size,
                                    args.gradient_accumulation_steps, args.n_epochs,
                                    args.warmup_steps, args.warmup_portion,
                                    args.from_checkpoint)

    if local_rank in [-1, 0]:
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", args.n_epochs)
        logger.info("  Training batch size = %d", args.train_batch_size)
        logger.info("  Evaluation batch size = %d", args.eval_batch_size)
        logger.info("  Accu. train batch size = %d",
                        args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Weight Decay = {}".format(args.weight_decay))
        logger.info("  Learning Rate = {}".format(args.lr))

        if args.no_first_eval or args.no_eval:
            best_metric = 0.0
        else:
            best_metric = evaluate(model, dev_dataloader, args.gpu_id, logger=logger)
        logger.info('start dev_metric = {}'.format(best_metric))
        tb_writer = SummaryWriter('{}/explog'.format(args.output_dir))
    else:
        best_metric = 0.0

    t1 = time.time()
    step, accu_step = 0, 0
    prev_acc_loss, acc_loss = 0.0, 0.0
    model.zero_grad()
    for i_epoch in range(args.n_epochs):
        t2 = time.time()
        logger.info('========== Epoch {} =========='.format(i_epoch))

        for batch in train_dataloader:
            batch = batch_sample_truncated_graphs(batch, rtype_distr, args)
            if batch is None:
                # sometimes happen
                logger.warning('unable to sample neg edges, skip this batch')
                continue

            batch = calculate_norms(batch)

            model.train()
            # to GPU
            if args.gpu_id != -1:
                batch = to_gpu(batch, args.gpu_id)

            # forward pass
            all_scores, all_ys, all_rtypes, all_embs = model(**batch)


            loss = criterion(
                all_scores, all_ys, all_rtypes, all_embs)
            if args.n_gpus > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # accumulation
            acc_loss += loss.item()
            accu_step += 1
            if accu_step % args.gradient_accumulation_steps == 0: # ignore the last accumulation
                # update params
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step += 1

                # loss
                if local_rank in [-1, 0]:
                    if args.logging_steps > 0 and step % args.logging_steps == 0:
                        cur_loss = (acc_loss - prev_acc_loss) / args.logging_steps
                        logger.info('task={}, train_loss={}, accu_step={}, step={}, time={}s'.format(
                            args.task,
                            cur_loss,
                            accu_step,
                            step,
                            time.time()-t1)
                         )
                        tb_writer.add_scalar('train_loss', cur_loss, step)
                        tb_writer.add_scalar('lr', scheduler.get_last_lr()[0])

                        # evaluate
                        if not args.no_eval:
                            dev_metric = evaluate(
                                model, dev_dataloader, args.gpu_id, logger=logger)
                            logger.info('dev_metric={}'.format(dev_metric))
                            if best_metric < dev_metric:
                                best_metric = dev_metric

                                # save
                                utils.save_model(model, optimizer, scheduler, args.output_dir, step)
                        prev_acc_loss = acc_loss

        logger.info('done epoch {}: {} s'.format(i_epoch, time.time()-t2))
        if local_rank in [-1, 0] and args.no_eval:
            logger.info('saving model for epoch {}'.format(i_epoch))
            utils.save_model(model, optimizer, scheduler, args.output_dir, step)
    if local_rank in [-1, 0]:
        tb_writer.close()
    logger.info('best_dev_metric = {}'.format(best_metric))
    logger.info('done training: {} s'.format(time.time() - t1))


def set_seed(gpu, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu != -1:
        torch.cuda.manual_seed_all(seed)


def get_dataset_inputs(prefix, args):
    if prefix in ['unsupervised', 'semisupervised']:
        indir = args.unsupervised_input_dir
    else:
        indir = args.supervised_input_dir
    input_f = os.path.join(indir, '{}_inputs.h5'.format(prefix))
    sample_f = os.path.join(args.sample_input_dir, '{}_sampled_edges.h5'.format(prefix))
    return input_f, sample_f


def main():
    args = utils.bin_config(get_arguments)

    if args.multi_gpus: # use all GPUs in parallel
        assert torch.cuda.is_available()
        args.n_gpus = torch.cuda.device_count()
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.master_port)
    elif args.gpu_id != -1:
        args.n_gpus = 1
    else:
        args.n_gpus = 0

    config = json.load(open(args.config_file))
    assert config['config_target'] == 'naive_psychology'

    # if args.mode == 'train':
    prefix = 'semisupervised' if args.is_semisupervised else 'unsupervised'
    train_inputs = get_dataset_inputs(prefix, args)
    dev_inputs = get_dataset_inputs('dev', args)
    if args.n_gpus > 1:
        mp.spawn(train, nprocs=args.n_gpus, args=(train_inputs, dev_inputs, config, args))
    else:
        train(args.gpu_id, train_inputs, dev_inputs, config, args)

    # test
    # test_inputs = load_dataset_inputs('test')
    # test(test_inputs, args.output_dir)

    # elif args.mode == 'test': # test
    #     test_inputs = load_dataset_inputs('test')
    #     test(test_inputs, args.from_checkpoint)
    # elif args.mode == 'test-dev': # test
    #     test_inputs = load_dataset_inputs('dev')
    #     test(test_inputs, args.from_checkpoint)
    # elif args.mode == 'test-train': # test
    #     test_inputs = load_dataset_inputs('train')
    #     test(test_inputs, args.from_checkpoint)


if __name__ == "__main__":
    main()
