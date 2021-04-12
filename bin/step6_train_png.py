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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from englib.common import utils
from englib.models.naive_psych import RGCNNaivePsychology
from englib.models.naive_psych import MASLOW_LABEL2IDX, REISS_LABEL2IDX, PLUTCHIK_LABEL2IDX


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='train PNG for NaivePsych')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='naive psych config file.')
    parser.add_argument('input_dir', metavar='INPUT_DIR',
                        help='input directory.')
    parser.add_argument("task", metavar='TASK', type=str,
                        choices=[
                            'maslow',
                            'reiss',
                            'plutchik'
                        ],
                        help="task objective")
    parser.add_argument('mode', metavar='MODE', choices=['train', 'test', 'test-dev', 'test-train'],
                        help='train or test')
    parser.add_argument('weight_name', metavar='WEIGHT_NAME',
                        choices=['bert-large-cased', 'roberta-large', 'bert-base-cased', 'roberta-base', 'elmo'],
                        help='model name')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    # parser.add_argument("--model_config", default=None, type=str,
    #                     help="model config")

    parser.add_argument("--patience", default=-1, type=int,
                        help="The stopping patience based on the number of epochs")
    parser.add_argument("--lr", default=2e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.95, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup steps.")
    parser.add_argument("--warmup_portion", default=0.0, type=float,
                        help="Linear warmup over warmup portion.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_epochs', default=150, type=int,
                        help='number of training epochs')
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--train_batch_size', default=2, type=int,
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
    parser.add_argument('--freeze_lm', action='store_true', default=False,
                        help='freeze LM')
    parser.add_argument('--from_unsupervised', type=str, default=None,
                        help='load the partial model from the unsupervised model specified in from_checkpoint.')
    parser.add_argument('--is_pretrained_lm', action='store_true', default=False,
                        help='from pretrain lm')

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


class PNGDataset(Dataset):
    def __init__(self, input_f, node_info_f, task):
        super(PNGDataset, self).__init__()
        self.task = task
        self.input_f = input_f
        self.node_info_f = node_info_f
        self.fp = None
        self.node_info = self.load_node_info(node_info_f)

        fp = h5py.File(input_f, 'r')
        self.sids = sorted(list(fp.keys()))
        fp.close()

    def load_node_info(self, fpath):
        node_info = {}
        with open(fpath, 'r') as fr:
            for line in fr:
                info = json.loads(line)
                node_info[info['sid']] = info['node_info']
        return node_info

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, idx):
        # a trick to solve picklability issue with torch.multiprocessing
        if self.fp is None:
            self.fp = h5py.File(self.input_f, 'r')

        sid = self.sids[idx]
        # wemb = torch.from_numpy(self.fp[sid][self.task]['word_embeddings'][:])
        input_ids = torch.from_numpy(self.fp[sid][self.task]['input_ids'][:])
        attention_mask = torch.from_numpy(self.fp[sid][self.task]['input_mask'][:])
        token_type_ids = torch.from_numpy(self.fp[sid][self.task]['token_type_ids'][:])

        edge_src = torch.from_numpy(self.fp[sid]['edge_src'][:])
        edge_dest = torch.from_numpy(self.fp[sid]['edge_dest'][:])
        edge_types = torch.from_numpy(self.fp[sid]['edge_types'][:])
        edge_norms = torch.from_numpy(self.fp[sid]['edge_norms'][:])

        labels = torch.FloatTensor(self.fp[sid][self.task]['labels'][:])

        # in case there is only one node
        if len(labels.shape) == 1:
            labels = labels.unsquzze(0)

        node_info = self.node_info[sid]
        return (input_ids, attention_mask, token_type_ids,
                edge_src, edge_dest, edge_types, edge_norms,
                labels, node_info)


def my_collate(samples):
    input_ids, attention_mask, token_type_ids, edge_src, edge_dest, edge_types, edge_norms, \
        labels, node_info = map(list, zip(*samples))

    # merge label first, because it's only for loss
    labels = torch.cat(labels, dim=0)

    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'edge_src': edge_src,
        'edge_dest': edge_dest,
        'edge_types': edge_types,
        'edge_norms': edge_norms,
        'labels': labels,
        'node_info': node_info
    }
    return batch


def micro_averaged_metrics(y_true, y_pred, prefix):
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    logger.info('{}: precision={}, recall={}, f1={}'.format(prefix, prec, recall, f1))
    return {'precision': prec, 'recall': recall, 'f1': f1}


def to_gpu(batch, gpu_id, move_labels):
    batch['input_ids'] = [im.cuda(gpu_id) for im in batch['input_ids']]
    batch['attention_mask'] = [im.cuda(gpu_id) for im in batch['attention_mask']]
    batch['token_type_ids'] = [im.cuda(gpu_id) for im in batch['token_type_ids']]
    batch['edge_types'] = [et.cuda(gpu_id) for et in batch['edge_types']]
    batch['edge_norms'] = [en.cuda(gpu_id) for en in batch['edge_norms']]
    if move_labels:
        batch['labels'] = batch['labels'].cuda(gpu_id)
    return batch


def get_predicted_scores(scores, idx2label):
    scores = {idx2label[i]: scores[i].item() for i in range(scores.shape[0])}
    return scores


def evaluate(model, dataloader, gpu, dump_predictions=False):
    model.eval()

    all_ys, all_scores = [], []
    all_node_info = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # to GPU
            if gpu != -1:
                batch = to_gpu(batch, gpu, move_labels=False)

            # forward pass
            score = model.predict(**batch)
            y_pred = (score >= 0.5).long()

            all_ys.append(batch['labels'])
            all_scores.append(score)

            if dump_predictions:
                all_node_info += [n for ns in batch['node_info'] for n in ns]

        all_ys = torch.cat(all_ys, dim=0)
        all_scores = torch.cat(all_scores, dim=0).cpu()
        all_preds = (all_scores >= 0.5).long()

    if dump_predictions:
        if args.task == 'maslow':
            idx2label = {idx: label for label, idx in MASLOW_LABEL2IDX.items()}
        elif args.task == 'reiss':
            idx2label = {idx: label for label, idx in REISS_LABEL2IDX.items()}
        elif args.task == 'plutchik':
            idx2label = {idx: label for label, idx in PLUTCHIK_LABEL2IDX.items()}
        fpath = os.path.join(args.output_dir, 'predictions.json')
        logger.info('dumping {}...'.format(fpath))
        with open(fpath, 'w') as fw:
            for i in range(len(all_node_info)):
                all_node_info[i]['predicted_{}'.format(args.task)] = \
                    get_predicted_scores(all_scores[i], idx2label)
                fw.write(json.dumps(all_node_info[i]) + '\n')

    # micro-average
    result = micro_averaged_metrics(all_ys, all_preds, args.task)
    return result['f1']


def test(test_inputs, model_dir):
    test_dataset = PNGDataset(test_inputs[0], test_inputs[1], args.task)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size, # fix 1 for sampling
        shuffle=False,
        collate_fn=my_collate,
        num_workers=1) # 1 is safe for hdf5

    model = get_model(model_dir)
    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)

    test_metric = evaluate(model, test_dataloader, args.gpu_id, dump_predictions=True)
    logger.info('test metric: {}'.format(test_metric))


def get_model(model_dir):
    if model_dir is not None:
        target_dir = utils.get_target_model_dir(model_dir)
        logger.info('loading model from {}...'.format(target_dir))
        model = RGCNNaivePsychology.from_pretrained(target_dir)
    else:
        # mconfig = json.load(open(args.model_config))
        if args.task == 'maslow':
            n_classes = len(MASLOW_LABEL2IDX)
        elif args.task == 'reiss':
            n_classes = len(REISS_LABEL2IDX)
        else:
            n_classes = len(PLUTCHIK_LABEL2IDX)

        mconfig = {
            'weight_name': args.weight_name,
            'n_classes': n_classes,
            'dropout': 0.2,
            # 'use_lm': True,
            'freeze_lm': args.freeze_lm,
            'n_hidden_layers': 2,
            'n_rtypes': 8, # relation type
            'reg_param': 0.0,
            'use_gate': False,
            'use_rgcn': True
        }
        model = RGCNNaivePsychology(**mconfig)
    return model


# def get_pos_weights(dataloader):

#     maslow_pos_count = torch.zeros(len(MASLOW_LABEL2IDX), dtype=torch.float32)
#     reiss_pos_count = torch.zeros(len(REISS_LABEL2IDX), dtype=torch.float32)
#     plutchik_pos_count = torch.zeros(len(PLUTCHIK_LABEL2IDX), dtype=torch.float32)
#     n_examples = 0
#     for bg, y_maslow, y_reiss, y_plutchik in dataloader:

#         n_examples += y_maslow.shape[0]

#         maslow_pos_count += y_maslow.sum(0)
#         reiss_pos_count += y_reiss.sum(0)
#         plutchik_pos_count += y_plutchik.sum(0)

#     maslow_neg_count = n_examples - maslow_pos_count
#     reiss_neg_count = n_examples - reiss_pos_count
#     plutchik_neg_count = n_examples - plutchik_pos_count

#     # smoothing
#     maslow_pos_count[maslow_pos_count == 0] = 1
#     reiss_pos_count[reiss_pos_count == 0] = 1
#     plutchik_pos_count[plutchik_pos_count == 0] = 1

#     maslow_pos_weight = maslow_neg_count / maslow_pos_count
#     reiss_pos_weight = reiss_neg_count / reiss_pos_count
#     plutchik_pos_weight = plutchik_neg_count / plutchik_pos_count

#     return maslow_pos_weight, reiss_pos_weight, plutchik_pos_weight


def train(train_inputs, dev_inputs, config):
    train_dataset = PNGDataset(train_inputs[0], train_inputs[1], args.task)
    dev_dataset = PNGDataset(dev_inputs[0], dev_inputs[1], args.task)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size, # fix 1 for sampling
        shuffle=True,
        collate_fn=my_collate,
        num_workers=1) # 1 is safe for hdf5

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size, # fix 1 for sampling
        shuffle=False,
        collate_fn=my_collate,
        num_workers=1) # 1 is safe for hdf5

    # model
    # pos_weight = torch.FloatTensor([args.pos_weight])
    model = get_model(args.from_checkpoint)
    if args.from_unsupervised:
        model = utils.load_from_unsupervised_pretraining(
            model, args.from_unsupervised, ignore_classifier=True,
            is_lm=args.is_pretrained_lm
        )

    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)
        # pos_weight = pos_weight.cuda(args.gpu_id)

    optimizer = utils.get_optimizer(model, args.weight_decay, args.lr,
                                    args.adam_epsilon, args.from_checkpoint)
    scheduler = utils.get_scheduler(len(train_dataset), optimizer, args.train_batch_size,
                                    args.gradient_accumulation_steps, args.n_epochs,
                                    args.warmup_steps, args.warmup_portion,
                                    args.from_checkpoint)

    if args.use_pos_weight:
        # maslow_pos_weight, reiss_pos_weight, plutchik_pos_weight = get_pos_weights(train_dataloader)
        pos_weight = torch.FloatTensor(config['class_weight'][self.task])
        if args.gpu_id != -1:
            pos_weight = pos_weight.cuda(args.gpu_id)
        logger.info('pos_weight = {}'.format(pos_weight))

        criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.n_epochs)
    logger.info("  Training batch size = %d", args.train_batch_size)
    logger.info("  Evaluation batch size = %d", args.eval_batch_size)
    logger.info("  Accu. train batch size = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Weight Decay = {}".format(args.weight_decay))
    logger.info("  Learning Rate = {}".format(args.lr))
    logger.info("  Patience = {}".format(args.patience))

    if args.no_first_eval:
        best_metric = 0.0
    else:
        best_metric = evaluate(model, dev_dataloader, args.gpu_id)
    logger.info('start dev_metric = {}'.format(best_metric))

    t1 = time.time()
    step, accu_step = 0, 0
    prev_acc_loss, acc_loss = 0.0, 0.0
    cur_patience = 0
    model.zero_grad()
    for i_epoch in range(args.n_epochs):
        t2 = time.time()
        logger.info('========== Epoch {} =========='.format(i_epoch))

        epoch_has_better_model = False
        for batch in train_dataloader:
            model.train()
            # to GPU
            if args.gpu_id != -1:
                batch = to_gpu(batch, args.gpu_id, move_labels=True)

            # forward pass
            logits = model(**batch)

            loss = criterion(logits, batch['labels'])
            logger.info('loss = {}'.format(loss))

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
                if args.logging_steps > 0 and step % args.logging_steps == 0:
                    cur_loss = (acc_loss - prev_acc_loss) / args.logging_steps
                    logger.info('task={}, train_loss={}, accu_step={}, step={}, time={}s'.format(
                        args.task,
                        cur_loss,
                        accu_step,
                        step,
                        time.time()-t1)
                    )
                    # tb_writer.add_scalar('train_loss', cur_loss, step)
                    # tb_writer.add_scalar('lr', scheduler.get_last_lr()[0])

                    # evaluate
                    if not args.no_eval:
                        dev_metric = evaluate(
                            model, dev_dataloader, args.gpu_id)
                        logger.info('dev_metric={}'.format(dev_metric))
                        if best_metric < dev_metric:
                            best_metric = dev_metric

                            # save
                            utils.save_model(model, optimizer, scheduler, args.output_dir, step)
                            epoch_has_better_model = True
                    prev_acc_loss = acc_loss
        logger.info('done epoch {}: {} s'.format(i_epoch, time.time()-t2))
        if args.no_eval:
            # if no_eval we save every epoch
            utils.save_model(model, optimizer, scheduler, args.output_dir, step)
        else:
            if epoch_has_better_model:
                cur_patience = 0
            else:
                cur_patience += 1
            logger.info('cur patience={}'.format(cur_patience))
            if args.patience != -1 and cur_patience >= args.patience:
                logger.info('Stop Training!! reach patience={}'.format(cur_patience))
                break # break training
    logger.info('best_dev_metric = {}'.format(best_metric))
    logger.info('done training: {} s'.format(time.time() - t1))


def set_seed(gpu, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu != -1:
        torch.cuda.manual_seed_all(seed)


def load_dataset_inputs(prefix):
    node_info_f = os.path.join(args.input_dir, '{}_node_info.json'.format(prefix))
    input_f = os.path.join(args.input_dir, '{}_inputs.h5'.format(prefix))
    return input_f, node_info_f


def main():
    config = json.load(open(args.config_file))
    assert config['config_target'] == 'naive_psychology'
    set_seed(args.gpu_id, args.seed) # in distributed training, this has to be same for all processes

    if args.mode == 'train':
        train_inputs = load_dataset_inputs('train')
        dev_inputs = load_dataset_inputs('dev')
        train(train_inputs, dev_inputs, config)

        # test
        test_inputs = load_dataset_inputs('test')
        test(test_inputs, args.output_dir)
    elif args.mode == 'test': # test
        test_inputs = load_dataset_inputs('test')
        test(test_inputs, args.from_checkpoint)
    elif args.mode == 'test-dev': # test
        test_inputs = load_dataset_inputs('dev')
        test(test_inputs, args.from_checkpoint)
    elif args.mode == 'test-train': # test
        test_inputs = load_dataset_inputs('train')
        test(test_inputs, args.from_checkpoint)


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
