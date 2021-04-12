import os
import sys
import json
import time
import h5py
import random
import pickle as pkl
import argparse
from copy import deepcopy

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
from englib.models.naive_psych import RGCNNaivePsychology
from englib.models.naive_psych import MASLOW_LABEL2IDX, REISS_LABEL2IDX, PLUTCHIK_LABEL2IDX
from englib.models.naive_psych import SENTIMENT_LABEL2IDX


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='train PNG for NaivePsych')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='naive psych config file.')
    parser.add_argument('unsupervised_input_dir', metavar='UNSUPERVISED_INPUT_DIR',
                        help='input directory.')
    parser.add_argument('task', metavar='TASK', choices=['maslow', 'reiss', 'plutchik'],
                        help='task')
    parser.add_argument('weight_name', metavar='WEIGHT_NAME',
                        choices=['bert-large-cased', 'roberta-large', 'bert-base-cased', 'roberta-base'],
                        help='model name')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

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
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--train_batch_size', default=1, type=int,
                        help='training batch size')
    # parser.add_argument('--eval_batch_size', default=4, type=int,
    #                     help='evaluating batch size')
    parser.add_argument('--from_checkpoint', default=None, type=str,
                        help='checkpoint directory')
    parser.add_argument('--freeze_lm', action='store_true', default=False,
                        help='freeze LM')
    parser.add_argument('--multi_gpus', action='store_true', default=False,
                        help='pararell gpus')
    parser.add_argument('--use_pos_weights', action='store_true', default=False,
                        help='use pos weights')
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
    def __init__(self, input_f, task):
        super(PNGPretrainDataset, self).__init__()
        self.input_f = input_f
        self.task = task
        self.fp = None

        # logger.info('loading {}...'.format(self.sample_f))
        fp = h5py.File(input_f, 'r')
        self.sids = sorted(list(fp.keys()))
        fp.close()

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, idx):
        # a trick to solve picklability issue with torch.multiprocessing
        if self.fp is None:
            self.fp = h5py.File(self.input_f, 'r')

        sid = self.sids[idx]
        input_ids = torch.from_numpy(self.fp[sid][self.task]['input_ids'][:])
        attention_mask = torch.from_numpy(self.fp[sid][self.task]['input_mask'][:])
        token_type_ids = torch.from_numpy(self.fp[sid][self.task]['token_type_ids'][:])

        edge_src = torch.from_numpy(self.fp[sid]['edge_src'][:])
        edge_dest = torch.from_numpy(self.fp[sid]['edge_dest'][:])
        edge_types = torch.from_numpy(self.fp[sid]['edge_types'][:])
        edge_norms = torch.from_numpy(self.fp[sid]['edge_norms'][:])

        labels = torch.FloatTensor(self.fp[sid]['sentiment'][:])
        label_idxs = torch.argmax(labels, dim=1)
        # labels = labels.zero_().scatter_(1, label_idxs.unsqueeze(1), 1.0)

        # in case there is only one node
        if len(labels.shape) == 1:
            labels = labels.unsquzze(0)

        return (input_ids, attention_mask, token_type_ids,
                edge_src, edge_dest, edge_types, edge_norms,
                label_idxs)


def my_train_collate(samples):
    input_ids, attention_mask, token_type_ids, edge_src, edge_dest, edge_types, edge_norms, \
        labels = map(list, zip(*samples))

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
        'labels': labels
    }
    return batch


def to_gpu(batch, gpu_id, move_labels):
    batch['input_ids'] = [im.cuda(gpu_id) for im in batch['input_ids']]
    batch['attention_mask'] = [im.cuda(gpu_id) for im in batch['attention_mask']]
    batch['token_type_ids'] = [im.cuda(gpu_id) for im in batch['token_type_ids']]
    batch['edge_types'] = [et.cuda(gpu_id) for et in batch['edge_types']]
    batch['edge_norms'] = [en.cuda(gpu_id) for en in batch['edge_norms']]
    if move_labels:
        batch['labels'] = batch['labels'].cuda(gpu_id)
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


def get_model(model_dir, weight_name, freeze_lm):
    if model_dir is not None:
        target_dir = utils.get_target_model_dir(model_dir)
        logger.info('loading model from {}...'.format(target_dir))
        model = RGCNNaivePsychology.from_pretrained(target_dir)
    else:
        n_classes = len(SENTIMENT_LABEL2IDX)
        mconfig = {
            'weight_name': weight_name,
            'n_classes': n_classes,
            'dropout': 0.2,
            # 'use_lm': True,
            'freeze_lm': freeze_lm,
            'n_hidden_layers': 2,
            'n_rtypes': 8, # relation type
            'reg_param': 0.0,
            'use_gate': False,
            'use_rgcn': True
        }
        model = RGCNNaivePsychology(**mconfig)
    return model


def get_pos_weights(dataset):
    # counts = [0] * len(SENTIMENT_LABEL2IDX)
    # for i in range(len(dataset)):
    #     ex = dataset[i]
    #     label_idxs = ex[7].tolist()
    #     for idx in label_idxs:
    #         counts[idx] += 1
    counts = [849716, 22343, 10325]
    m = min(counts)
    weight = torch.FloatTensor([m/float(c) for c in counts])
    return weight


def train(rank, train_inputs, config, args):
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

    # train dataset
    train_dataset = PNGPretrainDataset(train_inputs[0], args.task)
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
    model = get_model(args.from_checkpoint, args.weight_name, args.freeze_lm)
    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)
    if args.n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[args.gpu_id],
                                        find_unused_parameters=True)

    if args.use_pos_weights:
        pos_weights = get_pos_weights(train_dataset)
        if args.gpu_id != -1:
            pos_weights = pos_weights.cuda(args.gpu_id)
        logger.info('pos_weights = {}'.format(pos_weights))
        criterion = nn.CrossEntropyLoss(reduction='mean', weight=pos_weights)
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')

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
        # logger.info("  Evaluation batch size = %d", args.eval_batch_size)
        logger.info("  Accu. train batch size = %d",
                        args.train_batch_size * args.gradient_accumulation_steps)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Weight Decay = {}".format(args.weight_decay))
        logger.info("  Learning Rate = {}".format(args.lr))

    tb_writer = SummaryWriter('{}/explog'.format(args.output_dir))
    t1 = time.time()
    step, accu_step = 0, 0
    prev_acc_loss, acc_loss = 0.0, 0.0
    model.zero_grad()
    for i_epoch in range(args.n_epochs):
        t2 = time.time()
        logger.info('========== Epoch {} =========='.format(i_epoch))

        for batch in train_dataloader:
            model.train()
            # to GPU
            if args.gpu_id != -1:
                batch = to_gpu(batch, args.gpu_id, True)

            # forward pass
            logits = model(**batch)

            loss = criterion(logits, batch['labels'])
            # logger.info('accu={}, loss={}'.format(accu_step, loss))
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

                        prev_acc_loss = acc_loss

        logger.info('done epoch {}: {} s'.format(i_epoch, time.time()-t2))
        if local_rank in [-1, 0]:
            logger.info('saving model for epoch {}'.format(i_epoch))
            utils.save_model(model, optimizer, scheduler, args.output_dir, step)
    if local_rank in [-1, 0]:
        tb_writer.close()
    logger.info('done training: {} s'.format(time.time() - t1))


def set_seed(gpu, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu != -1:
        torch.cuda.manual_seed_all(seed)


def get_dataset_inputs(prefix, args):
    input_f = os.path.join(args.unsupervised_input_dir, '{}_inputs.h5'.format(prefix))
    return (input_f, )


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
    train_inputs = get_dataset_inputs('unsupervised_sentiment', args)
    if args.n_gpus > 1:
        mp.spawn(train, nprocs=args.n_gpus, args=(train_inputs, config, args))
    else:
        train(args.gpu_id, train_inputs, config, args)


if __name__ == "__main__":
    main()
