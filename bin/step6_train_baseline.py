import os
import sys
import json
import time
import h5py
import random
import pickle as pkl
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from nltk.tokenize import sent_tokenize
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

from englib.common import utils
from englib.models.naive_psych import LabelCorelationModel, SelfAttentionModel
from englib.models.naive_psych import MASLOW_LABEL2IDX, REISS_LABEL2IDX, PLUTCHIK_LABEL2IDX


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='feature extraction for NaivePsych')
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
    parser.add_argument('mode', metavar='MODE', choices=['train', 'test'],
                        help='train or test')
    parser.add_argument('weight_name', metavar='WEIGHT_NAME',
                        choices=['bert-large-cased', 'roberta-large', 'bert-base-cased', 'roberta-base', 'elmo'],
                        help='model name')
    parser.add_argument('model_class', metavar='MODEL_CLASS',
                        choices=['LabelCorelationModel', 'SelfAttentionModel'],
                        help='model class')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    # parser.add_argument("--model_config", default=None, type=str,
    #                     help="model config")
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='torch cache')

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
    parser.add_argument('--train_batch_size', default=20, type=int,
                        help='training batch size')
    parser.add_argument('--eval_batch_size', default=32, type=int,
                        help='evaluating batch size')
    parser.add_argument('--from_checkpoint', default=None, type=str,
                        help='checkpoint directory')
    parser.add_argument('--no_first_eval', action='store_true', default=False,
                        help='no first_evaluation (train only)')
    parser.add_argument('--no_eval', action='store_true', default=False,
                        help='no evaluation (train only)')
    parser.add_argument('--not_use_lm', action='store_true', default=False,
                        help='not use LM')
    parser.add_argument('--no_label_corelation', action='store_true', default=False,
                        help='not use label corelation')
    parser.add_argument("--dropout", default=0.2, type=float,
                        help="dropout rate.")
    parser.add_argument("--cor_reg_param", default=0.01, type=float,
                        help="corelation reg param.")

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


class LabelCorelationModelDataset(Dataset):
    def __init__(self, input_f, task, use_lm, **kwargs):
        super(LabelCorelationModelDataset, self).__init__()
        assert task in ['maslow', 'reiss', 'plutchik']
        self.input_f = input_f
        self.task = task
        self.use_lm = use_lm
        self.fp = None

        fp = h5py.File(input_f, 'r')
        self.exids = sorted(list(fp.keys()))
        fp.close()

    def __len__(self):
        return len(self.exids)

    def __getitem__(self, idx):
        if self.fp is None:
            self.fp = h5py.File(self.input_f, 'r')

        exid = self.exids[idx]
        if self.use_lm:
            input_ids = torch.LongTensor(self.fp[exid][self.task]['input_ids'][:])
            attention_mask = torch.LongTensor(self.fp[exid][self.task]['attention_mask'][:])
            token_type_ids = torch.LongTensor(self.fp[exid][self.task]['token_type_ids'][:])
            labels = torch.FloatTensor(self.fp[exid][self.task]['labels'][:])
            return (exid, input_ids, attention_mask, token_type_ids, labels)

        # not use lm
        wemb = torch.FloatTensor(self.fp[exid][self.task]['word_embeddings'][:])
        attention_mask = torch.LongTensor(self.fp[exid][self.task]['attention_mask'][:])
        labels = torch.FloatTensor(self.fp[exid][self.task]['labels'][:])
        return exid, wemb, attention_mask, labels


class SelfAttentionModelDataset(Dataset):
    def __init__(self, input_f, task, **kwargs):
        self.input_f = input_f
        self.task = task
        self.fp = None

        fp = h5py.File(input_f, 'r')
        self.exids = sorted(list(fp.keys()))
        fp.close()

    def __len__(self):
        return len(self.exids)

    def __getitem__(self, idx):
        if self.fp is None:
            self.fp = h5py.File(self.input_f, 'r')

        exid = self.exids[idx]
        s_wemb = torch.FloatTensor(self.fp[exid]['sentence_word_embeddings'][:])
        # s_input_ids = torch.LongTensor(self.fp[exid]['sentence_input_ids'][:])
        s_attention_mask = torch.LongTensor(self.fp[exid]['sentence_attention_mask'][:])
        # s_token_type_ids = torch.LongTensor(self.fp[exid]['sentence_token_type_ids'][:])

        c_wemb = torch.FloatTensor(self.fp[exid]['context_word_embeddings'][:])
        # c_input_ids = torch.LongTensor(self.fp[exid]['context_input_ids'][:])
        c_attention_mask = torch.LongTensor(self.fp[exid]['context_attention_mask'][:])
        # c_token_type_ids = torch.LongTensor(self.fp[exid]['context_token_type_ids'][:])

        labels = torch.FloatTensor(self.fp[exid]['{}_labels'.format(self.task)][:])
        # return (exid, s_wemb, s_input_ids, s_attention_mask, s_token_type_ids, c_wemb, c_input_ids, c_attention_mask, c_token_type_ids, labels)
        return (exid, s_wemb, s_attention_mask, c_wemb, c_attention_mask, labels)


def label_corelation_collate(samples):
    if args.not_use_lm:
        exid, wemb, attention_mask, labels = map(list, zip(*samples))

        wemb = torch.stack(wemb, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)

        batch = {
            'exid': exid,
            'wemb': wemb,
            'attention_mask': attention_mask,
            'labels': labels
        }
    else:
        exid, input_ids, attention_mask, token_type_ids, labels = map(list, zip(*samples))

        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
        labels = torch.stack(labels, dim=0)

        batch = {
            'exid': exid,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels
        }
    return batch


def self_attention_collate(samples):
    exid, s_wemb, s_attention_mask, c_wemb, c_attention_mask, labels = map(list, zip(*samples))
    # exid, s_wemb, s_input_ids, s_attention_mask, s_token_type_ids, c_wemb, c_input_ids, c_attention_mask, c_token_type_ids, labels = map(list, zip(*samples))

    s_wemb = torch.stack(s_wemb, dim=0)
    # s_input_ids = torch.stack(s_input_ids, dim=0)
    s_attention_mask = torch.stack(s_attention_mask, dim=0)
    # s_token_type_ids = torch.stack(s_token_type_ids, dim=0)

    c_wemb = torch.stack(c_wemb, dim=0)
    # c_input_ids = torch.stack(c_input_ids, dim=0)
    c_attention_mask = torch.stack(c_attention_mask, dim=0)
    # c_token_type_ids = torch.stack(c_token_type_ids, dim=0)

    labels = torch.stack(labels, dim=0)

    batch = {
        'exid': exid,
        's_wemb': s_wemb,
        # 's_input_ids': s_input_ids,
        's_attention_mask': s_attention_mask,
        # 's_token_type_ids': s_token_type_ids,
        'c_wemb': c_wemb,
        # 'c_input_ids': c_input_ids,
        'c_attention_mask': c_attention_mask,
        # 'c_token_type_ids': c_token_type_ids,
        'labels': labels
    }
    return batch


def set_seed(gpu, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu != -1:
        torch.cuda.manual_seed_all(seed)


def save_model(model, optimizer, scheduler, output_dir, step):
    dir_path = 'best_model_{}'.format(step)
    model_to_save = model.module \
        if hasattr(model, 'module') \
        else model
    dir_path = os.path.join(output_dir, dir_path)
    model_to_save.save_pretrained(dir_path)
    logger.info('save model {}...'.format(dir_path))
    torch.save(optimizer.state_dict(),
               os.path.join(dir_path, "optimizer.pt"))
    torch.save(scheduler.state_dict(),
               os.path.join(dir_path, "scheduler.pt"))


def micro_averaged_metrics(y_true, y_pred, prefix):
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    logger.info('precision={}, recall={}, f1={}'.format(prec, recall, f1))
    return {'precision': prec, 'recall': recall, 'f1': f1}


def to_gpu(batch, gpu):
    if args.model_class == 'LabelCorelationModel':
        if 'wemb' in batch:
            batch['wemb'] = batch['wemb'].cuda(gpu)
        if 'input_ids' in batch:
            batch['input_ids'] = batch['input_ids'].cuda(gpu)
        batch['attention_mask'] = batch['attention_mask'].cuda(gpu)
        if 'token_type_ids' in batch:
            batch['token_type_ids'] = batch['token_type_ids'].cuda(gpu)
        batch['labels'] = batch['labels'].cuda(gpu)
    elif args.model_class == 'SelfAttentionModel':
        batch['s_wemb'] = batch['s_wemb'].cuda(gpu)
        batch['s_attention_mask'] = batch['c_attention_mask'].cuda(gpu)
        batch['c_wemb'] = batch['s_wemb'].cuda(gpu)
        batch['c_attention_mask'] = batch['c_attention_mask'].cuda(gpu)
        batch['labels'] = batch['labels'].cuda(gpu)
    return batch


def evaluate(model, dataloader, gpu):
    model.eval()

    all_ys, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # to GPU
            if gpu != -1:
                batch = to_gpu(batch, gpu)

            y = batch['labels']
            # forward pass
            score = model.predict(**batch)
            y_pred = (score >= 0.5).long()

            all_ys.append(y)
            all_preds.append(y_pred)

        all_ys = torch.cat(all_ys, dim=0).cpu()
        all_preds = torch.cat(all_preds, dim=0).cpu()

    # micro-average
    result = micro_averaged_metrics(all_ys, all_preds, args.task)
    return result['f1']


def get_dataloader(dinputs, batch_size, shuffle):
    DatasetClass = eval('{}Dataset'.format(args.model_class))
    dataset = DatasetClass(
        dinputs, args.task, use_lm=(not args.not_use_lm))
    if args.model_class == 'LabelCorelationModel':
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size, # fix 1 for sampling
            shuffle=shuffle,
            collate_fn=label_corelation_collate,
            num_workers=1) # 1 is safe for hdf5
    elif args.model_class == 'SelfAttentionModel':
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size, # fix 1 for sampling
            shuffle=shuffle,
            collate_fn=self_attention_collate,
            num_workers=1) # 1 is safe for hdf5
    return dataset, dataloader


def test(test_dinputs, model_dir):
    test_dataset, test_dataloader = get_dataloader(test_dinputs, args.eval_batch_size, False)

    model = get_model(model_dir)
    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)

    test_metric = evaluate(model, test_dataloader, args.gpu_id)
    logger.info('test metric: {}'.format(test_metric))


def get_model(model_dir):
    ModelClass = eval(args.model_class)
    if model_dir is not None:
        # we don't use the given model config here
        target_dir = utils.get_target_model_dir(model_dir)
        logger.info('loading model from {}...'.format(target_dir))
        model = ModelClass.from_pretrained(target_dir)
    else:
        if args.task == 'maslow':
            n_classes = len(MASLOW_LABEL2IDX)
        elif args.task == 'reiss':
            n_classes = len(REISS_LABEL2IDX)
        else:
            n_classes = len(PLUTCHIK_LABEL2IDX)

        lm_dim = 768 if 'base' in args.weight_name else 1024

        mconfig = {
            'lm_dim': lm_dim,
            'weight_name': args.weight_name,
            'cache_dir': args.cache_dir,
            'n_classes': n_classes,
            'dropout': args.dropout,
            'no_label_corelation': args.no_label_corelation,
            'use_lm': (not args.not_use_lm)
        }
        model = ModelClass(**mconfig)
    return model


def my_criterion(bce1, bce2, e, yp, y, reg):
    loss1 = bce1(e, y)
    loss2 = bce2(e, yp)
    # logger.info('loss1={}, loss2={}, reg={}'.format(loss1, loss2, reg))
    logger.info('loss1={}, loss2={}'.format(loss1, loss2))
    return loss1 + args.cor_reg_param * loss2


def train(train_dinputs, dev_dinputs):
    train_dataset, train_dataloader = get_dataloader(train_dinputs, args.train_batch_size, True)
    dev_dataset, dev_dataloader = get_dataloader(dev_dinputs, args.eval_batch_size, False)

    # model
    # pos_weight = torch.FloatTensor([args.pos_weight])
    model = get_model(args.from_checkpoint)
    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)
        # pos_weight = pos_weight.cuda(args.gpu_id)

    optimizer = utils.get_optimizer(model, args.weight_decay, args.lr,
                                    args.adam_epsilon, args.from_checkpoint)
    n = len(train_dataset)
    scheduler = utils.get_scheduler(n, optimizer, args.train_batch_size,
                                    args.gradient_accumulation_steps, args.n_epochs,
                                    args.warmup_steps, args.warmup_portion,
                                    args.from_checkpoint)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce1 = nn.BCEWithLogitsLoss()
    bce2 = nn.BCEWithLogitsLoss()

    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.n_epochs)
    logger.info("  Training batch size = %d", args.train_batch_size)
    logger.info("  Evaluation batch size = %d", args.eval_batch_size)
    logger.info("  Accu. train batch size = %d",
                    args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Weight Decay = {}".format(args.weight_decay))
    logger.info("  Learning Rate = {}".format(args.lr))

    if args.no_first_eval:
        best_metric = 0.0
    else:
        best_metric = evaluate(model, dev_dataloader, args.gpu_id)
    logger.info('start dev_metric = {}'.format(best_metric))

    t1 = time.time()
    tb_writer = SummaryWriter('{}/explog'.format(args.output_dir))
    step = 0
    prev_acc_loss, acc_loss = 0.0, 0.0
    model.zero_grad()
    for i_epoch in range(args.n_epochs):
        t2 = time.time()
        logger.info('========== Epoch {} =========='.format(i_epoch))

        for batch in train_dataloader:
            model.train()
            if args.gpu_id != -1:
                batch = to_gpu(batch, args.gpu_id)

            y = batch['labels']
            if args.model_class == 'LabelCorelationModel':
                e, yp, reg = model(**batch)
                # loss
                if yp is None: # no label corelation
                    loss = bce1(e, y)
                else:
                    loss = my_criterion(bce1, bce2, e, yp, y, reg)
            elif args.model_class == 'SelfAttentionModel':
                logits = model(**batch)
                loss = bce1(logits, y)

            # joint loss
            logger.info('loss = {}'.format(loss))

            # backward pass
            loss.backward()

            acc_loss += loss.item()
            # update params
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            step += 1

            # loss
            if args.logging_steps > 0 and step % args.logging_steps == 0:
                cur_loss = (acc_loss - prev_acc_loss) / args.logging_steps
                logger.info('train_loss={}, step={}, time={}s'.format(cur_loss,
                                                                      step,
                                                                      time.time()-t1))
                tb_writer.add_scalar('train_loss', cur_loss, step)
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

                # evaluate
                if not args.no_eval:
                    dev_metric = evaluate(
                        model, dev_dataloader, args.gpu_id)
                    logger.info('dev_metric={}'.format(dev_metric))
                    tb_writer.add_scalar('train_loss', dev_metric, step)
                    if best_metric < dev_metric:
                        best_metric = dev_metric

                        # save
                        save_model(model, optimizer, scheduler, args.output_dir, step)
                else:
                    # simply save model
                    save_model(model, optimizer, scheduler, args.output_dir, step)
                prev_acc_loss = acc_loss

        logger.info('done epoch {}: {} s'.format(i_epoch, time.time()-t2))
    logger.info('best_dev_metric = {}'.format(best_metric))
    logger.info('done training: {} s'.format(time.time() - t1))
    tb_writer.close()


def load_dataset_inputs(prefix):
    input_f = os.path.join(args.input_dir, '{}_inputs.h5'.format(prefix))
    return input_f


def main():
    config = json.load(open(args.config_file))
    assert config['config_target'] == 'naive_psychology'
    set_seed(args.gpu_id, args.seed) # in distributed training, this has to be same for all processes

    if args.mode == 'train':
        train_dinputs = load_dataset_inputs('train')
        dev_dinputs = load_dataset_inputs('dev')
        train(train_dinputs, dev_dinputs)

        # test
        test_dinputs = load_dataset_inputs('test')
        test(test_dinputs, args.output_dir)
    else: # test
        test_dinputs = load_dataset_inputs('test')
        test(test_dinputs, args.from_checkpoint)


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
