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

from englib.common import utils
from englib.models.naive_psych import SimpleLM, SimpleFFN
from englib.models.naive_psych import MASLOW_LABEL2IDX, REISS_LABEL2IDX, PLUTCHIK_LABEL2IDX


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='feature extraction for NaivePsych')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='naive psych config file.')
    parser.add_argument('input_dir', metavar='INPUT_DIR',
                        help='input directory.')
    parser.add_argument('mode', metavar='MODE', choices=['train', 'test'],
                        help='train or test')
    parser.add_argument('model_class', metavar='MODEL_CLASS',
                        choices=['SimpleLM', 'SimpleFFN'],
                        help='model class')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    parser.add_argument("--model_config", default=None, type=str,
                        help="model config")
    parser.add_argument("--task", default='multi', type=str,
                        choices=[
                            'multi',
                            'maslow',
                            'reiss',
                            'plutchik'
                        ],
                        help="task objective")

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
    parser.add_argument('--n_epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--train_batch_size', default=32, type=int,
                        help='training batch size')
    parser.add_argument('--eval_batch_size', default=32, type=int,
                        help='evaluating batch size')
    parser.add_argument('--from_checkpoint', default=None, type=str,
                        help='checkpoint directory')
    parser.add_argument('--no_first_eval', action='store_true', default=False,
                        help='no first_evaluation (train only)')
    parser.add_argument('--no_eval', action='store_true', default=False,
                        help='no evaluation (train only)')
    # parser.add_argument('--use_ng_features', action='store_true', default=False,
    #                     help='use ng features')

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


class SimpleFFNDataset(Dataset):
    def __init__(self, fpath, input_mask, examples):
        super(SimpleFFNDataset, self).__init__()
        self.examples = examples
        self.input_mask = input_mask
        self.fpath = fpath
        self.fp = None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.fp is None:
            # a trick to solve picklability issue with torch.multiprocessing
            self.fp = h5py.File(self.fpath, 'r')
        sid, sent_idxs, c_name, maslow_labels, reiss_labels, plutchik_labels = \
            self.examples[idx]

        word_emb = torch.from_numpy(self.fp[sid][:])
        input_mask = self.input_mask[sid]

        # encode labels
        y_maslow = [0] * len(MASLOW_LABEL2IDX)
        for l in maslow_labels:
            idx = MASLOW_LABEL2IDX[l]
            y_maslow[idx] = 1

        y_reiss = [0] * len(REISS_LABEL2IDX)
        for l in reiss_labels:
            idx = REISS_LABEL2IDX[l]
            y_reiss[idx] = 1

        y_plutchik = [0] * len(PLUTCHIK_LABEL2IDX)
        for l in plutchik_labels:
            idx = PLUTCHIK_LABEL2IDX[l]
            y_plutchik[idx] = 1
        return (word_emb,
                input_mask,
                torch.LongTensor(sent_idxs),
                torch.LongTensor(y_maslow),
                torch.LongTensor(y_reiss),
                torch.LongTensor(y_plutchik)
                )


class SimpleLMDataset(Dataset):
    def __init__(self, input_ids, input_mask, examples):
        super(SimpleLMDataset, self).__init__()
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sid, sent_idxs, c_name, maslow_labels, reiss_labels, plutchik_labels = \
            self.examples[idx]

        # encode labels
        y_maslow = [0] * len(MASLOW_LABEL2IDX)
        for l in maslow_labels:
            idx = MASLOW_LABEL2IDX[l]
            y_maslow[idx] = 1

        y_reiss = [0] * len(REISS_LABEL2IDX)
        for l in reiss_labels:
            idx = REISS_LABEL2IDX[l]
            y_reiss[idx] = 1

        y_plutchik = [0] * len(PLUTCHIK_LABEL2IDX)
        for l in plutchik_labels:
            idx = PLUTCHIK_LABEL2IDX[l]
            y_plutchik[idx] = 1
        return (self.input_ids[sid], self.input_mask[sid], torch.LongTensor(sent_idxs),
                torch.LongTensor(y_maslow), torch.LongTensor(y_reiss), torch.LongTensor(y_plutchik))


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
    all_tp = 0
    all_fp = 0
    all_fn = 0
    for i in range(y_true.shape[1]):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        all_tp += tp.item()
        all_fp += fp.item()
        all_fn += fn.item()

    prec = all_tp / (all_tp + all_fp) if all_tp + all_fp != 0 else 0.0
    recall = all_tp / (all_tp + all_fn) if all_tp + all_fn != 0 else 0.0
    f1 = (2 * prec * recall) / (prec + recall) if prec + recall != 0 else 0.0
    logger.info('{}: precision={}, recall={}, f1={}'.format(prefix, prec, recall, f1))
    return {'precision': prec, 'recall': recall, 'f1': f1}


def evaluate(model, dataloader, gpu):
    model.eval()

    all_y_maslow, all_y_reiss, all_y_plutchik = [], [], []
    all_pred_maslow, all_pred_reiss, all_pred_plutchik = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # to GPU
            if gpu != -1:
                batch = [f.cuda(gpu) for f in batch]
            # forward pass
            maslow_score, reiss_score, plutchik_score = model.predict(*batch)
            y_maslow, y_reiss, y_plutchik = batch[-3:]

            maslow_pred = (maslow_score >= 0.5).long()
            reiss_pred = (reiss_score >= 0.5).long()
            plutchik_pred = (plutchik_score >= 0.5).long()

            all_y_maslow.append(y_maslow)
            all_y_reiss.append(y_reiss)
            all_y_plutchik.append(y_plutchik)

            all_pred_maslow.append(maslow_pred)
            all_pred_reiss.append(reiss_pred)
            all_pred_plutchik.append(plutchik_pred)

        all_y_maslow = torch.cat(all_y_maslow, dim=0).cpu()
        all_y_reiss = torch.cat(all_y_reiss, dim=0).cpu()
        all_y_plutchik = torch.cat(all_y_plutchik, dim=0).cpu()
        all_pred_maslow = torch.cat(all_pred_maslow, dim=0).cpu()
        all_pred_reiss = torch.cat(all_pred_reiss, dim=0).cpu()
        all_pred_plutchik = torch.cat(all_pred_plutchik, dim=0).cpu()

    # micro-average
    if args.task == 'multi':
        maslow_result = micro_averaged_metrics(all_y_maslow, all_pred_maslow, 'maslow')
        reiss_result = micro_averaged_metrics(all_y_reiss, all_pred_reiss, 'reiss')
        plutchik_result = micro_averaged_metrics(all_y_plutchik, all_pred_plutchik, 'plutchik')
        # dev on avg
        return (maslow_result['f1'] + reiss_result['f1'] + plutchik_result['f1']) / 3.0
    elif args.task == 'maslow':
        maslow_result = micro_averaged_metrics(all_y_maslow, all_pred_maslow, 'maslow')
        return maslow_result['f1']
    elif args.task == 'reiss':
        reiss_result = micro_averaged_metrics(all_y_reiss, all_pred_reiss, 'reiss')
        return reiss_result['f1']
    elif args.task == 'plutchik':
        plutchik_result = micro_averaged_metrics(all_y_plutchik, all_pred_plutchik, 'plutchik')
        return plutchik_result['f1']
    else:
        raise ValueError('task {}'.format(args.task))


def test(test_examples, test_inputs, model_dir):
    if args.model_class == 'SimpleLM':
        test_dataset = SimpleLMDataset(test_inputs[0], test_inputs[1], test_examples)
    elif args.model_class == 'SimpleFFN':
        test_dataset = SimpleFFNDataset(test_inputs[0], test_inputs[1], test_examples)
    else:
        raise ValueError('{}'.format(args.model_class))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size, # fix 1 for sampling
        shuffle=False,
        num_workers=1) # 1 is safe for hdf5

    model = get_model(model_dir)
    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)

    test_metric = evaluate(model, test_dataloader, args.gpu_id)
    logger.info('test metric: {}'.format(test_metric))


def get_model(model_dir):
    ModelClass = eval(args.model_class)
    if model_dir is not None:
        target_dir = utils.get_target_model_dir(model_dir)
        logger.info('loading model from {}...'.format(target_dir))
        model = ModelClass.from_pretrained(target_dir)
    else:
        mconfig = json.load(open(args.model_config))
        model = ModelClass(**mconfig)
    return model


def train(train_examples, dev_examples, train_inputs, dev_inputs):
    if args.model_class == 'SimpleLM':
        train_dataset = SimpleLMDataset(train_inputs[0], train_inputs[1], train_examples)
        dev_dataset = SimpleLMDataset(dev_inputs[0], dev_inputs[1], dev_examples)
    elif args.model_class == 'SimpleFFN':
        train_dataset = SimpleFFNDataset(train_inputs[0], train_inputs[1], train_examples)
        dev_dataset = SimpleFFNDataset(dev_inputs[0], dev_inputs[1], dev_examples)
    else:
        raise ValueError('{}'.format(args.model_class))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size, # fix 1 for sampling
        shuffle=True,
        num_workers=1) # 1 is safe for hdf5

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size, # fix 1 for sampling
        shuffle=False,
        num_workers=1) # 1 is safe for hdf5

    # model
    # pos_weight = torch.FloatTensor([args.pos_weight])
    model = get_model(args.from_checkpoint)
    if args.gpu_id != -1:
        model = model.cuda(args.gpu_id)
        # pos_weight = pos_weight.cuda(args.gpu_id)

    optimizer = utils.get_optimizer(model, args.weight_decay, args.lr,
                                    args.adam_epsilon, args.from_checkpoint)
    scheduler = utils.get_scheduler(len(train_dataset), optimizer, args.train_batch_size,
                                    args.gradient_accumulation_steps, args.n_epochs,
                                    args.warmup_steps, args.warmup_portion,
                                    args.from_checkpoint)

    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = nn.BCEWithLogitsLoss()

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
    step = 0
    prev_acc_loss, acc_loss = 0.0, 0.0
    model.zero_grad()
    for i_epoch in range(args.n_epochs):
        t2 = time.time()
        logger.info('========== Epoch {} =========='.format(i_epoch))

        for train_batch in train_dataloader:
            model.train()
            if args.gpu_id != -1:
                train_batch = [f.cuda(args.gpu_id) for f in train_batch]

            y_maslow, y_reiss, y_plutchik = train_batch[-3:]
            maslow_logits, reiss_logits, plutchik_logits = model(*train_batch)

            if args.task == 'multi':
                maslow_loss = criterion(maslow_logits, y_maslow.float())
                reiss_loss = criterion(reiss_logits, y_reiss.float())
                plutchik_loss = criterion(plutchik_logits, y_plutchik.float())
                loss = maslow_loss + reiss_loss + plutchik_loss
            elif args.task == 'maslow':
                loss = criterion(maslow_logits, y_maslow.float())
            elif args.task == 'reiss':
                loss = criterion(reiss_logits, y_reiss.float())
            elif args.task == 'plutchik':
                loss = criterion(plutchik_logits, y_plutchik.float())

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
                        save_model(model, optimizer, scheduler, args.output_dir, step)
                else:
                    # simply save model
                    save_model(model, optimizer, scheduler, args.output_dir, step)
                prev_acc_loss = acc_loss

        logger.info('done epoch {}: {} s'.format(i_epoch, time.time()-t2))
    logger.info('best_dev_metric = {}'.format(best_metric))
    logger.info('done training: {} s'.format(time.time() - t1))


def load_examples(prefix):
    example_f = os.path.join(args.input_dir, '{}_examples.pkl'.format(prefix))
    examples = pkl.load(open(example_f, 'rb'))
    return examples


def load_dataset_inputs(prefix):
    input_mask_f = os.path.join(args.input_dir, '{}_input_mask.pkl'.format(prefix))
    input_mask = pkl.load(open(input_mask_f, 'rb'))
    input_mask = {k: torch.from_numpy(v) for k, v in input_mask.items()}
    if args.model_class == 'SimpleLM':
        input_ids_f = os.path.join(args.input_dir, '{}_input_ids.pkl'.format(prefix))
        input_ids = pkl.load(open(input_ids_f, 'rb'))
        input_ids = {k: torch.from_numpy(v) for k, v in input_ids.items()}
        return inpud_ids, input_mask
    elif args.model_class == 'SimpleFFN':
        word_emb_f = os.path.join(args.input_dir, '{}_word_embeddings.h5'.format(prefix))
        return word_emb_f, input_mask
    else:
        raise ValueError('{}'.format(args.model_class))


def main():
    config = json.load(open(args.config_file))
    assert config['config_target'] == 'naive_psychology'
    set_seed(args.gpu_id, args.seed) # in distributed training, this has to be same for all processes

    if args.mode == 'train':
        train_examples = load_examples('train')
        logger.info("train {}".format(len(train_examples)))
        train_inputs = load_dataset_inputs('train')

        dev_examples = load_examples('dev')
        logger.info("dev {}".format(len(dev_examples)))
        dev_inputs = load_dataset_inputs('dev')
        train(train_examples, dev_examples, train_inputs, dev_inputs)

        # test
        test_examples = load_examples('test')
        logger.info("test {}".format(len(test_examples)))
        inputs = load_dataset_inputs('test')
        test(test_examples, inputs, args.output_dir)
    else: # test
        test_examples = load_examples('test')
        logger.info("test {}".format(len(test_examples)))
        inputs = load_dataset_inputs('test')
        test(test_examples, inputs, args.from_checkpoint)


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
