import os
import sys
import json
import logging
import argparse

import torch
from torch.optim import Adam
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel


logger = logging.getLogger(__name__)


def load_lined_json(fpath, key):
    js = {}
    with open(fpath, 'r') as fr:
        for line in fr:
            data = json.loads(line)
            js[data[key]] = data
    return js


def load_kfolds(fpath):
    kfolds = []
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.strip().rstrip('\n')
            kfolds.append(line.split(' '))
    return kfolds


def bin_config(get_arg_func):
    # get arguments
    args = get_arg_func(sys.argv[1:])

    # create output folder if not exist
    if hasattr(args, 'output_dir'):
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
    elif hasattr(args, 'output_folder'):
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)
    return args


def get_root_logger(args, log_fname=None):
    # set logger
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(levelname)-6s[%(name)s][%(filename)s:%(lineno)d] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    log_dir = './'
    if hasattr(args, 'output_dir'):
        log_dir = args.output_dir
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
    elif hasattr(args, 'output_folder'):
        log_dir = args.output_folder
        if not os.path.isdir(args.output_folder):
            os.mkdir(args.output_folder)

    fpath = os.path.join(log_dir, log_fname) if log_fname \
            else os.path.join(log_dir, 'log')

    fileHandler = logging.FileHandler(fpath)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger


def _bert_event_input(sent_words, tok_start_idx, tok_end_idx, tokenizer):
    bert_tokens, event_idxs = [], []
    for i_w, w in enumerate(sent_words):
        wtoks = tokenizer.tokenize(w)
        for wt in wtoks:
            if i_w >= tok_start_idx and i_w < tok_end_idx:
                event_idxs.append(len(bert_tokens))
            bert_tokens.append(wt)

    input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    return input_ids, event_idxs


def prepare_bert_event_input(words, pred_tidx, bert_tokenizer,
                             max_seq_len, padding_token_id=0,
                             pred_tidx_end=None):
    '''
    params:
        pred_tidx_end: sometimes we have to mark a span with 1 in token_type_ids
        max_seq_len: integer; -1 if no limit.
    '''
    if pred_tidx_end is None:
        pred_tidx_end = pred_tidx + 1
    input_ids, event_idxs = _bert_event_input(
        words, pred_tidx, pred_tidx_end, bert_tokenizer)

    if max_seq_len != -1 and event_idxs[-1] >= max_seq_len-2:
        # the event appears in the tail of a long sentence
        # try increase the max_seq_len or drop it
        return None, None, None, None

    if max_seq_len != -1 and len(input_ids) > max_seq_len-2:
        input_ids = input_ids[:max_seq_len-2]

    # insert CLS, SEP
    cls_ids, sep_ids = bert_tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
    # bert_tokenizer.cls_token, bert_tokenizer.sep_token
    # bert_tokenizer.cls_token_id, bert_tokenizer.sep_token_id
    input_ids = [cls_ids] + input_ids + [sep_ids]
    event_idxs = [ei+1 for ei in event_idxs] # shift for CLS

    # padding and
    # input mask to avoid attention on padding tokens
    input_mask = [1] * len(input_ids)
    if max_seq_len != -1 and len(input_ids) < max_seq_len:
        input_ids += [padding_token_id] * (max_seq_len - len(input_ids))
        input_mask += ([0] * (max_seq_len - len(input_mask)))

    # tokent type ids
    token_type_ids = [0] * len(input_ids)
    for idx in event_idxs:
        token_type_ids[idx] = 1
    return input_ids, input_mask, token_type_ids, event_idxs


def get_target_model_dir(model_dir):
    model_epochs = [int(d[11:]) for d in os.listdir(model_dir) if d.startswith('best_model_')]
    if len(model_epochs) == 0:
        # load directly
        target_model_dir = model_dir
    else: # load best model
        best_epoch = max(model_epochs)
        target_model_dir = os.path.join(model_dir, 'best_model_{}'.format(best_epoch))
    return target_model_dir


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


def get_optimizer(model, weight_decay, lr, adam_epsilon, from_checkpoint):
    # default adamw
    return get_optimizer_adamw(model, weight_decay, lr, adam_epsilon, from_checkpoint)


def get_optimizer_adamw(model, weight_decay, lr, adam_epsilon, from_checkpoint):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay},
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    if from_checkpoint:
        target_dir = get_target_model_dir(from_checkpoint)
        fpath = os.path.join(target_dir, 'optimizer.pt')
        if os.path.isfile(fpath):
            logger.info('loading optimizer from {}...'.format(fpath))
            optimizer.load_state_dict(torch.load(fpath,
                                                 map_location='cpu'))
    return optimizer


def get_optimizer_adam(model, weight_decay, lr, adam_epsilon, from_checkpoint, adam_betas):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay},
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
            ]
    optimizer = Adam(model.parameters(), lr=lr, eps=adam_epsilon, betas=adam_betas)
    if from_checkpoint:
        target_dir = get_target_model_dir(from_checkpoint)
        fpath = os.path.join(target_dir, 'optimizer.pt')
        if os.path.isfile(fpath):
            logger.info('loading optimizer from {}...'.format(fpath))
            optimizer.load_state_dict(torch.load(fpath,
                                                 map_location='cpu'))
    return optimizer


def get_scheduler(n_instances, optimizer,
                  train_batch_size, gradient_accumulation_steps,
                  n_epochs, warmup_steps, warmup_portion,
                  from_checkpoint):
    t_total = (n_instances // (train_batch_size * 1)
               // gradient_accumulation_steps) * n_epochs
    if warmup_portion > 0: # overwrte warmup_steps
        warmup_steps = int(t_total * warmup_portion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", warmup_steps)
    if from_checkpoint:
        target_dir = get_target_model_dir(from_checkpoint)
        fpath = os.path.join(target_dir, 'scheduler.pt')
        if os.path.isfile(fpath):
            logger.info('loading scheduler from {}...'.format(fpath))
            scheduler.load_state_dict(torch.load(fpath,
                                                 map_location='cpu'))
    return scheduler


def load_from_unsupervised_pretraining(model, model_dir, ignore_classifier=False, is_lm=False):
    # load partial model
    if is_lm:
        model.lm = AutoModel.from_pretrained(model_dir)
    else:
        target_dir = get_target_model_dir(model_dir)
        fpath = os.path.join(target_dir, 'model.pt')
        logger.info('loading model from {}...'.format(fpath))
        sd = torch.load(fpath, map_location='cpu')
        if ignore_classifier:
            keys = [k for k in sd if k.startswith('classifier')]
            for k in keys:
                del sd[k]
        model.load_state_dict(sd,
                              strict=False)
    return model
