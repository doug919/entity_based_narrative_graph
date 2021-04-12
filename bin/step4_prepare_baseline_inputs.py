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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import sent_tokenize
from transformers import AutoModel
from allennlp.modules.elmo import Elmo, batch_to_ids
from sacremoses import MosesTokenizer

from englib.common import utils
from englib.corpora import NaivePsychology
from englib.models.naive_psych import MASLOW_LABEL2IDX, REISS_LABEL2IDX, PLUTCHIK_LABEL2IDX
from englib.models.naive_psych import MASLOW_LABEL_SENTENCES
from englib.models.naive_psych import REISS_LABEL_SENTENCES
from englib.models.naive_psych import PLUTCHIK_LABEL_SENTENCES
from englib.models.naive_psych import encode_labels
from englib.models.naive_psych import get_label_sentences
from englib.narrative.psychology_narrative_graph import get_majority_labels
from englib.narrative.psychology_narrative_graph import get_context_sentence_list


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='feature extraction for NaivePsychology (example-based)')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='input config')
    parser.add_argument('class_name', metavar='CLASS_NAME',
                        choices=['LabelCorelationModel', 'SelfAttentionModel'],
                        help='model name')
    parser.add_argument('weight_name', metavar='WEIGHT_NAME',
                        choices=['bert-large-cased', 'roberta-large', 'bert-base-cased', 'roberta-base', 'elmo'],
                        help='model name')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    # parser.add_argument('--is_pretrain', action='store_true', default=False,
    #                     help='parse pretraining corpus')
    parser.add_argument('--max_seq_len', type=int, default=320,
                        help='BERT max sequence length')
    parser.add_argument('--seed', type=int, default=135,
                        help='seed for random')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='torch cache')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='gpu id')
    parser.add_argument('--no_embeddings', action='store_true', default=False,
                        help='no LM embeddings')
    parser.add_argument('--use_count_plutchik', action='store_true', default=False,
                        help='use count as labeling criteria for plutchik')
    parser.add_argument('--add_label_sent', action='store_true', default=False,
                        help='add label sentences')
    parser.add_argument('--elmo_weight_file', type=str, default='elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5',
                        help='ELMo weight file')
    parser.add_argument('--elmo_option_file', type=str, default='elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json',
                        help='ELMo option file')
    parser.add_argument('--no_paddings', action='store_true', default=False,
                        help='no paddings')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def process_elmo_word_embeddings(sent_num, sent, c_name, c_ann, context, fw_h5, tokenizer, lm, sid, msl):
    s_toks = tokenizer.tokenize(sent)
    c_toks = tokenizer.tokenize(context)

    if len(s_toks) > msl:
        msl = len(s_toks)
    if len(c_toks) > msl:
        msl = len(c_toks)

    ids = batch_to_ids([s_toks, c_toks])
    if args.gpu_id != -1:
        ids = ids.cuda(args.gpu_id)

    maslow_labels, reiss_labels, plutchik_labels = \
        get_majority_labels(c_ann, use_count_plutchik=args.use_count_plutchik)
    y_maslow, y_reiss, y_plutchik = encode_labels(maslow_labels, reiss_labels, plutchik_labels)

    exid = '{}_{}_{}'.format(sid, sent_num, c_name)
    if not args.no_embeddings:

        with torch.no_grad():
            out = lm(ids)
            wemb = out['elmo_representations'][0].cpu()
        n_paddings = args.max_seq_len - wemb.shape[1]
        padding = torch.zeros((wemb.shape[0], n_paddings, wemb.shape[2]), dtype=torch.float32)
        wemb = torch.cat((wemb, padding), dim=1).numpy()
        fw_h5.create_dataset('{}/sentence_word_embeddings'.format(exid), data=wemb[0])
        fw_h5.create_dataset('{}/context_word_embeddings'.format(exid), data=wemb[1])

    s_attention_mask = [1] * len(s_toks)
    if len(s_attention_mask) < args.max_seq_len:
        s_attention_mask += [0] * (args.max_seq_len - len(s_attention_mask))

    c_attention_mask = [1] * len(c_toks)
    if len(c_attention_mask) < args.max_seq_len:
        c_attention_mask += [0] * (args.max_seq_len - len(c_attention_mask))

    fw_h5.create_dataset('{}/sentence_attention_mask'.format(exid),
                         data=s_attention_mask)

    fw_h5.create_dataset('{}/context_attention_mask'.format(exid),
                         data=c_attention_mask)

    fw_h5.create_dataset('{}/maslow_labels'.format(exid),
                         data=y_maslow.numpy())
    fw_h5.create_dataset('{}/reiss_labels'.format(exid),
                         data=y_reiss.numpy())
    fw_h5.create_dataset('{}/plutchik_labels'.format(exid),
                         data=y_plutchik.numpy())
    return msl


def process_lm_word_embeddings(sent_num, sent, c_name, c_ann, context, fw_h5, tokenizer, lm, sid, msl):
    # note that we put c_name in the second sentence
    if args.no_paddings:
        s_inputs = tokenizer(
            sent, c_name, add_special_tokens=True,
            # max_length=args.max_seq_len, padding='max_length', truncation='only_second',
            return_token_type_ids=True, return_attention_mask=True)

        c_inputs = tokenizer(
            context, add_special_tokens=True,
            # max_length=args.max_seq_len, padding='max_length', truncation='only_second',
            return_token_type_ids=True, return_attention_mask=True)
    else:
        s_inputs = tokenizer(
            sent, c_name, add_special_tokens=True,
            max_length=args.max_seq_len, padding='max_length', truncation='only_second',
            return_token_type_ids=True, return_attention_mask=True)

        c_inputs = tokenizer(
            context, add_special_tokens=True,
            max_length=args.max_seq_len, padding='max_length', truncation='only_second',
            return_token_type_ids=True, return_attention_mask=True)

    if len(s_inputs['input_ids']) > msl:
        msl = len(s_inputs['input_ids'])
    if len(c_inputs['input_ids']) > msl:
        msl = len(c_inputs['input_ids'])

    maslow_labels, reiss_labels, plutchik_labels = \
        get_majority_labels(c_ann, use_count_plutchik=args.use_count_plutchik)
    y_maslow, y_reiss, y_plutchik = encode_labels(maslow_labels, reiss_labels, plutchik_labels)

    exid = '{}_{}_{}'.format(sid, sent_num, c_name)
    if not args.no_embeddings:
        input_ids = torch.LongTensor([s_inputs['input_ids'], c_inputs['input_ids']])
        attention_mask = torch.LongTensor([s_inputs['attention_mask'], c_inputs['attention_mask']])
        token_type_ids = torch.LongTensor([s_inputs['token_type_ids'], c_inputs['token_type_ids']])
        if args.gpu_id != -1:
            input_ids = input_ids.cuda(args.gpu_id)
            attention_mask = attention_mask.cuda(args.gpu_id)
            token_type_ids = token_type_ids.cuda(args.gpu_id)

        with torch.no_grad():
            outputs = lm(input_ids, attention_mask, token_type_ids=token_type_ids)
            wemb = outputs[0].cpu().numpy()
        fw_h5.create_dataset('{}/sentence_word_embeddings'.format(exid), data=wemb[0])
        fw_h5.create_dataset('{}/context_word_embeddings'.format(exid), data=wemb[1])

    fw_h5.create_dataset('{}/sentence_input_ids'.format(exid),
                         data=s_inputs['input_ids'])
    fw_h5.create_dataset('{}/sentence_attention_mask'.format(exid),
                         data=s_inputs['attention_mask'])
    fw_h5.create_dataset('{}/sentence_token_type_ids'.format(exid),
                         data=s_inputs['token_type_ids'])

    fw_h5.create_dataset('{}/context_input_ids'.format(exid),
                         data=c_inputs['input_ids'])
    fw_h5.create_dataset('{}/context_attention_mask'.format(exid),
                         data=c_inputs['attention_mask'])
    fw_h5.create_dataset('{}/context_token_type_ids'.format(exid),
                         data=c_inputs['token_type_ids'])

    fw_h5.create_dataset('{}/maslow_labels'.format(exid),
                         data=y_maslow.numpy())
    fw_h5.create_dataset('{}/reiss_labels'.format(exid),
                         data=y_reiss.numpy())
    fw_h5.create_dataset('{}/plutchik_labels'.format(exid),
                         data=y_plutchik.numpy())
    return msl


def process_self_attention_inputs(
        fw_h5, tokenizer, lm, sid, doc):

    msl = -1
    for sent_num in range(1, 6):

        sent = doc['lines'][str(sent_num)]['text']

        chars = doc['lines'][str(sent_num)]['characters']
        for c_name, c_ann in chars.items():
            if not c_ann['app']:
                continue

            context_texts = get_context_sentence_list(sent_num, doc, c_name)
            context = ' '.join(context_texts) if len(context_texts) > 0 else 'No context.'

            if args.weight_name == 'elmo':
                msl = process_elmo_word_embeddings(sent_num, sent, c_name, c_ann, context, fw_h5, tokenizer, lm, sid, msl)
            else:
                msl = process_lm_word_embeddings(sent_num, sent, c_name, c_ann, context, fw_h5, tokenizer, lm, sid, msl)
    return msl


def process_label_corelation_inputs(
        fw_h5, tokenizer, lm, sid, doc):

    maslow_idx2label = {v: k for k, v in MASLOW_LABEL2IDX.items()}
    reiss_idx2label = {v: k for k, v in REISS_LABEL2IDX.items()}
    plutchik_idx2label = {v: k for k, v in PLUTCHIK_LABEL2IDX.items()}

    msl = -1
    for sent_num in range(1, 6):

        sent = doc['lines'][str(sent_num)]['text']

        chars = doc['lines'][str(sent_num)]['characters']
        for c_name, c_ann in chars.items():
            if not c_ann['app']:
                continue

            context_texts = get_context_sentence_list(sent_num, doc, c_name)
            context = ' '.join(context_texts) if len(context_texts) > 0 else 'No context.'

            if args.add_label_sent:
                maslow_sents, reiss_sents, plutchik_sents = get_label_sentences(
                    c_name,
                    maslow_idx2label,
                    reiss_idx2label,
                    plutchik_idx2label
                )

                maslow_context = context + ' ' + ' '.join(maslow_sents)
                reiss_context = context + ' ' + ' '.join(reiss_sents)
                plutchik_context = context + ' ' + ' '.join(plutchik_sents)

                if args.no_paddings:
                    maslow_inputs = tokenizer(sent, maslow_context, add_special_tokens=True,
                                       # max_length=args.max_seq_len, padding='max_length', truncation='only_second',
                                       return_token_type_ids=True, return_attention_mask=True)
                    reiss_inputs = tokenizer(sent, reiss_context, add_special_tokens=True,
                                       # max_length=args.max_seq_len, padding='max_length', truncation='only_second',
                                       return_token_type_ids=True, return_attention_mask=True)
                    plutchik_inputs = tokenizer(sent, plutchik_context, add_special_tokens=True,
                                       # max_length=args.max_seq_len, padding='max_length', truncation='only_second',
                                       return_token_type_ids=True, return_attention_mask=True)
                else:
                    maslow_inputs = tokenizer(sent, maslow_context, add_special_tokens=True,
                                       max_length=args.max_seq_len, padding='max_length', truncation='only_second',
                                       return_token_type_ids=True, return_attention_mask=True)
                    reiss_inputs = tokenizer(sent, reiss_context, add_special_tokens=True,
                                       max_length=args.max_seq_len, padding='max_length', truncation='only_second',
                                       return_token_type_ids=True, return_attention_mask=True)
                    plutchik_inputs = tokenizer(sent, plutchik_context, add_special_tokens=True,
                                       max_length=args.max_seq_len, padding='max_length', truncation='only_second',
                                       return_token_type_ids=True, return_attention_mask=True)

            else:
                if args.no_paddings:
                    maslow_inputs = tokenizer(sent, context, add_special_tokens=True,
                                       # max_length=args.max_seq_len, padding='max_length', truncation='only_second',
                                       return_token_type_ids=True, return_attention_mask=True)
                else:
                    maslow_inputs = tokenizer(sent, context, add_special_tokens=True,
                                       max_length=args.max_seq_len, padding='max_length', truncation='only_second',
                                       return_token_type_ids=True, return_attention_mask=True)
                reiss_inputs = maslow_inputs
                plutchik_inputs = maslow_inputs

            if len(maslow_inputs['input_ids']) > msl:
                msl = len(maslow_inputs['input_ids'])
            if len(reiss_inputs['input_ids']) > msl:
                msl = len(reiss_inputs['input_ids'])
            if len(plutchik_inputs['input_ids']) > msl:
                msl = len(plutchik_inputs['input_ids'])

            maslow_labels, reiss_labels, plutchik_labels = \
                get_majority_labels(c_ann, use_count_plutchik=args.use_count_plutchik)

            y_maslow, y_reiss, y_plutchik = encode_labels(maslow_labels, reiss_labels, plutchik_labels)

            exid = '{}_{}_{}'.format(sid, sent_num, c_name)
            if not args.no_embeddings:
                input_ids = torch.LongTensor([maslow_inputs['input_ids'], reiss_inputs['input_ids'], plutchik_inputs['input_ids']])
                attention_mask = torch.LongTensor([maslow_inputs['attention_mask'], reiss_inputs['attention_mask'], plutchik_inputs['attention_mask']])
                token_type_ids = torch.LongTensor([maslow_inputs['token_type_ids'], reiss_inputs['token_type_ids'], plutchik_inputs['token_type_ids']])
                if args.gpu_id != -1:
                    input_ids = input_ids.cuda(args.gpu_id)
                    attention_mask = attention_mask.cuda(args.gpu_id)
                    token_type_ids = token_type_ids.cuda(args.gpu_id)

                with torch.no_grad():
                    outputs = lm(input_ids, attention_mask, token_type_ids=token_type_ids)
                    wemb = outputs[0].cpu().numpy()
                    fw_h5.create_dataset('{}/maslow/word_embeddings'.format(exid), data=wemb[0])
                    fw_h5.create_dataset('{}/reiss/word_embeddings'.format(exid), data=wemb[1])
                    fw_h5.create_dataset('{}/plutchik/word_embeddings'.format(exid), data=wemb[2])

            fw_h5.create_dataset('{}/maslow/input_ids'.format(exid),
                                 data=maslow_inputs['input_ids'])
            fw_h5.create_dataset('{}/maslow/attention_mask'.format(exid),
                                 data=maslow_inputs['attention_mask'])
            fw_h5.create_dataset('{}/maslow/token_type_ids'.format(exid),
                                 data=maslow_inputs['token_type_ids'])
            fw_h5.create_dataset('{}/maslow/labels'.format(exid),
                                 data=y_maslow.numpy())

            fw_h5.create_dataset('{}/reiss/input_ids'.format(exid),
                                 data=reiss_inputs['input_ids'])
            fw_h5.create_dataset('{}/reiss/attention_mask'.format(exid),
                                 data=reiss_inputs['attention_mask'])
            fw_h5.create_dataset('{}/reiss/token_type_ids'.format(exid),
                                 data=reiss_inputs['token_type_ids'])
            fw_h5.create_dataset('{}/reiss/labels'.format(exid),
                                 data=y_reiss.numpy())

            fw_h5.create_dataset('{}/plutchik/input_ids'.format(exid),
                                 data=plutchik_inputs['input_ids'])
            fw_h5.create_dataset('{}/plutchik/attention_mask'.format(exid),
                                 data=plutchik_inputs['attention_mask'])
            fw_h5.create_dataset('{}/plutchik/token_type_ids'.format(exid),
                                 data=plutchik_inputs['token_type_ids'])
            fw_h5.create_dataset('{}/plutchik/labels'.format(exid),
                                 data=y_plutchik.numpy())
    return msl


def process_split(gen, tokenizer, lm, prefix, target_sids):
    _max_seq_len = -1

    fpath = os.path.join(args.output_dir, '{}_inputs.h5'.format(prefix))
    fw_h5 = h5py.File(fpath, 'w')
    for sid, doc in tqdm(gen()):
        if sid not in target_sids:
            continue

        if args.class_name == 'LabelCorelationModel':
            msl = process_label_corelation_inputs(
                fw_h5, tokenizer, lm, sid, doc)
        elif args.class_name == 'SelfAttentionModel':
            msl = process_self_attention_inputs(
                fw_h5, tokenizer, lm, sid, doc)
        else:
            raise ValueError('unsupported {}'.format(args.class_name))

        if msl > _max_seq_len:
            _max_seq_len = msl

    fw_h5.close()
    logger.info('{} max_seq_len={}'.format(prefix, _max_seq_len))


def load_parses(parse_dir, prefix):
    parses = {}
    fpath = os.path.join(parse_dir, '{}_parses.json'.format(prefix))
    with open(fpath, 'r') as fr:
        for line in fr:
            d = json.loads(line)
            parses[d['sid']] = d
            if args.debug and len(parses) > 10:
                break
    return parses


def load_splits(split_dir):
    train_sids, dev_sids = set(), set()

    fpath = os.path.join(split_dir, 'train_sids.txt')
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.rstrip('\n')
            train_sids.add(line)

    fpath = os.path.join(split_dir, 'dev_sids.txt')
    with open(fpath, 'r') as fr:
        for line in fr:
            line = line.rstrip('\n')
            dev_sids.add(line)
    return train_sids, dev_sids


def get_marker2rtype(rtype2markers):
    dmarkers = {}
    for rtype, markers in rtype2markers.items():
        for m in markers:
            dmarkers[m] = rtype
    return dmarkers


def main():
    assert config['config_target'] == 'naive_psychology'

    if args.weight_name == 'elmo':
        lm = Elmo(args.elmo_option_file, args.elmo_weight_file, 1, dropout=0)
        tokenizer = MosesTokenizer(lang='en')
    else:
        # tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.weight_name, cache_dir=args.cache_dir
        )

        # language model
        lm = AutoModel.from_pretrained(args.weight_name,
                                       cache_dir=args.cache_dir)
    if args.gpu_id != -1:
        lm = lm.cuda(args.gpu_id)


    # dataset
    corpus = NaivePsychology(config['file_path'])
    # from the original dev, extract our train split
    train_sids, dev_sids = load_splits(config['split_dir'])
    process_split(corpus.dev_generator, tokenizer, lm, 'train', train_sids)

    # from the original dev, extract our dev split
    process_split(corpus.dev_generator, tokenizer, lm, 'dev', dev_sids)

    test_sids = set([sid for sid, _ in corpus.test_generator()])

    process_split(corpus.test_generator, tokenizer, lm, 'test', test_sids)


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    random.seed(args.seed)
    config = json.load(open(args.config_file))
    main()
