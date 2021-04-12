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
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from englib.common import utils
from englib.corpora import NaivePsychology
from englib.models.naive_psych import MASLOW_LABEL2IDX, REISS_LABEL2IDX, PLUTCHIK_LABEL2IDX
from englib.models.naive_psych import get_label_sentences
from englib.narrative.psychology_narrative_graph import create_psychology_narrative_graph
from englib.narrative.psychology_narrative_graph import create_psychology_narrative_graph_unsupervised


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='feature extraction for NaivePsychology (example-based)')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='input config')
    parser.add_argument('weight_name', metavar='WEIGHT_NAME',
                        choices=['bert-large-cased', 'roberta-large', 'bert-base-cased', 'roberta-base'],
                        help='model name')
    parser.add_argument('split_mode', metavar='SPLIT_MODE',
                        choices=['supervised', 'semisupervised', 'unsupervised', 'unsupervised_sentiment'],
                        help='model name')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    parser.add_argument('--max_seq_len', type=int, default=160,
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

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def process_unsupervised_sentiment(parses, tokenizer, lm, dmarkers, rtype2idx, prefix, sentiment_analyzer):
    max_n_nodes = -1
    fpath = os.path.join(args.output_dir, '{}_inputs.h5'.format(prefix))
    fw_h5 = h5py.File(fpath, 'w')
    failed_sids = []
    for sid, parse in tqdm(parses.items()):
        png = create_psychology_narrative_graph_unsupervised(
            sid, parse, dmarkers, rtype2idx)

        n_nodes = len(png.nodes)
        if n_nodes == 0:
            # no nodes; no coref
            logger.error('failed sid = {}'.format(sid))
            failed_sids.append(sid)
            continue

        if n_nodes > max_n_nodes:
            max_n_nodes = n_nodes

        wembs = {}
        all_bert_inputs = {}
        out_rgcn_inputs = None
        out_labels = None
        for task in ['maslow', 'reiss', 'plutchik']:
            # TODO: split bert_inputs and others' generation
            bert_inputs, rgcn_inputs, labels, node_info = png.to_dgl_inputs(
                tokenizer, task,
                max_seq_len=args.max_seq_len,
                has_label=True,
                use_sentiment_labels=True,
                sentiment_analyzer=sentiment_analyzer
            )
            out_rgcn_inputs = rgcn_inputs
            out_labels = labels

            if not args.no_embeddings:
                # wemb
                input_ids = torch.from_numpy(bert_inputs['input_ids'])
                input_mask = torch.from_numpy(bert_inputs['input_mask'])
                token_type_ids = torch.from_numpy(bert_inputs['token_type_ids'])
                if args.gpu_id != -1:
                    input_ids = input_ids.cuda(args.gpu_id)
                    input_mask = input_mask.cuda(args.gpu_id)
                    token_type_ids = token_type_ids.cuda(args.gpu_id)
                with torch.no_grad():
                    output = lm(input_ids, input_mask, token_type_ids=token_type_ids)
                    wemb = output[0].cpu().numpy()
                    wembs[task] = wemb

            # only bert_inputs is task-specific
            all_bert_inputs[task] = bert_inputs

        if len(all_bert_inputs) == 3: # write
            if not args.no_embeddings:
                for task, wemb in wembs.items():
                    fw_h5.create_dataset('{}/{}/word_embeddings'.format(sid, task), data=wemb)

            for task, bert_inputs in all_bert_inputs.items():
                for k, v in bert_inputs.items():
                    fw_h5.create_dataset('{}/{}/{}'.format(
                        sid, task, k), data=v)

            # these only write once
            # rgcn is the same for all graphs
            for k, v in out_rgcn_inputs.items():
                fw_h5.create_dataset('{}/{}'.format(
                    sid, k), data=v)
            for k, v in out_labels.items():
                fw_h5.create_dataset('{}/{}'.format(
                    sid, k), data=v)
    fw_h5.close()
    logger.info('{} max_n_nodes={}'.format(prefix, max_n_nodes))
    logger.info('#failed sids = {}'.format(len(failed_sids)))
    logger.info('failed_sids = {}'.format(failed_sids))


def process_semisupervised(gen, tokenizer, lm, prefix, parses, dmarkers, rtype2idx):
    max_n_nodes = -1
    fpath = os.path.join(args.output_dir, '{}_inputs.h5'.format(prefix))
    fw_h5 = h5py.File(fpath, 'w')

    fpath = os.path.join(args.output_dir,
                         '{}_node_info.json'.format(prefix))
    fw_node = open(fpath, 'w')
    for sid, doc in tqdm(gen()):

        parse = parses[sid]

        png = create_psychology_narrative_graph(
            sid, doc, parse, dmarkers, rtype2idx,
            use_count_plutchik=args.use_count_plutchik,
            has_label=False
        )

        n_nodes = len(png.nodes)
        if n_nodes > max_n_nodes:
            max_n_nodes = n_nodes

        for task in ['maslow', 'reiss', 'plutchik']:

            # TODO: split bert_inputs and others' generation
            bert_inputs, rgcn_inputs, labels, node_info = \
                png.to_dgl_inputs(
                    tokenizer, task,
                    max_seq_len=args.max_seq_len,
                    has_label=False
                )

            if task == 'maslow':
                # these only write once
                story_info = {
                    'sid': sid,
                    'node_info': node_info
                }
                fw_node.write(json.dumps(story_info) + '\n')

                # rgcn is the same for all graphs
                for k, v in rgcn_inputs.items():
                    fw_h5.create_dataset('{}/{}'.format(
                        sid, k), data=v)

                # for k, v in labels.items():
                #     fw_h5.create_dataset('{}/{}/labels'.format(
                #         sid, k), data=v)

            # only bert_inputs is task-specific
            for k, v in bert_inputs.items():
                fw_h5.create_dataset('{}/{}/{}'.format(
                    sid, task, k), data=v)

            # embeddings
            if not args.no_embeddings:
                input_ids = torch.from_numpy(bert_inputs['input_ids'])
                input_mask = torch.from_numpy(bert_inputs['input_mask'])
                token_type_ids = torch.from_numpy(bert_inputs['token_type_ids'])
                if args.gpu_id != -1:
                    input_ids = input_ids.cuda(args.gpu_id)
                    input_mask = input_mask.cuda(args.gpu_id)
                    token_type_ids = token_type_ids.cuda(args.gpu_id)
                with torch.no_grad():
                    output = lm(input_ids, input_mask, token_type_ids=token_type_ids)
                    wemb = output[0].cpu().numpy()
                    fw_h5.create_dataset('{}/{}/word_embeddings'.format(sid, task), data=wemb)
    fw_h5.close()
    fw_node.close()
    logger.info('{} max_n_nodes={}'.format(prefix, max_n_nodes))


def process_split(gen, tokenizer, lm, prefix, target_sids, parses, dmarkers, rtype2idx):
    max_n_nodes = -1
    fpath = os.path.join(args.output_dir, '{}_inputs.h5'.format(prefix))
    fw_h5 = h5py.File(fpath, 'w')

    fpath = os.path.join(args.output_dir,
                         '{}_node_info.json'.format(prefix))
    fw_node = open(fpath, 'w')
    for sid, doc in tqdm(gen()):
        if sid not in target_sids:
            continue

        parse = parses[sid]

        png = create_psychology_narrative_graph(
            sid, doc, parse, dmarkers, rtype2idx,
            use_count_plutchik=args.use_count_plutchik)

        n_nodes = len(png.nodes)
        if n_nodes > max_n_nodes:
            max_n_nodes = n_nodes

        for task in ['maslow', 'reiss', 'plutchik']:

            # TODO: split bert_inputs and others' generation
            bert_inputs, rgcn_inputs, labels, node_info = \
                png.to_dgl_inputs(
                    tokenizer, task,
                    max_seq_len=args.max_seq_len,
                    has_label=True
                )

            if task == 'maslow':
                # these only write once
                story_info = {
                    'sid': sid,
                    'node_info': node_info
                }
                fw_node.write(json.dumps(story_info) + '\n')

                # rgcn is the same for all graphs
                for k, v in rgcn_inputs.items():
                    fw_h5.create_dataset('{}/{}'.format(
                        sid, k), data=v)

                for k, v in labels.items():
                    fw_h5.create_dataset('{}/{}/labels'.format(
                        sid, k), data=v)

            # only bert_inputs is task-specific
            for k, v in bert_inputs.items():
                fw_h5.create_dataset('{}/{}/{}'.format(
                    sid, task, k), data=v)

            # embeddings
            if not args.no_embeddings:
                input_ids = torch.from_numpy(bert_inputs['input_ids'])
                input_mask = torch.from_numpy(bert_inputs['input_mask'])
                token_type_ids = torch.from_numpy(bert_inputs['token_type_ids'])
                if args.gpu_id != -1:
                    input_ids = input_ids.cuda(args.gpu_id)
                    input_mask = input_mask.cuda(args.gpu_id)
                    token_type_ids = token_type_ids.cuda(args.gpu_id)
                with torch.no_grad():
                    output = lm(input_ids, input_mask, token_type_ids=token_type_ids)
                    wemb = output[0].cpu().numpy()
                    fw_h5.create_dataset('{}/{}/word_embeddings'.format(sid, task), data=wemb)
    fw_h5.close()
    fw_node.close()
    logger.info('{} max_n_nodes={}'.format(prefix, max_n_nodes))


def process_unsupervised(parses, tokenizer, lm, dmarkers, rtype2idx, prefix):
    max_n_nodes = -1
    fpath = os.path.join(args.output_dir, '{}_inputs.h5'.format(prefix))
    fw_h5 = h5py.File(fpath, 'w')
    failed_sids = []
    for sid, parse in tqdm(parses.items()):
        png = create_psychology_narrative_graph_unsupervised(
            sid, parse, dmarkers, rtype2idx)

        n_nodes = len(png.nodes)
        if n_nodes > max_n_nodes:
            max_n_nodes = n_nodes

        wembs = {}
        all_bert_inputs = {}
        out_rgcn_inputs = None
        for task in ['maslow', 'reiss', 'plutchik']:
            # TODO: split bert_inputs and others' generation
            bert_inputs, rgcn_inputs, labels, node_info = png.to_dgl_inputs(
                tokenizer, task,
                max_seq_len=args.max_seq_len,
                has_label=False
            )
            out_rgcn_inputs = rgcn_inputs

            # no nodes; no coref
            if bert_inputs['input_ids'].size == 0:
                logger.error('failed sid = {}'.format(sid))
                failed_sids.append(sid)
                break

            if not args.no_embeddings:
                # wemb
                input_ids = torch.from_numpy(bert_inputs['input_ids'])
                input_mask = torch.from_numpy(bert_inputs['input_mask'])
                token_type_ids = torch.from_numpy(bert_inputs['token_type_ids'])
                if args.gpu_id != -1:
                    input_ids = input_ids.cuda(args.gpu_id)
                    input_mask = input_mask.cuda(args.gpu_id)
                    token_type_ids = token_type_ids.cuda(args.gpu_id)
                with torch.no_grad():
                    output = lm(input_ids, input_mask, token_type_ids=token_type_ids)
                    wemb = output[0].cpu().numpy()
                    wembs[task] = wemb

            # only bert_inputs is task-specific
            all_bert_inputs[task] = bert_inputs

        if len(all_bert_inputs) == 3: # write
            if not args.no_embeddings:
                for task, wemb in wembs.items():
                    fw_h5.create_dataset('{}/{}/word_embeddings'.format(sid, task), data=wemb)

            for task, bert_inputs in all_bert_inputs.items():
                for k, v in bert_inputs.items():
                    fw_h5.create_dataset('{}/{}/{}'.format(
                        sid, task, k), data=v)

            # these only write once
            # rgcn is the same for all graphs
            for k, v in out_rgcn_inputs.items():
                fw_h5.create_dataset('{}/{}'.format(
                    sid, k), data=v)

    fw_h5.close()
    logger.info('{} max_n_nodes={}'.format(prefix, max_n_nodes))
    logger.info('#failed sids = {}'.format(len(failed_sids)))
    logger.info('failed_sids = {}'.format(failed_sids))


def get_context_line_indexes_except(i, lines, c_name):
    context = [j-1 for j in range(1, 6) if j != i and lines[str(j)]['characters'][c_name]['app']]
    return context


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


def split_train_dev(gen):
    sids = [sid for sid, doc in gen()]
    random.shuffle(sids)

    n = len(sids)
    n_train = int(n * 0.8)

    train_sids = sids[:n_train]
    dev_sids = sids[n_train:]
    return set(train_sids), set(dev_sids)


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

    # discourse markers
    dmarkers = get_marker2rtype(config['discourse_markers'])

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
    if args.split_mode == 'unsupervised_sentiment':
        sa = SentimentIntensityAnalyzer()
        parses = load_parses(config['pretrain_parse_dir'], 'pretrain')
        process_unsupervised_sentiment(
            parses, tokenizer, lm, dmarkers, config['rtype2idx'], 'unsupervised_sentiment', sa)
    elif args.split_mode == 'unsupervised':
        # the prefix for the unsupervised was pretrain
        parses = load_parses(config['pretrain_parse_dir'], 'pretrain')
        process_unsupervised(parses, tokenizer, lm, dmarkers, config['rtype2idx'], 'unsupervised')
    elif args.split_mode == 'semisupervised':
        corpus = NaivePsychology(config['file_path'])
        train_parses = load_parses(config['parse_dir'], 'train')
        process_semisupervised(corpus.train_generator, tokenizer, lm, 'semisupervised', train_parses, dmarkers, config['rtype2idx'])
    else: # supervised
        corpus = NaivePsychology(config['file_path'])
        # from the original dev, extract our train split
        orig_dev_parses = load_parses(config['parse_dir'], 'dev')
        train_sids, dev_sids = load_splits(config['split_dir'])
        process_split(corpus.dev_generator, tokenizer, lm, 'train', train_sids, orig_dev_parses, dmarkers, config['rtype2idx'])

        # from the original dev, extract our dev split
        process_split(corpus.dev_generator, tokenizer, lm, 'dev', dev_sids, orig_dev_parses, dmarkers, config['rtype2idx'])

        del orig_dev_parses

        test_sids = set([sid for sid, _ in corpus.test_generator()])

        test_parses = load_parses(config['parse_dir'], 'test')
        process_split(corpus.test_generator, tokenizer, lm, 'test', test_sids, test_parses, dmarkers, config['rtype2idx'])


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    random.seed(args.seed)
    config = json.load(open(args.config_file))
    main()
