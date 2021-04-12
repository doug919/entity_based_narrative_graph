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

from englib.common import utils
from englib.corpora import NaivePsychology
from englib.models.naive_psych import MASLOW_LABEL2IDX, REISS_LABEL2IDX, PLUTCHIK_LABEL2IDX
from englib.narrative.psychology_narrative_graph import get_majority_labels


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='feature extraction for NaivePsychology')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='input config')
    parser.add_argument('weight_name', metavar='WEIGHT_NAME',
                        choices=['bert-large-cased', 'roberta-large', 'bert-base-cased', 'roberta-base'],
                        help='model name')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    parser.add_argument('--max_seq_len', type=int, default=32,
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


def process_split_sent(gen, tokenizer, lm, valid_sids, prefix):
    examples = []
    all_input_ids, all_input_mask = {}, {}
    n_sids = 0
    max_input_len = -1
    if not args.no_embeddings:
        fpath = os.path.join(args.output_dir, '{}_word_embeddings.h5'.format(prefix))
        fw_h5 = h5py.File(fpath, 'w')
    for sid, doc in tqdm(gen()):
        if sid not in valid_sids:
            continue
        n_sids += 1
        doc_input_ids, doc_input_mask = [], []
        for sent_num in range(1, 6):

            sent = doc['lines'][str(sent_num)]['text']
            input_ids = tokenizer.encode(sent)
            input_mask = [1] * len(input_ids)

            # padding
            if len(input_ids) > max_input_len:
                max_input_len = len(input_ids)
            if len(input_ids) < args.max_seq_len:
                k = args.max_seq_len - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * k
                input_mask += [0] * k
            doc_input_ids.append(input_ids)
            doc_input_mask.append(input_mask)

            # line-char examples
            chars = doc['lines'][str(sent_num)]['characters']
            for c_name, c_ann in chars.items():
                if not c_ann['app']:
                    continue
                logger.debug('sid={}, sent_num={}, c_name={}'.format(sid, sent_num, c_name))
                context_line_idxs = get_context_line_indexes_except(
                    sent_num, doc['lines'], c_name)

                # put sentence and contextual sentence indexes in one list
                sent_idxs = [sent_num-1] + context_line_idxs
                if len(sent_idxs) < 5:
                    sent_idxs += [-1] * (5 - len(sent_idxs))

                maslow_labels, reiss_labels, plutchik_labels = \
                    get_majority_labels(c_ann, use_count_plutchik=args.use_count_plutchik)
                examples.append((sid, sent_idxs, c_name, maslow_labels, reiss_labels, plutchik_labels))
        all_input_ids[sid] = np.array(doc_input_ids, dtype=np.int64)
        all_input_mask[sid] = np.array(doc_input_mask, dtype=np.int64)

        if not args.no_embeddings:
            input_ids = torch.from_numpy(all_input_ids[sid])
            input_mask = torch.from_numpy(all_input_mask[sid])
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.int64)
            if args.gpu_id != -1:
                input_ids = input_ids.cuda(args.gpu_id)
                input_mask = input_mask.cuda(args.gpu_id)
                token_type_ids = token_type_ids.cuda(args.gpu_id)
            with torch.no_grad():
                output = lm(input_ids, input_mask, token_type_ids=token_type_ids)
                wemb = output[0].cpu().numpy()
                fw_h5.create_dataset('{}'.format(sid), data=wemb)
    if not args.no_embeddings:
        fw_h5.close()
    logger.info('n_sids = {}'.format(n_sids))
    logger.info('{} max_input_len={}'.format(prefix, max_input_len))
    return all_input_ids, all_input_mask, examples


def get_context_line_indexes_except(i, lines, c_name):
    context = [j-1 for j in range(1, 6) if j != i and lines[str(j)]['characters'][c_name]['app']]
    return context


def save_pickle(prefix, input_ids, input_mask, examples):
    fpath = os.path.join(args.output_dir, '{}_input_ids.pkl'.format(prefix))
    logger.info('dumping {}...'.format(fpath))
    pkl.dump(input_ids, open(fpath, 'wb'))

    fpath = os.path.join(args.output_dir, '{}_input_mask.pkl'.format(prefix))
    logger.info('dumping {}...'.format(fpath))
    pkl.dump(input_mask, open(fpath, 'wb'))

    fpath = os.path.join(args.output_dir, '{}_examples.pkl'.format(prefix))
    logger.info('dumping {}...'.format(fpath))
    pkl.dump(examples, open(fpath, 'wb'))


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


def main():
    assert config['config_target'] == 'naive_psychology'

    # dataset
    corpus = NaivePsychology(config['file_path'])

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.weight_name, cache_dir=args.cache_dir
    )

    # language model
    lm = AutoModel.from_pretrained(args.weight_name,
                                   cache_dir=args.cache_dir)
    if args.gpu_id != -1:
        lm = lm.cuda(args.gpu_id)

    # splits
    train_sids, dev_sids = load_splits(config['split_dir'])

    # from the original dev, extract our train split
    train_input_ids, train_input_mask, train_examples = \
        process_split_sent(corpus.dev_generator, tokenizer, lm, train_sids, 'train')
    save_pickle('train', train_input_ids, train_input_mask, train_examples)

    # from the original dev, extract our dev split
    dev_input_ids, dev_input_mask, dev_examples = \
        process_split_sent(corpus.dev_generator, tokenizer, lm, dev_sids, 'dev')
    save_pickle('dev', dev_input_ids, dev_input_mask, dev_examples)

    del dev_input_ids, dev_input_mask, dev_examples

    # test
    test_sids = set([sid for sid, _ in corpus.test_generator()])
    test_input_ids, test_input_mask, test_examples = \
        process_split_sent(corpus.test_generator, tokenizer, lm, test_sids, 'test')
    save_pickle('test', test_input_ids, test_input_mask, test_examples)


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    random.seed(args.seed)
    config = json.load(open(args.config_file))
    main()
