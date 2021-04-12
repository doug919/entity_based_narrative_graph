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


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='feature extraction for NaivePsychology (example-based)')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='input config')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    parser.add_argument('--seed', type=int, default=135,
                        help='seed for random')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def split_train_dev(gen):
    sids = [sid for sid, doc in gen()]
    random.shuffle(sids)

    n = len(sids)
    n_train = int(n * 0.8)

    train_sids = sids[:n_train]
    dev_sids = sids[n_train:]
    return set(train_sids), set(dev_sids)


def save_splits(train, dev, test):
    fpath = os.path.join(args.output_dir, 'dev_sids.txt')
    with open(fpath, 'w') as fw:
        for sid in dev:
            fw.write('{}\n'.format(sid))

    fpath = os.path.join(args.output_dir, 'train_sids.txt')
    with open(fpath, 'w') as fw:
        for sid in train:
            fw.write('{}\n'.format(sid))

    fpath = os.path.join(args.output_dir, 'test_sids.txt')
    with open(fpath, 'w') as fw:
        for sid in test:
            fw.write('{}\n'.format(sid))


def main():
    assert config['config_target'] == 'naive_psychology'

    # dataset
    corpus = NaivePsychology(config['file_path'])


    # split train, dev from the original dev (80:20)
    train_sids, dev_sids = split_train_dev(corpus.dev_generator)

    test_sids = set([sid for sid, _ in corpus.test_generator()])
    save_splits(train_sids, dev_sids, test_sids) # save for records


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    random.seed(args.seed)
    config = json.load(open(args.config_file))
    main()
