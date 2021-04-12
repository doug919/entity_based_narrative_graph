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
from englib.models.naive_psych import sample_png_edges


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='sample PNG for NaivePsych')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='naive psych config file.')
    parser.add_argument('unsupervised_input_dir', metavar='UNSUPERVISED_INPUT_DIR',
                        help='input directory.')
    parser.add_argument('semisupervised_input_dir', metavar='SEMISUPERVISED_INPUT_DIR',
                        help='input directory.')
    parser.add_argument('supervised_input_dir', metavar='SUPERVISED_INPUT_DIR',
                        help='supervised input directory.')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    parser.add_argument("--edge_sample_rate", default=0.2, type=float,
                        help="proportion of missing edges for each PNG.")
    parser.add_argument('--n_neg_per_pos_edge', default=5, type=int,
                        help='number of negative edges for each positive edge')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size')
    parser.add_argument('--count_rtype_distr', action='store_true', default=False,
                        help='count rtype distribution')

    parser.add_argument('--seed', type=int, default=135,
                        help='seed for random')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


class PNGDataset(Dataset):
    def __init__(self, input_f):
        super(PNGDataset, self).__init__()
        self.input_f = input_f
        self.fp =  None

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
        n_nodes = self.fp[sid]['maslow']['input_mask'].shape[0]

        edge_src = torch.from_numpy(self.fp[sid]['edge_src'][:])
        edge_dest = torch.from_numpy(self.fp[sid]['edge_dest'][:])
        edge_types = torch.from_numpy(self.fp[sid]['edge_types'][:])
        edge_norms = torch.from_numpy(self.fp[sid]['edge_norms'][:])

        return (sid, n_nodes, edge_src, edge_dest, edge_types, edge_norms)


def my_collate(samples):
    sid, n_nodes, edge_src, edge_dest, edge_types, edge_norms = map(list, zip(*samples))
    batch = {
        'sid': sid,
        'n_nodes': n_nodes,
        'edge_src': edge_src,
        'edge_dest': edge_dest,
        'edge_types': edge_types,
        'edge_norms': edge_norms,
    }
    return batch


def set_seed(gpu, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu != -1:
        torch.cuda.manual_seed_all(seed)


def load_dataset_inputs(prefix):
    if prefix == 'unsupervised':
        indir = args.unsupervised_input_dir
    elif prefix == 'semisupervised':
        indir = args.semisupervised_input_dir
    else:
        indir = args.supervised_input_dir
    input_f = os.path.join(indir, '{}_inputs.h5'.format(prefix))
    return input_f


def _sample(dataset_inputs, rtype_distr, prefix):
    sample_dataset = PNGDataset(dataset_inputs)
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=my_collate,
        num_workers=1) # 1 is safe for hdf5

    n_failed = 0
    fpath = os.path.join(args.output_dir, '{}_sampled_edges.h5'.format(prefix))
    fw = h5py.File(fpath, 'w')
    for batch in tqdm(sample_dataloader):
        n_pngs = len(batch['edge_types'])
        for i_png in range(n_pngs):
            sid = batch['sid'][i_png]
            n_nodes = batch['n_nodes'][i_png]
            sampled = sample_png_edges(
                batch['edge_src'][i_png],
                batch['edge_dest'][i_png],
                batch['edge_types'][i_png],
                n_nodes,
                args.n_neg_per_pos_edge,
                edge_sample_rate=args.edge_sample_rate,
                rtype_distr=rtype_distr
            )
            if sampled is None:
                logger.warning('FAILED: {}, #edges={}'.format(sid, batch['edge_types'][i_png].shape[0]))
                n_failed += 1
                continue

            # n_edges = batch['edge_types'][i_png].shape[0]
            # logger.debug('n_edges={}'.format(n_edges))

            input_edges, pos_edges, neg_edges = sampled

            fw.create_dataset('{}/input_edges'.format(sid), data=input_edges.numpy())
            fw.create_dataset('{}/pos_edges'.format(sid), data=pos_edges.numpy())
            fw.create_dataset('{}/neg_edges'.format(sid), data=neg_edges.numpy())
    fw.close()
    logger.info('#dropped: {}'.format(n_failed))


def get_rtype_distr(_inputs):
    sample_dataset = PNGDataset(*_inputs)
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=my_collate,
        num_workers=1) # 1 is safe for hdf5

    counts = {}
    for batch in tqdm(sample_dataloader, desc='collecting rtype statistics'):
        for graph_rtypes in batch['edge_types']:
            for rt in graph_rtypes.tolist():
                if rt not in counts:
                    counts[rt] = 0
                counts[rt] += 1

    # normalize
    s = sum(counts.values())
    distr = {label: c/s for label, c in counts.items()}
    return distr


def main():
    config = json.load(open(args.config_file))
    assert config['config_target'] == 'naive_psychology'
    set_seed(-1, args.seed) # in distributed training, this has to be same for all processes

    uns_inputs = load_dataset_inputs('unsupervised')
    if args.count_rtype_distr:
        rtype_distr = get_rtype_distr(uns_inputs)
    else:
        rtype_distr = config['rtype_distr']
        rtype2idx = config['rtype2idx']
        rtype_distr = {rtype2idx[r]: d for r, d in rtype_distr.items()}
    logger.info('rtype_distr = {}'.format(rtype_distr))

    # logger.info('processing dev')
    # dev_inputs = load_dataset_inputs('dev')
    # _sample(dev_inputs, rtype_distr, 'dev')

    # logger.info('processing test')
    # test_inputs = load_dataset_inputs('test')
    # _sample(test_inputs, rtype_distr, 'test')

    # logger.info('processing train')
    # train_inputs = load_dataset_inputs('train')
    # _sample(train_inputs, rtype_distr, 'train')

    logger.info('processing semisupervised')
    semisupervised_inputs = load_dataset_inputs('semisupervised')
    _sample(semisupervised_inputs, rtype_distr, 'semisupervised')

    # logger.info('processing unsupervised')
    # unsupervised_inputs = load_dataset_inputs('unsupervised')
    # _sample(unsupervised_inputs, rtype_distr, 'unsupervised')


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    main()
