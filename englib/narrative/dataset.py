import os
import csv
import json
import h5py
import random
import logging
import linecache
from copy import deepcopy
from collections import OrderedDict

import dgl
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


class NarrativeSequenceDataset(Dataset):
    def __init__(self, fpath):
        super(NarrativeSequenceDataset, self).__init__()
        self.fpath = fpath
        self.fp = None

        fp = h5py.File(self.fpath, 'r')
        self.idx2graph = list(fp.keys())
        self.count_graphs = len(self.idx2graph)
        fp.close()

    def __getitem__(self, idx):
        if self.fp is None:
            # a trick to solve picklability issue with torch.multiprocessing
            self.fp = h5py.File(self.fpath, 'r')
            self.idx2graph = list(self.fp.keys())
            self.count_graphs = len(self.idx2graph)

        gn = self.idx2graph[idx]

        bert_inputs = torch.from_numpy(self.fp[gn]['bert_inputs'][:].astype('int64'))

        bert_target_idxs = torch.from_numpy(self.fp[gn]['bert_target_idxs'][:].astype('int64'))
        bert_nid2rows = torch.from_numpy(self.fp[gn]['bert_nid2rows'][:].astype('int64'))

        ng_edges = self.fp[gn]['ng_edges'][:].astype('float32')
        ng_edges = torch.from_numpy(ng_edges)

        coref_nids = self.fp[gn]['coref_nids'][:].astype('int64')
        coref_nids = torch.from_numpy(coref_nids)

        if 'negative_coref_nids' in self.fp[gn]:
            neg_coref_nids = self.fp[gn]['negative_coref_nids'][:].astype('int64')
            neg_coref_nids = torch.from_numpy(neg_coref_nids)
        else:
            neg_coref_nids = None
        return (ng_edges, bert_inputs, bert_target_idxs, bert_nid2rows,
                coref_nids, neg_coref_nids, gn)

    def __len__(self):
        return self.count_graphs

    @staticmethod
    def to_gpu(batch, gpu):
        nid2rows = [n2r.cuda(gpu) for n2r in batch['nid2rows']]
        input_ids = batch['input_ids'].cuda(gpu)
        input_masks = batch['input_masks'].cuda(gpu)
        token_type_ids = batch['token_type_ids'].cuda(gpu)
        target_idxs = batch['target_idxs'].cuda(gpu)
        coref_nids = [cnids.cuda(gpu) for cnids in batch['coref_nids']]

        neg_coref_nids = batch['neg_coref_nids']
        if neg_coref_nids:
            neg_coref_nids = [ncn.cuda(gpu) for ncn in neg_coref_nids]

        batch = {
            'input_ids': input_ids,
            'input_masks': input_masks,
            'token_type_ids': token_type_ids,
            'target_idxs': target_idxs,
            'nid2rows': nid2rows,
            'n_instances': batch['n_instances'],
            'coref_nids': coref_nids,
            'neg_coref_nids': neg_coref_nids
        }
        return batch


class NarrativeGraphDataset(Dataset):
    def __init__(self, fpath, has_target_edges=False):
        super(NarrativeGraphDataset, self).__init__()
        self.fpath = fpath
        self.fp = None
        self.has_target_edges = has_target_edges

        fp = h5py.File(self.fpath, 'r')
        self.idx2graph = list(fp.keys())
        self.count_graphs = len(self.idx2graph)
        fp.close()

    def __getitem__(self, idx):
        if self.fp is None:
            # a trick to solve picklability issue with torch.multiprocessing
            self.fp = h5py.File(self.fpath, 'r')
            self.idx2graph = list(self.fp.keys())
            self.count_graphs = len(self.idx2graph)

        gn = self.idx2graph[idx]

        bert_inputs = torch.from_numpy(self.fp[gn]['bert_inputs'][:].astype('int64'))
        # input_ids = bert_inputs[0]
        # input_masks = bert_inputs[1]
        # token_type_ids = bert_inputs[2]

        bert_target_idxs = torch.from_numpy(self.fp[gn]['bert_target_idxs'][:].astype('int64'))
        bert_nid2rows = torch.from_numpy(self.fp[gn]['bert_nid2rows'][:].astype('int64'))

        ng_edges = self.fp[gn]['ng_edges'][:].astype('float32')
        ng_edges = torch.from_numpy(ng_edges)

        if self.has_target_edges:
            target_edges = self.fp[gn]['target_edges'][:].astype('int64')
            target_edges = torch.from_numpy(target_edges)
            input_edges = self.fp[gn]['input_edges'][:].astype('float32')
            input_edges = torch.from_numpy(input_edges)
            return (ng_edges, bert_inputs, bert_target_idxs, bert_nid2rows,
                    target_edges, input_edges, gn)
        return ng_edges, bert_inputs, bert_target_idxs, bert_nid2rows

    def __len__(self):
        return self.count_graphs

    @staticmethod
    def to_gpu(batch, gpu):
        bg = batch['bg']
        if isinstance(bg, list):
            # same-node mode
            for i, bg_list in enumerate(bg):
                for g in bg_list:
                    g.edata['rel_type'] = g.edata['rel_type'].cuda(gpu)
                    g.edata['norm'] = g.edata['norm'].cuda(gpu)

            nid2rows = [n2r.cuda(gpu) for n2r in batch['nid2rows']]
        else:
            # batched-graph mode
            bg.edata['rel_type'] = bg.edata['rel_type'].cuda(gpu)
            bg.edata['norm'] = bg.edata['norm'].cuda(gpu)
            nid2rows = batch['nid2rows'].cuda(gpu)

        if 'target_edges' in batch:
            target_edges = [te.cuda(gpu) for te in batch['target_edges']]
        else:
            target_edges = None

        input_ids = batch['input_ids'].cuda(gpu)
        input_masks = batch['input_masks'].cuda(gpu)
        token_type_ids = batch['token_type_ids'].cuda(gpu)
        target_idxs = batch['target_idxs'].cuda(gpu)

        batch = {
            'bg': bg,
            'input_ids': input_ids,
            'input_masks': input_masks,
            'token_type_ids': token_type_ids,
            'target_idxs': target_idxs,
            'nid2rows': nid2rows,
            'target_edges': target_edges,
            'n_instances': batch['n_instances']
        }
        return batch


class SchemaNarrativeGraphDataset(Dataset):
    def __init__(self, fpath):
        super(SchemaNarrativeGraphDataset, self).__init__()
        self.fpath = fpath
        self.fp = None

        fp = h5py.File(self.fpath, 'r')
        self.idx2graph = list(fp.keys())
        self.count_graphs = len(self.idx2graph)
        fp.close()

    def __getitem__(self, idx):
        if self.fp is None:
            # a trick to solve picklability issue with torch.multiprocessing
            self.fp = h5py.File(self.fpath, 'r')
            self.idx2graph = list(self.fp.keys())
            self.count_graphs = len(self.idx2graph)

        gn = self.idx2graph[idx]
        # node_embs = self.fp[gn]['ng_node_embeddings'][:]

        bert_inputs = torch.from_numpy(self.fp[gn]['bert_inputs'][:].astype('int64'))
        # input_ids = bert_inputs[0]
        # input_masks = bert_inputs[1]
        # token_type_ids = bert_inputs[2]

        bert_target_idxs = torch.from_numpy(self.fp[gn]['bert_target_idxs'][:].astype('int64'))
        bert_nid2rows = torch.from_numpy(self.fp[gn]['bert_nid2rows'][:].astype('int64'))
        n_nodes = bert_nid2rows.shape[0]

        edges = self.fp[gn]['ng_edges'][:]

        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        g.add_edges(edges[0].astype('int64'), edges[2].astype('int64'))
        edge_types = torch.from_numpy(edges[1].astype('int64'))
        edge_norms = torch.from_numpy(edges[3]).unsqueeze(1).float()
        g.edata.update({'rel_type': edge_types})
        g.edata.update({'norm': edge_norms})

        # node_embs = torch.from_numpy(node_embs)
        # g.ndata['h'] = node_embs

        target_edges = self.fp[gn]['schema_edges'][:]
        target_edges = torch.from_numpy(target_edges.astype('int64'))
        return g, target_edges, bert_inputs, bert_target_idxs, bert_nid2rows

    def __len__(self):
        return self.count_graphs

    @staticmethod
    def to_gpu(batch, gpu):
        bg = batch['bg']
        bg.edata['rel_type'] = bg.edata['rel_type'].cuda(gpu)
        bg.edata['norm'] = bg.edata['norm'].cuda(gpu)

        input_ids = batch['input_ids'].cuda(gpu)
        input_masks = batch['input_masks'].cuda(gpu)
        token_type_ids = batch['token_type_ids'].cuda(gpu)
        target_idxs = batch['target_idxs'].cuda(gpu)
        nid2rows = batch['nid2rows'].cuda(gpu)
        target_edges = [te.cuda(gpu) for te in batch['target_edges']]
        batch = {
            'bg': bg,
            'target_edges': target_edges,
            'input_ids': input_ids,
            'input_masks': input_masks,
            'token_type_ids': token_type_ids,
            'target_idxs': target_idxs,
            'nid2rows': nid2rows
        }
        return batch

    @staticmethod
    def collate(samples):
        graphs, target_edges, bert_inputs, bert_target_idxs, bert_nid2rows = \
            map(list, zip(*samples))
        if len(graphs) <= 1:
            batched_graph = graphs[0]

            input_ids = bert_inputs[0][0]
            input_masks = bert_inputs[0][1]
            token_type_ids = bert_inputs[0][2]

            tidxs = bert_target_idxs[0]
            all_nid2rows = bert_nid2rows[0]
        else:
            batched_graph = dgl.batch(graphs)
            input_ids = torch.cat([bi[0] for bi in bert_inputs], dim=0)
            input_masks = torch.cat([bi[1] for bi in bert_inputs], dim=0)
            token_type_ids = torch.cat([bi[2] for bi in bert_inputs], dim=0)
            tidxs = torch.cat(bert_target_idxs, dim=0)

            n_instances = [bi[0].shape[0] for bi in bert_inputs]

            all_nid2rows = batch_nid2rows(bert_nid2rows, n_instances)
        batch = {
            'bg': batched_graph,
            'target_edges': target_edges,
            'input_ids': input_ids,
            'input_masks': input_masks,
            'token_type_ids': token_type_ids,
            'target_idxs': tidxs,
            'nid2rows': all_nid2rows
        }
        return batch


def batch_nid2rows(bert_nid2rows, n_instances):
    max_col = max([bn.shape[1] for bn in bert_nid2rows])
    n_nids = [bn.shape[0] for bn in bert_nid2rows]

    all_nid2rows = torch.ones((sum(n_nids), max_col), dtype=torch.int64)
    all_nid2rows *= -1
    cur_row = 0
    for i, bn in enumerate(bert_nid2rows):
        if i > 0:
            mask = (bn != -1)
            bn[mask] += sum(n_instances[:i])
        all_nid2rows[cur_row: cur_row+n_nids[i], :bn.shape[1]] = bn
        cur_row += n_nids[i]
    return all_nid2rows


def sample_missing_ep_edges(n_missing_ep_edges, ep_edge_idxs,
                            ng_edges, ep_rtype_rev, all_pos_ep_edges):
    # sample ep edges
    # randomly pick n_missing_ep_edges by indices
    # also collect their reversed edges' indices
    missing_ep_idxs = set()
    while True:
        ridx = random.randint(0, len(ep_edge_idxs)-1)
        candidate_idx = ep_edge_idxs[ridx]
        if candidate_idx in missing_ep_idxs:
            continue

        e = tuple(ng_edges[:3, candidate_idx].view(-1).long().tolist())
        rev_e = (e[2], ep_rtype_rev[e[1]], e[0])
        rev_candidate_idx = all_pos_ep_edges[rev_e]

        missing_ep_idxs.add(candidate_idx)
        missing_ep_idxs.add(rev_candidate_idx)
        if len(missing_ep_idxs) == n_missing_ep_edges*2:
            break
    return missing_ep_idxs


def sample_missing_pp_edges(pp_pool, n_missing_pp_edges, pp_ridx2distr):
    # original distributions
    #   idx,    rel,        distribution
    #   0,      next,       0.7894
    #   1,      cnext,      0.1899
    #   2,      before,     0.0047
    #   3,      after,      0.0043
    #   4,     simul,      0.0015
    #   5,     reason,     0.0026
    #   6,     result,     0.0006
    #   7,     contrast,   0.0070

    # use edges in this graph and re-normalize the prob.
    cands = [k for k in pp_pool.keys()]
    p = [pp_ridx2distr[c] for c in cands]
    s = sum(p)
    p = [x / s for x in p]

    missing_pp_idxs = set()
    k = n_missing_pp_edges
    while len(missing_pp_idxs) < n_missing_pp_edges:
        draw = np.random.choice(cands, k, p=p)
        for d in draw:
            candidate_idxs = pp_pool[d]
            ridx = random.randint(0, len(candidate_idxs)-1)
            missing_pp_idxs.add(candidate_idxs[ridx])

            if len(missing_pp_idxs) >= n_missing_pp_edges:
                break
        k = n_missing_pp_edges - len(missing_pp_idxs)
    return missing_pp_idxs


def _update_norms(edges, n_nodes):
    src = edges[0]
    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    g.add_edges(edges[0].long(), edges[2].long())
    for i in range(edges.shape[1]) :
        nid2 = int(edges[2][i])
        edges[3][i] = 1.0 / g.in_degree(nid2)
    return edges


def sample_mcnc(ng_edges, n_nodes, coref_nids, coref_ridx, n_choices):
    # target the last edge
    target_edge = (int(coref_nids[-2]), coref_ridx, int(coref_nids[-1]))

    # search edges
    all_pos_edges = {}
    for i in range(ng_edges.shape[1]):
        if isinstance(ng_edges, torch.Tensor):
            e = (int(ng_edges[0, i]),
                 int(ng_edges[1, i]),
                 int(ng_edges[2, i]))
        else: # numpy
            e = tuple(ng_edges[:3, i].astype('int64'))
        all_pos_edges[e] = i
    target_edge_idx = all_pos_edges[target_edge]
    assert target_edge_idx != -1

    # prepare input edges
    # input_edges = np.concatenate(
    #     (ng_edges[:, :target_edge_idx],
    #      ng_edges[:, target_edge_idx+1:]),
    #     axis=1
    # )

    # random choices
    choices = [target_edge]
    choice_set = set(choices)
    while len(choices) < n_choices:
        r_nid = random.randint(0, n_nodes-1)
        r_e = (target_edge[0], target_edge[1], r_nid)
        if r_e in all_pos_edges:
            continue
        if r_e in choice_set: # avoid duplications
            continue
        choice_set.add(r_e)
        choices.append(r_e)

    # shuffle choices
    choice_idxs = list(range(n_choices))
    random.shuffle(choice_idxs)
    correct = choice_idxs.index(0)
    choices = [choices[cidx] for cidx in choice_idxs]
    return correct, choices, target_edge_idx


def sample_truncated_ng(ng_edges,
                        n_nodes,
                        ent_pred_ridxs,
                        pred_pred_ridxs,
                        ep_rtype_rev,
                        n_truncated_ng,
                        edge_sample_rate,
                        n_neg_per_pos,
                        pp_ridx2distr,
                        coref_ridx,
                        sample_entity_only):
    n_rtypes = len(ent_pred_ridxs) + len(pred_pred_ridxs)
    n_edges = ng_edges.shape[1]
    all_pos_examples = {}
    all_pos_ep_edges, all_pos_pp_edges = {}, {}
    ep_edge_idxs = []
    # pp_edge_idxs = []
    pp_pool = {}
    for i in range(n_edges):
        e = tuple(ng_edges[:3, i].view(-1).long().tolist())
        all_pos_examples[e] = i
        if e[1] in ent_pred_ridxs:
            all_pos_ep_edges[e] = i
            ep_edge_idxs.append(i)
        else:
            assert e[1] in pred_pred_ridxs
            all_pos_pp_edges[e] = i
            if e[1] not in pp_pool:
                pp_pool[e[1]] = []
            pp_pool[e[1]].append(i)
            # pp_edge_idxs.append(i)
    n_ep_edges = len(all_pos_ep_edges)
    n_pp_edges = len(all_pos_pp_edges)

    # divided by 2 for reversed line
    n_missing_ep_edges = int(n_ep_edges / 2 * edge_sample_rate) # this could be zero

    n_coref_edges = 0
    if coref_ridx in pp_pool:
        n_coref_edges = len(pp_pool[coref_ridx])
    if pp_ridx2distr[coref_ridx] == 1.0:
        if n_coref_edges == 0:
            logger.warning('n_coref_edges == 0')
            return None, None
        n_missing_pp_edges = int(n_coref_edges * edge_sample_rate)
    else:
        n_missing_pp_edges = int(n_pp_edges * edge_sample_rate)

    if n_missing_pp_edges == 0:
        logger.debug(
            'found small ng: n_ep_edges={}, n_pp_edges={}, n_coref_edges={}'.format(
                n_ep_edges, n_pp_edges, n_coref_edges))
        n_missing_pp_edges = 1 # sample at least one edge
    n_missing_edges = n_missing_ep_edges*2 + n_missing_pp_edges

    all_target_edges = []
    all_input_edges = []
    for i in range(n_truncated_ng):
        # sample ep edges
        if n_missing_ep_edges > 0:
            missing_ep_idxs = sample_missing_ep_edges(n_missing_ep_edges,
                                                      ep_edge_idxs,
                                                      ng_edges,
                                                      ep_rtype_rev,
                                                      all_pos_ep_edges)
        else:
            missing_ep_idxs = set()

        # sample pp edges
        if n_missing_pp_edges > 0:
            missing_pp_idxs = sample_missing_pp_edges(
                pp_pool, n_missing_pp_edges, pp_ridx2distr)
        else:
            missing_pp_idxs = set()
        # random.shuffle(pp_edge_idxs)
        # missing_pp_idxs = set(pp_edge_idxs[:n_missing_pp_edges])

        # summarize idxs
        missing_idxs = missing_ep_idxs.union(missing_pp_idxs)
        remaining_idxs = sorted(list(set(range(n_edges)) - missing_idxs))
        missing_idxs = sorted(list(missing_idxs))

        pos_edges = ng_edges[:, missing_idxs]
        new_edges = deepcopy(ng_edges[:, remaining_idxs])

        # calculate new norms
        new_edges = _update_norms(new_edges, n_nodes)
        all_input_edges.append(new_edges)

        # sample negative
        neg_edges = sample_negative_edges(
            all_pos_examples, pos_edges, n_nodes, n_rtypes,
            n_neg_per_pos, sample_entity_only
        )

        new_pos_edges = torch.ones((4, n_missing_edges), dtype=torch.long)
        new_pos_edges[:3] = pos_edges[:3]

        target_edges = torch.cat((new_pos_edges, neg_edges), dim=1)
        all_target_edges.append(target_edges)

    all_target_edges = torch.stack(all_target_edges, dim=0)
    all_input_edges = torch.stack(all_input_edges, dim=0)
    return all_target_edges, all_input_edges


def sample_negative_edges(all_pos_examples, pos_edges, n_nodes, n_rtypes,
                          n_neg_per_pos, sample_entity_only):
    n_missing_edges = pos_edges.shape[1]

    neg_edges = torch.zeros(
        (4, n_missing_edges * n_neg_per_pos),
        dtype=torch.long
    )
    neg_examples = set()
    for j in range(n_missing_edges):
        e = tuple(pos_edges[:3, j].view(-1).long().tolist())

        n_tried = 0
        count = 0
        while count < n_neg_per_pos:
            # pick what to truncate
            if sample_entity_only:
                truncating_target = random.randint(0, 1)
            else:
                truncating_target = random.randint(0, 2)
            if truncating_target == 0:
                # sample head
                s = (random.randint(0, n_nodes-1), e[1], e[2])
                if s not in all_pos_examples and s not in neg_examples:
                    neg_examples.add(s)
                    for k in range(3):
                        neg_edges[k, j*n_neg_per_pos + count] = s[k]
                    count += 1

            elif truncating_target == 1:
                # sample tail
                s = (e[0], e[1], random.randint(0, n_nodes-1))
                if s not in all_pos_examples and s not in neg_examples:
                    neg_examples.add(s)
                    for k in range(3):
                        neg_edges[k, j*n_neg_per_pos + count] = s[k]
                    count += 1

            else:
                # sample rel
                s = (e[0], random.randint(0, n_rtypes-1), e[2])
                if s not in all_pos_examples and s not in neg_examples:
                    neg_examples.add(s)
                    for k in range(3):
                        neg_edges[k, j*n_neg_per_pos + count] = s[k]
                    count += 1
            n_tried += 1
    logger.debug('count={}, tried={}'.format(count, n_tried))
    return neg_edges
