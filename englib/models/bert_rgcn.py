'''
The implementation of R-GCN is based on DGL:
    https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
Thanks to their contribution
'''
import os
import sys
import time
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset
import dgl.function as fn
from dgl.nn.pytorch import RelGraphConv
from dgl.nn.pytorch import utils as dgl_utils
from transformers import BertModel, AutoModel
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


class BERTNarrativeGraph(nn.Module):
    def __init__(self,
                 weight_name='google/bert_uncased_L-2_H-128_A-2',
                 cache_dir=None):
        super(BERTNarrativeGraph, self).__init__()
        self.cache_dir = cache_dir
        self.bert_model = AutoModel.from_pretrained(
            weight_name,
            cache_dir=cache_dir
        )

    def forward(self, input_ids, input_masks, token_type_ids, target_idxs):
        outputs = self.bert_model(input_ids,
                                  attention_mask=input_masks,
                                  token_type_ids=token_type_ids,
                                  position_ids=None,
                                  head_mask=None)
        emb_dim = outputs[0].shape[-1]
        # get the interested token embedding for each sentence
        # https://discuss.pytorch.org/t/selecting-over-dimension-by-indices/14595
        embeddings = torch.gather(
            outputs[0],
            1,
            target_idxs.view(-1, 1).unsqueeze(2).repeat(1, 1, emb_dim)
        ).squeeze()
        return embeddings

    def merge_node_representations(self, embeddings, nid2rows):
        # merge node representations; average over instances
        node_embeddings = []
        for nid in range(nid2rows.shape[0]):
            idxs = nid2rows[nid]
            mask = (idxs != -1).nonzero().view(-1)
            masked_idxs = idxs[mask]
            ne = embeddings[masked_idxs].mean(0)
            node_embeddings.append(ne)
        node_embeddings = torch.stack(node_embeddings, dim=0)
        return node_embeddings


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels,
                 bias=True, activation=None, self_loop=False,
                 dropout=0.2, use_gate=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.use_gate = use_gate

        # relation weights
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # message func
        self.message_func = self.basis_message_func

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            logger.info('loop_weight: {}'.format(self.loop_weight.shape))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

        if self.use_gate:
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, 1))
            nn.init.xavier_uniform_(self.gate_weight, gain=nn.init.calculate_gain('sigmoid'))

    def basis_message_func(self, edges):
        msg = dgl_utils.bmm_maybe_select(edges.src['h'], self.weight, edges.data['rel_type'])
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        if self.use_gate:
            gate = dgl_utils.bmm_maybe_select(edges.src['h'], self.gate_weight, edges.data['rel_type']).reshape(-1, 1)
            gate = torch.sigmoid(gate)
            msg = msg * gate
        return {'msg': msg}

    def forward(self, g):
        assert g.is_homograph(), \
            "not a homograph; convert it with to_homo and pass in the edge type as argument"
        with g.local_scope():
            if self.self_loop:
                loop_message = dgl_utils.matmul_maybe_select(g.ndata['h'], self.loop_weight)

            # message passing
            g.update_all(self.message_func, fn.sum(msg='msg', out='h'))

            # apply bias and activation
            node_repr = g.ndata['h']
            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            return node_repr


class RGCNModel(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels,
                 num_hidden_layers=2, dropout=0.2,
                 use_self_loop=True, use_gate=True):
        super(RGCNModel, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_gate = use_gate

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        in_dim = self.in_dim if idx == 0 else self.h_dim
        logger.debug('RGCN Layer {}: {}, {}, {}'.format(idx, in_dim, self.h_dim, act))
        return RGCNLayer(in_dim, self.h_dim, self.num_rels,
                            activation=act, self_loop=True,
                            dropout=self.dropout, use_gate=self.use_gate)

    def forward(self, g):
        for layer in self.layers:
            h = layer(g)
            g.ndata['h'] = h
        return h


class BaseLinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim,
                 bert_weight_name,
                 dropout=0.2, class_weights=[],
                 cache_dir=None, **kwargs):
        super(BaseLinkPredict, self).__init__()
        self.bert_ng = BERTNarrativeGraph(bert_weight_name,
                                          cache_dir=cache_dir)

        self.dropout = dropout
        self.in_dim = in_dim
        self.h_dim = h_dim

    def save_pretrained(self, save_dir, model_name, save_config=False):
        d = os.path.join(save_dir, model_name)
        if not os.path.isdir(d):
            os.makedirs(d)
        if save_config:
            self.save_config(d)
        fpath = os.path.join(d, 'model.pt')
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), fpath)

    @classmethod
    def from_pretrained(cls, model_dir, config, strict=False, cache_dir=None):
        model = cls(**config, cache_dir=cache_dir)
        fpath = os.path.join(model_dir, 'model.pt')
        model.load_state_dict(torch.load(fpath,
                                         map_location='cpu'),
                              strict=strict)
        return model

    def forward(self, **kwargs):
        raise NotImplementedError

    def l2_regularization_loss(self):
        loss = None
        for p in self.parameters():
            if loss is None:
                loss = torch.norm(p, 2) ** 2
            else:
                loss += (torch.norm(p, 2) ** 2)
        return loss


class RGCNLinkPredict(BaseLinkPredict):
    def __init__(self, in_dim, h_dim, num_narrative_rels, num_output_rels,
                 bert_weight_name,
                 num_hidden_layers=2, dropout=0.2, reg_param=0,
                 use_gate=False, use_rgcn=True, class_weights=[],
                 cache_dir=None,
                 **kwargs):
        super(RGCNLinkPredict, self).__init__(
            in_dim, h_dim,
            bert_weight_name, dropout, cache_dir=cache_dir,
            **kwargs)

        self.num_output_rels = num_output_rels
        if len(class_weights) > 0:
            self.class_weights = nn.Parameter(torch.FloatTensor(class_weights),
                                              requires_grad=False)
            logger.info('class_weights={}'.format(self.class_weights))
        else:
            self.class_weights = None
        self.use_rgcn = use_rgcn
        self.num_narrative_rels = num_narrative_rels
        self.rgcn = RGCNModel(in_dim, h_dim, num_narrative_rels,
                              num_hidden_layers=num_hidden_layers, dropout=dropout,
                              use_gate=use_gate)

        self.w_relation = nn.Parameter(torch.Tensor(num_output_rels, h_dim, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        self.reg_param = reg_param
        self.dropout1 = nn.Dropout(dropout)

    def save_config(self, save_dir):
        fpath = os.path.join(save_dir, 'model_config.json')
        config = {
            'in_dim': self.rgcn.in_dim,
            'h_dim': self.rgcn.h_dim,
            'num_narrative_rels': self.num_narrative_rels,
            'num_output_rels': self.num_output_rels,
            'num_hidden_layers': self.rgcn.num_hidden_layers,
            'dropout': self.rgcn.dropout,
            'reg_param': self.reg_param,
            'use_gate': self.rgcn.use_gate,
            'use_rgcn': self.use_rgcn
        }
        json.dump(config, open(fpath, 'w'))

    def calc_score(self, embedding, target_edges):
        # DistMult
        s = embedding[target_edges[0]].unsqueeze(1)
        r = self.w_relation[target_edges[1]]
        o = embedding[target_edges[2]].unsqueeze(2)
        score = torch.bmm(torch.bmm(s, r), o).squeeze()
        if len(score.shape) == 0:
            score = score.view(1)
        return score

    def forward(self, bg, input_ids, input_masks, token_type_ids,
                target_idxs, nid2rows,
                target_edges=None, mode='embeddings', n_instances=None, **kwargs):
        # bert encodes
        instance_embs = self.bert_ng(input_ids, input_masks, token_type_ids, target_idxs)
        if self.use_rgcn:
            instance_embs = self.dropout1(F.relu(instance_embs))

        # split by nodes
        if isinstance(nid2rows, list):
            # list of nid2rows
            node_embs = []
            for i in range(len(nid2rows)):
                start = sum(n_instances[:i])
                end = start + n_instances[i]
                ne = self.bert_ng.merge_node_representations(
                    instance_embs[start:end], nid2rows[i])
                node_embs.append(ne)
        else:
            # batched nid2rows
            node_embs = self.bert_ng.merge_node_representations(instance_embs, nid2rows)

        # graph conv
        if isinstance(bg, list):
            # same-node mode (one graph with different edges)
            n_truncated_gs = len(bg[0])
            all_gs = []
            for i, bg_list in enumerate(bg):
                # each bg_list contains graphs with same set of nodes but different edges
                for g in bg_list:
                    g.ndata['h'] = node_embs[i]
                    all_gs.append(g)

            bbg = dgl.batch(all_gs)
            if self.use_rgcn:
                self.rgcn(bbg)

            bbg_embs = []
            for g in dgl.unbatch(bbg):
                bbg_embs.append(g.ndata['h'])

            if mode == 'loss':
                all_target_edges = []
                for i in range(len(target_edges)):
                    for j in range(n_truncated_gs):
                        all_target_edges.append(target_edges[i][j])

                out = self.get_loss(
                    bbg_embs,
                    all_target_edges)
            elif mode == 'embeddings':
                batch_size = len(bg)
                # reshape the list into nested list
                # rgcn_embs = []
                # for i in range(batch_size):
                #     rgcn_embs.append(bbg_embs[i*n_truncated_gs: (i+1)*n_truncated_gs])

                # (rgcn_embs, bert_embs)
                out = (bbg_embs, node_embs)
            else:
                all_target_edges = []
                for i in range(len(target_edges)):
                    for j in range(n_truncated_gs):
                        all_target_edges.append(target_edges[i][j])

                out = self.predict(
                    bbg_embs,
                    all_target_edges)
        else:
            # batched-graph mode
            bg.ndata['h'] = node_embs
            if args.use_rgcn:
                self.rgcn(bg)
            embs = []
            for g in dgl.unbatch(bg):
                embs.append(g.ndata['h'])

            # for DistributedDataParallel, we have to make sure
            # all these outputs are used for calculating loss
            if mode == 'loss':
                out = self.get_loss(embs, target_edges)
            elif mode == 'embeddings': # get embeddings
                # (rgcn_embs, bert_embs)
                out = (embs, node_embs)
            elif mode == 'predict':
                out = self.predict(embs, target_edges)
        return out

    # def regularization_loss(self, embedding):
    #     return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embeddings, target_edges):
        # embeddings should be unbatched so that the indices are correct
        loss = 0.0
        for emb, te in zip(embeddings, target_edges):
            labels = te[3].float()
            score = self.calc_score(emb, te)
            # consider class weights for scores
            if self.class_weights is not None:
                w = self.class_weights[te[1]]
                # re-scale logits with class weights
                predicted_loss = F.binary_cross_entropy_with_logits(
                    score, labels, weight=w)
            else:
                predicted_loss = F.binary_cross_entropy_with_logits(
                    score, labels)

            reg_loss = torch.mean(emb.pow(2)) # double check
            loss += (predicted_loss + self.reg_param * reg_loss)
        reg_loss = torch.mean(self.w_relation.pow(2))
        loss += (self.reg_param * reg_loss)
        return loss

    def predict(self, embeddings, target_edges):
        y_pred, y = [], []
        for emb, te in zip(embeddings, target_edges):
            labels = te[3]
            y.append(labels)

            score = self.calc_score(emb, te)
            score = torch.sigmoid(score)
            y_pred.append(score)
        y_pred = torch.cat(y_pred, dim=0)
        y = torch.cat(y, dim=0)
        return y_pred, y


class BertEventTransE(BaseLinkPredict):
    def __init__(self, in_dim, h_dim, num_output_rels,
                 bert_weight_name,
                 num_hidden_layers=2, dropout=0.2, reg_param=0,
                 use_gate=False, use_rgcn=True,
                 margin=1.0, distance_p=1,
                 n_negs = 20, **kwargs):
        super(BertEventTransE, self).__init__(
            in_dim, h_dim,
            bert_weight_name, dropout, cache_dir=cache_dir,
            **kwargs)

        self.num_output_rels = num_output_rels
        # TODO: this binds to the preprocessing, which is not good
        self.n_negs = n_negs
        self.distance_p = distance_p
        self.margin = margin
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_output_rels, h_dim))
        # nn.init.xavier_uniform_(self.w_relation,
        #                         gain=nn.init.calculate_gain('relu'))
        nn.init.uniform_(self.w_relation,
                         -6.0/math.sqrt(h_dim),
                         6.0/math.sqrt(h_dim))

        # self.l1 = nn.Linear(in_dim, h_dim)
        # nn.init.xavier_uniform_(self.l1.weight,
        #                         gain=nn.init.calculate_gain('relu'))

        # self.dropout1 = nn.Dropout(dropout)
        self._loss_y = nn.Parameter(torch.FloatTensor([-1]), requires_grad=False)
        self.criterion = nn.MarginRankingLoss(self.margin, reduction='sum')

    def forward(self, bg, input_ids, input_masks, token_type_ids,
                target_idxs, nid2rows,
                target_edges=None, mode='embeddings', n_instances=None, **kwargs):
        t0 = time.time()
        # bert encodes
        instance_embs = self.bert_ng(input_ids, input_masks, token_type_ids, target_idxs)
        # instance_embs = self.dropout1(F.relu(instance_embs))

        # list of nid2rows
        node_embs = []
        for i in range(len(nid2rows)):
            start = sum(n_instances[:i])
            end = start + n_instances[i]
            ne = self.bert_ng.merge_node_representations(
                instance_embs[start:end], nid2rows[i])
            node_embs.append(ne)

        # linear layer, logits will be normalized, so no act
        # node_embs = [self.l1(ne)
        #              for ne in node_embs]
        node_embs = [F.normalize(ne, p=2, dim=1) for ne in node_embs]

        bbg_embs = []
        for i, bg_list in enumerate(bg):
            # each bg_list contains graphs with same set of nodes but different edges
            for g in bg_list:
                bbg_embs.append(node_embs[i])

        n_truncated_gs = len(bg[0])
        if mode == 'loss':
            all_target_edges = []
            for i in range(len(target_edges)):
                for j in range(n_truncated_gs):
                    all_target_edges.append(target_edges[i][j])

            out = self.get_loss(
                bbg_embs,
                all_target_edges)
        elif mode == 'embeddings':
            # may be should returned normalized embs
            out = node_embs
        else: # predict
            all_target_edges = []
            for i in range(len(target_edges)):
                for j in range(n_truncated_gs):
                    all_target_edges.append(target_edges[i][j])

            out = self.predict(
                bbg_embs,
                all_target_edges)
        return out

    def distance_to_similarity(self, dist):
        return 1.0 / (1.0 + dist)

    def pn_scoring(self, emb, te):
        pos_idxs = (te[3] == 1).nonzero().flatten()

        # if pos_idxs.nelement() == 1:
        #     negs = te[:, 1:]
        #     poss = te[:, 0].view(-1, 1).expand(negs.shape)
        #     return poss, negs

        # may be we can simply assume it's 1:20 p/n rate
        # but let's check speed with this first
        p_score, n_score = [], []
        neg_start_idx = pos_idxs.max().item() + 1
        for pidx in pos_idxs:
            p_edge = te[:, pidx].view(-1, 1)

            start = neg_start_idx + self.n_negs * pidx
            end = start + self.n_negs
            neg_idxs = list(range(start, end))

            neg_edges = te[:, neg_idxs]

            edges = torch.cat((p_edge, neg_edges), dim=1)
            scores = self._energe(emb, edges)
            ns = scores[1:]
            ps = scores[0].expand(ns.shape)

            p_score.append(ps)
            n_score.append(ns)
        p_score = torch.cat(p_score, dim=0)
        n_score = torch.cat(n_score, dim=0)
        return p_score, n_score

    def _energe(self, emb, edges):
        sources = edges[0]
        rels = edges[1]
        dests = edges[2]

        s_embs = emb[sources]
        d_embs = emb[dests]
        r_embs = self.w_relation[rels]
        e = torch.norm(s_embs + r_embs - d_embs, p=self.distance_p, dim=1)
        return e

    def get_loss(self, embeddings, target_edges):
        t0 = time.time()
        p_scores, n_scores = [], []
        for emb, te in zip(embeddings, target_edges):
            p_score, n_score = self.pn_scoring(emb, te)
            p_scores.append(p_score)
            n_scores.append(n_score)
        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)
        loss = self.criterion(p_scores, n_scores, self._loss_y)
        return loss

    def predict(self, embeddings, target_edges):
        y_pred, y = [], []
        for emb, te in zip(embeddings, target_edges):
            labels = te[3]
            y.append(labels)

            e = self._energe(emb, te)
            s = self.distance_to_similarity(e)
            y_pred.append(s)
        y_pred = torch.cat(y_pred, dim=0)
        y = torch.cat(y, dim=0)
        return y_pred, y


class BertEventComp(BaseLinkPredict):
    def __init__(self, in_dim, h_dim,
                 bert_weight_name,
                 n_negs=10,
                 dropout=0.3, reg_param=0.0,
                 **kwargs):
        super(BertEventComp, self).__init__(
            in_dim, h_dim,
            bert_weight_name, dropout, cache_dir=cache_dir,
            **kwargs)

        self.n_negs = n_negs
        self.reg_param = reg_param

        self.arg_l1 = nn.Linear(in_dim, h_dim)
        nn.init.xavier_uniform_(self.arg_l1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        logger.info('arg_l1: {}->{}'.format(in_dim, h_dim))

        self.arg_l2 = nn.Linear(h_dim, h_dim//2)
        nn.init.xavier_uniform_(self.arg_l2.weight,
                                gain=nn.init.calculate_gain('tanh'))
        logger.info('arg_l2: {}->{}'.format(h_dim, h_dim//2))

        self.event_l1 = nn.Linear(h_dim, h_dim//2)
        nn.init.xavier_uniform_(self.event_l1.weight,
                                gain=nn.init.calculate_gain('tanh'))
        logger.info('event_l1: {}->{}'.format(h_dim, h_dim//2))

        self.event_l2 = nn.Linear(h_dim//2, h_dim//4)
        nn.init.xavier_uniform_(self.event_l2.weight,
                                gain=nn.init.calculate_gain('tanh'))
        logger.info('event_l2: {}->{}'.format(h_dim//2, h_dim//4))

        self.event_l3 = nn.Linear(h_dim//4, 1)
        nn.init.xavier_uniform_(self.event_l3.weight,
                                gain=nn.init.calculate_gain('sigmoid'))
        logger.info('event_l3: {}->{}'.format(h_dim//4, 1))

        self.dropout1 = nn.Dropout(dropout)

        pos_weight = torch.FloatTensor([n_negs])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, bg, input_ids, input_masks, token_type_ids,
                target_idxs, nid2rows,
                target_edges=None, mode='embeddings', n_instances=None, **kwargs):
        # bert encodes
        instance_embs = self.bert_ng(input_ids, input_masks, token_type_ids, target_idxs)

        # list of nid2rows
        node_embs = []
        for i in range(len(nid2rows)):
            start = sum(n_instances[:i])
            end = start + n_instances[i]
            ne = self.bert_ng.merge_node_representations(
                instance_embs[start:end], nid2rows[i])
            node_embs.append(ne)

        bbg_embs = []
        for i, bg_list in enumerate(bg):
            for g in bg_list:
                bbg_embs.append(node_embs[i])

        n_truncated_gs = len(bg[0])
        if mode == 'loss':
            all_target_edges = []
            for i in range(len(target_edges)):
                for j in range(n_truncated_gs):
                    all_target_edges.append(target_edges[i][j])

            out = self.get_loss(
                bbg_embs,
                all_target_edges)
        elif mode == 'embeddings':
            # may be should returned normalized embs
            out = node_embs
        else: # predict
            all_target_edges = []
            for i in range(len(target_edges)):
                for j in range(n_truncated_gs):
                    all_target_edges.append(target_edges[i][j])

            out = self.predict(
                bbg_embs,
                all_target_edges)
        return out

    def compose_event_pair(self, emb, te):
        ys = te[3, :]

        e1s = te[0, :]
        e2s = te[2, :]
        e1_embs = emb[e1s]
        e2_embs = emb[e2s]

        # arg-event composition
        e1_arg_h1 = self.dropout1(F.hardtanh(self.arg_l1(e1_embs)))
        e2_arg_h1 = self.dropout1(F.hardtanh(self.arg_l1(e2_embs)))

        e1_arg_h2 = self.dropout1(F.hardtanh(self.arg_l2(e1_arg_h1)))
        e2_arg_h2 = self.dropout1(F.hardtanh(self.arg_l2(e2_arg_h1)))

        # relation composition
        arg_h2 = torch.cat((e1_arg_h2, e2_arg_h2), dim=1)
        event_h1 = self.dropout1(F.hardtanh(self.event_l1(arg_h2)))
        event_h2 = self.dropout1(F.hardtanh(self.event_l2(event_h1)))
        logits = self.event_l3(event_h2).squeeze()
        return logits, ys

    def get_loss(self, embeddings, target_edges):
        all_logits, all_ys = [], []
        for emb, te in zip(embeddings, target_edges):
            logits, ys = self.compose_event_pair(emb, te)
            all_logits.append(logits)
            all_ys.append(ys)
        all_logits = torch.cat(all_logits, dim=0)
        all_ys = torch.cat(all_ys, dim=0).float()
        loss = self.criterion(all_logits, all_ys)
        reg = self.reg_param * self.l2_regularization_loss()
        return loss + reg

    def predict(self, embeddings, target_edges):
        all_logits, all_ys = [], []
        for emb, te in zip(embeddings, target_edges):
            logits, ys = self.compose_event_pair(emb, te)
            all_logits.append(logits)
            all_ys.append(ys)
        all_logits = torch.cat(all_logits, dim=0)
        all_scores = torch.sigmoid(all_logits)
        all_ys = torch.cat(all_ys, dim=0)
        return all_scores, all_ys


class BertEventMemory(BaseLinkPredict):
    def __init__(self, in_dim, h_dim,
                 bert_weight_name, num_layers=1,
                 dropout=0.2, reg_param=1e-8, n_negs=4,
                 mem_thr=0.1, max_mem_passes=3, n_steps=9,
                 **kwargs):
        super(BertEventMemory, self).__init__(
            in_dim, h_dim,
            bert_weight_name, dropout, cache_dir=cache_dir,
            **kwargs)

        self.n_steps = n_steps
        self.mem_thr = mem_thr
        self.max_mem_passes = max_mem_passes
        self.n_negs = n_negs
        self.reg_param = reg_param
        self.num_layers = num_layers

        self.mem = nn.GRU(h_dim, h_dim, batch_first=True, dropout=dropout, num_layers=num_layers)

        self.rnn = nn.LSTM(h_dim, h_dim,
                           batch_first=True,
                           dropout=dropout,
                           num_layers=num_layers)

        self.attns = nn.ModuleList([nn.Linear(h_dim, 1, bias=False) if i < n_steps-1
                      else nn.Linear(h_dim, 1) for i in range(self.n_steps)]) # h_c has bias
        for i in range(n_steps-1):
            nn.init.xavier_uniform_(self.attns[i].weight,
                                    gain=nn.init.calculate_gain('tanh'))
        self.score_layers = nn.ModuleList([nn.Linear(h_dim, 1, bias=False) if i < n_steps-1
                      else nn.Linear(h_dim, 1) for i in range(self.n_steps)]) # h_c has bias
        for i in range(n_steps-1):
            nn.init.xavier_uniform_(self.score_layers[i].weight,
                                    gain=nn.init.calculate_gain('sigmoid'))


        self.criterion = nn.BCELoss()

    def memory_hops(self, rnn_out):
        attn_logits = [self.attns[i](rnn_out[:, i, :]) for i in range(self.n_steps)]
        attn_ws = [F.hardtanh(attn_logits[i] + attn_logits[-1]) for i in range(self.n_steps-1)]
        attn_ws = F.softmax(torch.cat(attn_ws, dim=1), dim=1)

        h_e = torch.bmm(attn_ws.unsqueeze(1), rnn_out[:, :self.n_steps-1, :])
        v = rnn_out[:, self.n_steps-1, :].view(self.num_layers, -1, self.h_dim)
        diff = float('inf')
        count = 0
        while diff > self.mem_thr and count < self.max_mem_passes:
            new_h_e, new_v = self.mem(h_e, v)

            diff = torch.norm(new_v-v, p=1)
            # logger.debug('{}: diff = {}'.format(count, diff))
            h_e, v = new_h_e, new_v
            count += 1
        # logger.info('mem:{}: diff = {}'.format(count, diff))

        # assert self.num_layers == 1
        v_logits = self.attns[-1](v[-1])
        attn_ws = [F.hardtanh(attn_logits[i] + v_logits) for i in range(self.n_steps-1)]
        attn_ws = F.softmax(torch.cat(attn_ws, dim=1), dim=1)

        # not sure if the paper uses h_c or v_t for s_i, try both
        # v_t
        score_logits = [self.score_layers[i](rnn_out[:, i, :]) for i in range(self.n_steps-1)] \
            + [self.score_layers[-1](v[-1])]
        # h_c
        # score_logits = [self.score_layers[i](rnn_out[:, i, :]) for i in range(self.n_steps)]

        scores = [F.sigmoid(score_logits[i] + score_logits[-1]) for i in range(self.n_steps-1)]
        scores = torch.cat(scores, dim=1)
        weighted_scores = (attn_ws * scores).sum(1)
        return weighted_scores

    def forward(self, input_ids, input_masks, token_type_ids,
                target_idxs, nid2rows, coref_nids, neg_coref_nids,
                mode='embeddings', n_instances=None, **kwargs):
        # bert encodes
        instance_embs = self.bert_ng(input_ids, input_masks, token_type_ids, target_idxs)

        # list of nid2rows
        # node_embs = []
        batch_size = len(nid2rows)
        n_choices = neg_coref_nids[0].shape[0] + 1
        all_inputs = []
        for i in range(batch_size):
            start = sum(n_instances[:i])
            end = start + n_instances[i]
            ne = self.bert_ng.merge_node_representations(
                instance_embs[start:end], nid2rows[i])
            # node_embs.append(ne)

            context = coref_nids[i][:-1].view(1, -1).expand(n_choices, -1)

            # choices: 1 pos 4 neg
            choices = torch.cat(
                (coref_nids[i][-1].view(-1), neg_coref_nids[i]),
                dim=0
            ).view(-1, 1)
            inputs = torch.cat((context, choices), dim=1)
            inputs = ne[inputs]
            all_inputs.append(inputs)

        n_steps = all_inputs[0].shape[-2]
        all_inputs = torch.stack(all_inputs, dim=0)
        batched_inputs = all_inputs.view(-1, n_steps, self.h_dim)

        rnn_out, hidden = self.rnn(batched_inputs)

        if mode == 'loss':
            weighted_scores = self.memory_hops(rnn_out)
            logger.debug(weighted_scores)
            ys = torch.zeros(weighted_scores.shape, dtype=torch.float32)
            for j in range(0, weighted_scores.shape[0], n_choices):
                ys[j] = 1
            ys = ys.to(weighted_scores.device)
            out = self.get_loss(weighted_scores, ys)
        elif mode == 'embeddings':
            # may be should returned normalized embs
            out = rnn_out
        else: # predict
            weighted_scores = self.memory_hops(rnn_out)
            ys = torch.zeros(weighted_scores.shape, dtype=torch.float32)
            for j in range(0, weighted_scores.shape[0], n_choices):
                ys[j] = 1
            ys = ys.to(weighted_scores.device)
            out = (weighted_scores, ys)
        return out

    def get_loss(self, preds, ys):
        self.criterion.weight = (ys * (self.n_negs-1)) + 1
        loss = self.criterion(preds, ys)
        reg = self.reg_param * self.l2_regularization_loss()
        logger.debug('loss={}, reg_loss={}'.format(loss, reg))
        return loss + reg

    def save_pretrained(self, save_dir, model_name, save_config=False):
        d = os.path.join(save_dir, model_name)
        if not os.path.isdir(d):
            os.makedirs(d)
        if save_config:
            self.save_config(d)
        fpath = os.path.join(d, 'model.pt')
        model_to_save = self.module if hasattr(self, 'module') else self
        sd = model_to_save.state_dict()
        if 'criterion.weight' in sd:
            del sd['criterion.weight']
        torch.save(sd, fpath)


class BertEventLSTM(BaseLinkPredict):
    def __init__(self, in_dim, h_dim,
                 bert_weight_name, num_layers=1,
                 dropout=0.2, reg_param=1e-8, n_negs=4,
                 **kwargs):
        super(BertEventLSTM, self).__init__(
            in_dim, h_dim,
            bert_weight_name, dropout, cache_dir=cache_dir,
            **kwargs)

        self.n_negs = n_negs
        self.reg_param = reg_param
        self.num_layers = num_layers

        self.rnn = nn.LSTM(h_dim, h_dim,
                           batch_first=True,
                           dropout=dropout,
                           num_layers=num_layers)

        # self.attn = nn.Linear(h_dim*2, 1)
        self.out_l1 = nn.Linear(h_dim*2, 1)

        # nn.init.xavier_uniform_(self.attn.weight,
        #                         gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.out_l1.weight,
                                gain=nn.init.calculate_gain('sigmoid'))

        self.criterion = nn.BCELoss()

    def get_scores(self, rnn_out, get_logits=False):
        n_steps = rnn_out.shape[-2]

        # combine the last output from RNN and the candidate
        h_e = rnn_out[:, -2, :]
        h_c = rnn_out[:, -1, :]
        h = torch.cat((h_e, h_c), dim=1)
        if get_logits:
            s = self.out_l1(h)
        else:
            s = F.sigmoid(self.out_l1(h))
        return s.view(-1)

    def forward(self, input_ids, input_masks, token_type_ids,
                target_idxs, nid2rows, coref_nids, neg_coref_nids,
                mode='embeddings', n_instances=None, **kwargs):
        # bert encodes
        instance_embs = self.bert_ng(input_ids, input_masks, token_type_ids, target_idxs)

        # list of nid2rows
        # node_embs = []
        batch_size = len(nid2rows)
        n_choices = neg_coref_nids[0].shape[0] + 1
        all_inputs = []
        for i in range(batch_size):
            start = sum(n_instances[:i])
            end = start + n_instances[i]
            ne = self.bert_ng.merge_node_representations(
                instance_embs[start:end], nid2rows[i])
            # node_embs.append(ne)

            context = coref_nids[i][:-1].view(1, -1).expand(n_choices, -1)

            # choices: 1 pos 4 neg
            choices = torch.cat(
                (coref_nids[i][-1].view(-1), neg_coref_nids[i]),
                dim=0
            ).view(-1, 1)
            inputs = torch.cat((context, choices), dim=1)
            inputs = ne[inputs]
            all_inputs.append(inputs)

        n_steps = all_inputs[0].shape[-2]
        all_inputs = torch.stack(all_inputs, dim=0)
        batched_inputs = all_inputs.view(-1, n_steps, self.h_dim)

        rnn_out, hidden = self.rnn(batched_inputs)
        if mode == 'loss':
            scores = self.get_scores(rnn_out)
            ys = torch.zeros(scores.shape, dtype=torch.float32)
            for j in range(0, scores.shape[0], n_choices):
                ys[j] = 1
            ys = ys.to(scores.device)
            out = self.get_loss(scores, ys)
        elif mode == 'embeddings':
            # may be should returned normalized embs
            out = rnn_out
        else: # predict
            scores = self.get_scores(rnn_out)
            ys = torch.zeros(scores.shape, dtype=torch.float32)
            for j in range(0, scores.shape[0], n_choices):
                ys[j] = 1
            ys = ys.to(scores.device)
            out = (scores, ys)
        return out

    def get_loss(self, preds, ys):
        self.criterion.weight = (ys * (self.n_negs-1)) + 1
        loss = self.criterion(preds, ys)
        reg = self.reg_param * self.l2_regularization_loss()
        logger.debug('loss={}, reg_loss={}'.format(loss, reg))
        return loss + reg

    def save_pretrained(self, save_dir, model_name, save_config=False):
        d = os.path.join(save_dir, model_name)
        if not os.path.isdir(d):
            os.makedirs(d)
        if save_config:
            self.save_config(d)
        fpath = os.path.join(d, 'model.pt')
        model_to_save = self.module if hasattr(self, 'module') else self
        sd = model_to_save.state_dict()
        if 'criterion.weight' in sd:
            del sd['criterion.weight']
        torch.save(sd, fpath)


class BertEventAttnLSTM(BaseLinkPredict):
    def __init__(self, in_dim, h_dim,
                 bert_weight_name, num_layers=1,
                 dropout=0.2, reg_param=1e-8, n_negs=4,
                 **kwargs):
        super(BertEventAttnLSTM, self).__init__(
            in_dim, h_dim,
            bert_weight_name, dropout, cache_dir=cache_dir,
            **kwargs)

        self.n_negs = n_negs
        self.reg_param = reg_param
        self.num_layers = num_layers

        self.rnn = nn.LSTM(h_dim, h_dim,
                           batch_first=True,
                           dropout=dropout,
                           num_layers=num_layers)

        self.attn = nn.Linear(h_dim*2, 1)
        self.out_l1 = nn.Linear(h_dim*2, 1)

        nn.init.xavier_uniform_(self.attn.weight,
                                gain=nn.init.calculate_gain('tanh'))
        nn.init.xavier_uniform_(self.out_l1.weight,
                                gain=nn.init.calculate_gain('sigmoid'))

        self.criterion = nn.BCELoss()

    def attn_scores(self, rnn_out, get_logits=False):
        n_steps = rnn_out.shape[-2]
        scores, attn_ws = [], []
        for i in range(n_steps-1):
            h_i = rnn_out[:, i, :]
            h_c = rnn_out[:, -1, :]
            h = torch.cat((h_i, h_c), dim=1)

            w = F.hardtanh(self.attn(h))
            attn_ws.append(w)

            # attention on scores
            if get_logits:
                s = self.out_l1(h)
            else:
                s = F.sigmoid(self.out_l1(h))
            scores.append(s)

        attn_ws = F.softmax(torch.cat(attn_ws, dim=1), dim=1)
        # attention on scores
        scores = torch.cat(scores, dim=1)
        weighted_scores = (attn_ws * scores).sum(1)

        # attention on h
        # h_e = torch.bmm(attn_ws.unsqueeze(1), rnn_out[:, :n_steps-1, :]).squeeze()
        # h_c = rnn_out[:, -1, :]
        # h = torch.cat((h_e, h_c), dim=1)
        # weighted_scores = F.sigmoid(self.out_l1(h))
        return weighted_scores

    def forward(self, input_ids, input_masks, token_type_ids,
                target_idxs, nid2rows, coref_nids, neg_coref_nids,
                mode='embeddings', n_instances=None, **kwargs):
        # bert encodes
        instance_embs = self.bert_ng(input_ids, input_masks, token_type_ids, target_idxs)

        # list of nid2rows
        # node_embs = []
        batch_size = len(nid2rows)
        n_choices = neg_coref_nids[0].shape[0] + 1
        all_inputs = []
        for i in range(batch_size):
            start = sum(n_instances[:i])
            end = start + n_instances[i]
            ne = self.bert_ng.merge_node_representations(
                instance_embs[start:end], nid2rows[i])
            # node_embs.append(ne)

            context = coref_nids[i][:-1].view(1, -1).expand(n_choices, -1)

            # choices: 1 pos 4 neg
            choices = torch.cat(
                (coref_nids[i][-1].view(-1), neg_coref_nids[i]),
                dim=0
            ).view(-1, 1)
            inputs = torch.cat((context, choices), dim=1)
            inputs = ne[inputs]
            all_inputs.append(inputs)

        n_steps = all_inputs[0].shape[-2]
        all_inputs = torch.stack(all_inputs, dim=0)
        batched_inputs = all_inputs.view(-1, n_steps, self.h_dim)

        rnn_out, hidden = self.rnn(batched_inputs)

        if mode == 'loss':
            weighted_scores = self.attn_scores(rnn_out, get_logits=False)
            ys = torch.zeros(weighted_scores.shape, dtype=torch.float32)
            for j in range(0, weighted_scores.shape[0], n_choices):
                ys[j] = 1
            ys = ys.to(weighted_scores.device)
            out = self.get_loss(weighted_scores, ys)
        elif mode == 'embeddings':
            # may be should returned normalized embs
            out = rnn_out
        else: # predict
            weighted_scores = self.attn_scores(rnn_out, get_logits=False)
            ys = torch.zeros(weighted_scores.shape, dtype=torch.float32)
            for j in range(0, weighted_scores.shape[0], n_choices):
                ys[j] = 1
            ys = ys.to(weighted_scores.device)
            out = (weighted_scores, ys)
        return out

    def get_loss(self, preds, ys):
        self.criterion.weight = (ys * (self.n_negs-1)) + 1
        loss = self.criterion(preds, ys)
        reg = self.reg_param * self.l2_regularization_loss()
        logger.debug('loss={}, reg_loss={}'.format(loss, reg))
        return loss + reg

    def save_pretrained(self, save_dir, model_name, save_config=False):
        d = os.path.join(save_dir, model_name)
        if not os.path.isdir(d):
            os.makedirs(d)
        if save_config:
            self.save_config(d)
        fpath = os.path.join(d, 'model.pt')
        model_to_save = self.module if hasattr(self, 'module') else self
        sd = model_to_save.state_dict()
        if 'criterion.weight' in sd:
            del sd['criterion.weight']
        torch.save(sd, fpath)


class BertEventSGNN(BaseLinkPredict):
    def __init__(self, in_dim, h_dim,
                 bert_weight_name, num_layers=1,
                 dropout=0.2, reg_param=1e-8, n_negs=4,
                 n_steps=9, margin=0.015, recurrent_passes=2,
                 is_cased=False, bigram_fpath=None,
                 **kwargs):
        super(BertEventSGNN, self).__init__(
            in_dim, h_dim,
            bert_weight_name, dropout, cache_dir=cache_dir,
            **kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_weight_name, do_lower_case=(not is_cased), cache_dir=cache_dir)
        self.bigram = self.load_bigram(bigram_fpath)

        self.n_steps = n_steps
        self.margin = margin
        self.recurrent_passes = recurrent_passes
        self.n_negs = n_negs
        self.reg_param = reg_param
        self.num_layers = num_layers

        self.rnn = nn.GRU(h_dim, h_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
        self.adj_bias = nn.Parameter(torch.zeros((h_dim, ), dtype=torch.float32))
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.bias.data.uniform_(-stdv, stdv)

        self.attn = nn.Linear(h_dim*2, 1)
        nn.init.xavier_uniform_(self.attn.weight,
                                gain=nn.init.calculate_gain('tanh'))
        self.softmax_d2 = nn.Softmax(dim=2)

        self._loss_y = nn.Parameter(torch.FloatTensor([-1]), requires_grad=False)
        self.criterion = nn.MarginRankingLoss(self.margin, reduction='sum')
        # self.criterion = nn.MultiMarginLoss(margin=self.margin)

    def load_bigram(self, fpath):
        t1 = time.time()
        bigram = {}
        with open(fpath, 'r') as fr:
            for line in fr:
                sp = line.rstrip('\n').split('\t')
                if sp[0] not in bigram:
                    bigram[sp[0]] = {}
                bigram[sp[0]][sp[1]] = int(sp[2])
        logger.info('{} loaded in {} s'.format(fpath, time.time()-t1))
        return bigram

    def get_all_tokens(self, input_ids, target_idxs, n_instances):
        pred_ids = torch.gather(input_ids,
                                1,
                                target_idxs.view(-1, 1)).view(-1)
        toks = self.tokenizer.convert_ids_to_tokens(pred_ids)
        all_toks = [toks[sum(n_instances[:i]): sum(n_instances[:i])+n]
                    for i, n in enumerate(n_instances)]
        return all_toks

    def build_adj_matrix(self, ev_nids, toks, nid2rows):
        k = self.n_steps + self.n_negs
        adj_m = torch.zeros((k, k), dtype=torch.float32)

        assert nid2rows.shape[1] == 1, 'row index != nid'
        nid_toks = [toks[nid] for nid in ev_nids]

        for i in range(len(nid_toks)):
            tok1 = nid_toks[i]
            if tok1 not in self.bigram:
                continue
            s = sum(self.bigram[tok1].values())
            for j in range(len(nid_toks)):
                if i == j:
                    continue
                tok2 = nid_toks[j]
                if tok2 in self.bigram[tok1]:
                    adj_m[i, j] = self.bigram[tok1][tok2] / s
        return adj_m.to(nid2rows.device)

    def metric_euclid(self, v0, v1):
        dist = torch.norm(v0-v1, 2, 1)
        # return -dist
        return 1.0 / (1.0 + dist) # distance to similarity

    def distance_to_similarity(self, dist):
        return 1.0 / (1.0 + dist)

    def forward(self, input_ids, input_masks, token_type_ids,
                target_idxs, nid2rows, coref_nids, neg_coref_nids,
                mode='embeddings', n_instances=None, **kwargs):
        # bert encodes
        instance_embs = self.bert_ng(input_ids, input_masks, token_type_ids, target_idxs)

        # get tokens
        all_toks = self.get_all_tokens(input_ids, target_idxs, n_instances)

        batch_size = len(nid2rows)
        n_choices = neg_coref_nids[0].shape[0] + 1

        # all_adj_ms, all_ev_embs = [], []
        all_hs, all_as = [], []
        for i in range(batch_size):
            start = sum(n_instances[:i])
            end = start + n_instances[i]
            ne = self.bert_ng.merge_node_representations(
                instance_embs[start:end], nid2rows[i])

            context = coref_nids[i][:-1]

            # choices: 1 pos 4 neg
            choices = torch.cat(
                (coref_nids[i][-1].view(-1), neg_coref_nids[i]),
                dim=0
            )

            ev_nids = torch.cat((context, choices), dim=0)
            ev_embs = ne[ev_nids]

            # adj matrix
            adj_m = self.build_adj_matrix(
                ev_nids, all_toks[i], nid2rows[i])

            all_hs.append(ev_embs)
            a = torch.matmul(adj_m.transpose(0, 1), ev_embs) + self.adj_bias
            # a = torch.matmul(adj_m, ev_embs) + self.adj_bias
            all_as.append(a)

        # pass GRU
        all_hs = torch.cat(all_hs, dim=0).view(self.num_layers, -1, self.h_dim)
        all_as = torch.cat(all_as, dim=0).view(-1, 1, self.h_dim)
        for i_pass in range(self.recurrent_passes):
            all_as, all_hs = self.rnn(all_as, all_hs)

        n = self.n_steps + self.n_negs
        all_hs = all_hs.view(-1, n, self.h_dim)

        # pairwise scores
        if mode == 'loss':
            weighted_scores = self.get_weighted_scores(all_hs, batch_size)
            out = self.get_loss(weighted_scores, None)

            # weighted_scores = self.get_weighted_scores(all_hs, batch_size)
            # weighted_scores = weighted_scores.view(-1)
            # n_choices = self.n_negs + 1
            # ys = torch.zeros(weighted_scores.shape, dtype=torch.int64)
            # for j in range(0, weighted_scores.shape[0], n_choices):
            #     ys[j] = 1
            # ys = ys.to(weighted_scores.device)
            # out = self.get_loss(weighted_scores, ys)
        elif mode == 'embeddings':
            out = all_hs
        else: # predict
            weighted_scores = self.get_weighted_scores(all_hs, batch_size)
            weighted_scores = weighted_scores.view(-1)

            n_choices = self.n_negs + 1
            ys = torch.zeros(weighted_scores.shape, dtype=torch.float32)
            for j in range(0, weighted_scores.shape[0], n_choices):
                ys[j] = 1
            ys = ys.to(weighted_scores.device)
            out = (weighted_scores, ys)
        return out

    def get_weighted_scores(self, all_hs, batch_size):
        context = all_hs[:, :self.n_steps-1]
        choices = all_hs[:, self.n_steps-1:]

        v0s, v1s = [], []
        for i_batch in range(batch_size):
            for i_ch in range(choices.shape[1]):
                for i_ctx in range(context.shape[1]):
                    v0s.append(context[i_batch, i_ctx])
                    v1s.append(choices[i_batch, i_ch])

        v0s = torch.stack(v0s, dim=0)
        v1s = torch.stack(v1s, dim=0)
        scores = self.metric_euclid(v0s, v1s).view(batch_size, choices.shape[1], context.shape[1])

        alphas = self.attn(torch.cat((v0s, v1s), dim=1)).view(scores.shape)
        alphas = self.softmax_d2(alphas)

        weighted_scores = (alphas * scores).sum(2)
        return weighted_scores

    def get_loss(self, scores, ys):
        p_scores, n_scores = [], []
        for i in range(scores.shape[0]):
            p_scores.append(scores[i][0].expand(self.n_negs))
            n_scores.append(scores[i][1:])

        p_scores = torch.cat(p_scores, dim=0)
        n_scores = torch.cat(n_scores, dim=0)
        loss = self.criterion(p_scores, n_scores, self._loss_y)
        # loss = self.criterion(scores.view(-1, 1), ys)

        reg = self.reg_param * self.l2_regularization_loss()
        logger.debug('loss={}, reg_loss={}'.format(loss, reg))
        loss += reg
        return loss
