import os
import sys
import json
import time
import math
import random
import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dgl
from dgl import DGLGraph
from dgl.data import MiniGCDataset
import dgl.function as fn
from dgl.nn.pytorch import RelGraphConv
from dgl.nn.pytorch import utils as dgl_utils
from transformers import BertModel, AutoModel

from .bert_rgcn import RGCNModel


MASLOW_LABEL_SENTENCES = {
    'spiritual growth': 'spiritual growth',
    'esteem': 'esteem',
    'love': 'love',
    'stability': 'stability',
    'physiological': 'physiological'
}

REISS_LABEL_SENTENCES = {
    'status': 'status',
    'idealism': 'idealism',
    'power': 'power',
    'family': 'family',
    'food': 'food',
    'independence': 'independence',
    'belonging': 'belonging',
    'competition': 'competition',
    'honor': 'honor',
    'romance': 'romance',
    'savings': 'savings',
    'contact': 'contact',
    'health': 'health',
    'serenity': 'serenity',
    'curiosity': 'curiosity',
    'approval': 'approval',
    'rest': 'rest',
    'tranquility': 'tranquility',
    'order': 'order'
}

PLUTCHIK_LABEL_SENTENCES = {
    'disgust': 'disgusted',
    'surprise': 'surprised',
    'anger': 'angry',
    'trust': 'trustful',
    'sadness': 'sad',
    'anticipation': 'anticipated',
    'joy': 'joyful',
    'fear': 'fearful'
}


# MASLOW_LABEL_SENTENCES = {
#     'spiritual growth': 'wants the spiritual growth.',
#     'esteem': 'wants the esteem.',
#     'love': 'wants the love.',
#     'stability': 'wants the stability.',
#     'physiological': 'wants the physiological.'
# }

# REISS_LABEL_SENTENCES = {
#     'status': 'wants the status.',
#     'idealism': 'wants the idealism.',
#     'power': 'wants the power.',
#     'family': 'wants the family.',
#     'food': 'wants the food.',
#     'independence': 'wants the independence.',
#     'belonging': 'wants the belonging.',
#     'competition': 'wants the competition.',
#     'honor': 'wants the honor.',
#     'romance': 'wants the romance.',
#     'savings': 'wants the savings.',
#     'contact': 'wants the contact.',
#     'health': 'wants the health.',
#     'serenity': 'wants the serenity.',
#     'curiosity': 'wants the curiosity.',
#     'approval': 'wants the approval.',
#     'rest': 'wants the rest.',
#     'tranquility': 'wants the tranquility.',
#     'order': 'wants the order.'
# }

# PLUTCHIK_LABEL_SENTENCES = {
#     'disgust': 'is disgusted.',
#     'surprise': 'is surprised.',
#     'anger': 'is angry.',
#     'trust': 'is trustful.',
#     'sadness': 'is sad.',
#     'anticipation': 'is anticipated.',
#     'joy': 'is joyful.',
#     'fear': 'is fearful.'
# }


SENTIMENT_LABEL2IDX = {
    'neu': 0,
    'pos': 1,
    'neg': 2
}

MASLOW_LABEL2IDX = {
    'spiritual growth': 0,
    'esteem': 1,
    'love': 2,
    'stability': 3,
    'physiological': 4
}

REISS_LABEL2IDX = {
    'status': 0,
    'idealism': 1,
    'power': 2,
    'family': 3,
    'food': 4,
    'independence': 5,
    'belonging': 6,
    'competition': 7,
    'honor': 8,
    'romance': 9,
    'savings': 10,
    'contact': 11,
    'health': 12,
    'serenity': 13,
    'curiosity': 14,
    'approval': 15,
    'rest': 16,
    'tranquility': 17,
    'order': 18
}

PLUTCHIK_LABEL2IDX = {
    'disgust': 0,
    'surprise': 1,
    'anger': 2,
    'trust': 3,
    'sadness': 4,
    'anticipation': 5,
    'joy': 6,
    'fear': 7
}

logger = logging.getLogger(__name__)


class BaseNaivePsychologyModel(nn.Module):
    def __init__(self):
        super(BaseNaivePsychologyModel, self).__init__()

    def forward(*pargs):
        raise NotImplementedError('{}'.format(self.__class__.__name__))

    @classmethod
    def from_pretrained(cls, model_dir, strict=False):
        fpath = os.path.join(model_dir, 'model_config.json')
        model_config = json.load(open(fpath))
        model = cls(**model_config)
        fpath = os.path.join(model_dir, 'model.pt')
        model.load_state_dict(torch.load(fpath,
                                         map_location='cpu'),
                              strict=strict)
        return model

    def save_pretrained(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_config(save_dir)
        fpath = os.path.join(save_dir, 'model.pt')
        model_to_save = self.module if hasattr(self, 'module') else self
        torch.save(model_to_save.state_dict(), fpath)

    def save_config(self, save_dir):
        raise NotImplementedError('{}'.format(self.__class__.__name__))

    def average_words(self, word_emb, input_mask):
        word_emb[input_mask == 0] = 0
        avg = word_emb.sum(1) / input_mask.sum(1).unsqueeze(1)
        return avg

    def lstm_forward(self, word_emb, input_mask, lstm_layer, pad_total_length=False):
        # LSTM packed forward
        orig_len = word_emb.shape[1] if pad_total_length else None
        unsorted_seq_lengths = input_mask.sum(1)
        seq_lengths, sorted_idx = unsorted_seq_lengths.sort(0, descending=True)
        _, unsorted_idx = torch.sort(sorted_idx, dim=0)
        sorted_word_emb = word_emb.index_select(0, sorted_idx)

        packed_input = pack_padded_sequence(sorted_word_emb, seq_lengths, batch_first=True)
        packed_out, (rnn_h, rnn_c) = lstm_layer(packed_input)
        unpacked_out, input_sizes = pad_packed_sequence(packed_out, batch_first=True, total_length=orig_len)

        unsorted_out = unpacked_out.index_select(0, unsorted_idx)
        return unsorted_out, unsorted_seq_lengths


# class RGCNNaivePsychologyLinkPredict(BaseNaivePsychologyModel):
#     def __init__(self, in_dim, h_dim, n_rtypes,
#                  n_hidden_layers=2, dropout=0.2, reg_param=0,
#                  use_gate=False, use_rgcn=True, use_lstm=False,
#                  class_weights=[],
#                  **kwargs):
#         super(RGCNNaivePsychologyLinkPredict, self).__init__()
#         self.in_dim = in_dim
#         self.h_dim = h_dim
#         self.n_rtypes = n_rtypes
#         self.use_gate = use_gate
#         self.use_rgcn = use_rgcn
#         self.use_lstm = use_lstm
#         self.dropout = dropout
#         self.n_hidden_layers = n_hidden_layers

#         if self.use_lstm:
#             self.lstm = nn.LSTM(self.in_dim, self.in_dim // 2, 1, batch_first=True, bidirectional=True)
#         self.rgcn = RGCNModel(in_dim, h_dim, n_rtypes,
#                               num_hidden_layers=n_hidden_layers,
#                               dropout=dropout,
#                               use_gate=use_gate)

#         self.w_relation = nn.Parameter(torch.Tensor(n_rtypes, h_dim, h_dim))
#         nn.init.xavier_uniform_(self.w_relation,
#                                 gain=nn.init.calculate_gain('relu'))
#         self.reg_param = reg_param
#         if len(class_weights) > 0:
#             self.class_weights = nn.Parameter(torch.FloatTensor(class_weights),
#                                               requires_grad=False)
#             logger.info('class_weights={}'.format(self.class_weights))
#         else:
#             self.class_weights = None

#     def save_config(self, save_dir):
#         fpath = os.path.join(save_dir, 'model_config.json')
#         config = {
#             'in_dim': self.in_dim,
#             'h_dim': self.h_dim,
#             'n_rtypes': self.n_rtypes,
#             'n_hidden_layers': self.n_hidden_layers,
#             'dropout': self.dropout,
#             'use_gate': self.use_gate,
#             'use_rgcn': self.use_rgcn,
#             'reg_param': self.reg_param,
#             'use_lstm': self.use_lstm
#         }
#         json.dump(config, open(fpath, 'w'))

#     def forward(self, wemb, input_mask, input_edges, edge_types, edge_norms, pos_edges, neg_edges, **kwargs):
#         n_instances = [we.shape[0] for we in wemb]

#         # LSTM embeddings, batch
#         batched_wemb = torch.cat(wemb, dim=0)
#         batched_input_mask = torch.cat(input_mask, dim=0)
#         b, t, dim = batched_wemb.shape

#         if self.use_lstm:
#             unsorted_out, unsorted_seq_lengths = self.lstm_forward(batched_wemb, batched_input_mask, self.lstm)

#             # average words
#             sent_emb = unsorted_out.sum(1) / unsorted_seq_lengths.unsqueeze(1)
#             sent_emb = sent_emb
#         else:
#             # average word embeddings to represent a sentence
#             sent_emb = self.average_words(batched_wemb, batched_input_mask)

#         # unbatch
#         node_feat = [sent_emb[sum(n_instances[:i]): sum(n_instances[:i+1])] for i in range(len(n_instances))]

#         # create graphs
#         n_truncates = []
#         gs = []
#         target_edges = []
#         for i_batch in range(len(wemb)):
#             pos_es = pos_edges[i_batch]
#             neg_es = neg_edges[i_batch]
#             input_es = input_edges[i_batch]
#             enorm = edge_norms[i_batch]

#             n_nodes = wemb[i_batch].shape[0]
#             # n_edges = input_es.shape[1]

#             # create graph
#             g = dgl.DGLGraph()
#             g.add_nodes(n_nodes)
#             g.add_edges(input_es[0], input_es[2])

#             g.ndata['h'] = node_feat[i_batch]
#             g.edata.update(
#                 {
#                     'rel_type': edge_types[i_batch],
#                     'norm': enorm.unsqueeze(1)
#                 }
#             )
#             gs.append(g)

#         # batch graph
#         bg = dgl.batch(gs)

#         self.rgcn(bg)

#         # unbatch
#         bg_embs = []
#         for g in dgl.unbatch(bg):
#             bg_embs.append(g.ndata['h'])

#         # emb = bg.ndata['h']

#         # link scores
#         all_ys, all_scores = [], []
#         all_rtypes = []
#         for emb, pe, ne in zip(bg_embs, pos_edges, neg_edges):
#             n_pos = pe.shape[1]
#             n_neg = ne.shape[1]
#             y = [1] * n_pos + [0] * n_neg

#             target_edges = torch.cat((pe, ne), dim=1)
#             score = self.calc_score(emb, target_edges)

#             all_rtypes.append(target_edges[1])

#             # ys moves to scores' device
#             all_ys.append(torch.FloatTensor(y).to(score.device))
#             all_scores.append(score)
#         return all_scores, all_ys, all_rtypes, bg_embs

#     def criterion(self, all_scores, all_ys, all_rtypes, all_embs):
#         loss = None
#         for score, ys, rtypes, emb in zip(all_scores, all_ys, all_rtypes, all_embs):
#             if self.class_weights is not None:
#                 w = self.class_weights[rtypes]
#                 # re-scale logits with class weights
#                 predicted_loss = F.binary_cross_entropy_with_logits(
#                     score, ys, weight=w)
#             else:
#                 predicted_loss = F.binary_cross_entropy_with_logits(
#                     score, ys)
#             reg_loss = torch.mean(emb.pow(2)) # double check
#             if loss is None:
#                 loss = (predicted_loss + self.reg_param * reg_loss)
#             else:
#                 loss += (predicted_loss + self.reg_param * reg_loss)
#         reg_loss = torch.mean(self.w_relation.pow(2))
#         loss += (self.reg_param * reg_loss)
#         return loss

#     def calc_score(self, embedding, target_edges):
#         # DistMult
#         s = embedding[target_edges[0]].unsqueeze(1)
#         r = self.w_relation[target_edges[1]]
#         o = embedding[target_edges[2]].unsqueeze(2)
#         score = torch.bmm(torch.bmm(s, r), o).squeeze()
#         if len(score.shape) == 0:
#             score = score.view(1)
#         return score

#     def predict(self, wemb, input_mask, input_edges, edge_types, edge_norms, pos_edges, neg_edges, **kwargs):
#         all_scores, all_ys, all_rtypes, bg_embs = \
#             self.forward(wemb, input_mask, input_edges, edge_types, edge_norms, pos_edges, neg_edges)
#         return (
#             torch.cat(all_scores, dim=0),
#             torch.cat(all_ys, dim=0),
#             torch.cat(all_rtypes, dim=0)
#         )


# class RGCNNaivePsychology(BaseNaivePsychologyModel):
#     def __init__(self, in_dim, h_dim, n_rtypes,
#                  n_hidden_layers=2, dropout=0.2, reg_param=0,
#                  use_gate=False, use_rgcn=True, use_lstm=False,
#                  **kwargs):
#         super(RGCNNaivePsychology, self).__init__()
#         self.in_dim = in_dim
#         self.h_dim = h_dim
#         self.n_rtypes = n_rtypes
#         self.use_gate = use_gate
#         self.use_rgcn = use_rgcn
#         self.use_lstm = use_lstm
#         self.dropout = dropout
#         self.n_hidden_layers = n_hidden_layers

#         logger.info('use_lstm = {}'.format(self.use_lstm))
#         if self.use_lstm:
#             self.lstm = nn.LSTM(self.in_dim, self.in_dim // 2, 1, batch_first=True, bidirectional=True)
#         self.rgcn = RGCNModel(in_dim, h_dim, n_rtypes,
#                               num_hidden_layers=n_hidden_layers,
#                               dropout=dropout,
#                               use_gate=use_gate)

#         self.h1_dim = h_dim // 2
#         self.h2_dim = self.h1_dim // 2
#         self.maslow_classifier = TwoLayerClassifier(
#             h_dim, self.h1_dim, self.h2_dim, len(MASLOW_LABEL2IDX),
#             self.dropout
#         )
#         self.reiss_classifier = TwoLayerClassifier(
#             h_dim, self.h1_dim, self.h2_dim, len(REISS_LABEL2IDX),
#             self.dropout
#         )
#         self.plutchik_classifier = TwoLayerClassifier(
#             h_dim, self.h1_dim, self.h2_dim, len(PLUTCHIK_LABEL2IDX),
#             self.dropout
#         )

#         self.reg_param = reg_param

#     def save_config(self, save_dir):
#         fpath = os.path.join(save_dir, 'model_config.json')
#         config = {
#             'in_dim': self.in_dim,
#             'h_dim': self.h_dim,
#             'n_rtypes': self.n_rtypes,
#             'n_hidden_layers': self.n_hidden_layers,
#             'dropout': self.dropout,
#             'use_gate': self.use_gate,
#             'use_rgcn': self.use_rgcn,
#             'reg_param': self.reg_param,
#             'use_lstm': self.use_lstm
#         }
#         json.dump(config, open(fpath, 'w'))

#     def forward(self, wemb, input_mask, edge_src, edge_dest, edge_types, edge_norms, **kwargs):
#         n_instances = [we.shape[0] for we in wemb]

#         # batch
#         batched_wemb = torch.cat(wemb, dim=0)
#         batched_input_mask = torch.cat(input_mask, dim=0)
#         b, t, dim = batched_wemb.shape

#         if self.use_lstm:
#             unsorted_out, unsorted_seq_lengths = self.lstm_forward(batched_wemb, batched_input_mask, self.lstm)

#             # average words
#             sent_emb = unsorted_out.sum(1) / unsorted_seq_lengths.unsqueeze(1)
#             sent_emb = sent_emb
#         else:
#             # average word embeddings to represent a sentence
#             sent_emb = self.average_words(batched_wemb, batched_input_mask)

#         # unbatch
#         node_feat = [sent_emb[sum(n_instances[:i]): sum(n_instances[:i+1])] for i in range(len(n_instances))]

#         gs = []
#         for i in range(len(wemb)):
#             g = dgl.DGLGraph()
#             n_nodes = wemb[i].shape[0]
#             g.add_nodes(n_nodes)
#             if edge_src[i].nelement() != 0:
#                 g.add_edges(edge_src[i], edge_dest[i])

#             g.ndata['h'] = node_feat[i]

#             g.edata.update(
#                 {
#                     'rel_type': edge_types[i],
#                     'norm': edge_norms[i].unsqueeze(1)
#                 }
#             )
#             gs.append(g)

#         bg = dgl.batch(gs)

#         self.rgcn(bg)
#         # bg_embs = []
#         # for g in dgl.unbatch(bg):
#         #     bg_embs.append(g.ndata['h'])

#         emb = bg.ndata['h']

#         maslow_logits = self.maslow_classifier(emb)
#         reiss_logits = self.reiss_classifier(emb)
#         plutchik_logits = self.plutchik_classifier(emb)
#         return maslow_logits, reiss_logits, plutchik_logits

#     def predict(self, wemb, input_mask, edge_src, edge_dest, edge_types, edge_norms, **kwargs):
#         maslow_logits, reiss_logits, plutchik_logits = \
#             self.forward(wemb, input_mask, edge_src, edge_dest, edge_types, edge_norms)

#         maslow_probs = torch.sigmoid(maslow_logits)
#         reiss_probs = torch.sigmoid(reiss_logits)
#         plutchik_probs = torch.sigmoid(plutchik_logits)

#         # maslow_pred = (maslow_probs >= 0.5).long()
#         # reiss_pred = (reiss_probs >= 0.5).long()
#         # plutchik_pred = (plutchik_probs >= 0.5).long()
#         return maslow_probs, reiss_probs, plutchik_probs


class RGCNNaivePsychologyLinkPredict(BaseNaivePsychologyModel):
    '''Version 2
    '''
    def __init__(self, weight_name, n_rtypes,
                 n_hidden_layers=2, reg_param=0.0,
                 use_gate=False, use_rgcn=True,
                 dropout=0.2, freeze_lm=False, class_weights=[],
                 cache_dir=None, **kwargs):
        super(RGCNNaivePsychologyLinkPredict, self).__init__()

        self.n_hidden_layers = n_hidden_layers
        self.freeze_lm = freeze_lm
        self.weight_name = weight_name
        self.dropout = dropout
        self.n_rtypes = n_rtypes
        self.use_gate = use_gate
        self.use_rgcn = use_rgcn
        self.reg_param = reg_param
        self.class_weights = class_weights

        # language model
        self.lm = AutoModel.from_pretrained(
            weight_name,
            cache_dir=cache_dir
        )
        self.lm_dim = self.lm.config.hidden_size
        if self.freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        # rgcn
        self.rgcn = RGCNModel(self.lm_dim, self.lm_dim, n_rtypes,
                              num_hidden_layers=n_hidden_layers,
                              dropout=dropout,
                              use_gate=use_gate)

        self.w_relation = nn.Parameter(torch.Tensor(n_rtypes, self.lm_dim, self.lm_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

        if len(class_weights) > 0:
            self.cw = nn.Parameter(torch.FloatTensor(class_weights),
                                              requires_grad=False)
            logger.info('class_weights={}'.format(self.cw))


    def save_config(self, save_dir):
        fpath = os.path.join(save_dir, 'model_config.json')
        config = {
            'weight_name': self.weight_name,
            'dropout': self.dropout,
            'freeze_lm': self.freeze_lm,
            'n_rtypes': self.n_rtypes,
            'n_hidden_layers': self.n_hidden_layers,
            'use_gate': self.use_gate,
            'use_rgcn': self.use_rgcn,
            'reg_param': self.reg_param,
            'class_weights': self.class_weights
        }
        json.dump(config, open(fpath, 'w'))

    def forward(self, input_ids, attention_mask, token_type_ids,
                input_edges, edge_types, edge_norms, pos_edges, neg_edges, **kwargs):
        # language model
        n_instances = [ii.shape[0] for ii in input_ids]

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        lm_output = self.lm(
            input_ids, attention_mask, token_type_ids=token_type_ids)
        pooled = lm_output[1]
        node_feat = [pooled[sum(n_instances[:i]): sum(n_instances[:i+1])] for i in range(len(n_instances))]

        # build graphs
        gs = []
        for i_batch in range(len(n_instances)):
            pos_es = pos_edges[i_batch]
            neg_es = neg_edges[i_batch]
            input_es = input_edges[i_batch]
            enorm = edge_norms[i_batch]

            n_nodes = n_instances[i_batch]

            # create graph
            g = dgl.DGLGraph()
            g.add_nodes(n_nodes)
            g.add_edges(input_es[0], input_es[2])

            g.ndata['h'] = node_feat[i_batch]
            g.edata.update(
                {
                    'rel_type': edge_types[i_batch],
                    'norm': enorm.unsqueeze(1)
                }
            )
            gs.append(g)

        # batch graphs
        bg = dgl.batch(gs)

        # rgcn forward
        self.rgcn(bg)

        # unbatch
        bg_embs = []
        for g in dgl.unbatch(bg):
            bg_embs.append(g.ndata['h'])

        # link scores
        all_ys, all_scores = [], []
        all_rtypes = []
        for emb, pe, ne in zip(bg_embs, pos_edges, neg_edges):
            n_pos = pe.shape[1]
            n_neg = ne.shape[1]
            y = [1] * n_pos + [0] * n_neg

            target_edges = torch.cat((pe, ne), dim=1)
            score = self.calc_score(emb, target_edges)

            all_rtypes.append(target_edges[1])

            # ys moves to scores' device
            all_ys.append(torch.FloatTensor(y).to(score.device))
            all_scores.append(score)
        return all_scores, all_ys, all_rtypes, bg_embs

    def criterion(self, all_scores, all_ys, all_rtypes, all_embs):
        loss = None
        for score, ys, rtypes, emb in zip(all_scores, all_ys, all_rtypes, all_embs):
            if len(self.class_weights) != 0:
                w = self.cw[rtypes]
                # re-scale logits with class weights
                predicted_loss = F.binary_cross_entropy_with_logits(
                    score, ys, weight=w)
            else:
                predicted_loss = F.binary_cross_entropy_with_logits(
                    score, ys)
            reg_loss = torch.mean(emb.pow(2)) # double check
            if loss is None:
                loss = (predicted_loss + self.reg_param * reg_loss)
            else:
                loss += (predicted_loss + self.reg_param * reg_loss)
        reg_loss = torch.mean(self.w_relation.pow(2))
        loss += (self.reg_param * reg_loss)
        return loss

    def calc_score(self, embedding, target_edges):
        # DistMult
        s = embedding[target_edges[0]].unsqueeze(1)
        r = self.w_relation[target_edges[1]]
        o = embedding[target_edges[2]].unsqueeze(2)
        score = torch.bmm(torch.bmm(s, r), o).squeeze()
        if len(score.shape) == 0:
            score = score.view(1)
        return score

    def predict(self, input_ids, attention_mask, token_type_ids,
                input_edges, edge_types, edge_norms, pos_edges, neg_edges, **kwargs):
        all_scores, all_ys, all_rtypes, bg_embs = \
            self.forward(input_ids, attention_mask, token_type_ids, input_edges, edge_types, edge_norms, pos_edges, neg_edges)
        return (
            torch.cat(all_scores, dim=0),
            torch.cat(all_ys, dim=0),
            torch.cat(all_rtypes, dim=0)
        )


class RGCNNaivePsychology(BaseNaivePsychologyModel):
    '''Version 2
    '''
    def __init__(self, weight_name, n_classes, n_rtypes,
                 n_hidden_layers=2, reg_param=0.0,
                 use_gate=False, use_rgcn=True,
                 dropout=0.2, freeze_lm=False,
                 cache_dir=None, **kwargs):
        super(RGCNNaivePsychology, self).__init__()

        self.n_hidden_layers = n_hidden_layers
        self.freeze_lm = freeze_lm
        self.weight_name = weight_name
        self.n_classes = n_classes
        self.dropout = dropout
        self.n_rtypes = n_rtypes
        self.use_gate = use_gate
        self.use_rgcn = use_rgcn
        self.reg_param = reg_param

        # language model
        self.lm = AutoModel.from_pretrained(
            weight_name,
            cache_dir=cache_dir
        )
        self.lm_dim = self.lm.config.hidden_size
        if self.freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        # rgcn
        self.rgcn = RGCNModel(self.lm_dim, self.lm_dim, n_rtypes,
                              num_hidden_layers=n_hidden_layers,
                              dropout=dropout,
                              use_gate=use_gate)

        # classifier
        self.h1_dim = self.lm_dim // 2
        self.h2_dim = self.h1_dim // 2
        self.classifier = TwoLayerClassifier(
            self.lm_dim, self.h1_dim, self.h2_dim, self.n_classes,
            self.dropout
        )

    def get_embedding_dim(self):
        return self.lm_dim

    def save_config(self, save_dir):
        fpath = os.path.join(save_dir, 'model_config.json')
        config = {
            'weight_name': self.weight_name,
            'n_classes': self.n_classes,
            'dropout': self.dropout,
            'freeze_lm': self.freeze_lm,
            'n_rtypes': self.n_rtypes,
            'n_hidden_layers': self.n_hidden_layers,
            'use_gate': self.use_gate,
            'use_rgcn': self.use_rgcn,
            'reg_param': self.reg_param
        }
        json.dump(config, open(fpath, 'w'))

    def get_embeddings(self, input_ids, attention_mask, token_type_ids,
                edge_src, edge_dest, edge_types, edge_norms, **kwargs):
        # language model
        n_instances = [ii.shape[0] for ii in input_ids]

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        lm_output = self.lm(
            input_ids, attention_mask, token_type_ids=token_type_ids)
        pooled = lm_output[1]
        node_feat = [pooled[sum(n_instances[:i]): sum(n_instances[:i+1])] for i in range(len(n_instances))]

        # build graphs
        gs = []
        for i in range(len(n_instances)):
            g = dgl.DGLGraph()
            g.add_nodes(n_instances[i])
            if edge_src[i].nelement() != 0:
                g.add_edges(edge_src[i], edge_dest[i])

            g.ndata['h'] = node_feat[i]

            g.edata.update(
                {
                    'rel_type': edge_types[i],
                    'norm': edge_norms[i].unsqueeze(1)
                }
            )
            gs.append(g)

        bg = dgl.batch(gs)

        # rgcn forward
        self.rgcn(bg)

        rgcn_embs = []
        for g in dgl.unbatch(bg):
            rgcn_embs.append(g.ndata['h'])

        bert_embs = node_feat
        return rgcn_embs, bert_embs

    def forward(self, input_ids, attention_mask, token_type_ids, labels,
                edge_src, edge_dest, edge_types, edge_norms, **kwargs):
        # language model
        n_instances = [ii.shape[0] for ii in input_ids]

        input_ids = torch.cat(input_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        lm_output = self.lm(
            input_ids, attention_mask, token_type_ids=token_type_ids)
        pooled = lm_output[1]
        node_feat = [pooled[sum(n_instances[:i]): sum(n_instances[:i+1])] for i in range(len(n_instances))]

        # build graphs
        gs = []
        for i in range(len(n_instances)):
            g = dgl.DGLGraph()
            g.add_nodes(n_instances[i])
            if edge_src[i].nelement() != 0:
                g.add_edges(edge_src[i], edge_dest[i])

            g.ndata['h'] = node_feat[i]

            g.edata.update(
                {
                    'rel_type': edge_types[i],
                    'norm': edge_norms[i].unsqueeze(1)
                }
            )
            gs.append(g)

        bg = dgl.batch(gs)

        # rgcn forward
        self.rgcn(bg)

        emb = bg.ndata['h']

        # output layer
        z = self.classifier(emb)
        return z

    def predict(self, input_ids, attention_mask, token_type_ids, labels,
                edge_src, edge_dest, edge_types, edge_norms, **kwargs):
        logits = self.forward(
            input_ids, attention_mask, token_type_ids, labels,
            edge_src, edge_dest, edge_types, edge_norms)
        score = torch.sigmoid(logits)
        return score


class TwoLayerClassifier(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, n_classes, dropout=0.2):
        super(TwoLayerClassifier, self).__init__()
        self.in_dim = in_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.n_classes = n_classes

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)

        self.l1 = nn.Linear(self.in_dim, self.h1_dim)
        self.l2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.lout = nn.Linear(self.h2_dim, self.n_classes)
        nn.init.xavier_uniform_(self.l1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.l2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.lout.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, h_e):
        h_1 = F.relu(self.l1(self.d1(h_e)))
        h_2 = F.relu(self.l2(self.d2(h_1)))
        out = self.lout(self.d3(h_2))
        return out


class LabelCorelationModel(BaseNaivePsychologyModel):
    def __init__(self, weight_name, n_classes, dropout=0.2, freeze_lm=False, no_label_corelation=False, cache_dir=None, use_lm=True, **kwargs):
        super(LabelCorelationModel, self).__init__()

        self.use_lm = use_lm
        self.freeze_lm = freeze_lm
        self.weight_name = weight_name
        self.lm = AutoModel.from_pretrained(
            weight_name,
            cache_dir=cache_dir
        )
        self.n_classes = n_classes
        self.dropout = dropout
        self.no_label_corelation = no_label_corelation

        if self.freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        self.lm_dim = self.lm.config.hidden_size
        self.h1_dim = self.lm_dim // 2
        self.h2_dim = self.h1_dim // 2

        self.classifier = TwoLayerClassifier(
            self.lm_dim, self.h1_dim, self.h2_dim, self.n_classes,
            self.dropout
        )

        if not self.no_label_corelation:
            self.corelation_m = nn.Parameter(torch.Tensor(self.n_classes, self.n_classes))
            nn.init.xavier_uniform_(self.corelation_m,
                                    gain=nn.init.calculate_gain('relu'))

        if not self.use_lm:
            # use an LSTM on words
            self.lstm_dim = self.lm_dim // 2
            self.lstm = nn.LSTM(self.lm_dim, self.lstm_dim, 2, batch_first=True, bidirectional=True)


    def save_config(self, save_dir):
        fpath = os.path.join(save_dir, 'model_config.json')
        config = {
            'weight_name': self.weight_name,
            'n_classes': self.n_classes,
            'dropout': self.dropout,
            'freeze_lm': self.freeze_lm,
            'no_label_corelation': self.no_label_corelation,
            'use_lm': self.use_lm
        }
        json.dump(config, open(fpath, 'w'))

    def forward(self, **kwargs):
        if self.use_lm:
            return self.lm_forward(**kwargs)
        return self.no_lm_forward(**kwargs)

    def no_lm_forward(self, wemb, attention_mask, labels, **kwargs):
        # use an LSTM to encode word embeddings
        lstm_out, lstm_seq_lengths = self.lstm_forward(wemb, attention_mask, self.lstm)

        # average words
        sent_emb = lstm_out.sum(1) / lstm_seq_lengths.unsqueeze(1)

        z = self.classifier(sent_emb)
        return self.corelation_matrix_forward(z, labels)

    def lm_forward(self, input_ids, attention_mask, token_type_ids, labels, **kwargs):
        lm_output = self.lm(
            input_ids, attention_mask, token_type_ids=token_type_ids)
        pooled = lm_output[1]
        z = self.classifier(pooled)
        return self.corelation_matrix_forward(z, labels)

    def corelation_matrix_forward(self, z, labels):
        if not self.no_label_corelation:
            batch_size = z.shape[0]
            g = self.corelation_m.unsqueeze(0).expand(
                batch_size, self.n_classes, self.n_classes)
            e = torch.bmm(z.view(batch_size, 1, -1), g).view(batch_size, -1)
            yp = torch.bmm(labels.view(batch_size, 1, -1), g).view(batch_size, -1)

            reg_loss = torch.mean(self.corelation_m.pow(2))
            return e, yp, reg_loss
        else:
            return z, None, None

    def predict(self, **kwargs):
        if self.use_lm:
            return self.lm_predict(**kwargs)
        return self.no_lm_predict(**kwargs)

    def no_lm_predict(self, wemb, attention_mask, labels, **kwargs):
        logits, _, _ = self.no_lm_forward(wemb, attention_mask, labels)
        score = torch.sigmoid(logits)
        return score

    def lm_predict(self, input_ids, attention_mask, token_type_ids, labels, **kwargs):
        logits, _, _ = self.lm_forward(input_ids, attention_mask, token_type_ids, labels)
        score = torch.sigmoid(logits)
        return score


class SelfAttentionModel(BaseNaivePsychologyModel):
    def __init__(self, lm_dim, n_classes, dropout=0.2, cache_dir=None, **kwargs):
        super(SelfAttentionModel, self).__init__()

        self.lm_dim = lm_dim
        self.n_classes = n_classes
        self.dropout = dropout

        self.lstm_dim = self.lm_dim // 2
        self.s_lstm = nn.LSTM(self.lm_dim, self.lstm_dim, 1, batch_first=True, bidirectional=True)
        self.c_lstm = nn.LSTM(self.lm_dim, self.lstm_dim, 1, batch_first=True, bidirectional=True)

        w_dim = self.lstm_dim * 2 # bidirectional
        self.s_attn1 = nn.Linear(w_dim, w_dim // 2)
        self.s_attn2 = nn.Linear(w_dim // 2, 1)
        self.c_attn1 = nn.Linear(w_dim, w_dim // 2)
        self.c_attn2 = nn.Linear(w_dim // 2, 1)

        self.out1 = nn.Linear(w_dim * 2, w_dim)
        self.out2 = nn.Linear(w_dim, n_classes)

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)
        self.d4 = nn.Dropout(dropout)

        # init
        nn.init.xavier_uniform_(self.s_attn1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.s_attn2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.c_attn1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.c_attn2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.out1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.out2.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, s_wemb, s_attention_mask, c_wemb, c_attention_mask, **kwargs):
        s_lstm_out, s_lstm_seq_len = self.lstm_forward(s_wemb, s_attention_mask, self.s_lstm, pad_total_length=True)
        c_lstm_out, c_lstm_seq_len = self.lstm_forward(c_wemb, c_attention_mask, self.c_lstm, pad_total_length=True)

        # sent-char attn
        batch_size, n_steps, dim = s_lstm_out.shape
        s_a = F.relu(self.s_attn1(self.d1(s_lstm_out.view(-1, dim))))
        s_v = self.s_attn2(self.d2(s_a))
        s_v = s_v.view(batch_size, n_steps)
        s_v[s_attention_mask == 0] = float('-inf')
        s_alpha = torch.softmax(s_v, dim=1)

        # context attn
        c_a = F.relu(self.c_attn1(self.d1(c_lstm_out.view(-1, dim))))
        c_v = self.c_attn2(self.d2(c_a))
        c_v = c_v.view(batch_size, n_steps)
        c_v[c_attention_mask == 0] = float('-inf')
        c_alpha = torch.softmax(c_v, dim=1)

        s_x = (s_alpha.unsqueeze(-1).expand(s_lstm_out.shape) * s_lstm_out).sum(1)
        c_x = (c_alpha.unsqueeze(-1).expand(c_lstm_out.shape) * c_lstm_out).sum(1)
        x = torch.cat((s_x, c_x), dim=1)

        h1 = F.relu(self.out1(self.d3(x)))
        logits = self.out2(self.d4(h1))
        return logits

    def predict(self, s_wemb, s_attention_mask, c_wemb, c_attention_mask, **kwargs):
        logits = self.forward(s_wemb, s_attention_mask, c_wemb, c_attention_mask)
        score = torch.sigmoid(logits)
        return score

    def save_config(self, save_dir):
        fpath = os.path.join(save_dir, 'model_config.json')
        config = {
            'lm_dim': self.lm_dim,
            'n_classes': self.n_classes,
            'dropout': self.dropout
        }
        json.dump(config, open(fpath, 'w'))


class SimpleLM(BaseNaivePsychologyModel):
    def __init__(self, weight_name, dropout=0.2, freeze_lm=True, use_lstm=True, **kwargs):
        super(SimpleLM, self).__init__()

        self.maslow_n_classes = len(MASLOW_LABEL2IDX)
        self.reiss_n_classes = len(REISS_LABEL2IDX)
        self.plutchik_n_classes = len(PLUTCHIK_LABEL2IDX)

        self.dropout = dropout

        self.use_lstm = use_lstm
        self.freeze_lm = freeze_lm
        self.weight_name = weight_name
        self.lm = AutoModel.from_pretrained(weight_name)

        if self.freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False

        if self.use_lstm:
            self.in_dim = self.lm.config.hidden_size // 2
            self.lstm = nn.LSTM(self.lm.config.hidden_size, self.in_dim, 1, batch_first=True, bidirectional=True)
            self_in_dim *= 2 # bidirectional
        else:
            self.in_dim = self.lm.config.hidden_size
        self.h1_dim = self.in_dim // 2
        self.h2_dim = self.h1_dim // 2
        self.context_attn = nn.Linear(self.in_dim*2, 1)

        self.maslow_classifier = TwoLayerClassifier(
            self.in_dim, self.h1_dim, self.h2_dim, self.maslow_n_classes,
            self.dropout
        )
        self.reiss_classifier = TwoLayerClassifier(
            self.in_dim, self.h1_dim, self.h2_dim, self.reiss_n_classes,
            self.dropout
        )
        self.plutchik_classifier = TwoLayerClassifier(
            self.in_dim, self.h1_dim, self.h2_dim, self.plutchik_n_classes,
            self.dropout
        )

    def forward(self, input_ids, input_mask, sent_idxs, *pargs):
        b, t, w = input_ids.shape
        input_ids = input_ids.view(-1, w)
        input_mask = input_mask.view(-1, w)

        output = self.lm(input_ids, input_mask)
        word_emb = output[0]

        if self.use_lstm:
            unsorted_out, unsorted_seq_lengths = self.lstm_forward(word_emb, input_mask, self.lstm)

            # average words
            sent_emb = unsorted_out.sum(1) / unsorted_seq_lengths.unsqueeze(1)
            sent_emb = sent_emb.view(b, t, -1)
        else:
            # average word embeddings to represent a sentence
            sent_emb = self.average_words(word_emb, input_mask).view(b, t, -1)
        dim = sent_emb.shape[-1]

        # target sentence
        target_sent_idxs = sent_idxs[:, 0]
        target_sent_idxs = target_sent_idxs.unsqueeze(-1).expand(b, dim)
        target_sent_idxs = target_sent_idxs.unsqueeze(1)
        target_sent_emb = sent_emb.gather(1, target_sent_idxs).view(b, dim)

        # context
        # context_sent_idxs = sent_idxs[:, 1:]

        # because there could be no contextual sentence for the attention layer
        # to avoid NaN we include the target sentence in the context
        # so that there should be at least one
        context_sent_idxs = sent_idxs

        # we fill the paddings with index 0, which will be masked during attention
        tmp_context_sent_idxs = context_sent_idxs.clone()
        tmp_context_sent_idxs[tmp_context_sent_idxs == -1] = 0
        n_context = tmp_context_sent_idxs.shape[-1]
        context_embs = \
            sent_emb.gather(
                1,
                tmp_context_sent_idxs.unsqueeze(-1).expand(b, n_context, dim)
            )

        # attention
        pair_emb = torch.cat(
            (
                target_sent_emb.unsqueeze(1).expand(b, n_context, dim),
                context_embs
            ),
            dim=2
        )
        alpha = F.hardtanh(self.context_attn(pair_emb)).view(b, n_context)
        alpha[context_sent_idxs==-1] = float('-inf')
        alpha = torch.softmax(alpha, dim=1)
        # when there is no contextual sentence, we'll get nan
        # if torch.isnan(alpha).any():
        #     alpha[torch.isnan(alpha)] = 0

        h_c = (alpha.unsqueeze(2).expand(context_embs.shape) * context_embs).sum(1)

        # classification
        h_s = target_sent_emb
        h_e = torch.cat((h_s, h_c), dim=1)

        maslow_logits = self.maslow_classifier(h_e)
        reiss_logits = self.reiss_classifier(h_e)
        plutchik_logits = self.plutchik_classifier(h_e)

        return maslow_logits, reiss_logits, plutchik_logits

    def predict(self, input_ids, input_mask, sent_idxs, **kwargs):
        maslow_logits, reiss_logits, plutchik_logits = \
            self.forward(input_ids, input_mask, sent_idxs)

        maslow_probs = torch.sigmoid(maslow_logits)
        reiss_probs = torch.sigmoid(reiss_logits)
        plutchik_probs = torch.sigmoid(plutchik_logits)

        # maslow_pred = (maslow_probs >= 0.5).long()
        # reiss_pred = (reiss_probs >= 0.5).long()
        # plutchik_pred = (plutchik_probs >= 0.5).long()
        return maslow_probs, reiss_probs, plutchik_probs

    def save_config(self, save_dir):
        fpath = os.path.join(save_dir, 'model_config.json')
        config = {
            'weight_name': self.weight_name,
            'dropout': self.dropout,
            'freeze_lm': self.freeze_lm,
            'use_lstm': self.use_lstm
        }
        json.dump(config, open(fpath, 'w'))


class SimpleFFN(BaseNaivePsychologyModel):
    def __init__(self, lm_dim=1024, dropout=0.2, use_lstm=True, use_lstm_last_state=False,
                 use_context=True, **kwargs):
        super(SimpleFFN, self).__init__()

        self.use_context = use_context
        self.maslow_n_classes = 5
        self.reiss_n_classes = 19
        self.plutchik_n_classes = 8

        self.dropout = dropout

        self.use_lstm = use_lstm
        self.use_lstm_last_state = use_lstm_last_state

        self.lm_dim = lm_dim

        if self.use_lstm:
            self.in_dim = self.lm_dim // 2
            self.lstm = nn.LSTM(self.lm_dim, self.in_dim, 1, batch_first=True, bidirectional=True)
            self.in_dim *= 2  # bidirectional
        else:
            self.in_dim = self.lm_dim

        self.h1_dim = self.in_dim // 2
        self.h2_dim = self.h1_dim // 2
        self.context_attn = nn.Linear(self.in_dim*2, 1)

        cls_in_dim = self.in_dim*2 if self.use_context else self.in_dim
        self.maslow_classifier = TwoLayerClassifier(
            cls_in_dim, self.h1_dim, self.h2_dim, self.maslow_n_classes,
            self.dropout
        )
        self.reiss_classifier = TwoLayerClassifier(
            cls_in_dim, self.h1_dim, self.h2_dim, self.reiss_n_classes,
            self.dropout
        )
        self.plutchik_classifier = TwoLayerClassifier(
            cls_in_dim, self.h1_dim, self.h2_dim, self.plutchik_n_classes,
            self.dropout
        )

    def forward(self, word_emb, input_mask, sent_idxs, *pargs):
        b, t, w = input_mask.shape
        word_emb = word_emb.view(-1, w, self.lm_dim)
        input_mask = input_mask.view(-1, w)
        if self.use_lstm:
            unsorted_out, unsorted_seq_lengths = self.lstm_forward(word_emb, input_mask, self.lstm)

            if self.use_lstm_last_state:
                # get the first token's second half
                lstm_h_dim = self.lm_dim // 2
                right_left_last = unsorted_out[:, 0, lstm_h_dim:] # second half

                # get the last token's first half
                idxs = (unsorted_seq_lengths - 1).unsqueeze(-1).expand(b*t, self.lm_dim).unsqueeze(1)
                left_right_last = unsorted_out.gather(1, idxs).view(b*t, self.lm_dim)[:, :lstm_h_dim]

                sent_emb = torch.cat((left_right_last, right_left_last), dim=1).view(b, t, -1)
            else:
                # average words
                sent_emb = unsorted_out.sum(1) / unsorted_seq_lengths.unsqueeze(1)
                sent_emb = sent_emb.view(b, t, -1)
        else:
            # average word embeddings to represent a sentence
            sent_emb = self.average_words(word_emb, input_mask).view(b, t, -1)
        dim = sent_emb.shape[-1]

        # target sentence
        target_sent_idxs = sent_idxs[:, 0]
        target_sent_idxs = target_sent_idxs.unsqueeze(-1).expand(b, dim)
        target_sent_idxs = target_sent_idxs.unsqueeze(1)
        target_sent_emb = sent_emb.gather(1, target_sent_idxs).view(b, dim)

        if self.use_context:
            # context
            # context_sent_idxs = sent_idxs[:, 1:]

            # because there could be no contextual sentence for the attention layer
            # to avoid NaN we include the target sentence in the context
            # so that there should be at least one
            context_sent_idxs = sent_idxs

            # we fill the paddings with index 0, which will be masked during attention
            tmp_context_sent_idxs = context_sent_idxs.clone()
            tmp_context_sent_idxs[tmp_context_sent_idxs == -1] = 0
            n_context = tmp_context_sent_idxs.shape[-1]
            context_embs = \
                sent_emb.gather(
                    1,
                    tmp_context_sent_idxs.unsqueeze(-1).expand(b, n_context, dim)
                )

            # attention
            pair_emb = torch.cat(
                (
                    target_sent_emb.unsqueeze(1).expand(b, n_context, dim),
                    context_embs
                ),
                dim=2
            )
            alpha = F.hardtanh(self.context_attn(pair_emb)).view(b, n_context)
            alpha[context_sent_idxs==-1] = float('-inf')
            alpha = torch.softmax(alpha, dim=1)
            # when there is no contextual sentence, we'll get nan
            # if torch.isnan(alpha).any():
            #     alpha[torch.isnan(alpha)] = 0

            h_c = (alpha.unsqueeze(2).expand(context_embs.shape) * context_embs).sum(1)

            # classification
            h_s = target_sent_emb
            h_e = torch.cat((h_s, h_c), dim=1)
        else:
            h_e = target_sent_emb

        maslow_logits = self.maslow_classifier(h_e)
        reiss_logits = self.reiss_classifier(h_e)
        plutchik_logits = self.plutchik_classifier(h_e)
        return maslow_logits, reiss_logits, plutchik_logits

    def predict(self, word_emb, input_mask, sent_idxs, *pargs):
        maslow_logits, reiss_logits, plutchik_logits = \
            self.forward(word_emb, input_mask, sent_idxs)

        maslow_probs = torch.sigmoid(maslow_logits)
        reiss_probs = torch.sigmoid(reiss_logits)
        plutchik_probs = torch.sigmoid(plutchik_logits)

        # maslow_pred = (maslow_probs >= 0.5).long()
        # reiss_pred = (reiss_probs >= 0.5).long()
        # plutchik_pred = (plutchik_probs >= 0.5).long()
        return maslow_probs, reiss_probs, plutchik_probs

    def save_config(self, save_dir):
        fpath = os.path.join(save_dir, 'model_config.json')
        config = {
            'lm_dim': self.lm_dim,
            'dropout': self.dropout,
            'use_lstm': self.use_lstm,
            'use_lstm_last_state': self.use_lstm_last_state,
            'use_context': self.use_context
        }
        json.dump(config, open(fpath, 'w'))


def sample_png_edges(edge_src, edge_dest, edge_types,
                     n_nodes, n_neg_per_pos_edges,
                     edge_sample_rate,
                     rtype_distr):

    n_edges = edge_types.shape[0]
    n_missing_edges = int(n_edges * edge_sample_rate)
    if n_missing_edges == 0:
        logger.warning('number of missing edges = 0')
        return None

    # logger.debug('n_edges={}, n_missing_edges={}'.format(n_edges, n_missing_edges))

    rtype_pool = {}
    for idx, r in enumerate(edge_types.tolist()):
        if r not in rtype_pool:
            rtype_pool[r] = []
        rtype_pool[r].append(idx)

    # cands and dist
    cands = list(rtype_pool.keys())
    p = [rtype_distr[c] for c in cands]
    s = sum(p)
    p = [x / s for x in p]

    # randomly sample missing edges
    missing_idxs = set()
    k = n_missing_edges
    while len(missing_idxs) < n_missing_edges:
        draw = np.random.choice(cands, k, p=p)
        for d in draw:
            candidate_idxs = rtype_pool[d]
            ridx = random.randint(0, len(candidate_idxs)-1)
            missing_idxs.add(candidate_idxs[ridx])

        k = n_missing_edges - len(missing_idxs)

    # collect edges
    png_edges = torch.cat(
        (edge_src.view(1, -1), edge_types.view(1, -1), edge_dest.view(1, -1)),
         dim=0
    )
    remaining_idxs = sorted(list(set(range(n_edges)) - missing_idxs))
    missing_idxs = sorted(list(missing_idxs))
    pos_edges = png_edges[:, missing_idxs]
    input_edges = deepcopy(png_edges[:, remaining_idxs])

    n_rtypes = len(rtype_distr)
    # negative edges
    neg_edges = _sample_negative_edges(
        input_edges, pos_edges, n_nodes, n_rtypes, n_neg_per_pos_edges)

    return input_edges, pos_edges, neg_edges


def _sample_negative_edges(input_edges, pos_edges, n_nodes, n_rtypes, n_neg_per_pos_edge,
                           max_n_tried=10000):
    n_pos_edges = pos_edges.shape[1]
    n_neg_edges = n_pos_edges * n_neg_per_pos_edge
    neg_edges = torch.zeros((3, n_neg_edges), dtype=torch.long)

    # collect existed
    existed_edges = set()
    for i in range(input_edges.shape[1]):
        e = tuple(input_edges[:, i].view(-1).long().tolist())
        existed_edges.add(e)
    for i in range(pos_edges.shape[1]):
        e = tuple(pos_edges[:, i].view(-1).long().tolist())
        existed_edges.add(e)

    sampled_neg = set()
    for i in range(n_pos_edges):
        e = tuple(pos_edges[:, i].view(-1).long().tolist())

        n_tried = 0
        count = 0
        while count < n_neg_per_pos_edge:
            # pick what to truncate
            truncating_target = random.randint(0, 2)
            if truncating_target == 0:
                # sample head
                s = (random.randint(0, n_nodes-1), e[1], e[2])
                if s not in existed_edges and s not in sampled_neg:
                    sampled_neg.add(s)
                    for k in range(3):
                        neg_edges[k, i*n_neg_per_pos_edge + count] = s[k]
                    count += 1

            elif truncating_target == 1:
                # sample tail
                s = (e[0], e[1], random.randint(0, n_nodes-1))
                if s not in existed_edges and s not in sampled_neg:
                    sampled_neg.add(s)
                    for k in range(3):
                        neg_edges[k, i*n_neg_per_pos_edge + count] = s[k]
                    count += 1

            else:
                # sample rel
                s = (e[0], random.randint(0, n_rtypes-1), e[2])
                if s not in existed_edges and s not in sampled_neg:
                    sampled_neg.add(s)
                    for k in range(3):
                        neg_edges[k, i*n_neg_per_pos_edge + count] = s[k]
                    count += 1
            n_tried += 1
            if n_tried >= max_n_tried:
                return None
    logger.debug('count={}, tried={}'.format(count, n_tried))
    return neg_edges


def encode_labels(maslow_labels, reiss_labels, plutchik_labels):
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
    return (
        torch.LongTensor(y_maslow),
        torch.LongTensor(y_reiss),
        torch.LongTensor(y_plutchik)
    )


def _get_label_sentence(c_name, idx2label, label_sentence_map, task):
    if task == 'plutchik':
        sent = '{} is '.format(c_name)
    else:
        sent = '{} wants the '.format(c_name)
    for i in range(len(idx2label)):
        label = idx2label[i]
        s = label_sentence_map[label]
        if i == 0:
            sent += s
        else:
            sent += ', {}'.format(s)
    sent += '.'
    return sent


def get_label_sentences(c_name, maslow_idx2label, reiss_idx2label, plutchik_idx2label):
    # def _get_label_sentences(idx2label, label_sentences):
    #     sents = []
    #     for i in range(len(idx2label)):
    #         label = idx2label[i]
    #         s = label_sentences[label]
    #         sents.append('{} {}'.format(c_name, s))
    #     return sents
    # maslow_sents = _get_label_sentences(maslow_idx2label, MASLOW_LABEL_SENTENCES)
    # reiss_sents = _get_label_sentences(reiss_idx2label, REISS_LABEL_SENTENCES)
    # plutchik_sents = _get_label_sentences(plutchik_idx2label, PLUTCHIK_LABEL_SENTENCES)
    # return maslow_sents, reiss_sents, plutchik_sents

    maslow_sents = [_get_label_sentence(c_name, maslow_idx2label, MASLOW_LABEL_SENTENCES, 'maslow')]
    reiss_sents = [_get_label_sentence(c_name, reiss_idx2label, REISS_LABEL_SENTENCES, 'reiss')]
    plutchik_sents = [_get_label_sentence(c_name, plutchik_idx2label, PLUTCHIK_LABEL_SENTENCES, 'plutchik')]
    return maslow_sents, reiss_sents, plutchik_sents


def prepare_png_model_inputs(bert_inputs, rgcn_inputs, gpu_id):
    input_ids = torch.from_numpy(bert_inputs['input_ids'])
    attention_mask = torch.from_numpy(bert_inputs['input_mask'])
    token_type_ids = torch.from_numpy(bert_inputs['token_type_ids'])
    edge_src = torch.LongTensor(rgcn_inputs['edge_src'])
    edge_dest = torch.LongTensor(rgcn_inputs['edge_dest'])
    edge_types = torch.LongTensor(rgcn_inputs['edge_types'])
    edge_norms = torch.FloatTensor(rgcn_inputs['edge_norms'])
    if gpu_id != -1:
        input_ids = input_ids.cuda(gpu_id)
        attention_mask = attention_mask.cuda(gpu_id)
        token_type_ids = token_type_ids.cuda(gpu_id)

        # edge_src = edge_src.cuda(gpu_id)
        # edge_dest = edge_dest.cuda(gpu_id)
        edge_types = edge_types.cuda(gpu_id)
        edge_norms = edge_norms.cuda(gpu_id)

    batch = {
        'input_ids': [input_ids],
        'attention_mask': [attention_mask],
        'token_type_ids': [token_type_ids],
        'edge_src': [edge_src],
        'edge_dest': [edge_dest],
        'edge_types': [edge_types],
        'edge_norms': [edge_norms]
    }
    return batch
