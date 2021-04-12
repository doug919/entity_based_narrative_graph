import logging
from collections import OrderedDict

import torch
import numpy as np
import dgl
from transformers import AutoTokenizer

from ..common import utils
from ..common import discourse as disc
from ..models.naive_psych import MASLOW_LABEL2IDX
from ..models.naive_psych import REISS_LABEL2IDX
from ..models.naive_psych import PLUTCHIK_LABEL2IDX
from ..models.naive_psych import SENTIMENT_LABEL2IDX
from ..models.naive_psych import MASLOW_LABEL_SENTENCES
from ..models.naive_psych import REISS_LABEL_SENTENCES
from ..models.naive_psych import PLUTCHIK_LABEL_SENTENCES
from ..models.naive_psych import _get_label_sentence


logger = logging.getLogger(__name__)


class PNGNode:
    def __init__(self, nid, char, sent1, sent2, story_id, sent_idx,
                 maslow_labels, reiss_labels, plutchik_labels):
        self.nid = nid
        self.char = char
        self.sent1 = sent1
        self.sent2 = sent2
        self.sid = story_id
        self.sent_idx = sent_idx

        self.maslow_labels = maslow_labels
        self.reiss_labels = reiss_labels
        self.plutchik_labels = plutchik_labels

    def __repr__(self):
        return '{}:{}:{}:{}'.format(self.nid, self.char, self.sent1, self.sent2)

    def get_node_info(self):
        return {
            'sid': self.sid,
            'nid': self.nid,
            'character': self.char,
            'sent_idx': self.sent_idx,
            'sentence1': self.sent1,
            'sentence2': self.sent2,
            'maslow': self.maslow_labels,
            'reiss': self.reiss_labels,
            'plutchik': self.plutchik_labels
        }

    def get_bert_inputs(self, tokenizer, label_sent_task, max_seq_len, idx2label, label_sentence_map, task):
        sent2 = self.sent2
        if label_sent_task is not None:
            label_sent = _get_label_sentence(self.char, idx2label, label_sentence_map, task)
            sent2 = sent2 + ' ' + label_sent
        if max_seq_len == -1:
            inputs = tokenizer(self.sent1, sent2, add_special_tokens=True,
                               return_token_type_ids=True, return_attention_mask=True)
        else:
            # only_first: the target sentence
            inputs = tokenizer(self.sent1, sent2, add_special_tokens=True,
                               max_length=max_seq_len, padding='max_length', truncation='longest_first',
                               return_token_type_ids=True, return_attention_mask=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        return input_ids, attention_mask, token_type_ids

    def get_labels(self):
        maslow_y = np.zeros(len(MASLOW_LABEL2IDX), dtype=np.int64)
        reiss_y = np.zeros(len(REISS_LABEL2IDX), dtype=np.int64)
        plutchik_y = np.zeros(len(PLUTCHIK_LABEL2IDX), dtype=np.int64)
        for l in self.maslow_labels:
            maslow_y[MASLOW_LABEL2IDX[l]] = 1
        for l in self.reiss_labels:
            reiss_y[REISS_LABEL2IDX[l]] = 1
        for l in self.plutchik_labels:
            plutchik_y[PLUTCHIK_LABEL2IDX[l]] = 1
        return maslow_y, reiss_y, plutchik_y

    def get_sentiment_labels(self, analyzer, idx2label):
        ss = analyzer.polarity_scores(self.sent1)
        labels = []
        for i in range(len(idx2label)):
            labels.append(ss[idx2label[i]])
        return labels


class PsychologyNarrativeGraph:
    def __init__(self, gid, rtype2idx):
        self.gid = gid
        self.nodes = {}
        self.edges = set()
        self.rtype2idx = rtype2idx

    def add_node(self, n):
        if n.nid in self.nodes:
            logger.warning('failed to add node: {}'.format(n))
            return False
        self.nodes[n.nid] = n
        return True

    def add_edge(self, nid1, nid2, rtype):
        t = (nid1, nid2, rtype)
        if t in self.edges:
            logger.debug('failed add_edge: {} existed.'.format(t))
            return False
        self.edges.add(t)
        return True

    def truncate_graph(self, max_nid):
        new_nodes = {k: v for k, v in self.nodes.items() if k <= max_nid}
        new_edges = set([e for e in self.edges if e[0] <= max_nid and e[1] <= max_nid])
        self.nodes = new_nodes
        self.edges = new_edges

    def get_edge_stats(self):
        counts = {}
        for nid1, nid2, rtype in self.edges:
            if rtype not in counts:
                counts[rtype] = 0
            counts[rtype] += 1
        return counts

    def to_dgl_inputs(self, tokenizer, label_sent_task, max_seq_len,
                      has_label=True, use_sentiment_labels=False, sentiment_analyzer=None):
        assert label_sent_task in ['maslow', 'reiss', 'plutchik']
        if label_sent_task == 'maslow':
            idx2label = {v: k for k, v in MASLOW_LABEL2IDX.items()}
            label_sentence_map = MASLOW_LABEL_SENTENCES
        elif label_sent_task == 'reiss':
            idx2label = {v: k for k, v in REISS_LABEL2IDX.items()}
            label_sentence_map = REISS_LABEL_SENTENCES
        else:
            idx2label = {v: k for k, v in PLUTCHIK_LABEL2IDX.items()}
            label_sentence_map = PLUTCHIK_LABEL_SENTENCES
        sentiment_idx2label = {v: k for k, v in SENTIMENT_LABEL2IDX.items()}


        g = dgl.DGLGraph()
        g.add_nodes(len(self.nodes))

        # edge inputs
        edge_list = list(self.edges)
        nid1s = [e[0] for e in edge_list]
        nid2s = [e[1] for e in edge_list]
        edge_types = [self.rtype2idx[e[2]] for e in edge_list]
        g.add_edges(nid1s, nid2s)

        edge_norms = []
        for nid1, nid2, rtype in edge_list:
            assert nid1 != nid2, \
                'should not include self loop, which is handled in the model'
            edge_norms.append(1.0 / g.in_degree(nid2))

        edge_types = np.array(edge_types, dtype=np.int64)
        # edge_norms = torch.from_numpy(np.array(edge_norms)).unsqueeze(1).float()
        edge_norms = np.array(edge_norms, dtype=np.float32)
        # g.edata.update({'rel_type': edge_types})
        # g.edata.update({'norm': edge_norms})

        all_node_info = []
        all_sentiment_labels = []
        all_maslow_labels, all_reiss_labels, all_plutchik_labels = [], [], []
        all_input_ids, all_input_masks, all_token_type_ids = [], [], []
        for nid in range(len(self.nodes)):
            n = self.nodes[nid]
            node_input_ids, node_input_masks, node_token_type_ids = \
                n.get_bert_inputs(
                    tokenizer, label_sent_task, max_seq_len, idx2label, label_sentence_map, label_sent_task)

            node_info = n.get_node_info()
            all_node_info.append(node_info)

            all_input_ids.append(node_input_ids)
            all_input_masks.append(node_input_masks)
            all_token_type_ids.append(node_token_type_ids)

            if has_label:
                if use_sentiment_labels:
                    all_sentiment_labels.append(
                        n.get_sentiment_labels(sentiment_analyzer, sentiment_idx2label)
                    )
                else:
                    maslow_labels, reiss_labels, plutchik_labels = n.get_labels()
                    all_maslow_labels.append(maslow_labels)
                    all_reiss_labels.append(reiss_labels)
                    all_plutchik_labels.append(plutchik_labels)
        if max_seq_len != -1: # not testing mode
            all_input_ids = np.array(all_input_ids, dtype=np.int64)
            all_input_masks = np.array(all_input_masks, dtype=np.int64)
            all_token_type_ids = np.array(all_token_type_ids, dtype=np.int64)

        bert_inputs = {
            'input_ids': all_input_ids,
            'input_mask': all_input_masks,
            'token_type_ids': all_token_type_ids,
        }
        rgcn_inputs = {
            'num_nodes': len(self.nodes),
            'edge_src': nid1s,
            'edge_dest': nid2s,
            'edge_types': edge_types,
            'edge_norms': edge_norms
        }
        if not has_label:
            labels = None
        else:
            if use_sentiment_labels:
                try:
                    labels = {
                        'sentiment': np.vstack(all_sentiment_labels)
                    }
                except:
                    import pdb; pdb.set_trace()
            else:
                all_maslow_labels = np.vstack(all_maslow_labels)
                all_reiss_labels = np.vstack(all_reiss_labels)
                all_plutchik_labels = np.vstack(all_plutchik_labels)
                labels = {
                    'maslow': np.vstack(all_maslow_labels),
                    'reiss': np.vstack(all_reiss_labels),
                    'plutchik': np.vstack(all_plutchik_labels)
                }
        return bert_inputs, rgcn_inputs, labels, all_node_info


def get_majority_labels(c_ann, use_count_plutchik=False):
    mlabels = {'maslow': [], 'reiss': [], 'plutchik': []}

    # motiv
    maslow_votes = {}
    reiss_votes = {}
    for i in range(3):
        ann_id = 'ann{}'.format(i)
        # maslow
        if ann_id in c_ann['motiv'] and 'maslow' in c_ann['motiv'][ann_id]:
            label_list = c_ann['motiv'][ann_id]['maslow']
            for l in label_list:
                if l in ['none', 'na', 'indep']:
                    continue
                if l not in MASLOW_LABEL2IDX:
                    logger.warning('maslow {} not fuond'.format(l))
                    continue
                if l not in maslow_votes:
                    maslow_votes[l] = 1
                else:
                    maslow_votes[l] += 1
        # reiss
        if ann_id in c_ann['motiv'] and 'reiss' in c_ann['motiv'][ann_id]:
            label_list = c_ann['motiv'][ann_id]['reiss']
            for l in label_list:
                if l in ['none', 'na', 'indep']:
                    continue
                if l not in REISS_LABEL2IDX:
                    logger.warning('reiss {} not fuond'.format(l))
                    continue
                if l not in reiss_votes:
                    reiss_votes[l] = 1
                else:
                    reiss_votes[l] += 1
    # maslow
    for l, cnt in maslow_votes.items():
        if cnt >= 2:
            mlabels['maslow'].append(l)
            # assert l in MASLOW_LABEL2IDX, '{}'.format(l)
    # reiss
    for l, cnt in reiss_votes.items():
        if cnt >= 2:
            mlabels['reiss'].append(l)
            # assert l in REISS_LABEL2IDX, '{}'.format(l)

    # emotion
    plutchik_votes = {l: (0, 0) for l, idx in PLUTCHIK_LABEL2IDX.items() }
    n_annotators = 0
    for i in range(3):
        ann_id = 'ann{}'.format(i)
        if ann_id in c_ann['emotion'] and 'plutchik' in c_ann['emotion'][ann_id]:
            n_annotators += 1
            label_list = c_ann['emotion'][ann_id]['plutchik']
            for l in label_list:
                em, sc = l.split(':')
                sc = int(sc)
                plutchik_votes[em] = (plutchik_votes[em][0] + sc, plutchik_votes[em][1] + 1)
    for l, (sc, cnt) in plutchik_votes.items():
        if l in ['none', 'na', 'indep']:
            continue
        if l not in PLUTCHIK_LABEL2IDX:
            logger.warning('plutchik {} not fuond'.format(l))
            continue

        ## this paper use the wrong labeling scheme
        ## https://github.com/StonyBrookNLP/emotion-label-semantics
        if use_count_plutchik:
            if cnt >= 2:
                mlabels['plutchik'].append(l)
        else:
            avg = float(sc) / n_annotators if n_annotators > 0 else 0.0
            if avg >= 2.0:
                mlabels['plutchik'].append(l)
                # assert l in PLUTCHIK_LABEL2IDX, '{}'.format(l)
    return mlabels['maslow'], mlabels['reiss'], mlabels['plutchik']



def create_psychology_narrative_graph_unsupervised(
        sid, parse, dmarkers, rtype2idx, is_rocstories=True):
    ng = PsychologyNarrativeGraph(sid, rtype2idx)

    # use all coref mentions
    # ToDo: check is all roc stories have 5 sentences
    n_sents = 5 if is_rocstories else len(parse['sentences'])
    sent_nids = {i: [] for i in range(n_sents)}
    cur_nid = 0
    for cid, cchain in parse['corefs'].items():
        chain_nids = []
        cchain_sent_nums = sorted(list(set([m['sentNum'] for m in cchain])))
        for m in cchain:
            sent_idx = m['sentNum'] - 1
            toks = [tok['word'] for tok in parse['sentences'][sent_idx]['tokens']]
            sent = ' '.join(toks)
            c_name = m['text']

            # sent1 is the current sentence
            sent1 = sent

            # sentence2 is a concatenation of char-based context sentences and label sentences
            context_texts = get_context_sentence_list_unsupervised(m, cchain_sent_nums, parse['sentences'])
            context = ' '.join(context_texts) if len(context_texts) > 0 else 'No context.'
            sent2 = context

            # create a node
            node = PNGNode(cur_nid, c_name, sent1, sent2, sid, sent_idx,
                           None, None, None)
            cur_nid += 1
            ng.add_node(node)

            chain_nids.append(node.nid)

            sent_nids[sent_idx].append(node.nid)

        # CNEXT
        for i in range(len(chain_nids)-1):
            nid1, nid2 = chain_nids[i], chain_nids[i+1]
            ng.add_edge(nid1, nid2, 'cnext')

    # NEXT
    for i in range(n_sents - 1):
        for nid1 in sent_nids[i]:
            for nid2 in sent_nids[i+1]:
                ng.add_edge(nid1, nid2, 'next')

    # DISCOURSE_NEXT:
    disc_rels = []
    for i_sent in range(n_sents):
        matched = disc._find_matched_markers(parse['sentences'][i_sent], dmarkers)
        if matched is None:
            continue

        if i_sent > 0:
            # assume we has the discourse relation between the current and previous sentence
            for nid1 in sent_nids[i_sent-1]:
                for nid2 in sent_nids[i_sent]:
                    ng.add_edge(nid1, nid2, matched[2])
    return ng


def create_psychology_narrative_graph(
        sid, doc, parse, dmarkers, rtype2idx, use_count_plutchik=False, has_label=True):

    # doc: is in NaivePsychology dataset's json format
    ng = PsychologyNarrativeGraph(sid, rtype2idx)

    sent_nids = {}
    c_name_nids = {}
    cur_nid = 0
    for sent_num in range(1, 6):
        sent = doc['lines'][str(sent_num)]['text']
        sent_nids[sent_num-1] = []

        # line-char examples
        chars = doc['lines'][str(sent_num)]['characters']
        for c_name, c_ann in chars.items():
            if not c_ann['app']:
                continue
            if has_label:
                maslow_labels, reiss_labels, plutchik_labels = \
                    get_majority_labels(c_ann, use_count_plutchik=use_count_plutchik)
            else:
                maslow_labels = None
                reiss_labels = None
                plutchik_labels = None
            # sent1 is the current sentence
            sent1 = sent

            # sentence2 is a concatenation of char-based context sentences and label sentences
            context_texts = get_context_sentence_list(sent_num, doc, c_name)
            context = ' '.join(context_texts) if len(context_texts) > 0 else 'No context.'
            sent2 = context

            node = PNGNode(cur_nid, c_name, sent1, sent2, sid, sent_num-1,
                           maslow_labels, reiss_labels, plutchik_labels)
            cur_nid += 1
            ng.add_node(node)

            sent_nids[sent_num-1].append(node.nid)

            if c_name not in c_name_nids:
                c_name_nids[c_name] = []
            c_name_nids[c_name].append(node.nid)

    # NEXT
    for i in range(4):
        for nid1 in sent_nids[i]:
            for nid2 in sent_nids[i+1]:
                ng.add_edge(nid1, nid2, 'next')

    # CNEXT:
    for c_name, cnids in c_name_nids.items():
        for i in range(len(cnids)-1):
            for j in range(i+1, len(cnids)):
                ng.add_edge(cnids[i], cnids[j], 'cnext')

    # DISCOURSE_NEXT:
    disc_rels = []
    for i_sent in range(5):
        matched = disc._find_matched_markers(parse['sentences'][i_sent], dmarkers)
        if matched is None:
            continue

        if i_sent > 0:
            # assume we has the discourse relation between the current and previous sentence
            for nid1 in sent_nids[i_sent-1]:
                for nid2 in sent_nids[i_sent]:
                    ng.add_edge(nid1, nid2, matched[2])
    return ng


def get_context_sentence_list_unsupervised(m, cchain_sent_nums, sentences):
    context_idxs = [sn-1 for sn in cchain_sent_nums if sn < m['sentNum']]

    context_sents = []
    for idx in context_idxs:
        toks = [t['word'] for t in sentences[idx]['tokens']]
        context_sents.append(' '.join(toks))
    return context_sents


def get_context_line_nums_except(i, lines, c_name):
    context = [j for j in range(1, 6) if j < i and lines[str(j)]['characters'][c_name]['app']]
    return context


def get_context_sentence_list(sent_num, doc, c_name):
    context_line_nums = get_context_line_nums_except(
        sent_num, doc['lines'], c_name)
    context_texts = [doc['lines'][str(cln)]['text'] for cln in context_line_nums]
    return context_texts
