import logging
from collections import OrderedDict

import torch
import numpy as np
import dgl
from transformers import AutoTokenizer

from ..common import utils
from ..common import discourse as disc


MASK_DISC_CONN = '[MASK]'


class NGNode:
    def __init__(self, nid, ntype, name=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nid = nid
        assert ntype in ['entity', 'predicate']
        self.ntype = ntype
        self.name = name

        # (sent, pred_tidx, role_tidx)
        self.instances = []
        # self.logger.debug('create node {}:{}:{}'.format(self.nid, self.ntype, self.rep_token))

    def __repr__(self):
        return '({}:{}:{}:{})'.format(
            self.nid, self.name, self.ntype, len(self.instances))

    def add_instance(self, rep, sent, i_sent, pred_tidx, role_tidx):
        self.instances.append((rep, sent, i_sent, pred_tidx, role_tidx))
        if self.ntype == 'predicate':
            # predicate node should have unique instance
            assert len(self.instances) == 1
        # self.logger.debug('add instance {}:{}:{}: {}'.format(self.nid, self.ntype, self.rep_token, self.instances[-1]))

    def get_bert_inputs(self, bert_tokenizer, max_seq_len, give_up_long_sentence=True):
        all_input_ids, all_input_masks, all_token_type_ids = [], [], []
        all_target_idxs = []
        for inst in self.instances:
            pred_tidx = inst[3]
            entity_tidx = inst[4]
            words = inst[1].split(' ')

            # TODO: too dirty, refactor
            input_ids, input_mask, token_type_ids, event_idxs = \
                utils.prepare_bert_event_input(
                    words, pred_tidx, bert_tokenizer, max_seq_len=max_seq_len)
            if input_ids is None:
                if give_up_long_sentence:
                    # event appears in the tail of a long sentences
                    return None, None, None, None
                else:
                    # get the max_seq_len tokens around the event
                    input_ids, input_mask, token_type_ids, event_idxs = \
                        utils.prepare_bert_event_input(
                            words, pred_tidx, bert_tokenizer, max_seq_len=-1)

                    # remove CLS, SEP
                    tmp_input_ids = input_ids[1:-1]
                    tmp_input_mask = input_mask[1:-1]
                    tmp_token_type_ids = token_type_ids[1:-1]
                    eidx = event_idxs[0] - 1

                    # get second half
                    half_len = (max_seq_len-3) // 2
                    new_input_ids = tmp_input_ids[eidx: eidx + half_len]
                    # get first half
                    ahead = max_seq_len - 2 - len(new_input_ids)
                    new_input_ids = tmp_input_ids[eidx - ahead: eidx + half_len]
                    new_input_mask = tmp_input_mask[eidx - ahead: eidx + half_len]
                    new_token_type_ids = tmp_token_type_ids[eidx - ahead: eidx + half_len]
                    new_eidx = ahead

                    # attach CLS, SEP
                    new_input_ids = [bert_tokenizer.cls_token_id] + \
                        new_input_ids + [bert_tokenizer.sep_token_id]
                    new_input_mask = [1] + new_input_mask + [1]
                    new_token_type_ids = [0] + new_token_type_ids + [0]
                    new_eidx += 1
                    new_event_idxs = [i for i in range(new_eidx, new_eidx + len(event_idxs))]

                    # replace
                    input_ids = new_input_ids
                    input_mask = new_input_mask
                    token_type_ids = new_token_type_ids
                    event_idxs = new_event_idxs

            # target idxs
            # this is the output embedding's index that we want
            if entity_tidx is None:
                all_target_idxs.append(event_idxs[0])
            else:
                _, entity_idxs = utils._bert_event_input(
                    words, entity_tidx, entity_tidx+1, bert_tokenizer)
                if entity_idxs is None or entity_idxs[0] >= max_seq_len-2:
                    # entity appears in the tail of a long sentences
                    return None, None, None, None
                all_target_idxs.append(entity_idxs[0]+1) # plus 1 for CLS

            all_input_ids.append(input_ids)
            all_input_masks.append(input_mask)
            all_token_type_ids.append(token_type_ids)
        return all_input_ids, all_input_masks, all_token_type_ids, all_target_idxs


class NarrativeGraph:
    def __init__(self, doc_id, rtypes):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.doc_id = doc_id
        self.nodes = {}
        self.edges = set()
        if rtypes is None:
            self.logger.warning('rtype2idx is None. using default.')
            self.rtypes = {
                "next": 0,
                "cnext": 1,
                "Temporal.Asynchronous.Precedence": 2,
                "Temporal.Asynchronous.Succession": 3,
                "Temporal.Synchrony": 4,
                "Contingency.Cause.Reason": 5,
                "Contingency.Cause.Result": 6,
                "Comparison.Contrast": 7,
                "s": 8,
                "o": 9,
                "prep": 10,
                "rev_s": 11,
                "rev_o": 12,
                "rev_prep": 13
            }
        else:
            self.rtypes = rtypes

    def add_node(self, n):
        if n.nid in self.nodes:
            self.logger.warning('failed to add node: {}'.format(n.nid))
            return False
        assert n.nid == len(self.nodes)
        self.nodes[n.nid] = n
        # if add_self_loop_edge:
        #     self.add_edge(n.nid, n.nid, 'self')
        # self.logger.debug('add_node: {}'.format(n.nid))
        return True

    def add_edge(self, nid1, nid2, rtype):
        if (nid1, nid2, rtype) in self.edges:
            self.logger.debug('failed add_edge: {} existed.'.format((nid1, nid2, rtype)))
            return False
        # self.logger.debug('add_edge: ({},{}): {}'.format(nid1, nid2, rtype))
        self.edges.add((nid1, nid2, rtype))
        return True

    def get_edge_stats(self):
        counts = {}
        for nid1, nid2, rtype in self.edges:
            if rtype not in counts:
                counts[rtype] = 0
            counts[rtype] += 1
        return counts

    def get_loc2nid(self):
        loc2nid = {}
        for nid, n in self.nodes.items():
            for inst in n.instances:
                if inst[2] not in loc2nid:
                    loc2nid[inst[2]] = {}

                if n.ntype == 'predicate':
                    loc2nid[inst[2]][inst[3]] = nid
                else: # entity
                    assert n.ntype == 'entity'
                    loc2nid[inst[2]][inst[4]] = nid
        return loc2nid

    def finalize(self):
        # sort entity node instances so that the first mention can be retrieved easily
        for nid, n in self.nodes.items():
            if n.ntype != 'entity' or len(n.instances) == 1:
                continue
            n.instances = sorted(n.instances, key=lambda x: (x[2], x[3]))

    def get_predicate2nids(self):
        pred2nids = {}
        for nid, n in self.nodes.items():
            if n.ntype == 'predicate':
                for ni in n.instances:

                    if ni[0] not in pred2nids:
                        pred2nids[ni[0]] = []
                    pred2nids[ni[0]].append(nid)
        return pred2nids

    def to_dgl_inputs(self, bert_tokenizer, max_seq_len=128, give_up_long_sentence=True):
        self.finalize()
        g = dgl.DGLGraph()
        g.add_nodes(len(self.nodes))

        edge_list = list(self.edges)
        nid1s = [e[0] for e in edge_list]
        nid2s = [e[1] for e in edge_list]
        edge_types = [self.rtypes[e[2]] for e in edge_list]
        g.add_edges(nid1s, nid2s)

        # node inputs
        nid2rows = {} # TODO: seek for efficient tensor version
        all_input_ids, all_input_masks, all_token_type_ids, all_target_idxs = \
            [], [], [], []
        for nid in range(len(self.nodes)):
            n = self.nodes[nid]
            node_input_ids, node_input_masks, node_token_type_ids, node_target_idxs = \
                n.get_bert_inputs(bert_tokenizer, max_seq_len,
                                  give_up_long_sentence=give_up_long_sentence)

            if node_input_ids is None:
                # event appears in the tail of a long sentences
                return None, None

            nid2rows[nid] = np.arange(len(all_input_ids),
                                      len(all_input_ids)+len(node_input_ids)).tolist()

            all_input_ids += node_input_ids
            all_input_masks += node_input_masks
            all_token_type_ids += node_token_type_ids
            all_target_idxs += node_target_idxs
        #     g.nodes[nid].data['h'] = x

        # edge inputs
        edge_norms = []
        for nid1, nid2, rtype in edge_list:
            assert nid1 != nid2, \
                'should not include self loop, which is handled in the model'
            edge_norms.append(1.0 / g.in_degree(nid2))

        edge_types = torch.from_numpy(np.array(edge_types))
        # edge_norms = torch.from_numpy(np.array(edge_norms)).unsqueeze(1).float()
        edge_norms = torch.from_numpy(np.array(edge_norms)).float()

        # g.edata.update({'rel_type': edge_types})
        # g.edata.update({'norm': edge_norms})

        bert_inputs = {
            'input_ids': torch.LongTensor(all_input_ids),
            'input_masks': torch.LongTensor(all_input_masks),
            'token_type_ids': torch.LongTensor(all_token_type_ids),
            'target_idxs': torch.LongTensor(all_target_idxs),
            'nid2rows': nid2rows,
        }
        rgcn_inputs = {
            'num_nodes': len(self.nodes),
            'edge_src': nid1s,
            'edge_dest': nid2s,
            'edge_types': edge_types,
            'edge_norms': edge_norms
        }
        return bert_inputs, rgcn_inputs


# helper functions
def pad_nid2rows(nid2rows):
    # turn nid2rows into numpy array with -1 padding
    max_len = 0
    for nid in range(len(nid2rows)):
        idxs = nid2rows[nid]
        if len(idxs) > max_len:
            max_len = len(idxs)
    new_nid2rows = -1 * np.ones((len(nid2rows), max_len), dtype=np.int64)
    for nid in range(len(nid2rows)):
        idxs = nid2rows[nid]
        new_nid2rows[nid, :len(idxs)] = idxs
    return new_nid2rows


def get_sent_predicates(sent, dep_key='enhancedPlusPlusDependencies'):
    '''get predicates for a sentence using our predicate definition: verb
    parameters:
        sent: Stanford CoreNLP sentence in json
    '''
    deps = sent[dep_key]
    sent_predicates = OrderedDict() # make it deterministic
    for dep in deps:
        if not dep['dep'].startswith(
                ('nsubj', 'dobj', 'iobj', 'nsubjpass',
                 'nmod', 'obj', 'pobj', 'obl', 'prep')):
            continue

        # predicate definition: verbs (exclude be-verbs) with interested edges
        pred_tidx = dep['governor'] - 1
        if not sent['tokens'][pred_tidx]['pos'].startswith('V'): # only verb events
            continue
        if sent['tokens'][pred_tidx]['lemma'] == 'be': # no be verbs
            continue

        if pred_tidx not in sent_predicates:
            sent_predicates[pred_tidx] = {}

        if dep['dep'].startswith('nsubjpass'): # exclude this from nsubj
            d = 'o'
        elif dep['dep'].startswith(('nsubj', 'nmod:agent', 'obl:agent')):
            d = 's'
        elif dep['dep'].startswith(('dobj', 'obj')):
            d = 'o'
        else: # iobj, nmod, pobj, obl, prep
            d = 'prep'
        if d in sent_predicates[pred_tidx]:
            # it is possible that a predicate has multiple subjects or objects.
            # we only take one of them for simplicity
            continue
        sent_predicates[pred_tidx][d] = dep['dependent'] - 1

    for dep in deps:
        if (dep['dep'] == 'compound:prt'
                and dep['governor'] - 1 in sent_predicates):
            pred_tidx = dep['governor'] - 1
            if pred_tidx not in sent_predicates:
                continue
            sent_predicates[pred_tidx]['prt'] = dep['dependent'] - 1
    return sent_predicates


def _events_in_arg(all_sent_predicates, arg):
    start_sent_idx, start_tok_idx, end_sent_idx, end_tok_idx = arg
    if start_sent_idx == end_sent_idx:
        events = [(start_sent_idx, pidx) for pidx in all_sent_predicates[start_sent_idx].keys()
                  if pidx >= start_tok_idx and pidx < end_tok_idx]
    else: # this is only possible for arg1
        # all events in the prev sentence
        prev_events = [(start_sent_idx, pidx) for pidx in all_sent_predicates[start_sent_idx].keys()]
        # cur sentense before marker
        events = [(end_sent_idx, pidx) for pidx in all_sent_predicates[end_sent_idx].keys()
                  if pidx >= 0 and pidx < end_tok_idx]
        events = events + prev_events
    return events


def _disc_relations(all_sent_predicates, arg1, arg2, rtype, parse):
    rels = []
    e1s = _events_in_arg(all_sent_predicates, arg1)
    e2s = _events_in_arg(all_sent_predicates, arg2)
    if len(e1s) > 0 and len(e2s) > 0:
        # we get all possible pairs
        for e1 in e1s:
            for e2 in e2s:
                # (rtype, i_sent1, pred_tidx1, i_sent2, pred_tidx2)
                r = (rtype, e1[0], e1[1], e2[0], e2[1])
                rels.append(r)
    return rels


def get_disc_relations(all_sent_predicates, parse, dmarkers):
    disc_rels = []
    for i_sent, sent in enumerate(parse['sentences']):
        matched = disc._find_matched_markers(sent, dmarkers)
        if matched is None:
            continue
        # (i_sent, begin_tidx, end_tidx, rtype, dmarker)
        conn = (i_sent, matched[0], matched[1], matched[2], matched[3])
        arg1, arg2 = disc._find_args(parse, conn)
        if arg1 is None:
            continue

        # event pairs
        # [(rtype, i_sent1, pred_tidx1, i_sent2, pred_tidx2), ...]
        rels = _disc_relations(all_sent_predicates, arg1, arg2, conn[3], parse)
        if len(rels) > 0:
            disc_rels += rels
    return disc_rels


def get_next_relations(all_sent_predicates):
    rels = []
    for i in range(len(all_sent_predicates)-1):
        cur_predicates = all_sent_predicates[i]
        next_predicates = all_sent_predicates[i+1]

        for pred_tidx1, deps1 in cur_predicates.items():
            for pred_tidx2, deps2 in next_predicates.items():
                rels.append((i, pred_tidx1, i+1, pred_tidx2))
    return rels


def predicate_rep(parse, i_sent, pred_tidx, deps):
    tok = parse['sentences'][i_sent]['tokens'][pred_tidx]
    if 'prt' in deps:
        prt_tok = parse['sentences'][i_sent]['tokens'][deps['prt']]
        rep = '{}_{}'.format(tok['lemma'], prt_tok['lemma'])
    else:
        rep = tok['lemma']
    return rep


def entity_rep(parse, i_sent, pred_tidx, deps, entity_dep):
    pred_rep = predicate_rep(parse, i_sent, pred_tidx, deps)
    return '{}-{}'.format(pred_rep, entity_dep)


def find_chain(i_sent, tidx, corefs):
    for cid, chain in corefs.items():
        for m in chain:
            cur_sent_idx = m['sentNum'] - 1
            cur_tok_start_idx = m['startIndex'] - 1
            cur_tok_end_idx = m['endIndex'] - 1
            if (i_sent == cur_sent_idx
                    and tidx >= cur_tok_start_idx
                    and tidx < cur_tok_end_idx):
                return cid
    return None


def get_ng_bert_tokenizer(bert_weight_name, is_cased):
    bert_tokenizer = AutoTokenizer.from_pretrained(
        bert_weight_name, do_lower_case=(not is_cased))
    return bert_tokenizer


def replace_discourse_markers(sent_parse, dmarkers):
    match = disc._find_matched_markers(
        sent_parse, dmarkers)
    toks = [t['word'] for t in sent_parse['tokens']]
    if match is None:
        return ' '.join(toks)

    begin_tidx = match[0]
    end_tidx = match[1]
    for i in range(begin_tidx, end_tidx):
        toks[i] = MASK_DISC_CONN
    return ' '.join(toks)


def get_pp_ridx2distr_coref(config):
    rtype2idx = config['rtype2idx']
    ret = {}
    for rtype, idx in rtype2idx.items():
        if rtype == 'cnext':
            ret[idx] = 1.0
        else:
            ret[idx] = 0.0
    return ret


def get_pp_ridx2distr(config):
    rtype2idx = config['rtype2idx']
    distr = config['pp_sampling_distribution']
    ret = {}
    for rtype, p in distr.items():
        ret[rtype2idx[rtype]] = p
    return ret


def create_narrative_graph(parse,
                           sent_predicates=None,
                           next_rels=None,
                           rtypes=None,
                           dmarkers=None,
                           disc_rels=None,
                           mask_connectives=True,
                           no_entity=False,
                           min_coref_chain_len=-1,
                           ):
    '''
    parameters:
        parse: CoreNLP
        sent_predicates: list of sentence dict
                        {pred_tidx: {'s': s_tidx, 'o': o_tidx, 'prt': prt_tidx,
                                    'name': 'e1', 'prep': prep_tidx}}
        next_rels: list of (sent_idx1, pred_tidx1, sent_idx2, pred_tidx2)
        rtypes: {"before": 0, ...}
        dmarkers: {"before": "Temporal.Asynchronous.Precedence", ...}
    '''
    doc_id = parse['doc_id']
    corefs = parse['corefs']
    g = NarrativeGraph(doc_id, rtypes=rtypes)

    if sent_predicates is None:
        sent_predicates = [get_sent_predicates(sent) for sent in parse['sentences']]
    if next_rels is None:
        next_rels = get_next_relations(sent_predicates)
    if dmarkers is not None and disc_rels is None:
        disc_rels = get_disc_relations(sent_predicates, parse, dmarkers)
    if mask_connectives:
        masked_sentences = [replace_discourse_markers(sent_parse, dmarkers) for sent_parse in parse['sentences']]

    # doc predicate nodes
    cur_nid = 0
    loc2pnode = {}
    for i_sent, predicates in enumerate(sent_predicates):
        if mask_connectives: # mask discourse markers
            sent = masked_sentences[i_sent]
        else:
            toks = [t['word'] for t in parse['sentences'][i_sent]['tokens']]
            sent = ' '.join(toks)

        for pred_tidx, deps in predicates.items():
            rep = predicate_rep(parse, i_sent, pred_tidx, deps)
            name = deps['name'] if 'name' in deps else None
            n = NGNode(cur_nid, 'predicate', name=name)

            cur_nid += 1
            n.add_instance(rep, sent, i_sent, pred_tidx, None)
            loc2pnode[(i_sent, pred_tidx)] = n
            g.add_node(n)

    # next relations
    for i_sent1, pred_tidx1, i_sent2, pred_tidx2 in next_rels:
        n1 = loc2pnode[(i_sent1, pred_tidx1)]
        n2 = loc2pnode[(i_sent2, pred_tidx2)]
        g.add_edge(n1.nid, n2.nid, 'next')

    # discourse relations
    if disc_rels is not None:
        for rtype, i_sent1, pred_tidx1, i_sent2, pred_tidx2 in disc_rels:
            n1 = loc2pnode[(i_sent1, pred_tidx1)]
            n2 = loc2pnode[(i_sent2, pred_tidx2)]
            g.add_edge(n1.nid, n2.nid, rtype)

    # doc entity nodes
    loc2ent_node = {}
    chainid2node = {}
    chainid2plocs = {}
    for i_sent, predicates in enumerate(sent_predicates):
        if mask_connectives: # mask discourse markers
            sent = masked_sentences[i_sent]
        else:
            toks = [t['word'] for t in parse['sentences'][i_sent]['tokens']]
            sent = ' '.join(toks)

        for pred_tidx, deps in predicates.items():
            pred_n = loc2pnode[(i_sent, pred_tidx)]
            for d, ent_tidx in deps.items():
                if d == 'prt' or d == 'name':
                    continue
                chain_id = find_chain(i_sent, ent_tidx, corefs)
                # for cnext
                if chain_id is not None:
                    if chain_id not in chainid2plocs:
                        chainid2plocs[chain_id] = set()
                    chainid2plocs[chain_id].add((i_sent, pred_tidx, d))

                # logger.debug('chain_id={}'.format(chain_id))
                rep = entity_rep(parse, i_sent, pred_tidx, deps, d)
                if not no_entity:
                    # create entity nodes and links
                    if chain_id is None: # no coreference
                        loc = (i_sent, ent_tidx)
                        # if two predicates share an entity and the entity has no chain_id,
                        # we have to access the entity node through location
                        if loc in loc2ent_node:
                            n = loc2ent_node[loc]
                        else:
                            n = NGNode(cur_nid, 'entity')
                            loc2ent_node[loc] = n
                            g.add_node(n)
                            cur_nid += 1
                    elif chain_id in chainid2node:
                        # coreferent mentions share a node
                        n = chainid2node[chain_id]
                    else:
                        n = NGNode(cur_nid, 'entity')
                        chainid2node[chain_id] = n
                        g.add_node(n)
                        cur_nid += 1

                    n.add_instance(rep, sent, i_sent, pred_tidx, ent_tidx)

                    # note the direction
                    g.add_edge(pred_n.nid, n.nid, d)
                    g.add_edge(n.nid, pred_n.nid, 'rev_'+d)

    # coref_next relations
    coref_chains = []
    for chain_id, plocs in chainid2plocs.items():
        coref_ploc_list = list(plocs)
        coref_ploc_list = sorted(coref_ploc_list, key=lambda x: (x[0], x[1]))
        if len(coref_ploc_list) <= 1:
            continue

        # design choice: coref_next window=1
        chain = []
        for i in range(len(coref_ploc_list)-1):
            n1 = loc2pnode[coref_ploc_list[i][:2]]
            n2 = loc2pnode[coref_ploc_list[i+1][:2]]
            if n1.nid == n2.nid:
                # two mentions that connects to the same predicate
                # refer to the same entity
                # print('same predicate coref mentions. drop it.')
                continue
            g.add_edge(n1.nid, n2.nid, 'cnext')

            if i == 0:
                e = {
                    'nid': n1.nid,
                    'sent_idx': n1.instances[0][2],
                    'tok_idx': n1.instances[0][3],
                    'sent': n1.instances[0][1],
                    'repr': n1.instances[0][0],
                    'dep': coref_ploc_list[i][2]
                }
                chain.append(e)
            e = {
                'nid': n2.nid,
                'sent_idx': n2.instances[0][2],
                'tok_idx': n2.instances[0][3],
                'sent': n2.instances[0][1],
                'repr': n2.instances[0][0],
                'dep': coref_ploc_list[i+1][2]
            }
            chain.append(e)

        if (min_coref_chain_len != -1
                and len(chain) >= min_coref_chain_len):
            coref_chains.append(chain)
    # print(g.get_edge_stats())
    return g, coref_chains


def prepare_ng_model_inputs(bert_inputs, rgcn_inputs):
    edge_src = torch.LongTensor(rgcn_inputs['edge_src'])
    edge_types = torch.LongTensor(rgcn_inputs['edge_types'])
    edge_dest = torch.LongTensor(rgcn_inputs['edge_dest'])
    # back_edge_norms = torch.FloatTensor(rgcn_inputs['edge_norms']).unsqueeze(1)

    input_ids = bert_inputs['input_ids']
    input_masks = bert_inputs['input_masks']
    token_type_ids = bert_inputs['token_type_ids']
    target_idxs = bert_inputs['target_idxs']

    nid2rows = torch.from_numpy(pad_nid2rows(bert_inputs['nid2rows']))

    n_nodes = nid2rows.shape[0]
    n_instances = [input_ids.shape[0]]

    g = dgl.DGLGraph()
    g.add_nodes(n_nodes)
    g.add_edges(edge_src, edge_dest)
    edge_norms = []
    for i in range(edge_dest.shape[0]) :
        nid2 = int(edge_dest[i])
        edge_norms.append(1.0 / g.in_degree(nid2))
    edge_norms = torch.FloatTensor(edge_norms).unsqueeze(1)
    g.edata.update({'rel_type': edge_types})
    g.edata.update({'norm': edge_norms})

    inputs = {
        'bg': [[g]],
        'input_ids': input_ids,
        'input_masks': input_masks,
        'token_type_ids': token_type_ids,
        'target_idxs': target_idxs,
        'nid2rows': [nid2rows],
        'n_instances': n_instances
    }
    return inputs
