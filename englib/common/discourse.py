import re
import logging
from copy import deepcopy

from nltk import Tree

from .confusion_matrix import Alphabet, ConfusionMatrix


logger = logging.getLogger(__name__)

CONLL16_REL2IDX = {
    "Comparison.Contrast": 0,
    "Contingency.Cause.Reason": 1,
    "Contingency.Cause.Result": 2,
    "Expansion.Restatement": 3,
    "Expansion.Conjunction": 4,
    "Expansion.Instantiation": 5,
    "Temporal.Synchrony": 6,
    "EntRel": 7
    # "Contingency.Condition": 3,
    # "Temporal.Asynchronous": 8,
}
CONLL16_IDX2REL = {
    0: "Comparison.Contrast",
    1: "Contingency.Cause.Reason",
    2: "Contingency.Cause.Result",
    3: "Expansion.Restatement",
    4: "Expansion.Conjunction",
    5: "Expansion.Instantiation",
    6: "Temporal.Synchrony",
    7: "EntRel"
    # 3: "Contingency.Condition",
    # 8: "Temporal.Asynchronous",
}


def create_cm(rels, rel2idx):
    valid_senses = set()
    sense_alphabet = Alphabet()
    for i, rel in enumerate(rels):
        s = rel['Sense'][0]
        if s in rel2idx:
            sense_alphabet.add(s)
            valid_senses.add(s)
    sense_alphabet.add(ConfusionMatrix.NEGATIVE_CLASS)
    cm = ConfusionMatrix(sense_alphabet)
    return cm, valid_senses


def scoring_cm(y, y_pred, cm, valid_senses, idx2rel):
    assert y.shape == y.shape
    cm.matrix.fill(0)
    for gold, pred in zip(y.tolist(), y_pred.tolist()):
        gold_s, pred_s = idx2rel[gold], idx2rel[pred]
        if gold_s not in valid_senses:
            gold_s = ConfusionMatrix.NEGATIVE_CLASS
        if pred_s not in valid_senses:
            pred_s = ConfusionMatrix.NEGATIVE_CLASS
        cm.add(pred_s, gold_s)
    prec, recall, f1 = cm.compute_micro_average_f1()
    return prec, recall, f1


def _find_matched_markers(sent, markers, output_one_marker=True):
    toks = [tok["word"].lower() for tok in sent["tokens"]]
    cidx2tidx = {0: 0}
    acc_clen = 0
    for j, tok in enumerate(sent["tokens"]):
        if j == 0:
            acc_clen += (len(tok["word"]) + 1)
            continue
        cidx2tidx[acc_clen] = j
        acc_clen += (len(tok["word"]) + 1)

    text = ' '.join(toks)
    matched = []
    for dmarker, rtype in markers.items():
        dlen = len(dmarker.split(" "))
        tidxs = []
        for m in re.finditer(dmarker, text):
            if m.start() not in cidx2tidx:
                continue
            begin_tidx = cidx2tidx[m.start()]
            end_tidx = cidx2tidx[m.start()] + dlen
            tmp_toks = [sent["tokens"][k]["word"].lower() for k in range(begin_tidx, end_tidx)]
            tmp_text = ' '.join(tmp_toks)
            if tmp_text == dmarker:
                # avoid substring matched
                tidxs.append((begin_tidx, end_tidx, rtype, dmarker))
        matched += tidxs

    # we pick the one that has the longest connective matched (more specific).
    # assuming each sentence should have at most one discourse marker
    if output_one_marker:
        if len(matched) > 0:
            picked_idx = -1
            cur_len = -1
            for idx, (_, _, _, dm) in enumerate(matched):
                if len(dm) > cur_len:
                    cur_len = len(dm)
                    picked_idx = idx
            matched = matched[picked_idx]
        else:
            matched = None
    return matched


def _is_noisy_marker(marker, noisy_markers=set(['next', 'as'])):
    return marker in noisy_markers


def _get_clauses(sent):
    t = Tree.fromstring(sent['parse'])

    subtexts = []
    for subtree in t.subtrees():
        if subtree.label() == "S" or subtree.label() == "SBAR":
            subtexts.append(' '.join(subtree.leaves()).lower())

    presubtexts = deepcopy(subtexts)
    for i in range(len(subtexts)-1):
        try:
            idx = subtexts[i].index(subtexts[i+1])
            subtexts[i] = subtexts[i][0: idx].strip().lower()
        except:
            subtexts[i] = subtexts[i].strip()
    # for i in reversed(range(len(subtexts)-1)):
    #     subtexts[i] = subtexts[i][0:subtexts[i].index(subtexts[i+1])]
    subtexts = list(filter(None, subtexts))
    try:
        leftover = presubtexts[0][presubtexts[0].index(presubtexts[1])+len(presubtexts[1]):]
        subtexts.append(leftover.strip())
    except:
        pass
    if len(subtexts) == 0:
        # if cannot get clauses from parse tree
        toks = [t['word'] for t in sent['tokens']]
        text = ' '.join(toks)
        subtexts = [text.lower()]
    return subtexts


def _find_args(parse, conn, direction='prev'):
    # conn: (i_sent, begin_tidx, end_tidx, rtype, dmarker)
    i_sent = conn[0]
    cur_sent = parse['sentences'][i_sent]

    marker = conn[4]
    cur_toks = [t['word'].lower() for t in cur_sent['tokens']]
    cur_text = ' '.join(cur_toks)
    clauses = _get_clauses(cur_sent)
    if ' '.join(clauses) != cur_text:
        # invalid parse tree
        return None, None

    if len(clauses) > 1:
        # sent_idx -> [start, end] inclusive
        # tok_idx -> [start, end) exclusive, always for sent[1]
        arg1_start_sent_idx = i_sent
        arg2_start_sent_idx = i_sent
        arg1_end_sent_idx = i_sent
        arg2_end_sent_idx = i_sent
        if cur_text.startswith(marker):
            # logger.debug('case1')
            # case 1: CONNECTIVE ARG2, ARG1
            # arg2: connective to the first comma
            arg2_start_tok_idx = len(marker.split(' '))
            arg2_end_tok_idx = next((i for i, x in enumerate(cur_toks) if x == ','), None)
            if arg2_end_tok_idx is None:
                return None, None

            # arg1: after the first comma
            arg1_start_tok_idx = arg2_end_tok_idx + 1
            arg1_end_tok_idx = len(cur_toks)
        else:
            # logger.debug('case2')
            # case 2: ARG1, CONNECTIVE ARG2
            if _is_noisy_marker(marker):
                # we blacklist some markers because they tend to be noisy
                return None, None
            arg1_start_tok_idx = 0
            arg1_end_tok_idx = conn[1]
            arg2_start_tok_idx = conn[2]
            if arg1_end_tok_idx is None:
                return None, None
            arg2_end_tok_idx = len(cur_toks)
    else: # no clauses
        # logger.debug('case3')
        # case 3: inter-sentence, (ARG1. CONNECTIVE ARG2.) or (CONNECTIVE ARG2. ARG1)
        # has to determine which sentence is ARG1: prev or next
        # according to literature it's almost always prev
        # craft rules for each connective
        # we blacklist some markers because they tend to be noisy
        if _is_noisy_marker(marker):
            # we blacklist some markers because they tend to be noisy
            return None, None

        arg2_start_sent_idx = i_sent
        arg2_end_sent_idx = i_sent
        if direction == 'prev': # in most cases, it's (ARG1. CONNECTIVE ARG2.)
            arg1_start_sent_idx = i_sent-1 if i_sent > 0 else i_sent
            arg1_end_sent_idx = i_sent
        else:
            raise ValueError("Unknown direction type: {}".format(direction))
        arg1_start_tok_idx = 0
        arg1_end_tok_idx = conn[1]
        arg2_start_tok_idx = conn[2]
        arg2_end_tok_idx = len(cur_toks)

    arg1 = (arg1_start_sent_idx, arg1_start_tok_idx, arg1_end_sent_idx, arg1_end_tok_idx)
    arg2 = (arg2_start_sent_idx, arg2_start_tok_idx, arg2_end_sent_idx, arg2_end_tok_idx)
    return arg1, arg2
