import os
import sys
import json
import time
import h5py
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from stanza.server import CoreNLPClient

from englib.common import utils
from englib.corpora import NaivePsychology, RocStories


def get_arguments(argv):
    parser = argparse.ArgumentParser(description='Parse CoreNLP and save NG for NaivePsych')
    parser.add_argument('config_file', metavar='CONFIG_FILE',
                        help='input config')
    parser.add_argument('output_dir', metavar='OUTPUT_DIR',
                        help='output directory.')

    parser.add_argument('-p', '--nlp_server_port', type=int, default=9002,
                        help='Server port for Stanford CoreNLP')
    parser.add_argument('--seed', type=int, default=135,
                        help='seed for random')
    parser.add_argument('--is_pretrain', action='store_true', default=False,
                        help='parse pretraining corpus')

    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='show info messages')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='show debug messages')
    args = parser.parse_args(argv)
    return args


def run_parsing(gen, prefix):
    logger.info('start parsing {}'.format(prefix))
    fpath = os.path.join(args.output_dir, '{}_parses.json'.format(prefix))
    fw = open(fpath, 'w')
    # java -Xmx16G -cp "/homes/lee2226/scratch2/stanford-corenlp-full-2020-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9002 -timeout 300000 -threads 5 -maxCharLength 100000 -preload tokenize,ssplit,pos,lemma,ner,parse,depparse,coref -outputFormat json

    cnt = 0
    failed_sids = []
    properties = {
        # 'tokenize.whitespace': True,
        'tokenize.keepeol': True,
        'ssplit.eolonly': True,
        # 'coref.algorithm': 'statistical',
        'ner.useSUTime': False
    }
    annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']
    with CoreNLPClient(
        annotators=annotators,
        properties=properties,
        timeout=1200000,
        endpoint='http://localhost:{}'.format(args.nlp_server_port),
        start_server=False
    ) as client:

        t1 = time.time()
        for sid, doc in tqdm(gen()):
            logger.info('last processing time: {} s'.format(time.time()-t1))
            t1 = time.time()

            text = [doc['lines'][str(i)]['text'] for i in range(1, 6)]
            text = '\n'.join(text)

            # parsing
            try:
                ann = client.annotate(text,
                                      annotators=annotators,
                                      properties=properties,
                                      output_format='json')
            except:
                logger.warning('failed parsing {}'.format(sid))
                failed_sids.append(sid)
                continue

            if len(ann['sentences']) != 5:
                logger.warning('failed sentence length {}'.format(sid))
                failed_sids.append(sid)
                continue
            ann['sid'] = sid
            line = json.dumps(ann)
            fw.write(line+'\n')
            cnt += 1
    fw.close()
    logger.info('failed sids={}'.format(failed_sids))
    logger.info('done: {} files,  {} s'.format(cnt, time.time()-t1))


def run_pretrain_parsing(gen, excluded_sids):
    fpath = os.path.join(args.output_dir, 'pretrain_parses.json')
    fw = open(fpath, 'w')
    # java -Xmx16G -cp "/homes/lee2226/scratch2/stanford-corenlp-full-2020-04-20/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9002 -timeout 300000 -threads 5 -maxCharLength 100000 -preload tokenize,ssplit,pos,lemma,ner,parse,depparse,coref -outputFormat json

    cnt = 0
    failed_sids = []
    properties = {
        # 'tokenize.whitespace': True,
        'tokenize.keepeol': True,
        'ssplit.eolonly': True,
        # 'coref.algorithm': 'statistical',
        'ner.useSUTime': False
    }
    annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref']
    with CoreNLPClient(
        annotators=annotators,
        properties=properties,
        timeout=1200000,
        endpoint='http://localhost:{}'.format(args.nlp_server_port),
        start_server=False
    ) as client:

        t1 = time.time()
        for sid, title, sents in tqdm(gen()):
            if sid in excluded_sids:
                continue
            t2 = time.time()
            logger.info('last processing time: {} s'.format(time.time()-t2))

            text = '\n'.join(sents)

            # parsing
            try:
                ann = client.annotate(text,
                                      annotators=annotators,
                                      properties=properties,
                                      output_format='json')
            except:
                logger.warning('failed parsing {}'.format(sid))
                failed_sids.append(sid)
                continue

            if len(ann['sentences']) != 5:
                logger.warning('failed sentence length {}'.format(sid))
                failed_sids.append(sid)
                continue
            ann['sid'] = sid
            line = json.dumps(ann)
            fw.write(line+'\n')
            cnt += 1
    fw.close()
    logger.info('failed sids={}'.format(failed_sids))
    logger.info('done: {} files,  {} s'.format(cnt, time.time()-t1))


def main():
    assert config['config_target'] == 'naive_psychology'
    corpus = NaivePsychology(config['file_path'])


    if args.is_pretrain:
        story_corpus = RocStories(config['pretrain_files'])

        # exclude NaivePsychology dev, test
        dev_sids = set([sid for sid, _ in corpus.dev_generator()])
        test_sids = set([sid for sid, _ in corpus.test_generator()])
        excluded_sids = dev_sids.union(test_sids)
        run_pretrain_parsing(story_corpus.story_generator, excluded_sids)
    else:

        run_parsing(corpus.dev_generator, 'dev')
        run_parsing(corpus.test_generator, 'test')

        run_parsing(corpus.train_generator, 'train')


if __name__ == "__main__":
    args = utils.bin_config(get_arguments)
    logger = utils.get_root_logger(args)
    random.seed(args.seed)
    config = json.load(open(args.config_file))
    main()
