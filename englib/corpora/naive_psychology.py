import os
import json
import logging

import numpy as np


logger = logging.getLogger(__name__)


class NaivePsychology:
    def __init__(self, fpath):
        self.fpath = fpath
        self.stories = self._load(fpath)

    def _load(self, fpath):
        return json.load(open(fpath, 'r'))

    def __len__(self):
        return len(self.stories)

    def _generator(self, split):
        for sid, doc in self.stories.items():
            if doc['partition'] == split:
                yield sid, doc

    def train_generator(self):
        return self._generator('train')

    def dev_generator(self):
        return self._generator('dev')

    def test_generator(self):
        return self._generator('test')
