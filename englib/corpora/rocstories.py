import os
import csv
import logging


logger = logging.getLogger(__name__)


class RocStories:
    def __init__(self, fpaths):
        self.fpaths = fpaths
        self.stories = self._load(fpaths)

    def _load(self, fpaths):
        stories = []
        for fpath in fpaths:
            logger.info('loading {}...'.format(fpath))
            with open(fpath, 'r') as fr:
                reader = csv.reader(fr, delimiter=',', quotechar='"')
                for i, row in enumerate(reader):
                    if i == 0:
                        # ['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
                        continue
                    sid = row[0]
                    title = row[1]
                    sents = row[2:]

                    stories.append((sid, title, sents))
            logger.info('{} storeis loaded so far'.format(len(stories)))
        return stories

    def story_generator(self):
        for s in self.stories:
            yield s
