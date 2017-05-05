import os
import logging
from scipy.stats import pearsonr
import utils  # Sets up useful loggers
from data import base
import shutil
from time import localtime, strftime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="darkgrid", color_codes=True)


class Model():
    def __init__(self, logdir, dataset, config):
        self.training_set = dataset
        if not os.path.isabs(logdir):
            self.logdir = os.path.join(base, 'log', logdir)
        self.logger = logging.getLogger(os.path.split(logdir)[-1])
        if os.path.exists(self.logdir):
            self.logger.warn("Logdir %s exists", self.logdir)
            if config.get("log_overwrite", False):
                self.logger.warn("Overwriting logdir %s", self.logdir)
                shutil.rmtree(self.logdir)
            else:
                self.logdir += strftime(".%B_%d__%H:%M", localtime())
                self.logger.warn("Renamed logdir to %s", self.logdir)
                self.logger = logging.getLogger(os.path.split(self.logdir)[-1])

        os.makedirs(self.logdir, exist_ok=True)
        self.logger.info("Initialized model; logdir %s", self.logdir)
        fh = logging.FileHandler(os.path.join(self.logdir, 'model.log'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(utils.log_fmt))
        self.logger.addHandler(fh)

    def train(self, num_steps):
        raise NotImplemented

    def _predict(self, sa, sb):
        raise NotImplemented

    def predict(self, sa, sb):
        sa = self.preprocessLine(sa)
        sb = self.preprocessLine(sb)
        return self._predict(sa, sb)

    def preprocessLine(self, sentence):
        raise NotImplemented

    def evalScore(self, eval_set=None):
        if eval_set is None:
            eval_set = self.training_set
        self.logger.info("Evaluating on %s", eval_set.name)
        scores = []
        for sentences in eval_set.input:
            score = self.predict(*sentences)
            self.logger.debug("%s %s --> %.3f", sentences[0], sentences[1], score)
            scores.append(score)
        r = pearsonr(scores, eval_set.target)[0]
        self.logger.info("Correlation coefficient %.3f", r)
        df = pd.DataFrame.from_dict({
            "model": scores,
            "target": eval_set.target
        })

        g = sns.jointplot(x="target", y="model", data=df, kind="reg", color="r", size=7)
        g.savefig(os.path.join(self.logdir, 'correlation.png'))
        return r
