import os
import logging
from scipy.stats import pearsonr
log_fmt = '\r[%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_fmt)


class Model():
    def __init__(self, logdir, training_set, config):
        self.training_set = training_set
        self.logdir = logdir
        self.logger = logging.getLogger(os.path.split(logdir)[-1])
        self.logger.info("Initialized model")

    def train(self, num_steps):
        raise NotImplemented

    def _predict(self, sa, sb):
        raise NotImplemented

    def predict(self, sa, sb):
        sa = self.training_set.preprocessLine(sa)
        sb = self.training_set.preprocessLine(sb)
        return self._predict(sa, sb)

    def evalScore(self, eval_set=None):
        if eval_set is None:
            self.logger.info("Evaluating on train set")
            eval_set = self.training_set
        scores = []
        for sentences in eval_set.raw_input:
            score = self.predict(*sentences)
            self.logger.debug("%s %s --> %.3f", sentences[0], sentences[1], score)
            scores.append(score)
        r = pearsonr(scores, eval_set.target)[0]
        self.logger.info("Correlation coefficient %.3f", r)
        return r
