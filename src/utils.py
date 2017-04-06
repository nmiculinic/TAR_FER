import os

__all__ = ["proj_path"]


class ProjectPath:
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, logdir):
        # Will be important later for TF logging
        self.logdir = logdir

        from time import localtime, strftime
        self.timestamp = strftime("%B_%d__%H:%M", localtime())

        self.logpath = os.path.join(ProjectPath.base, self.logdir, self.timestamp)
        self.w2v = os.path.join(ProjectPath.base, "data", "word_embeddings", "wiki.en.vec")
        self.eval_data_in = os.path.join(ProjectPath.base, "data", "train-en-en.in")
        self.eval_data_out = os.path.join(ProjectPath.base, "data", "train-en-en.out")


proj_path = ProjectPath("log")
