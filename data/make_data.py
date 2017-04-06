#!/usr/bin/env python3
import logging
from urllib.request import urlretrieve
import shutil
import os
import progressbar

logger = logging.getLogger("Requesting data")
log_fmt = '[%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger.info("Started gathering data")

file_dir = os.path.normpath(os.path.dirname(__file__))


download_list = [
    ('http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.gs.zip', 'sts2017.gs.zip', '.'),
    ('http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.eval.v1.1.zip', 'sts2017.eval.v1.1.zip', '.'),
    ('https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec', 'wiki.en.vec', None),
]

copy_files = [
    ('STS2017.eval.v1.1/STS.input.track5.en-en.txt', 'train-en-en.in'),
    ('STS2017.gs/STS.gs.track5.en-en.txt', 'train-en-en.out')
]


def maybeDownload(url, fname):
    if os.path.isfile(fname):
        logger.info("File %s exists, skipping", fname)
    else:
        bar = progressbar.ProgressBar(widgets=[
            progressbar.Timer(),
            progressbar.Bar(),
            progressbar.AdaptiveETA(),
            progressbar.FileTransferSpeed()
        ])

        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                bar.maxval = totalsize
                readsofar = min(readsofar, totalsize)
            else:
                bar.maxval = progressbar.UnknownLength
            bar.update(readsofar)

        logger.info("Downloading from %s to %s", url, fname)
        bar.start()
        urlretrieve(url, fname + ".part", reporthook)
        bar.finish()
        shutil.move(fname + ".part", fname)


def extractArhive(fname, dir):
    logger.info("Extracting %s in %s", fname, dir)
    try:
        shutil.unpack_archive(fname, dir)
    except Exception as ex:
        logger.error("Exception happened during extracting %s", fname, exc_info=True)


for url, fname, dd in download_list:
    maybeDownload(url, os.path.join(file_dir, fname))
    if dd is None:
        logger.info("%s is not meant to be extracted, skpping", fname)
    else:
        extractArhive(os.path.join(file_dir, fname), os.path.join(file_dir, dd))


for src, dest in copy_files:
    src = os.path.join(file_dir, src)
    dest = os.path.join(file_dir, dest)
    shutil.copy(src, dest)
    logger.info("Copying %s to %s", src, dest)
