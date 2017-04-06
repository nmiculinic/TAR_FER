import logging
import sys
log_fmt = '\r[%(levelname)s] %(name)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_fmt)
logging.getLogger().handlers = []

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter(log_fmt))
logging.getLogger().addHandler(handler)
