# coding: utf-8
import logging
def get_logger(log_path):
    """Returns a logger.
    这个函数要转化为自己的代码
    Args:
        log_path (str): Path to the log file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%a, %d %b %Y %H:%M:%S'))
    logger.addHandler(fh)
    return logger
