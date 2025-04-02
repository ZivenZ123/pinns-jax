"""日志记录器模块，提供兼容 TensorFlow 2 的 Python 命令行日志记录功能。"""

import logging


def get_pylogger(name: str = __name__) -> logging.Logger:
    """初始化一个兼容 TensorFlow 2 的 Python 命令行日志记录器。

    :param name: 日志记录器的名称，默认为 "__name__"。

    :return: 一个日志记录器对象。
    """
    logger = logging.getLogger(name)

    return logger
