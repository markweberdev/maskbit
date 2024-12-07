"""This files contains util functions supporting logging to terminal and files.

We thank the following public implementations for inspiring this code:
    https://github.com/facebookresearch/detectron2
"""

import atexit
import functools
import os
import sys
from accelerate.logging import MultiProcessAdapter
import logging
from termcolor import colored

from iopath.common.file_io import PathManager as PathManagerClass

__all__ = ["setup_logger", "PathManager"]

PathManager = PathManagerClass()


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", self._root_name)
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(name="VQGAN", log_level: str = None, color=True, use_accelerate=True, output_dir=None):
    logger = logging.getLogger(name)
    if log_level is None:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level.upper())

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_dir is not None:
        filename = os.path.join(output_dir, "log.txt")
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    if use_accelerate:
        return MultiProcessAdapter(logger, {})
    else:
        return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = PathManager.open(filename, "a", buffering=_get_log_stream_buffer_size(filename))
    atexit.register(io.close)
    return io


def _get_log_stream_buffer_size(filename: str) -> int:
    if "://" not in filename:
        # Local file, no extra caching is necessary
        return -1
    # Remote file requires a larger cache to avoid many small writes.
    return 1024*1024 # 1MB