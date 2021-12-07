# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import sys

import ska_helpers

__version__ = ska_helpers.get_version(__package__)


def _get_kadi_logger():
    """Define top-level logger for the kadi package.

    Defaults to WARNING level.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    fmt = logging.Formatter('%(asctime)s %(funcName)s: %(message)s', datefmt=None)
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(fmt)
    logger.addHandler(hdlr)
    return logger


logger = _get_kadi_logger()


def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
