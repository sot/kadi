# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pyyaks.logger

# TODO: make it easier to set the log level (e.g. add a set_level method() to
# logger object that sets all handlers to that level)
logger = pyyaks.logger.get_logger(
    name='kadi.commands',
    format='%(asctime)s %(funcName)s - %(message)s')
for handler in logger.handlers:
    handler.setLevel(pyyaks.logger.DEBUG)

from .core import *  # noqa
from .commands import *  # noqa
