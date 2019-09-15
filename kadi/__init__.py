# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .version import __version__, __git_version__  # noqa


def test(*args, **kwargs):
    '''
    Run py.test unit tests.
    '''
    import testr
    return testr.test(*args, **kwargs)
