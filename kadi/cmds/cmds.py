# Licensed under a 3-clause BSD style license - see LICENSE.rst
import tables
from collections import OrderedDict

import numpy as np

from astropy.table import Table
from Chandra.Time import DateTime
import six
from six.moves import cPickle as pickle

from ..paths import IDX_CMDS_PATH, PARS_DICT_PATH

__all__ = ['filter']


class LazyVal(object):
    def __init__(self, load_func):
        self._load_func = load_func

    def __getattribute__(self, name):
        try:
            val = object.__getattribute__(self, '_val')
        except AttributeError:
            val = object.__getattribute__(self, '_load_func')()
            self._val = val

        if name == '_val':
            return val
        else:
            return val.__getattribute__(name)

    def __getitem__(self, item):
        return self._val[item]

    def __repr__(self):
        return repr(self._val)

    def __str__(self):
        return str(self._val)

    def __len__(self):
        return self._val.__len__()


def load_idx_cmds():
    h5 = tables.openFile(IDX_CMDS_PATH(), mode='r')
    idx_cmds = h5.root.data[:]
    h5.close()
    return idx_cmds


def load_pars_dict():
    with open(PARS_DICT_PATH(), 'r') as fh:
        pars_dict = pickle.load(fh)
    return pars_dict

# Globals that contain the entire commands table and the parameters index
# dictionary.
idx_cmds = LazyVal(load_idx_cmds)
pars_dict = LazyVal(load_pars_dict)
rev_pars_dict = LazyVal(lambda: {v: k for k, v in pars_dict.items()})


def filter(start=None, stop=None, **kwargs):
    """
    Get commands between ``start`` and ``stop``.  Additional ``key=val`` pairs
    can be supplied to further filter the results.  Both ``key`` and ``val``
    are case insensitive.  In addition to the any of the command parameters
    such as TLMSID, MSID, SCS, STEP, or POS, the ``key`` can be:

    date : Exact date of command e.g. '2013:003:22:11:45.530'
    type : Command type e.g. COMMAND_SW, COMMAND_HW, ACISPKT, SIMTRANS

    Examples::

      >>> from kadi import cmds
      >>> cs = cmds.filter('2012:001', '2012:030')
      >>> cs = cmds.filter('2012:001', '2012:030', type='simtrans')
      >>> cs = cmds.filter(type='acispkt', tlmsid='wsvidalldn')
      >>> cs = cmds.filter(msid='aflcrset')
      >>> print(cs.table)

    Parameters
    ----------
    start : DateTime format (optional)
        Start time, defaults to beginning of available commands (2002:001)

    stop : DateTime format (optional)
        Stop time, defaults to end of available commands

    **kwargs : any key=val keyword argument pairs

    Returns
    -------
    cmds : CmdList object (list of commands)
    """
    cmds = _find(start, stop, **kwargs)
    return CmdList(cmds)


def _find(start=None, stop=None, **kwargs):
    """
    Get commands between ``start`` and ``stop``.  Additional ``key=val`` pairs
    can be supplied to further filter the results.  Both ``key`` and ``val``
    are case insensitive.  In addition to the any of the command parameters
    such as TLMSID, MSID, SCS, STEP, or POS, the ``key`` can be:

    date : Exact date of command e.g. '2013:003:22:11:45.530'
    type : Command type e.g. COMMAND_SW, COMMAND_HW, ACISPKT, SIMTRANS

    Examples::

      >>> from kadi import cmds
      >>> cs = cmds._find('2012:001', '2012:030')
      >>> cs = cmds._find('2012:001', '2012:030', type='simtrans')
      >>> cs = cmds._find(type='acispkt', tlmsid='wsvidalldn')
      >>> cs = cmds._find(msid='aflcrset')

    Parameters
    ----------
    start : DateTime format (optional)
        Start time, defaults to beginning of available commands (2002:001)

    stop : DateTime format (optional)
        Stop time, defaults to end of available commands

    **kwargs : any key=val keyword argument pairs

    Returns
    -------
    cmds : numpy structured array of commands
    """
    ok = np.ones(len(idx_cmds), dtype=bool)
    par_ok = np.zeros(len(idx_cmds), dtype=bool)

    if start:
        ok &= idx_cmds['date'] >= DateTime(start).date
    if stop:
        ok &= idx_cmds['date'] <= DateTime(stop).date
    for key, val in kwargs.items():
        key = key.lower()
        if isinstance(val, six.string_types):
            val = val.upper()
        if key in idx_cmds.dtype.names:
            ok &= idx_cmds[key] == val
        else:
            par_ok[:] = False
            for pars_tuple, idx in pars_dict.items():
                pars = dict(pars_tuple)
                if pars.get(key) == val:
                    par_ok |= (idx_cmds['idx'] == idx)
            ok &= par_ok
    cmds = idx_cmds[ok]
    return Table(cmds)


class Cmd(OrderedDict):
    def __init__(self, cmd):
        # Create ordered dict from field values in idx_cmd structured array row.
        # Skip the first field "idx" because that is not relevant here.
        super(Cmd, self).__init__()
        self.update(OrderedDict((name, cmd[name].tolist()) for name in cmd.dtype.names[1:]))
        self.update(OrderedDict(rev_pars_dict[cmd['idx']]))
        if self['tlmsid'] is None:
            del self['tlmsid']

    def __repr__(self):
        out = ('{} {:11s} '.format(self['date'], self['type']) +
               ' '.join('{}={}'.format(key, val) for key, val in self.items()
                        if key not in ('type', 'date')))
        return out

    def __str__(self):
        return repr(self)


class CmdList(object):
    def __init__(self, cmds):
        self.cmds = cmds

    @property
    def table(self):
        if not hasattr(self, '_table'):
            self._table = Table(self.cmds)
            self._table.remove_column('idx')
        return self._table

    def __getitem__(self, item):
        cmds = self.cmds
        if isinstance(item, six.string_types):
            out = []
            for cmd in cmds:
                if item in cmds.dtype.names:
                    out.append(cmd[item].tolist())
                else:
                    # Find the parameters tuple for this command from the reverse
                    # lookup table which maps index to the params tuple.
                    pars_tuple = rev_pars_dict[cmd['idx']]
                    pars = dict(pars_tuple)
                    out.append(pars.get(item))
        elif isinstance(item, int):
            out = Cmd(cmds[item])
        elif isinstance(item, (slice, np.ndarray)):
            out = [Cmd(cmd) for cmd in cmds[item]]
        return out

    def __repr__(self):
        return '\n'.join(repr(Cmd(cmd)) for cmd in self.cmds)

    def __str__(self):
        return repr(self)
