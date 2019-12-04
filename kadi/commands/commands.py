# Licensed under a 3-clause BSD style license - see LICENSE.rst
import tables
from pathlib import Path

import numpy as np

from astropy.table import Table, Row, Column, vstack
from Chandra.Time import DateTime
import pickle

from ..paths import IDX_CMDS_PATH, PARS_DICT_PATH

__all__ = ['get_cmds', 'get_cmds_from_backstop', 'CommandTable']


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
    h5 = tables.open_file(IDX_CMDS_PATH(), mode='r')
    idx_cmds = Table(h5.root.data[:])
    h5.close()
    return idx_cmds


def load_pars_dict():
    with open(PARS_DICT_PATH(), 'rb') as fh:
        pars_dict = pickle.load(fh, encoding='ascii')
    return pars_dict


# Globals that contain the entire commands table and the parameters index
# dictionary.
idx_cmds = LazyVal(load_idx_cmds)
pars_dict = LazyVal(load_pars_dict)
rev_pars_dict = LazyVal(lambda: {v: k for k, v in pars_dict.items()})


def get_cmds(start=None, stop=None, **kwargs):
    """
    Get commands with ``start`` <= date < ``stop``.  Additional ``key=val`` pairs
    can be supplied to further filter the results.  Both ``key`` and ``val``
    are case insensitive.  In addition to the any of the command parameters
    such as TLMSID, MSID, SCS, STEP, or POS, the ``key`` can be:

    type
      Command type e.g. COMMAND_SW, COMMAND_HW, ACISPKT, SIMTRANS
    date
      Exact date of command e.g. '2013:003:22:11:45.530'

    If ``date`` is provided then ``start`` and ``stop`` values are ignored.

    Examples::

      >>> from kadi import commands
      >>> cmds = commands.get_cmds('2012:001', '2012:030')
      >>> cmds = commands.get_cmds('2012:001', '2012:030', type='simtrans')
      >>> cmds = commands.get_cmds(type='acispkt', tlmsid='wsvidalldn')
      >>> cmds = commands.get_cmds(msid='aflcrset')
      >>> print(cmds)

    :param start: DateTime format (optional)
        Start time, defaults to beginning of available commands (2002:001)
    :param stop: DateTime format (optional)
        Stop time, defaults to end of available commands
    :param kwargs: key=val keyword argument pairs

    :returns: :class:`~kadi.commands.commands.CommandTable` of commands
    """
    cmds = _find(start, stop, **kwargs)
    out = CommandTable(cmds)
    out['params'] = None if len(out) > 0 else Column([], dtype=object)

    return out


def get_cmds_from_backstop(backstop, remove_starcat=True):
    """
    Initialize a ``CommandTable`` from ``backstop``, which can either
    be a string file name or a backstop table from ``parse_cm.read_backstop``.

    :param backstop: str or Table
    :param remove_starcat: remove star catalog commands (default=True)
    :returns: :class:`~kadi.commands.commands.CommandTable` of commands
    """
    if isinstance(backstop, Path):
        backstop = str(backstop)

    if isinstance(backstop, str):
        from parse_cm import read_backstop
        bs = read_backstop(backstop)
    elif isinstance(backstop, Table):
        bs = backstop
    else:
        raise ValueError(f'`backstop` arg must be a string filename or '
                         f'a backstop Table')

    if remove_starcat:
        ok = bs['type'] != 'MP_STARCAT'
        bs = bs[ok]

    n_bs = len(bs)
    out = {}
    # Set idx to max (2**16 -1) so it does not match any real idx
    out['idx'] = np.full(n_bs, fill_value=65535, dtype=np.uint16)
    out['date'] = np.chararray.encode(bs['date'])
    out['type'] = np.chararray.encode(bs['type'])
    out['tlmsid'] = np.chararray.encode(bs['tlmsid'])
    out['scs'] = bs['scs'].astype(np.uint8)
    out['step'] = bs['step'].astype(np.uint16)
    # Set timeline_id to 0, does not match any real timeline id
    out['timeline_id'] = np.zeros(n_bs, dtype=np.uint32)
    out['vcdu'] = bs['vcdu'].astype(np.int32)
    out['params'] = bs['params']

    # Backstop has redundant param keys, get rid of them here
    for params in out['params']:
        for key in ('tlmsid', 'step', 'scs'):
            params.pop(key, None)

        # Match the hack in update_commands which swaps in "event_type" for "type"
        # for orbit event commands
        if 'type' in params:
            params['event_type'] = params['type']
            del params['type']

    return CommandTable(out)


def _find(start=None, stop=None, **kwargs):
    """
    Get commands ``start`` <= date < ``stop``.  Additional ``key=val`` pairs
    can be supplied to further filter the results.  Both ``key`` and ``val``
    are case insensitive.  In addition to the any of the command parameters
    such as TLMSID, MSID, SCS, STEP, or POS, the ``key`` can be:

    date : Exact date of command e.g. '2013:003:22:11:45.530'
    type : Command type e.g. COMMAND_SW, COMMAND_HW, ACISPKT, SIMTRANS

    Examples::

      >>> from kadi import commands
      >>> cmds = commands._find('2012:001', '2012:030')
      >>> cmds = commands._find('2012:001', '2012:030', type='simtrans')
      >>> cmds = commands._find(type='acispkt', tlmsid='wsvidalldn')
      >>> cmds = commands._find(msid='aflcrset')

    :param start: DateTime format (optional)
        Start time, defaults to beginning of available commands (2002:001)
    :param stop: DateTime format (optional)
        Stop time, defaults to end of available commands
    :param kwargs: key=val keyword argument pairs

    :returns: astropy Table of commands
    """
    date = kwargs.pop('date', None)
    if date:
        start = DateTime(date).date  # clip resolution to nearest msec
        stop = DateTime(start) + 0.001 / 86400  # exactly 1 msec later

    ok = np.ones(len(idx_cmds), dtype=bool)
    par_ok = np.zeros(len(idx_cmds), dtype=bool)

    if start:
        ok &= idx_cmds['date'] >= DateTime(start).date
    if stop:
        ok &= idx_cmds['date'] < DateTime(stop).date
    for key, val in kwargs.items():
        key = key.lower()
        if isinstance(val, str):
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
    return cmds


class CommandRow(Row):
    def __getitem__(self, item):
        if item == 'params':
            out = super(CommandRow, self).__getitem__(item)
            if out is None:
                idx = super(CommandRow, self).__getitem__('idx')
                out = self['params'] = dict(rev_pars_dict[idx])
        elif item not in self.colnames:
            out = self['params'][item]
        else:
            out = super(CommandRow, self).__getitem__(item)
        return out

    def keys(self):
        out = [name for name in self.colnames if name != 'params']
        params = [key.lower() for key in sorted(self['params'])]

        return out + params

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, value) for key, value in zip(self.keys(), self.values())]

    def __repr__(self):
        out = ('<Cmd '.format(self.__class__.__name__) + str(self) + '>')
        return out

    def __str__(self):
        keys = self.keys()
        keys.remove('date')
        keys.remove('type')
        if 'idx' in keys:
            keys.remove('idx')

        out = ('{} {} '.format(self['date'], self['type']) +
               ' '.join('{}={}'.format(key, self[key]) for key in keys
                        if key not in ('type', 'date')))
        return out

    def __sstr__(self):
        return str(self._table[self.index:self.index + 1])


class CommandTable(Table):
    """
    Astropy Table subclass that is specialized to handle commands via a
    ``params`` column that is expected to be ``None`` or a dict of params.
    """

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.colnames:
                return self.columns[item]
            else:
                return Column([cmd['params'].get(item) for cmd in self], name=item)

        elif isinstance(item, int):
            return CommandRow(self, item)

        elif isinstance(item, (tuple, list)) and all(x in self.colnames
                                                     for x in item):
            from copy import deepcopy
            from astropy.table import groups
            out = self.__class__([self[x] for x in item], meta=deepcopy(self.meta))
            out._groups = groups.TableGroups(out, indices=self.groups._indices,
                                             keys=self.groups._keys)
            return out

        elif (isinstance(item, slice) or
              isinstance(item, np.ndarray) or
              isinstance(item, list) or
              isinstance(item, tuple) and all(isinstance(x, np.ndarray)
                                              for x in item)):
            # here for the many ways to give a slice; a tuple of ndarray
            # is produced by np.where, as in t[np.where(t['a'] > 2)]
            # For all, a new table is constructed with slice of all columns
            return self._new_from_slice(item)

        else:
            raise ValueError('Illegal type {0} for table item access'
                             .format(type(item)))

    def __str__(self):
        # Cut out params column for printing
        colnames = self.colnames
        if 'idx' in colnames:
            colnames.remove('idx')

        # Nice repr of parameters.  This forces all cmd params to get resolved.
        tmp_params = None
        if 'params' in colnames:
            params_list = []
            for params in self['params']:
                if params is None:
                    params_list.append('N/A')
                else:
                    param_strs = ['{}={}'.format(key, val) for key, val in params.items()]
                    params_list.append(' '.join(param_strs))
            tmp_params = params_list
            colnames.remove('params')

        tmp = self[colnames]
        if tmp_params:
            tmp['params'] = tmp_params

        lines = tmp.pformat(max_width=-1)
        return '\n'.join(lines)

    def __bytes__(self):
        return str(self).encode('utf-8')

    def fetch_params(self):
        """
        Fetch all ``params`` for every row and force resolution of actual values.

        This is handy for printing a command table and seeing all the parameters at once.
        """
        for cmd in self:
            cmd['params']

    def add_cmds(self, cmds):
        """
        Add CommandTable ``cmds`` to self and return the new CommandTable. The
        commands table is maintained in order (date, step, scs).

        :param cmds: :class:`~kadi.commands.commands.CommandTable` of commands
        :returns: :class:`~kadi.commands.commands.CommandTable` of commands
        """
        out = vstack([self, cmds])
        out.sort(['date', 'step', 'scs'])

        return out
