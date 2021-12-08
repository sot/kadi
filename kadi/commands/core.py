# Licensed under a 3-clause BSD style license - see LICENSE.rst
import tables
from pathlib import Path
import logging
import pickle

import numpy as np
from ska_helpers import retry

from astropy.table import Table, Row, Column, vstack, TableAttribute
from cxotime import CxoTime

from kadi.paths import IDX_CMDS_PATH, PARS_DICT_PATH

__all__ = ['read_backstop', 'get_cmds_from_backstop', 'CommandTable']

logger = logging.getLogger(__name__)


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


@retry.retry(tries=4, delay=0.5, backoff=4)
def load_idx_cmds(version=None, file=None):
    """Load the cmds.h5 file, trying up to 3 times

    It seems that one can occasionally get:

    File "tables/hdf5extension.pyx", line 492, in tables.hdf5extension.File._g_new
tables.exceptions.HDF5ExtError: HDF5 error back trace
     ...
    File "H5FDsec2.c", line 941, in H5FD_sec2_lock
    unable to lock file, errno = 11, error message = 'Resource temporarily unavailable'
    """
    if file is None:
        file = IDX_CMDS_PATH(version)
    logger.info(f'Loading {file}')
    with tables.open_file(file, mode='r') as h5:
        idx_cmds = CommandTable(h5.root.data[:])

    return idx_cmds


@retry.retry(tries=4, delay=0.5, backoff=4)
def load_pars_dict(version=None, file=None):
    if file is None:
        file = PARS_DICT_PATH(version)
    logger.info(f'Loading {file}')
    with open(PARS_DICT_PATH(version), 'rb') as fh:
        pars_dict = pickle.load(fh, encoding='ascii')
    return pars_dict


def read_backstop(backstop):
    """Read ``backstop`` and return a ``CommandTable``.

    The ``backstop`` argument can be either be a string file name or a backstop
    table from ``parse_cm.read_backstop``.

    This function is a wrapper around ``get_cmds_from_backstop`` but follows a
    more typical naming convention.

    :param backstop: str or Table
    :returns: :class:`~kadi.commands.commands.CommandTable` of commands
    """
    return get_cmds_from_backstop(backstop)


def get_cmds_from_backstop(backstop, remove_starcat=False):
    """
    Initialize a ``CommandTable`` from ``backstop``, which can either
    be a string file name or a backstop table from ``parse_cm.read_backstop``.

    :param backstop: str or Table
    :param remove_starcat: remove star catalog command parameters (default=False)
    :returns: :class:`~kadi.commands.commands.CommandTable` of commands
    """
    if isinstance(backstop, Path):
        backstop = str(backstop)

    if isinstance(backstop, (str, list)):
        from parse_cm import read_backstop
        bs = read_backstop(backstop)
    elif isinstance(backstop, Table):
        bs = backstop
    else:
        raise ValueError(f'`backstop` arg must be a string filename or '
                         f'a backstop Table')

    n_bs = len(bs)
    out = {}
    # Set idx to -1 so it does not match any real idx
    out['idx'] = np.full(n_bs, fill_value=-1, dtype=np.int32)
    out['date'] = np.chararray.encode(bs['date'])
    out['type'] = np.chararray.encode(bs['type'])
    out['tlmsid'] = np.chararray.encode(bs['tlmsid'])
    out['scs'] = bs['scs'].astype(np.uint8)
    out['step'] = bs['step'].astype(np.uint16)
    out['time'] = CxoTime(bs['date'], format='date').secs
    # Set timeline_id to 0, does not match any real timeline id
    out['timeline_id'] = np.zeros(n_bs, dtype=np.uint32)
    out['vcdu'] = bs['vcdu'].astype(np.int32)
    out['params'] = bs['params']

    if remove_starcat:
        # Remove the lengthy parameters in star catalog but leave the command
        for idx in np.flatnonzero(bs['type'] == 'MP_STARCAT'):
            out['params'][idx] = {}

    # Backstop has redundant param keys, get rid of them here
    for params in out['params']:
        for key in ('tlmsid', 'step', 'scs'):
            params.pop(key, None)

        # Match the hack in update_commands which swaps in "event_type" for "type"
        # for orbit event commands
        if 'type' in params:
            params['event_type'] = params['type']
            del params['type']

    out = CommandTable(out)
    out['time'].info.format = '.3f'

    return out


def _find(start=None, stop=None, inclusive_stop=False, idx_cmds=None,
          pars_dict=None, **kwargs):
    """
    Get commands beteween ``start`` and ``stop``.

    By default the interval is ``start`` <= date < ``stop``, but if
    ``inclusive_stop=True`` then the interval is ``start`` <= date <= ``stop``.

    Additional ``key=val`` pairs can be supplied to further filter the results.
    Both ``key`` and ``val`` are case insensitive.  In addition to the any of
    the command parameters such as TLMSID, MSID, SCS, STEP, or POS, the ``key``
    can be:

    date : Exact date of command e.g. '2013:003:22:11:45.530'
    type : Command type e.g. COMMAND_SW, COMMAND_HW, ACISPKT, SIMTRANS

    Examples::

      >>> from kadi import commands
      >>> cmds = commands._find('2012:001', '2012:030')
      >>> cmds = commands._find('2012:001', '2012:030', type='simtrans')
      >>> cmds = commands._find(type='acispkt', tlmsid='wsvidalldn')
      >>> cmds = commands._find(msid='aflcrset')

    :param start: CxoTime format (optional)
        Start time, defaults to beginning of available commands (2002:001)
    :param stop: CxoTime format (optional)
        Stop time, defaults to end of available commands
    :param kwargs: key=val keyword argument pairs

    :returns: astropy Table of commands
    """
    ok = np.ones(len(idx_cmds), dtype=bool)
    par_ok = np.zeros(len(idx_cmds), dtype=bool)

    date = kwargs.pop('date', None)
    if date:
        ok &= idx_cmds['date'] == CxoTime(date).date
    else:
        if start:
            ok &= idx_cmds['date'] >= CxoTime(start).date
        if stop:
            if inclusive_stop:
                ok &= idx_cmds['date'] <= CxoTime(stop).date
            else:
                ok &= idx_cmds['date'] < CxoTime(stop).date

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
                # self.parent.rev_pars_dict should be a weakref to the reverse
                # pars dict for this CommandTable. But it might not be defined,
                # in which case just leave it as None.
                if rev_pars_dict := self.table.rev_pars_dict:
                    params = dict(rev_pars_dict()[idx])
                else:
                    raise KeyError('params cannot be mapped because the rev_pars_dict '
                                   'attribute is not defined (needs to be a weakref.ref '
                                   'of REV_PARS_DICT)')
                out = self['params'] = params
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
        out = (f'<Cmd {str(self)}>')
        return out

    def __str__(self):
        keys = self.keys()
        keys.remove('date')
        keys.remove('type')
        if 'idx' in keys:
            keys.remove('idx')

        out = ('{} {} '.format(self['date'], self['type'])
               + ' '.join('{}={}'.format(key, self[key]) for key in keys
                          if key not in ('type', 'date', 'time')))
        return out

    def __sstr__(self):
        return str(self._table[self.index:self.index + 1])


class CommandTable(Table):
    """
    Astropy Table subclass that is specialized to handle commands via a
    ``params`` column that is expected to be ``None`` or a dict of params.
    """
    rev_pars_dict = TableAttribute()

    COL_TYPES = {'idx': np.int32,
                 'date': 'S21',
                 'type': 'S12',
                 'tlmsid': 'S10',
                 'scs': np.uint8,
                 'step': np.uint16,
                 'source': 'S8',
                 'timeline_id': np.uint32,
                 'vcdu': np.int32,
                 'params': object}

    def _convert_data_to_col(self, *args, **kwargs):
        col = super()._convert_data_to_col(*args, **kwargs)
        if col.info.name in self.COL_TYPES:
            col = col.astype(self.COL_TYPES[col.info.name])
        return col

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.colnames:
                return self.columns[item]
            else:
                return Column([cmd['params'].get(item) for cmd in self], name=item)

        elif isinstance(item, (int, np.integer)):
            return CommandRow(self, item)

        elif isinstance(item, (tuple, list)) and all(x in self.colnames
                                                     for x in item):
            from copy import deepcopy
            from astropy.table import groups
            out = self.__class__([self[x] for x in item], meta=deepcopy(self.meta))
            out._groups = groups.TableGroups(out, indices=self.groups._indices,
                                             keys=self.groups._keys)
            return out

        elif (isinstance(item, slice)
              or isinstance(item, np.ndarray)
              or isinstance(item, list)
              or isinstance(item, tuple) and all(isinstance(x, np.ndarray)
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

    def find_date(self, date, side='left'):
        """Find row in table corresponding to ``date``.

        This is a thin wrapper around np.searchsorted that converts ``date`` to
        a byte string before calling searchsorted. This is necessary because
        the ``date`` column is a byte string and the astropy unicode machinery
        ends up getting called a lot in a way that impacts performance badly.

        :param date: str, sequence of str
            Date(s) to search for.
        :param side: {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `a`).
        :returns: int
            Index of row(s) corresponding to ``date``.
        """
        if isinstance(date, CxoTime):
            date = date.date
        date = np.char.encode(date, encoding='ascii')
        idxs = np.searchsorted(self['date'], date, side=side)
        return idxs

    def fetch_params(self):
        """
        Fetch all ``params`` for every row and force resolution of actual values.

        This is handy for printing a command table and seeing all the parameters at once.
        """
        if 'params' not in self.colnames:
            self['params'] = None
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
        out.sort_in_backstop_order()

        return out

    def sort_in_backstop_order(self):
        """Sort table in order (date, step, scs)

        This matches the order in backstop.
        """
        sort_keys = ['date', 'step', 'scs', 'source']
        self.sort(sort_keys)

    def as_list_of_dict(self, ska_parsecm=False):
        """Convert CommandTable to a list of dict.

        The command ``params`` are embedded as a dict for each command.

        If ``ska_parsecm`` is True then the output is made more compatible with
        the legacy output from ``Ska.ParseCM.read_backstop()``, namely:

        - Add ``cmd`` key which is set to the ``type`` key
        - Make ``params`` keys uppercase.

        :param ska_parsecm: bool, make output more Ska.ParseCM compatible
        :return: list of dict
        """
        self.fetch_params()

        names = self.colnames
        cmds_list = [{name: cmd[name] for name in names} for cmd in self]

        if ska_parsecm:
            for cmd in cmds_list:
                if 'type' in cmd:
                    cmd['cmd'] = cmd['type']
                if 'params' in cmd:
                    cmd['params'] = {key.upper(): val for key, val in cmd['params'].items()}

        return cmds_list

    def pformat_like_backstop(self):
        """Format the table in a human-readable format that is similar to backstop"""
        lines = []
        has_params = 'params' in self.colnames
        for cmd in self:
            if has_params:
                # Make a single string of params like POS= 75624, SCS= 130, STEP= 9
                fmtvals = []
                for key, val in cmd['params'].items():
                    if key == 'aoperige':
                        fmt = '{}={:.13e}'
                    elif isinstance(val, float):
                        fmt = '{}={:.8e}'
                    elif key == 'packet(40)':
                        continue
                    elif (key.startswith('aopcads') or
                          key.startswith('co')
                          or key.startswith('afl')
                          or key.startswith('2s1s')
                          or key.startswith('2s2s')
                          ):
                        fmt = '{}={:d} '
                    else:
                        fmt = '{}={}'
                    fmtvals.append(fmt.format(key, val))

                if cmd['scs'] != 0:
                    fmtvals.append(f'scs={cmd["scs"]}')

                params_str = ', '.join(fmtvals)
            else:
                params_str = 'N/A'

            if 'source' in self.colnames:
                lines.append('{} | {:16s} | {:10s} | {:8s} | {}'.format(
                    cmd['date'], cmd['type'], cmd['tlmsid'], cmd['source'], params_str))
            else:
                lines.append('{} | {:16s} | {:10s} | {:8d} | {}'.format(
                    cmd['date'], cmd['type'], cmd['tlmsid'], cmd['timeline_id'], params_str))

        return lines

    def pprint_like_backstop(self, logger_func=None, logger_text=''):
        if logger_func is None:
            lines = self.pformat_like_backstop()
            for line in lines:
                print(line)
        else:
            lines = self.pformat_like_backstop()
            logger_func(logger_text + '\n' + '\n'.join(lines) + '\n')


def get_par_idx_update_pars_dict(pars_dict, cmd):
    """Get par_idx representing index into pars tuples dict.

    This is used internally in updating the commands H5 and commands PARS_DICT
    pickle files. The ``pars_dict`` input is updated in place.

    This code was factored out verbatim from kadi.update_cmds.py.

    :param pars_dict: dict of pars tuples
    :param cmd: dict or CommandRow
        Command for updated par_idx
    :returns: int
        Params index (value of corresponding pars tuple dict key)
    """
    # Define a consistently ordered tuple that has all command parameter information
    pars = cmd['params']
    keys = set(pars.keys()) - set(('SCS', 'STEP', 'TLMSID'))
    if cmd['tlmsid'] == 'AOSTRCAT':
        # Skip star catalog command because that has many (uninteresting) parameters
        # and increases the file size and load speed by an order of magnitude.
        pars_tup = ()
    else:
        pars_tup = tuple((key.lower(), pars[key]) for key in sorted(keys))

    try:
        par_idx = pars_dict[pars_tup]
    except KeyError:
        # Along with transition to 32-bit idx in #190, ensure that idx=65535
        # never gets used. Prior to #190 this value was being used by
        # get_cmds_from_backstop() assuming that it will never occur as a
        # key in the pars_dict. Adding 65536 allows older versions to work
        # with the new cmds.pkl pars_dict.
        par_idx = len(pars_dict) + 65536
        pars_dict[pars_tup] = par_idx

    return par_idx
