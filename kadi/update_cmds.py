# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import argparse
import difflib
import pickle
from pathlib import Path

import numpy as np
import tables

import pyyaks.logger
import Ska.DBI
import Ska.File
from Chandra.Time import DateTime
from ska_helpers.run_info import log_run_info

from .paths import IDX_CMDS_PATH, PARS_DICT_PATH
from . import __version__

MIN_MATCHING_BLOCK_SIZE = 500
BACKSTOP_CACHE = {}
CMDS_DTYPE = [('idx', np.uint32),
              ('date', '|S21'),
              ('type', '|S12'),
              ('tlmsid', '|S10'),
              ('scs', np.uint8),
              ('step', np.uint16),
              ('timeline_id', np.uint32),
              ('vcdu', np.int32)]

logger = None  # This is set as a global in main.  Define here for pyflakes.


class UpdatedDict(dict):
    """
    Dict with an ``n_updated`` attribute that gets incremented when any key value is set.
    """
    n_updated = 0

    def __setitem__(self, *args, **kwargs):
        self.n_updated += 1
        super(UpdatedDict, self).__setitem__(*args, **kwargs)


def get_opt(args=None):
    """
    Get options for command line interface to update_cmd_states.
    """
    parser = argparse.ArgumentParser(description='Update HDF5 cmds table')
    parser.add_argument("--mp-dir",
                        help=("MP load directory (default=/data/mpcrit1/mplogs) "
                              "or $SKA/data/mpcrit1/mplogs)"))
    parser.add_argument("--start",
                        help="Start date for update (default=stop-42 days)")
    parser.add_argument("--stop",
                        help="Stop date for update (default=Now+21 days)")
    parser.add_argument("--log-level",
                        type=int,
                        default=10,
                        help='Log level (10=debug, 20=info, 30=warnings)')
    parser.add_argument("--data-root",
                        default='.',
                        help="Data root (default='.')")
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))

    args = parser.parse_args(args)
    return args


def fix_nonload_cmds(nl_cmds):
    """
    Convert non-load commands commands dict format from Chandra.cmd_states
    to the values/structure needed here.  A typical value is shown below:
    {'cmd': u'SIMTRANS',               # Needs to be 'type'
    'date': u'2017:066:00:24:22.025',
    'id': 371228,                      # Store as params['nonload_id'] for provenence
    'msid': None,                      # Goes into params
    'params': {u'POS': -99616},
    'scs': None,                       # Set to 0
    'step': None,                      # Set to 0
    'time': 605233531.20899999,        # Ignored
    'timeline_id': None,               # Set to 0
    'tlmsid': None,                    # 'None' if None
    'vcdu': None},                     # Set to -1
    """
    new_cmds = []
    for cmd in nl_cmds:
        new_cmd = {}
        new_cmd['date'] = str(cmd['date'])
        new_cmd['type'] = str(cmd['cmd'])
        new_cmd['tlmsid'] = str(cmd['tlmsid'])
        for key in ('scs', 'step', 'timeline_id'):
            new_cmd[key] = 0
        new_cmd['vcdu'] = -1

        new_cmd['params'] = {}
        new_cmd['params']['nonload_id'] = int(cmd['id'])
        if cmd['msid'] is not None:
            new_cmd['params']['msid'] = str(cmd['msid'])

        # De-numpy (otherwise unpickling on PY3 has problems).
        if 'params' in cmd:
            params = new_cmd['params']
            for key, val in cmd['params'].items():
                key = str(key)
                try:
                    val = val.item()
                except AttributeError:
                    pass
                params[key] = val

        new_cmds.append(new_cmd)

    return new_cmds


def _tl_to_bs_cmds(tl_cmds, tl_id, db):
    """
    Convert the commands ``tl_cmds`` (numpy recarray) that occur in the
    timeline ``tl_id'' to a format mimicking backstop commands from
    Ska.ParseCM.read_backstop().  This includes reading parameter values
    from the ``db``.

    :param tl_cmds: numpy recarray of commands from timeline load segment
    :param tl_id: timeline id
    :param db: Ska.DBI db object

    :returns: list of command dicts
    """
    bs_cmds = [dict((col, row[col]) for col in tl_cmds.dtype.names)
               for row in tl_cmds]
    cmd_index = dict((x['id'], x) for x in bs_cmds)

    # Add 'params' dict of command parameter key=val pairs to each tl_cmd
    for par_table in ('cmd_intpars', 'cmd_fltpars'):
        tl_params = db.fetchall("SELECT * FROM %s WHERE timeline_id %s" %
                                (par_table,
                                 '= %d' % tl_id if tl_id else 'IS NULL'))

        # Build up the params dict for each command in timeline load segment
        for par in tl_params:
            # I.e. cmd_index[par.cmd_id]['params'][par.name] = par.value
            # but create the ['params'] dict as needed.
            if par.cmd_id in cmd_index:
                cmd_index[par.cmd_id].setdefault('params', {})[par.name] = par.value

    return bs_cmds


def get_cmds(start, stop, mp_dir='/data/mpcrit1/mplogs'):
    """
    Get backstop commands corresponding to the supplied timeline load segments.
    The timeline load segments must be ordered by 'id'.

    Return cmds in the format defined by Ska.ParseCM.read_backstop().
    """
    # Get timeline_loads within date range.  Also get non-load commands
    # within the date range covered by the timelines.
    server = os.path.join(os.environ['SKA'], 'data', 'cmd_states', 'cmd_states.db3')
    with Ska.DBI.DBI(dbi='sqlite', server=server) as db:
        timeline_loads = db.fetchall("""SELECT * from timeline_loads
                                        WHERE datestop > '{}' AND datestart < '{}'
                                        ORDER BY id"""
                                     .format(start.date, stop.date))

        # Get non-load commands (from autonomous or ground SCS107, NSM, etc) in the
        # time range that the timelines span.
        tl_datestart = min(timeline_loads['datestart'])
        nl_cmds = db.fetchall('SELECT * from cmds where timeline_id IS NULL and '
                              'date >= "{}" and date <= "{}"'
                              .format(tl_datestart, stop.date))

        # Private method from cmd_states.py fetches the actual int/float param values
        # and returns list of dict.
        nl_cmds = _tl_to_bs_cmds(nl_cmds, None, db)
        nl_cmds = fix_nonload_cmds(nl_cmds)
        logger.info(f'Found {len(nl_cmds)} non-load commands between {tl_datestart} : {stop.date}')

    logger.info('Found {} timelines included within {} to {}'
                .format(len(timeline_loads), start.date, stop.date))

    if np.min(np.diff(timeline_loads['id'])) < 1:
        raise ValueError('Timeline loads id not monotonically increasing')

    cmds = []
    orbit_cmds = []
    orbit_cmd_files = set()

    for tl in timeline_loads:
        bs_file = Ska.File.get_globfiles(os.path.join(mp_dir + tl.mp_dir,
                                                      '*.backstop'))[0]
        if bs_file not in BACKSTOP_CACHE:
            bs_cmds = read_backstop(bs_file)
            logger.info('Read {} commands from {}'.format(len(bs_cmds), bs_file))
            BACKSTOP_CACHE[bs_file] = bs_cmds
        else:
            bs_cmds = BACKSTOP_CACHE[bs_file]

        # Process ORBPOINT (orbit event) pseudo-commands in backstop.  These
        # have scs=0 and need to be treated separately since during a replan
        # or shutdown we still want these ORBPOINT to be in the cmds archive
        # and not be excluded by timeline intervals.
        if bs_file not in orbit_cmd_files:
            bs_orbit_cmds = [x for x in bs_cmds if x['type'] == 'ORBPOINT']
            for orbit_cmd in bs_orbit_cmds:
                orbit_cmd['timeline_id'] = tl['id']
                if 'EVENT_TYPE' not in orbit_cmd['params']:
                    orbit_cmd['params']['EVENT_TYPE'] = orbit_cmd['params']['TYPE']
                    del orbit_cmd['params']['TYPE']
            orbit_cmds.extend(bs_orbit_cmds)
            orbit_cmd_files.add(bs_file)

        # Only store commands for this timeline (match SCS and date)
        bs_cmds = [x for x in bs_cmds
                   if tl['datestart'] <= x['date'] <= tl['datestop']
                   and x['scs'] == tl['scs']]

        for bs_cmd in bs_cmds:
            bs_cmd['timeline_id'] = tl['id']

        logger.info('  Got {} backstop commands for timeline_id={} and SCS={}'
                    .format(len(bs_cmds), tl['id'], tl['scs']))
        cmds.extend(bs_cmds)

    orbit_cmds = get_unique_orbit_cmds(orbit_cmds)
    logger.debug('Read total of {} orbit commands'
                 .format(len(orbit_cmds)))

    cmds.extend(nl_cmds)
    cmds.extend(orbit_cmds)

    # Sort by date and SCS step number.
    cmds = sorted(cmds, key=lambda y: (y['date'], y['step']))
    logger.debug('Read total of {} commands ({} non-load commands)'
                 .format(len(cmds), len(nl_cmds)))

    return cmds


def get_unique_orbit_cmds(orbit_cmds):
    """
    Given list of ``orbit_cmds`` find the quasi-unique set.  In the event of a
    replan/reopen or other schedule oddity, it can happen that there are multiple cmds
    that describe the same orbit event.  Since the detailed timing might change between
    schedule runs, cmds are considered the same if the date is within 3 minutes.
    """
    if len(orbit_cmds) == 0:
        return []

    # Sort by (event_type, date)
    orbit_cmds.sort(key=lambda y: (y['params']['EVENT_TYPE'], y['date']))

    uniq_cmds = [orbit_cmds[0]]
    # Step through one at a time and add to uniq_cmds only if the candidate is
    # "different" from uniq_cmds[-1].
    for cmd in orbit_cmds:
        last_cmd = uniq_cmds[-1]
        if (cmd['params']['EVENT_TYPE'] == last_cmd['params']['EVENT_TYPE']
                and abs(DateTime(cmd['date']).secs - DateTime(last_cmd['date']).secs) < 180):
            # Same event as last (even if date is a bit different).  Now if this one
            # has a larger timeline_id that means it is from a more recent schedule, so
            # use that one.
            if cmd['timeline_id'] > last_cmd['timeline_id']:
                uniq_cmds[-1] = cmd
        else:
            uniq_cmds.append(cmd)

    uniq_cmds.sort(key=lambda y: y['date'])

    return uniq_cmds


def get_idx_cmds(cmds, pars_dict):
    """
    For the input `cmds` (list of dicts), convert to the indexed command format where
    parameters are specified as an index into `pars_dict`, a dict of unique parameter
    values.

    Returns `idx_cmds` as a list of tuples:
       (par_idx, date, time, cmd, tlmsid, scs, step, timeline_id, vcdu)
    """
    idx_cmds = []

    for i, cmd in enumerate(cmds):
        if i % 10000 == 9999:
            logger.info('   Iteration {}'.format(i))

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
            par_idx = len(pars_dict)
            pars_dict[pars_tup] = par_idx

        idx_cmds.append((par_idx, cmd['date'], cmd['type'], cmd.get('tlmsid'),
                         cmd['scs'], cmd['step'], cmd['timeline_id'], cmd['vcdu']))

    return idx_cmds


def add_h5_cmds(h5file, idx_cmds):
    """
    Add `idx_cmds` to HDF5 file `h5file` of indexed spacecraft commands.
    If file does not exist then create it.
    """
    # Note: reading this file uncompressed is about 5 times faster, so sacrifice file size
    # for read speed and do not use compression.
    h5 = tables.open_file(h5file, mode='a')

    # Convert cmds (list of tuples) to numpy structured array.  This also works for an
    # existing structured array.
    cmds = np.array(idx_cmds, dtype=CMDS_DTYPE)

    # TODO : make sure that changes in non-load commands triggers an update

    try:
        h5d = h5.root.data
        logger.info('Opened h5 cmds table {}'.format(h5file))
    except tables.NoSuchNodeError:
        h5.create_table(h5.root, 'data', cmds, "cmds", expectedrows=2e6)
        logger.info('Created h5 cmds table {}'.format(h5file))
    else:
        date0 = min(idx_cmd[1] for idx_cmd in idx_cmds)
        h5_date = h5d.cols.date[:]
        idx_recent = np.searchsorted(h5_date, date0)
        logger.info('Selecting commands from h5d[{}:]'.format(idx_recent))
        logger.info('  {}'.format(str(h5d[idx_recent])))
        h5d_recent = h5d[idx_recent:]  # recent h5d entries

        # Define the column names that specify a complete and unique row
        key_names = ('date', 'type', 'tlmsid', 'scs', 'step', 'timeline_id', 'vcdu')

        h5d_recent_vals = [tuple(
            row[x].decode('ascii') if isinstance(row[x], bytes) else str(row[x])
            for x in key_names)
            for row in h5d_recent]
        idx_cmds_vals = [tuple(str(x) for x in row[1:]) for row in idx_cmds]

        diff = difflib.SequenceMatcher(a=h5d_recent_vals, b=idx_cmds_vals, autojunk=False)
        blocks = diff.get_matching_blocks()
        logger.info('Matching blocks for existing HDF5 and timeline commands')
        for block in blocks:
            logger.info('  {}'.format(block))
        opcodes = diff.get_opcodes()
        logger.info('Diffs between existing HDF5 and timeline commands')
        for opcode in opcodes:
            logger.info('  {}'.format(opcode))
        # Find the first matching block that is sufficiently long
        for block in blocks:
            if block.size > MIN_MATCHING_BLOCK_SIZE:
                break
        else:
            raise ValueError('No matching blocks at least {} long'
                             .format(MIN_MATCHING_BLOCK_SIZE))

        # Index into idx_cmds at the end of the large matching block.  block.b is the
        # beginning of the match.
        idx_cmds_idx = block.b + block.size

        if idx_cmds_idx < len(cmds):
            # Index into h5d at the point of the first diff after the large matching block
            h5d_idx = block.a + block.size + idx_recent

            if h5d_idx < len(h5d):
                logger.debug('Deleted relative cmds indexes {} .. {}'.format(h5d_idx - idx_recent,
                                                                             len(h5d) - idx_recent))
                logger.debug('Deleted cmds indexes {} .. {}'.format(h5d_idx, len(h5d)))
                h5d.truncate(h5d_idx)

            h5d.append(cmds[idx_cmds_idx:])
            logger.info('Added {} commands to HDF5 cmds table'.format(len(cmds[idx_cmds_idx:])))
        else:
            logger.info('No new timeline commands, HDF5 cmds table not updated')

    h5.flush()
    logger.info('Upated HDF5 cmds table {}'.format(h5file))
    h5.close()


def main(args=None):
    global logger

    opt = get_opt(args)

    logger = pyyaks.logger.get_logger(name='kadi', level=opt.log_level,
                                      format="%(asctime)s %(message)s")

    log_run_info(logger.info, opt)

    # Set the global root data directory.  This gets used in ..paths to
    # construct file names.  The use of an env var is needed to allow
    # configurability of the root data directory within django.
    os.environ['KADI'] = os.path.abspath(opt.data_root)
    idx_cmds_path = IDX_CMDS_PATH()
    pars_dict_path = PARS_DICT_PATH()

    try:
        with open(pars_dict_path, 'rb') as fh:
            pars_dict = pickle.load(fh)
        logger.info('Read {} pars_dict values from {}'.format(len(pars_dict), pars_dict_path))
    except IOError:
        logger.info('No pars_dict file {} found, starting from empty dict'
                    .format(pars_dict_path))
        pars_dict = {}

    if not opt.mp_dir:
        for prefix in ('/', os.environ['SKA']):
            pth = Path(prefix, 'data', 'mpcrit1', 'mplogs')
            if pth.exists():
                opt.mp_dir = str(pth)
                break
        else:
            raise FileNotFoundError('no mission planning directories found (need --mp-dir)')
    logger.info(f'Using mission planning files at {opt.mp_dir}')

    # Recast as dict subclass that remembers if any element was updated
    pars_dict = UpdatedDict(pars_dict)

    stop = DateTime(opt.stop) if opt.stop else DateTime() + 21
    start = DateTime(opt.start) if opt.start else stop - 42

    cmds = get_cmds(start, stop, opt.mp_dir)
    idx_cmds = get_idx_cmds(cmds, pars_dict)
    add_h5_cmds(idx_cmds_path, idx_cmds)

    if pars_dict.n_updated > 0:
        with open(pars_dict_path, 'wb') as fh:
            pickle.dump(pars_dict, fh, protocol=2)
            logger.info('Wrote {} pars_dict values ({} new) to {}'
                        .format(len(pars_dict), pars_dict.n_updated, pars_dict_path))
    else:
        logger.info('pars_dict was unmodified, not writing')


def _coerce_type(val):
    """Coerce the supplied ``val`` (typically a string) into an int or float if
    possible, otherwise as a string.
    """
    try:
        val = int(val)
    except ValueError:
        try:
            val = float(val)
        except ValueError:
            val = str(val)
    return val


def parse_params(paramstr):
    """
    Parse parameters key1=val1,key2=val2,... from ``paramstr``

    Parameter values are cast to the first type (int, float, or str) that
    succeeds.

    :param paramstr: Comma separated string of key=val pairs
    :rtype: dict of key=val pairs
    """
    params = {}
    for opt in paramstr.split(','):
        try:
            key, val = opt.split('=')
            params[key] = val if key == 'HEX' else _coerce_type(val)
        except Exception:
            pass  # backstop has some quirks like blank or '??????' fields

    return params


def read_backstop(filename):
    """
    Read commands from backstop file.

    Create dict with keys as follows for each command.  ``paramstr`` is the
    actual string with comma-separated parameters and ``params`` is the
    corresponding dict of key=val pairs.

    :param filename: Backstop file name
    :returns: list of dict for each command
    """
    bs = []
    for bs_line in open(filename):
        bs_line = bs_line.replace(' ', '')
        date, vcdu, cmd_type, paramstr = [x for x in bs_line.split('|')]
        vcdu = int(vcdu[:-1])  # Get rid of final '0' from '8023268 0' (where space was stripped)
        params = parse_params(paramstr)
        bs.append({'date': date,
                   'type': cmd_type,
                   'params': params,
                   'tlmsid': params.get('TLMSID'),
                   'scs': params.get('SCS'),
                   'step': params.get('STEP'),
                   'vcdu': vcdu
                   })
    return bs


if __name__ == '__main__':
    main()
