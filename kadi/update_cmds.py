# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import argparse
import difflib

import numpy as np
import tables
from six.moves import cPickle as pickle

import pyyaks.logger
import Ska.DBI
import Ska.File
from Chandra.Time import DateTime
from Chandra.cmd_states.cmd_states import _tl_to_bs_cmds
from . import occweb
from .paths import IDX_CMDS_PATH, PARS_DICT_PATH

MIN_MATCHING_BLOCK_SIZE = 500
BACKSTOP_CACHE = {}
CMDS_DTYPE = [('idx', np.uint16),
              ('date', '|S21'),
              ('type', '|S12'),
              ('tlmsid', '|S10'),
              ('scs', np.uint8),
              ('step', np.uint16),
              ('timeline_id', np.uint32)]

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
    OCC_SOT_ACCOUNT = os.environ['USER'].lower() == 'sot'
    parser = argparse.ArgumentParser(description='Update HDF5 cmds table')
    parser.add_argument("--mp-dir",
                        default='/data/mpcrit1/mplogs',
                        help="MP load directory")
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
    parser.add_argument("--occ",
                        default=OCC_SOT_ACCOUNT,
                        action='store_true',
                        help="Running at OCC as copy-only client")
    parser.add_argument("--ftp",
                        default=False,
                        action='store_true',
                        help="Store or get files via ftp (implied for --occ)")

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
    'vcdu': None},                     # Ignored
    """
    for cmd in nl_cmds:
        cmd['type'] = cmd['cmd']
        cmd.setdefault('params', {})
        cmd['params']['nonload_id'] = cmd['id']
        if cmd['msid'] is not None:
            cmd['params']['msid'] = cmd['msid']
        if cmd['tlmsid'] is None:
            cmd['tlmsid'] = 'None'
        for key in ('scs', 'step', 'timeline_id'):
            cmd[key] = 0


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
        tl_datestop = max(timeline_loads['datestop'])
        nl_cmds = db.fetchall('SELECT * from cmds where timeline_id IS NULL and '
                              'date >= "{}" and date <= "{}"'
                              .format(tl_datestart, tl_datestop))

        # Private method from cmd_states.py fetches the actual int/float param values
        # and returns list of dict.
        nl_cmds = _tl_to_bs_cmds(nl_cmds, None, db)
        fix_nonload_cmds(nl_cmds)

    logger.info('Found {} timelines included within {} to {}'
                .format(len(timeline_loads), start.date, stop.date))

    if np.min(np.diff(timeline_loads['id'])) < 1:
        raise ValueError('Timeline loads id not monotonically increasing')

    cmds = []
    for tl in timeline_loads:
        bs_file = Ska.File.get_globfiles(os.path.join(mp_dir + tl.mp_dir,
                                                      '*.backstop'))[0]
        if bs_file not in BACKSTOP_CACHE:
            bs_cmds = read_backstop(bs_file)
            logger.info('Read {} commands from {}'.format(len(bs_cmds), bs_file))
            BACKSTOP_CACHE[bs_file] = bs_cmds
        else:
            bs_cmds = BACKSTOP_CACHE[bs_file]

        # Only store commands for this timeline (match SCS and date)
        bs_cmds = [x for x in bs_cmds
                   if tl['datestart'] <= x['date'] <= tl['datestop'] and
                   x['scs'] == tl['scs']]

        for bs_cmd in bs_cmds:
            bs_cmd['timeline_id'] = tl['id']

        logger.info('  Got {} backstop commands for timeline_id={} and SCS={}'
                    .format(len(bs_cmds), tl['id'], tl['scs']))
        cmds.extend(bs_cmds)

    cmds.extend(nl_cmds)

    # Sort by date and SCS step number.
    cmds = sorted(cmds, key=lambda y: (y['date'], y['step']))
    logger.debug('Read total of {} commands ({} non-load commands)'
                 .format(len(cmds), len(nl_cmds)))

    return cmds


def get_idx_cmds(cmds, pars_dict):
    """
    For the input `cmds` (list of dicts), convert to the indexed command format where
    parameters are specified as an index into `pars_dict`, a dict of unique parameter
    values.

    Returns `idx_cmds` as a list of tuples:
       (par_idx, date, time, cmd, tlmsid, scs, step)
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
                         cmd['scs'], cmd['step'], cmd['timeline_id']))

    return idx_cmds


def add_h5_cmds(h5file, idx_cmds):
    """
    Add `idx_cmds` to HDF5 file `h5file` of indexed spacecraft commands.
    If file does not exist then create it.
    """
    # Note: reading this file uncompressed is about 5 times faster, so sacrifice file size
    # for read speed and do not use compression.
    h5 = tables.openFile(h5file, mode='a')

    # Convert cmds (list of tuples) to numpy structured array.  This also works for an
    # existing structured array.
    cmds = np.array(idx_cmds, dtype=CMDS_DTYPE)

    # TODO : make sure that changes in non-load commands triggers an update

    try:
        h5d = h5.root.data
        logger.info('Opened h5 cmds table {}'.format(h5file))
    except tables.NoSuchNodeError:
        h5.createTable(h5.root, 'data', cmds, "cmds", expectedrows=2e6)
        logger.info('Created h5 cmds table {}'.format(h5file))
    else:
        date0 = min(idx_cmd[1] for idx_cmd in idx_cmds)
        h5_date = h5d.cols.date[:]
        idx_recent = np.searchsorted(h5_date, date0)
        logger.info('Selecting commands from h5d[{}:]'.format(idx_recent))
        logger.info('  {}'.format(str(h5d[idx_recent])))
        h5d_recent = h5d[idx_recent:]  # recent h5d entries

        # Define the column names that specify a complete and unique row
        key_names = ('date', 'type', 'tlmsid', 'scs', 'step', 'timeline_id')

        h5d_recent_vals = [tuple(str(row[x]) for x in key_names) for row in h5d_recent]
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

    # Set the global root data directory.  This gets used in ..paths to
    # construct file names.  The use of an env var is needed to allow
    # configurability of the root data directory within django.
    os.environ['KADI'] = os.path.abspath(opt.data_root)
    idx_cmds_path = IDX_CMDS_PATH()
    pars_dict_path = PARS_DICT_PATH()

    if opt.occ:
        # Get cmds files from HEAD via lucky ftp
        occweb.ftp_get_from_lucky('kadi', [idx_cmds_path, pars_dict_path], logger=logger)
        return

    try:
        with open(pars_dict_path, 'rb') as fh:
            pars_dict = pickle.load(fh)
        logger.info('Read {} pars_dict values from {}'.format(len(pars_dict), pars_dict_path))
    except IOError:
        logger.info('No pars_dict file {} found, starting from empty dict'
                    .format(pars_dict_path))
        pars_dict = {}

    # Recast as dict subclass that remembers if any element was updated
    pars_dict = UpdatedDict(pars_dict)

    stop = DateTime(opt.stop) if opt.stop else DateTime() + 21
    start = DateTime(opt.start) if opt.start else stop - 42

    cmds = get_cmds(start, stop, opt.mp_dir)
    idx_cmds = get_idx_cmds(cmds, pars_dict)
    add_h5_cmds(idx_cmds_path, idx_cmds)

    if pars_dict.n_updated > 0:
        with open(pars_dict_path, 'wb') as fh:
            pickle.dump(pars_dict, fh, protocol=-1)
            logger.info('Wrote {} pars_dict values ({} new) to {}'
                        .format(len(pars_dict), pars_dict.n_updated, pars_dict_path))
    else:
        logger.info('pars_dict was unmodified, not writing')

    if opt.ftp:
        # Push cmds files to OCC via lucky ftp
        occweb.ftp_put_to_lucky('kadi', [idx_cmds_path, pars_dict_path], logger=logger)


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
        except:
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
        params = parse_params(paramstr)
        bs.append({'date': date,
                   'type': cmd_type,
                   'params': params,
                   'tlmsid': params.get('TLMSID'),
                   'scs': params.get('SCS'),
                   'step': params.get('STEP'),
                   })
    return bs


if __name__ == '__main__':
    main()
