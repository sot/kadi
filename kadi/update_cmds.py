import os
import argparse
import cPickle as pickle

import numpy as np
import tables

import pyyaks.logger
import Ska.DBI
import Ska.File
from Chandra.Time import DateTime
from . import occweb

CMDS_DTYPE = [('idx', np.uint16),
              ('date', '|S21'),
              ('type', '|S12'),
              ('tlmsid', '|S10'),
              ('scs', np.uint8),
              ('step', np.uint16),
              ('timeline_id', np.uint32)]

logger = None  # This is set as a global in main.  Define here for pyflakes.


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
                        help="Start date for update (default=stop-21 days)")
    parser.add_argument("--stop",
                        help="Stop date for update (default=Now)")
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

    args = parser.parse_args(args)
    return args


def get_cmds(timeline_loads, mp_dir='/data/mpcrit1/mplogs'):
    """
    Get backstop commands corresponding to the supplied timeline load segments.
    The timeline load segments must be ordered by 'id'.

    Return cmds in the format defined by Ska.ParseCM.read_backstop().
    """
    if np.min(np.diff(timeline_loads['id'])) < 1:
        raise ValueError('Timeline loads id not monotonically increasing')

    cache = {}
    cmds = []
    for tl in timeline_loads:
        bs_file = Ska.File.get_globfiles(os.path.join(mp_dir + tl.mp_dir,
                                                      '*.backstop'))[0]
        if bs_file not in cache:
            bs_cmds = read_backstop(bs_file)
            logger.info('Read {} commands from {}'.format(len(bs_cmds), bs_file))
            cache[bs_file] = bs_cmds
        else:
            bs_cmds = cache[bs_file]

        # Only store commands for this timeline (match SCS and date)
        bs_cmds = [x for x in bs_cmds
                   if tl['datestart'] <= x['date'] <= tl['datestop']
                   and x['scs'] == tl['scs']]

        for bs_cmd in bs_cmds:
            bs_cmd['timeline_id'] = tl['id']

        logger.info('  Got {} backstop commands for timeline_id={} and SCS={}'
                    .format(len(bs_cmds), tl['id'], tl['scs']))
        cmds.extend(bs_cmds)

    # Sort by date and SCS step number.
    cmds = sorted(cmds, key=lambda y: (y['date'], y['step']))
    logger.debug('Read total of {} commands'.format(len(cmds)))

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

    try:
        h5d = h5.root.data
        logger.info('Opened h5 cmds table {}'.format(h5file))
    except tables.NoSuchNodeError:
        h5.createTable(h5.root, 'data', cmds, "cmds", expectedrows=2e6)
        logger.info('Created h5 cmds table {}'.format(h5file))
    else:
        h5_timeline_ids = h5d.cols.timeline_id[:]
        ok = np.zeros(len(h5_timeline_ids), dtype=bool)
        for timeline_id in set(cmds['timeline_id']):
            ok |= h5_timeline_ids == timeline_id
        del_idxs = np.flatnonzero(ok)
        if not np.all(np.diff(del_idxs) == 1):
            raise ValueError('Inconsistency in timeline_ids')
        h5d.truncate(np.min(del_idxs))
        logger.debug('Deleted cmds indexes {} .. {}'.format(np.min(del_idxs), np.max(del_idxs)))
        h5d.append(cmds)
        logger.info('Added {} commands to HDF5 cmds table'.format(len(cmds)))

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
    from .paths import IDX_CMDS_PATH, PARS_DICT_PATH

    if opt.occ:
        # Get cmds files from HEAD via lucky ftp
        occweb.ftp_get_from_lucky('kadi', [IDX_CMDS_PATH, PARS_DICT_PATH], logger=logger)
        return

    stop = DateTime(opt.stop)
    start = DateTime(opt.start) if opt.start else stop - 21

    # Get timeline_loads including and after start
    db = Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read')
    timeline_loads = db.fetchall("""SELECT * from timeline_loads
                                    WHERE datestop > '{}' AND datestart < '{}'
                                    ORDER BY id"""
                                 .format(start.date, stop.date))
    db.conn.close()

    logger.info('Found {} timelines included within {} to {}'
                .format(len(timeline_loads), start.date, stop.date))

    try:
        with open(PARS_DICT_PATH, 'r') as fh:
            pars_dict = pickle.load(fh)
        logger.info('Read {} pars_dict values from {}'.format(len(pars_dict), PARS_DICT_PATH))
    except IOError:
        logger.info('No pars_dict file {} found, starting from empty dict'
                    .format(PARS_DICT_PATH))
        pars_dict = {}

    cmds = get_cmds(timeline_loads, opt.mp_dir)
    idx_cmds = get_idx_cmds(cmds, pars_dict)
    add_h5_cmds(IDX_CMDS_PATH, idx_cmds)

    with open(PARS_DICT_PATH, 'w') as fh:
        pickle.dump(pars_dict, fh, protocol=-1)
        logger.info('Wrote {} pars_dict values to {}'.format(len(pars_dict), PARS_DICT_PATH))

    # Push cmds files to OCC via lucky ftp
    occweb.ftp_put_to_lucky('kadi', [IDX_CMDS_PATH, PARS_DICT_PATH], logger=logger)


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
