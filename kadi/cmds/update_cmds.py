import sys
import os
import logging
import argparse
import cPickle as pickle

import numpy as np
import tables

import Ska.DBI
import Ska.File
import Ska.ParseCM
from Chandra.Time import DateTime


CMDS_DTYPE = [('idx', np.uint16),
              ('date', '|S21'),
              ('time', np.float64),
              ('cmd', '|S12'),
              ('tlmsid', '|S10'),
              ('scs', np.uint8),
              ('step', np.uint16),
              ('timeline_id', np.uint32)]


def get_opt(args=None):
    """Get options for command line interface to update_cmd_states.
    """
    parser = argparse.ArgumentParser(description='Update HDF5 cmds table')
    parser.add_argument("--outroot",
                        default='cmds',
                        help="root filename for HDF5 (.h5) and "
                             "pickle (.pkl) files (default='cmds')")
    parser.add_argument("--mp-dir",
                        default='/data/mpcrit1/mplogs',
                        help="MP load directory")
    parser.add_argument("--start",
                        help="Start date for update (default=stop-21 days)")
    parser.add_argument("--stop",
                        help="Stop date for update (default=Now)")
    parser.add_argument("--loglevel",
                        type=int,
                        default=10,
                        help='Log level (10=debug, 20=info, 30=warnings)')

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
            bs_cmds = Ska.ParseCM.read_backstop(bs_file)
            logging.info('Read {} commands from {}'.format(len(bs_cmds), bs_file))
            cache[bs_file] = bs_cmds
        else:
            bs_cmds = cache[bs_file]

        # Only store commands for this timeline (match SCS and date)
        bs_cmds = [x for x in bs_cmds
                   if tl['datestart'] <= x['date'] <= tl['datestop']
                   and x['scs'] == tl['scs']]

        for bs_cmd in bs_cmds:
            bs_cmd['timeline_id'] = tl['id']

        logging.info('  Got {} backstop commands for timeline_id={} and SCS={}'
                     .format(len(bs_cmds), tl['id'], tl['scs']))
        cmds.extend(bs_cmds)

    # Sort by date and SCS step number.
    cmds = sorted(cmds, key=lambda y: (y['date'], y['step']))
    logging.debug('Read total of {} commands'.format(len(cmds)))

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
            logging.info('   Iteration {}'.format(i))

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

        idx_cmds.append((par_idx, cmd['date'], cmd['time'], cmd['cmd'], cmd.get('tlmsid'),
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
        logging.info('Opened h5 cmds table {}'.format(h5file))
    except tables.NoSuchNodeError:
        h5.createTable(h5.root, 'data', cmds, "cmds", expectedrows=2e6)
        logging.info('Created h5 cmds table {}'.format(h5file))
    else:
        h5_timeline_ids = h5d.cols.timeline_id[:]
        ok = np.zeros(len(h5_timeline_ids), dtype=bool)
        for timeline_id in set(cmds['timeline_id']):
            ok |= h5_timeline_ids == timeline_id
        del_idxs = np.flatnonzero(ok)
        if not np.all(np.diff(del_idxs) == 1):
            raise ValueError('Inconsistency in timeline_ids')
        h5d.truncate(np.min(del_idxs))
        logging.debug('Deleted cmds indexes {} .. {}'.format(np.min(del_idxs), np.max(del_idxs)))
        h5d.append(cmds)
        logging.info('Added {} commands to HDF5 cmds table'.format(len(cmds)))

    h5.flush()
    logging.info('Upated HDF5 cmds table {}'.format(h5file))
    h5.close()


def main(args=None):
    opt = get_opt(args)

    # Configure logging to emit msgs to stdout
    logging.basicConfig(level=opt.loglevel,
                        format='%(message)s',
                        stream=sys.stdout)

    stop = DateTime(opt.stop)
    start = DateTime(opt.start) if opt.start else stop - 21

    # Get timeline_loads including and after start
    db = Ska.DBI.DBI(dbi='sybase', server='sybase', user='aca_read')
    timeline_loads = db.fetchall("""SELECT * from timeline_loads
                                    WHERE datestop > '{}' AND datestart < '{}'
                                    ORDER BY id"""
                                 .format(start.date, stop.date))
    db.conn.close()

    logging.info('Found {} timelines included within {} to {}'
                 .format(len(timeline_loads), start.date, stop.date))

    h5file = opt.outroot + '.h5'
    pars_dict_file = opt.outroot + '.pkl'
    try:
        with open(pars_dict_file, 'r') as fh:
            pars_dict = pickle.load(fh)
        logging.info('Read {} pars_dict values from {}'.format(len(pars_dict), pars_dict_file))
    except IOError:
        logging.info('No pars_dict file {} found, starting from empty dict'
                     .format(pars_dict_file))
        pars_dict = {}

    cmds = get_cmds(timeline_loads, opt.mp_dir)
    idx_cmds = get_idx_cmds(cmds, pars_dict)
    add_h5_cmds(h5file, idx_cmds)

    with open(pars_dict_file, 'w') as fh:
        pickle.dump(pars_dict, fh, protocol=-1)
        logging.info('Wrote {} pars_dict values to {}'.format(len(pars_dict), pars_dict_file))

    return cmds


if __name__ == '__main__':
    main()
