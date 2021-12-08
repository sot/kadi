# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import os
import shutil
import weakref

import numpy as np
from astropy.table import Table, vstack, Column

from kadi.commands.core import (get_cmds_from_backstop, CommandTable,
                                load_idx_cmds, load_pars_dict)
from kadi.commands.commands_v2 import update_cmds_archive
from kadi import paths
import Ska.DBI
from cxotime import CxoTime
import astropy.units as u

SKA = Path(os.environ['SKA'])
CMD_STATES_PATH = SKA / 'data' / 'cmd_states' / 'cmd_states.db3'


RLTT_ERA_START = 'APR1420B'

CMDS_DTYPE = [('idx', np.int32),
              ('date', '|S21'),
              ('type', '|S12'),
              ('tlmsid', '|S10'),
              ('scs', np.uint8),
              ('step', np.uint16),
              ('source', '|S8'),
              ('vcdu', np.int32)]


def make_cmds2_through_date(stop=None, step=7 * u.day):
    """Make initial cmds2 and then do archive updates every ``step`` days"""
    migrate_cmds_to_cmds2()
    update_cmds_archive(stop='2020-04-28', v1_v2_transition=True)

    date = CxoTime('2020-04-28')
    stop = CxoTime(stop)
    while date < stop:
        print('*' * 80)
        print(f'Updating cmds2 to {date}')
        print('*' * 80)
        update_cmds_archive(stop=date)
        date += step


def migrate_cmds_to_cmds2(start=0):
    """Migrate the legacy cmds.h5 through APR1320A to the new cmds2.h5 format.

    Key change is migrating from timeline_id to source, which is either the load
    name or "CMD_EVT" for commands from the event table.

    This create ``cmds2.h5`` and ``cmds2.pkl`` in the current directory.

    This includes all commands prior to APR1420B, which is the first
    load of the RLTT era that includes LOAD_EVENT commands. After generating
    cmds2.h5 and cmds2.pkl with this, run::

       >>> %run utils/migrate_cmds_to_cmds2.py
       >>> migrate_cmds_to_cmds2()
       >>> from kadi.commands.commands_v2 import update_cmds_archive
       >>> update_cmds_archive(stop='2020-04-28', v1_v2_transition=True)

    After this running the ``update_cmds_archive`` command as normal will work.

    :param start: int
        Index into existing loads to start at. Used in debugging by setting
        start=-200 to include just 200 commands from APR1320A.
    """
    cmds = load_idx_cmds(version=1)

    # This code is to get the load name ("source") for each cmd
    with Ska.DBI.DBI(dbi='sqlite', server=str(CMD_STATES_PATH)) as db:
        timelines = db.fetchall("""SELECT * from timelines""")
    timelines = Table(timelines)

    timeline_id_to_load_name = {0: 'CMD_EVT'}
    for timeline in timelines:
        # dir looks like /2002/JAN0702/oflsd/
        load_name = timeline['dir'][6:13] + timeline['dir'][-2].upper()
        timeline_id_to_load_name[timeline['id']] = load_name

    sources = []
    for cmd in cmds:
        sources.append(timeline_id_to_load_name[cmd['timeline_id']])

    col_index = cmds.colnames.index('timeline_id')
    cmds.add_column(Column(sources, name='source', dtype='S8'), index=col_index)
    del cmds['timeline_id']

    idx_start = np.flatnonzero(cmds['source'] == RLTT_ERA_START)[0]
    cmds = cmds[:idx_start]

    cmds[start:].write('cmds2.h5', path='data', overwrite=True)
    shutil.copy2(paths.DATA_DIR() / 'cmds.pkl', 'cmds2.pkl')


###############################################################################
# Stuff after here was used in initial testing / dev of the commands v2 code.
# Probably not useful going forward.
###############################################################################

def ska_load_dir(load_name):
    root = SKA / 'data' / 'mpcrit1' / 'mplogs'
    year = load_name[5:7]
    if year == '99':
        year = 1999
    else:
        year = 2000 + int(year)
    load_rev = load_name[-1].lower()
    load_dir = load_name[:-1]
    load_dir = root / str(year) / load_dir / f'ofls{load_rev}'
    return load_dir


def get_backstop_cmds_from_load_legacy(load):
    """This also updates the load cmd_start and cmd_stop as a side effect."""
    # THIS WILL BE MADE FASTER by using pre-generated gzipped CommandTable files
    load_name = load if isinstance(load, str) else load['name']
    load_dir = ska_load_dir(load_name)
    backstop_files = list(load_dir.glob('CR*.backstop'))
    if len(backstop_files) != 1:
        raise ValueError(f'Expected 1 backstop file for {load_name}')
    bs = get_cmds_from_backstop(backstop_files[0], remove_starcat=True)
    return bs


def fix_load_based_on_backstop_legacy(load, bs):
    # Get the first and last cmds for the load which are not the RLTT and
    # scheduled_stop pseudo-cmds.
    for cmd in bs:
        if cmd['type'] != 'LOAD_EVENT':
            load['cmd_start'] = cmd['date']
            break
    for cmd in bs[::-1]:
        if cmd['type'] != 'LOAD_EVENT':
            load['cmd_stop'] = cmd['date']
            break
    for cmd in bs:
        if (cmd['type'] == 'LOAD_EVENT'
                and cmd['params']['event_type'] == 'RUNNING_LOAD_TERMINATION_TIME'):
            load['rltt'] = cmd['date']
            break
    for cmd in bs[::-1]:
        if (cmd['type'] == 'LOAD_EVENT'
                and cmd['params']['event_type'] == 'SCHEDULED_STOP_TIME'):
            load['scheduled_stop_time'] = cmd['date']
            break

    if load['observing_stop'] == load['cmd_stop']:
        del load['observing_stop']
    if load['vehicle_stop'] == load['cmd_stop']:
        del load['vehicle_stop']


def get_backstop_cmds_from_loads_legacy(loads):
    """Get all the commands using LEGACY products, specifically loads includes
    interrupt times from legacy timelines.
    """
    bs_list = []
    for load in loads:
        bs = get_backstop_cmds_from_load_legacy(load)
        fix_load_based_on_backstop_legacy(load, bs)

        bs = interrupt_load_commands_legacy(load, bs)

        bs_list.append(bs)

    bs_cmds = vstack(bs_list)
    bs_cmds.sort(['date', 'step', 'scs'])
    return bs_cmds


def interrupt_load_commands_legacy(load, cmds):
    # Cut commands beyond stop times
    bad = np.zeros(len(cmds), dtype=bool)
    if 'observing_stop' in load:
        bad |= ((cmds['date'] > load['observing_stop'])
                & (cmds['scs'] > 130))
    if 'vehicle_stop' in load:
        bad |= ((cmds['date'] > load['vehicle_stop'])
                & (cmds['scs'] < 131))
    if np.any(bad):
        cmds = cmds[~bad]
    return cmds
