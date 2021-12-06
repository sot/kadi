# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path
import calendar
import re
import gzip
import pickle
import itertools
import functools
import weakref

import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
import requests

from kadi.commands import get_cmds_from_backstop
from kadi.commands.core import (load_idx_cmds, load_pars_dict, LazyVal,
                                get_par_idx_update_pars_dict)
from kadi.command_sets import get_cmds_from_event
from kadi import occweb
from kadi import paths
from cxotime import CxoTime
import logging


# TODO configuration options, but use DEFAULT_* in the mean time
# - cache loads (backstop) downloads from OCCweb (useful for development but
#   likely to just take disk space for users)
# - commands_dir ?
# - commands_version (v1, v2)
# - default_lookback_time
# - update_from_network (default True)
# - remove_old_loads (default True) Remove older loads from local directory.
DEFAULT_LOOKBACK = 30  # Lookback time for recent loads
UPDATE_FROM_NETWORK = True
CACHE_LOADS_IN_ASTROPY_CACHE = True  # Or maybe just be clever about cleaning old files?
MATCHING_BLOCK_SIZE = 50

# TODO: cache translation from cmd_events to CommandTable's  [Probably not]

APPROVED_LOADS_OCCWEB_DIR = Path('FOT/mission_planning/PRODUCTS/APPR_LOADS')

# https://docs.google.com/spreadsheets/d/<document_id>/export?format=csv&gid=<sheet_id>
CMD_EVENTS_SHEET_ID = '19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ'
CMD_EVENTS_SHEET_URL = f'https://docs.google.com/spreadsheets/d/{CMD_EVENTS_SHEET_ID}/export?format=csv'  # noqa

CMDS_DTYPE = [('idx', np.int32),
              ('date', '|S21'),
              ('type', '|S12'),
              ('tlmsid', '|S10'),
              ('scs', np.uint8),
              ('step', np.uint16),
              ('source', '|S8'),
              ('vcdu', np.int32)]

# Cached values of the full mission commands archive (cmds_v2.h5, cmds_v2.pkl).
# These are loaded on demand.
IDX_CMDS = LazyVal(functools.partial(load_idx_cmds, version=2))
PARS_DICT = LazyVal(functools.partial(load_pars_dict, version=2))
REV_PARS_DICT = LazyVal(lambda: {v: k for k, v in PARS_DICT.items()})

# Cache of recent commands keyed by scenario
CMDS_RECENT = {}

# APR1420B was the first load set to have RLTT (backstop 6.9)
RLTT_ERA_START = CxoTime('2020-04-14')

logger = logging.getLogger('kadi.commands')


def load_name_to_cxotime(name):
    """Convert load name to date"""
    mon = name[:3].capitalize()
    imon = list(calendar.month_abbr).index(mon)
    day = name[3:5]
    yr = name[5:7]
    if int(yr) > 50:
        year = f'19{yr}'
    else:
        year = f'20{yr}'
    out = CxoTime(f'{year}-{imon:02d}-{day}')
    out.format = 'date'
    return out


def interrupt_load_commands(load, cmds):
    # Cut commands beyond stop times
    bad = np.zeros(len(cmds), dtype=bool)
    if load['observing_stop'] != '':
        bad |= ((cmds['date'] > load['observing_stop'])
                & (cmds['scs'] > 130))
    if load['vehicle_stop'] != '':
        bad |= ((cmds['date'] > load['vehicle_stop'])
                & (cmds['scs'] < 131))
    if np.any(bad):
        logger.info(f'Cutting {bad.sum()} commands from {load["name"]}')
        cmds = cmds[~bad]
    return cmds


def merge_cmds_archive_recent(start, cmds_recent):
    logger.info('Merging cmds_recent with archive commands from {start}')
    # First get the part of the mission archive commands from `start`
    i0 = np.searchsorted(IDX_CMDS['date'], start.date)
    cmds_arch = IDX_CMDS[i0:]

    i0_arch, i0_recent = get_matching_block_idx(cmds_arch, cmds_recent)

    # Stored archive commands HDF5 has no `params` column, instead storing an
    # index to the param values which are in PARS_DICT. Add `params` object
    # column with None values and then stack with cmds_recent (which has
    # `params` already as dicts).
    cmds_arch.add_column(None, name='params')
    cmds = vstack([cmds_arch[:i0_arch], cmds_recent[i0_recent:]],
                  join_type='exact')

    # Need to give CommandTable a ref to REV_PARS_DICT so it can tranlate from
    # params index to the actual dict of values. Stored as a weakref so that
    # pickling and other serialization doesn't break.
    cmds.rev_pars_dict = weakref.ref(REV_PARS_DICT)

    return cmds


def get_matching_block_idx_simple(cmds_recent, cmds_arch, min_match):
    # Find the first command in cmd_arch that starts at the same date as the
    # block of recent commands. There might be multiple commands at the same
    # date, so we walk through this getting a block match of `min_match` size.
    # Matching block is defined by all `key_names` columns matching.
    date0 = cmds_recent['date'][0]
    i0_arch = np.searchsorted(cmds_arch['date'], date0)
    key_names = ('date', 'type', 'tlmsid', 'scs', 'step', 'vcdu')

    # Find block of commands in cmd_arch that match first min_match of
    # cmds_recent. Special case is min_match=0, which means we just want to
    # append the cmds_recent to the end of cmds_arch. This is the case for
    # the transition from pre-RLTT (APR1420B) to post, for the one-time
    # migration from version 1 to version 2.
    while min_match > 0:
        if all(np.all(cmds_arch[name][i0_arch:i0_arch + min_match]
                      == cmds_recent[name][:min_match])
               for name in key_names):
            break
        # No joy, step forward and make sure date still matches
        i0_arch += 1
        if cmds_arch['date'][i0_arch] != date0:
            raise ValueError(f'No matching commands block in archive found for recent_commands')

    logger.info(f'Found matching commands block in archive at {i0_arch}')
    return i0_arch


def get_cmds(start=None, stop=None, inclusive_stop=False, scenario=None):
    """Get commands using loads table, relying entirely on RLTT.

    :param start: CxoTime-like
        Start time for cmds
    :param stop: CxoTime-like
        Stop time for cmds
    :param scenario: str, None
        Scenario name
    :param inclusive_stop: bool
        Include commands at exactly ``stop`` if True.
    :param loads_stop: CxoTime-like, None
        Stop time for loads table (default is all available loads, but useful
        for development/testing work)
    :returns: CommandTable
    """
    if scenario not in CMDS_RECENT:
        cmds_recent = update_archive_and_get_cmds_recent(scenario)
        CMDS_RECENT[scenario] = cmds_recent
    else:
        cmds_recent = CMDS_RECENT[scenario]

    start = CxoTime(start or '1999:001')
    stop = CxoTime(stop or '2099:001')

    if start < CxoTime(cmds_recent['date'][0]) + 1 * u.day:
        cmds = merge_cmds_archive_recent(start, cmds_recent)
    else:
        cmds = cmds_recent

    idx0 = np.searchsorted(cmds['date'], start.date)
    idx1 = np.searchsorted(cmds['date'], stop.date,
                           side=('right' if inclusive_stop else 'left'))
    cmds = cmds[idx0:idx1]

    return cmds


def update_archive_and_get_cmds_recent(scenario=None, *, lookback=None, stop=None):
    """Update local loads table and downloaded loads and return all recent cmds.

    This also caches the recent commands in the global CMDS_RECENT dict.

    This relies entirely on RLTT and load_events to assemble the commands.

    :param scenario: str, None
        Scenario name
    :param lookback: int, Quantity, None
        Lookback time from ``stop`` for recent loads. If None, use DEFAULT_LOOKBACK.
    :param stop: CxoTime-like, None
        Stop time for loads table (default is now + 21 days)

    :returns: CommandTable
    """
    cmds_list = []  # List of CommandTable objects from loads and cmd_events
    rltts = []  # Corresponding list of RLTTs, where cmd_events use None for RLTT

    # Update local cmds_events.csv from Google Sheets
    cmd_events = update_cmd_events(scenario)

    # Update loads table and download/archive backstop files from OCCweb
    loads = update_loads(scenario, cmd_events=cmd_events,
                         lookback=lookback, stop=stop)
    logger.info(f'Including loads {", ".join(loads["name"])}')

    for load in loads:
        loads_backstop_path = paths.LOADS_BACKSTOP_PATH(load['name'])
        with gzip.open(loads_backstop_path, 'rb') as fh:
            cmds = pickle.load(fh)

        # Apply load interrupts (SCS-107, NSM) from the loads table to this
        # command load. This assumes that loads.csv has been updated
        # appropriately from cmd_events.csv (which might have come from the
        # Command Events sheet).
        cmds = interrupt_load_commands(load, cmds)
        if len(cmds) > 0:
            logger.info(f'Load {load["name"]} has {len(cmds)} commands')
            cmds_list.append(cmds)
            rltts.append(load['rltt'])
        else:
            logger.info(f'Load {load["name"]} has no commands, skipping')

    # Filter events outside the time interval, assuming command event cannot
    # last more than 2 weeks.
    start = CxoTime(min(loads['cmd_start']))
    stop = CxoTime(max(loads['cmd_stop']))
    bad = ((cmd_events['Date'] < (start - 14 * u.day).date)
           | (cmd_events['Date'] > stop.date))
    cmd_events = cmd_events[~bad]
    cmd_events_ids = [evt['Event'] + '-' + evt['Date'][:8] for evt in cmd_events]
    if len(cmd_events) > 0:
        logger.info(f'Including cmd_events {", ".join(cmd_events_ids)}')
    else:
        logger.info('No cmd_events to include')

    for cmd_event in cmd_events:
        cmds = get_cmds_from_event(
            cmd_event['Date'],
            cmd_event['Event'],
            cmd_event['Params'])

        # Events that do not generate commands (e.g. load interrupt) return
        # None from get_cmds_from_event.
        if cmds is not None and len(cmds) > 0:
            cmds_list.append(cmds)
            rltts.append(None)

    # Sort cmds_list and rltts by the date of the first cmd in each cmds Table
    cmds_starts = np.array([cmds['date'][0] for cmds in cmds_list])
    idx_sort = np.argsort(cmds_starts)
    cmds_list = [cmds_list[ii] for ii in idx_sort]
    rltts = [rltts[ii] for ii in idx_sort]

    for ii, cmds, rltt in zip(itertools.count(), cmds_list, rltts):
        if rltt is None:
            continue

        # Apply RLTT from this load to the current running loads (cmds_all).
        # Remove commands with date greater than the RLTT date. In most
        # cases this does not cut anything.
        for jj in range(0, ii):
            prev_cmds = cmds_list[jj]
            if prev_cmds['date'][-1] > rltt:
                # See Boolean masks in
                # https://occweb.cfa.harvard.edu/twiki/bin/view/Aspect/SkaPython#Ska_idioms_and_style
                idx_rltt = np.searchsorted(prev_cmds['date'], rltt, side='right')
                logger.info(f'Removing {len(prev_cmds) - idx_rltt} '
                            f'cmds from {prev_cmds["source"][0]}')
                prev_cmds.remove_rows(slice(idx_rltt, None))

        if len(cmds) > 0:
            logger.info(f'Adding {len(cmds)} commands from {cmds["source"][0]}')

    cmds_recent = vstack(cmds_list)
    cmds_recent.sort_in_backstop_order()

    # Cache recent commands so future requests for the same scenario are fast
    CMDS_RECENT[scenario] = cmds_recent

    return cmds_recent


def update_cmd_events(scenario=None):
    if scenario is not None:
        # Ensure the scenario directory exists
        scenario_dir = paths.SCENARIO_DIR(scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)

    cmd_events_path = paths.CMD_EVENTS_PATH(scenario)
    url = CMD_EVENTS_SHEET_URL
    logger.info(f'Getting cmd_events from {url}')
    req = requests.get(url, timeout=30)
    if req.status_code != 200:
        raise ValueError(f'Failed to get cmd events sheet: {req.status_code}')

    cmd_events = Table.read(req.text, format='csv')
    ok = cmd_events['Valid'] == 'Yes'
    cmd_events = cmd_events[ok]
    del cmd_events['Valid']
    logger.info(f'Writing {len(cmd_events)} cmd_events to {cmd_events_path}')
    cmd_events.write(cmd_events_path, format='csv', overwrite=True)
    return cmd_events


def get_cmd_events(scenario=None):
    cmd_events_path = paths.CMD_EVENTS_PATH(scenario)
    logger.info(f'Reading command events {cmd_events_path}')
    cmd_events = Table.read(cmd_events_path, format='csv', fill_values=[])
    return cmd_events


def get_loads(scenario=None):
    loads_path = paths.LOADS_TABLE_PATH(scenario)
    loads = Table.read(loads_path, format='csv')
    return loads


def update_loads(scenario=None, *, cmd_events=None, lookback=None, stop=None):
    """Update or create loads.csv and loads/ archive though ``lookback`` days

    CSV table file with column names in the first row and data in subsequent rows.

    - load_name: name of load products containing this load segment (e.g. "MAY2217B")
    - cmd_start: time of first command in load segment
    - cmd_stop: time of last command in load segment
    - rltt: running load termination time (terminates previous running loads).
    - schedule_stop_observing: activity end time for loads (propagation goes to this point).
    - schedule_stop_vehicle: activity end time for loads (propagation goes to this point).
    """
    if lookback is None:
        lookback = DEFAULT_LOOKBACK

    if scenario is not None:
        # Ensure the scenario directory exists
        scenario_dir = paths.SCENARIO_DIR(scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)

    if cmd_events is None:
        cmd_events = get_cmd_events(scenario)

    # TODO for performance when we have decent testing:
    # Read in the existing loads table and grab the RLTT and scheduled stop
    # dates. Those are the only reason we need to read an existing set of
    # commands that are already locally available, but they are fixed and cannot
    # change. The only thing that might change is an interrupt time, e.g. if
    # the SCS time gets updated.
    # So maybe read in the loads table and make a dict of rltts and ssts keyed
    # by load name and use those to avoid reading in the cmds table.
    # For now just get things working reliably.

    loads_table_path = paths.LOADS_TABLE_PATH(scenario)
    loads_rows = []

    # Probably too complicated, but this bit of code generates a list of dates
    # that are guaranteed to sample all the months in the lookback period with
    # two weeks of margin on the tail end.
    dt = 21 * u.day
    if stop is None:
        stop = CxoTime.now() + dt
    else:
        stop = CxoTime(stop)
    start = CxoTime(stop) - lookback * u.day
    n_sample = int(np.ceil((stop - start) / dt))
    dates = start + np.arange(n_sample + 1) * (stop - start) / n_sample
    dirs_tried = set()

    # Get the directory listing for each unique Year/Month and find loads.
    # For each load not already in the table, download the backstop and add the
    # load to the table.
    for date in dates:
        year, month = str(date.ymdhms.year), date.ymdhms.month
        month_name = calendar.month_abbr[month].upper()

        dir_year_month = APPROVED_LOADS_OCCWEB_DIR / year / month_name
        if dir_year_month in dirs_tried:
            continue
        dirs_tried.add(dir_year_month)

        # Get directory listing for Year/Month
        contents = occweb.get_occweb_dir(dir_year_month)

        # Find each valid load name in the directory listing and process:
        # - Find and download the backstop file
        # - Parse the backstop file and save to the loads/ archive as a gzipped
        #   pickle file
        # - Add the load to the table
        for content in contents:
            if re.match(r'[A-Z]{3}\d{4}[A-Z]/', content['Name']):
                load_name = content['Name'][:8]  # chop the /
                load_date = load_name_to_cxotime(load_name)
                if load_date < RLTT_ERA_START:
                    logger.warning(f'Skipping {load_name} which is before '
                                   f'{RLTT_ERA_START} start of RLTT era')
                    continue
                if load_date >= start and load_date <= stop:
                    cmds = get_load_cmds_from_occweb_or_local(dir_year_month, load_name)
                    load = get_load_dict_from_cmds(load_name, cmds, cmd_events)
                    loads_rows.append(load)

    # Finally, save the table to file
    loads_table = Table(loads_rows)
    logger.info(f'Saving {len(loads_table)} loads to {loads_table_path}')
    loads_table.sort('cmd_start')
    loads_table.write(loads_table_path, format='csv', overwrite=True)
    loads_table.write(loads_table_path.with_suffix('.dat'), format='ascii.fixed_width',
                      overwrite=True)

    clean_loads_dir(loads_table)

    return loads_table


def clean_loads_dir(loads):
    """Remove load-like files from loads directory if not in ``loads``"""
    for file in Path(paths.LOADS_ARCHIVE_DIR()).glob('*.pkl.gz'):
        if (re.match(r'[A-Z]{3}\d{4}[A-Z]\.pkl\.gz', file.name)
                and file.name[:8] not in loads['name']):
            logger.info(f'Removing load file {file}')
            file.unlink()


def get_load_dict_from_cmds(load_name, cmds, cmd_events):
    """Update ``load`` dict in place from the backstop commands.
    """
    load = {'name': load_name,
            'cmd_start': cmds['date'][0],
            'cmd_stop': cmds['date'][-1],
            'observing_stop': '',
            'vehicle_stop': ''}
    for cmd in cmds:
        if (cmd['type'] == 'LOAD_EVENT'
                and cmd['params']['event_type'] == 'RUNNING_LOAD_TERMINATION_TIME'):
            load['rltt'] = cmd['date']
            break
    else:
        raise ValueError(f'No RLTT found')

    for idx in range(len(cmds), 0, -1):
        cmd = cmds[idx - 1]
        if (cmd['type'] == 'LOAD_EVENT'
                and cmd['params']['event_type'] == 'SCHEDULED_STOP_TIME'):
            load['scheduled_stop_time'] = cmd['date']
            break
    else:
        raise ValueError(f'No scheduled stop time found')

    # CHANGE THIS to use LOAD_EVENT entries in commands. Or NOT?? Probably
    # provides good visibility into what's going on. But this hard-coding is
    # annoying.
    for cmd_event in cmd_events:
        cmd_event_date = cmd_event['Date']

        if (cmd_event_date >= load['cmd_start']
                and cmd_event_date <= load['cmd_stop']
                and cmd_event['Event'] in ('SCS-107', 'NSM')):
            logger.info(f'{cmd_event["Event"]} at {cmd_event_date} found for {load_name}')
            load['observing_stop'] = cmd_event['Date']
            if cmd_event['Event'] == 'NSM':
                load['vehicle_stop'] = cmd_event['Date']

        if cmd_event['Event'] == 'Load not run' and cmd_event['Params'] == load_name:
            logger.info(f'{cmd_event["Event"]} at {cmd_event_date} found for {load_name}')
            load['observing_stop'] = '1998:001'
            load['vehicle_stop'] = '1998:001'

        if cmd_event['Event'] == 'Observing not run' and cmd_event['Params'] == load_name:
            logger.info(f'{cmd_event["Event"]} at {cmd_event_date} found for {load_name}')
            load['observing_stop'] = '1998:001'

    return load


def get_load_cmds_from_occweb_or_local(dir_year_month, load_name):
    """Get the load cmds (backstop) for ``load_name`` within ``dir_year_month``

    If the backstop file is already available locally, use that. Otherwise, the
    file is downloaded from OCCweb and is then parsed and saved as a gzipped
    pickle file of the corresponding CommandTable object.

    :param dir_year_month: Path
        Path to the directory containing the ``load_name`` directory.
    :param load_name: str
        Load name in the usual format e.g. JAN0521A.
    :returns: CommandTable
        Backstop commands for the load.
    """
    # Determine output file name and make directory if necessary.
    loads_dir = paths.LOADS_ARCHIVE_DIR()
    loads_dir.mkdir(parents=True, exist_ok=True)
    cmds_filename = loads_dir / f'{load_name}.pkl.gz'

    # If the output file already exists, read the commands and return them.
    if cmds_filename.exists():
        logger.info(f'Already have {cmds_filename}')
        with gzip.open(cmds_filename, 'rb') as fh:
            cmds = pickle.load(fh)
        return cmds

    load_dir_contents = occweb.get_occweb_dir(dir_year_month / load_name)
    for filename in load_dir_contents['Name']:
        if re.match(r'CR\d{3}_\d{4}\.backstop', filename):

            # Download the backstop file from OCCweb
            logger.info(f'Getting {dir_year_month / load_name / filename}')
            backstop_text = occweb.get_occweb_page(dir_year_month / load_name / filename,
                                                   cache=CACHE_LOADS_IN_ASTROPY_CACHE)
            backstop_lines = backstop_text.splitlines()
            cmds = get_cmds_from_backstop(backstop_lines, remove_starcat=True)

            # Fix up the commands to be in the right format
            idx = cmds.colnames.index('timeline_id')
            cmds.add_column(load_name, index=idx, name='source')
            del cmds['timeline_id']
            del cmds['time']

            logger.info(f'Saving {cmds_filename}')
            with gzip.open(cmds_filename, 'wb') as fh:
                pickle.dump(cmds, fh)
            return cmds
    else:
        raise ValueError(f'Could not find backstop file in {dir_year_month / load_name}')


def update_cmds_archive(*, lookback=None, stop=None, log_level=logging.INFO,
                        data_root='.', v1_v2_transition=False):
    """Update cmds2.h5 and cmds2.pkl archive files.

    This updates the archive though ``stop`` date, where is required that the
    ``stop`` date is within ``lookback`` days of existing data in the archive.

    :param lookback: int, None
        Number of days to look back to get recent load commands from OCCweb.
        Default is ``DEFAULT_LOOKBACK`` (currently 30).
    :param stop: CxoTime-like, None
        Stop date to update the archive to. Default is NOW + 21 days.
    :param log_level: int
        Logging level. Default is ``logging.INFO``.
    :param data_root: str, Path
        Root directory where cmds2.h5 and cmds2.pkl are stored. Default is '.'.
    :param v1_v2_transition: bool
        One-time use flag set to True to update the cmds archive near the v1/v2
        transition of APR1420B. See ``utils/migrate_cmds_to_cmds2.py`` for
        details.
    """
    # Local context manager for log_level and data_root
    kadi_orig = os.environ.get('KADI')
    kadi_logger = logging.getLogger('kadi')
    log_level_orig = kadi_logger.level
    try:
        os.environ['KADI'] = data_root
        kadi_logger.setLevel(log_level)
        _update_cmds_archive(lookback=lookback, stop=stop,
                             v1_v2_transition=v1_v2_transition)
    finally:
        if kadi_orig is not None:
            os.environ['KADI'] = kadi_orig
        kadi_logger.setLevel(log_level_orig)


def _update_cmds_archive(*, lookback=None, stop=None, v1_v2_transition=False):
    idx_cmds_path = paths.IDX_CMDS_PATH(version=2)
    pars_dict_path = paths.PARS_DICT_PATH(version=2)

    cmds_arch = load_idx_cmds(version=2)
    pars_dict = load_pars_dict(version=2)

    cmds_recent = update_archive_and_get_cmds_recent(stop=stop, lookback=lookback)
    if v1_v2_transition:
        idx0_arch = len(cmds_arch)
        idx0_recent = 0
    else:
        idx0_arch, idx0_recent = get_matching_block_idx(cmds_arch, cmds_recent)

    # Convert from `params` col of dicts to index into same params in pars_dict.
    for cmd in cmds_recent:
        cmd['idx'] = get_par_idx_update_pars_dict(pars_dict, cmd)
    del cmds_recent['params']

    # If the length of the updated table will be the same as the existing table,
    # it might be that there is no update so we can skip writing the file. But
    # we need to check that the new data values are exactly the same.
    if False and len(cmds_arch) == len(cmds_recent) + idx0_arch:
        for name in cmds_arch.colnames:
            if np.any(cmds_arch[name][idx0_arch:] != cmds_recent[name]):
                break
        else:
            logger.info(f'No new commands found, skipping writing {idx_cmds_path}')
            return

    # Merge the recent commands with the existing archive.
    logger.info(f'Merging {len(cmds_recent)} new commands with existing archive')
    logger.info(f' starting with cmds_arch[:{idx0_arch}] and adding '
                f'cmds_recent[{idx0_recent}:{len(cmds_recent)}]')

    # Save the updated archive and pars_dict.
    cmds_arch = vstack([cmds_arch[:idx0_arch], cmds_recent[idx0_recent:]],
                       join_type='exact')
    logger.info(f'Writing {len(cmds_arch)} commands to {idx_cmds_path}')
    cmds_arch.write(str(idx_cmds_path), path='data', format='hdf5', overwrite=True)

    logger.info(f'Writing updated pars_dict to {pars_dict_path}')
    pickle.dump(pars_dict, open(pars_dict_path, 'wb'))


import difflib


def get_matching_block_idx(cmds_arch, cmds_recent):
    # Find place in archive where the recent commands start.
    idx_arch_recent = np.searchsorted(cmds_arch['date'], cmds_recent['date'][0])
    logger.info('Selecting commands from cmds_arch[{}:]'.format(idx_arch_recent))
    # logger.info('  {}'.format(str(cmds_arch[idx_recent])))
    cmds_arch_recent = cmds_arch[idx_arch_recent:]

    cmds_arch_recent[:10].pprint_like_backstop(
        logger.info, 'Start of archive commands from beginning of recent commands')
    cmds_recent[:10].pprint_like_backstop(
        logger.info, 'Start of recent commands')

    # Define the column names that specify a complete and unique row
    key_names = ('date', 'type', 'tlmsid', 'scs', 'step', 'source', 'vcdu')

    cmds_arch_recent[key_names].write('cmds_arch_recent.dat', format='ascii.fixed_width', overwrite=True)
    cmds_recent[key_names].write('cmds_recent.dat', format='ascii.fixed_width', overwrite=True)

    recent_vals = [tuple(
        row[x].decode('ascii') if isinstance(row[x], bytes) else str(row[x])
        for x in key_names)
        for row in cmds_arch_recent]
    arch_vals = [tuple(
        row[x].decode('ascii') if isinstance(row[x], bytes) else str(row[x])
        for x in key_names)
        for row in cmds_recent]

    diff = difflib.SequenceMatcher(a=arch_vals, b=recent_vals, autojunk=False)

    matching_blocks = diff.get_matching_blocks()
    logger.info('Matching blocks for (a) existing HDF5 and (b) recent commands')
    for block in matching_blocks:
        logger.info('  {}'.format(block))
    opcodes = diff.get_opcodes()
    logger.info('Diffs between (a) existing HDF5 and (b) recent commands')
    for opcode in opcodes:
        logger.info('  {}'.format(opcode))
    # Find the first matching block that is sufficiently long
    for block in matching_blocks:
        if block.size > MATCHING_BLOCK_SIZE:
            break
    else:
        raise ValueError('No matching blocks at least {} long'
                         .format(MATCHING_BLOCK_SIZE))

    # Index into idx_cmds at the end of the large matching block.  block.b is the
    # beginning of the match.
    idx_cmds_idx = block.b + block.size
    h5d_idx = idx_arch_recent + block.a + block.size

    return h5d_idx, idx_cmds_idx

def get_opt(args=None):
    """
    Get options for command line interface to update_
    """
    from kadi import __version__
    import argparse
    parser = argparse.ArgumentParser(description='Update HDF5 cmds v2 table')
    parser.add_argument("--lookback",
                        help="Lookback (default=30 days)")
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
