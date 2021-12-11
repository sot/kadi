# Licensed under a 3-clause BSD style license - see LICENSE.rst
import difflib
import os
from pathlib import Path
import calendar
import platform
import re
import gzip
import pickle
import itertools
import functools
import weakref
import logging

import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
import requests

from kadi.commands import get_cmds_from_backstop, conf
from kadi.commands.core import (load_idx_cmds, load_pars_dict, LazyVal,
                                get_par_idx_update_pars_dict, _find,
                                ska_load_dir, CommandTable)
from kadi.commands.command_sets import get_cmds_from_event
from kadi import occweb, paths
from cxotime import CxoTime

# TODO configuration options, but use DEFAULT_* in the mean time
# - commands_version (v1, v2)

MATCHING_BLOCK_SIZE = 100

# TODO: cache translation from cmd_events to CommandTable's  [Probably not]

APPROVED_LOADS_OCCWEB_DIR = Path('FOT/mission_planning/PRODUCTS/APPR_LOADS')

# https://docs.google.com/spreadsheets/d/<document_id>/export?format=csv&gid=<sheet_id>
CMD_EVENTS_SHEET_ID = '19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ'
CMD_EVENTS_SHEET_URL = f'https://docs.google.com/spreadsheets/d/{CMD_EVENTS_SHEET_ID}/export?format=csv'  # noqa

# Cached values of the full mission commands archive (cmds_v2.h5, cmds_v2.pkl).
# These are loaded on demand.
IDX_CMDS = LazyVal(functools.partial(load_idx_cmds, version=2))
PARS_DICT = LazyVal(functools.partial(load_pars_dict, version=2))
REV_PARS_DICT = LazyVal(lambda: {v: k for k, v in PARS_DICT.items()})

# Cache of recent commands keyed by scenario
CMDS_RECENT = {}
MATCHING_BLOCKS = {}

# APR1420B was the first load set to have RLTT (backstop 6.9)
RLTT_ERA_START = CxoTime('2020-04-14')

logger = logging.getLogger(__name__)

# DEBUG: remove this for production
logging.getLogger('kadi').setLevel(1)


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


def _merge_cmds_archive_recent(start, scenario):
    """Merge cmds archive from ``start`` onward with recent cmds for ``scenario``

    This assumes:
    - CMDS_RECENT cache has been set with that scenario.
    - Recent commands overlap the cmds archive

    :parameter start: CxoTime-like,
        Start time for returned commands
    :parameter scenario: str
        Scenario name
    :returns: CommandTable
        Commands from cmds archive and all recent commands
    """
    cmds_recent = CMDS_RECENT[scenario]

    logger.info('Merging cmds_recent with archive commands from {start}')

    if scenario not in MATCHING_BLOCKS:
        # Get index for start of cmds_recent within the cmds archive
        i0_arch_recent = IDX_CMDS.find_date(cmds_recent['date'][0])

        # Find the end of the first large (MATCHING_BLOCK_SIZE) block of
        # cmds_recent that overlap with archive cmds. Look for the matching
        # block in a subset of archive cmds that starts at the start of
        # cmds_recent. `arch_recent_offset` is the offset from `i0_arch_recent`
        # to the end of the matching block. `i0_recent` is the end of the
        # matching block in recent commands.
        arch_recent_offset, recent_block_end = get_matching_block_idx(
            IDX_CMDS[i0_arch_recent:], cmds_recent)
        arch_block_end = i0_arch_recent + arch_recent_offset
        MATCHING_BLOCKS[scenario] = arch_block_end, recent_block_end, i0_arch_recent
    else:
        arch_block_end, recent_block_end, i0_arch_recent = MATCHING_BLOCKS[scenario]

    # Get archive commands from the requested start time (or start of the overlap
    # with recent commands) to the end of the matching block in recent commands.
    i0_arch_start = min(IDX_CMDS.find_date(start), i0_arch_recent)
    cmds_arch = IDX_CMDS[i0_arch_start:arch_block_end]

    # Stored archive commands HDF5 has no `params` column, instead storing an
    # index to the param values which are in PARS_DICT. Add `params` object
    # column with None values and then stack with cmds_recent (which has
    # `params` already as dicts).
    cmds = vstack([cmds_arch, cmds_recent[recent_block_end:]],
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
    i0_arch = cmds_arch.find_date(date0)
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


def get_cmds(start=None, stop=None, inclusive_stop=False, scenario=None, **kwargs):
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
    :param **kwargs: dict
        key=val keyword argument pairs for filtering

    :returns: CommandTable
    """
    if scenario not in CMDS_RECENT:
        cmds_recent = update_archive_and_get_cmds_recent(scenario, cache=True)
    else:
        cmds_recent = CMDS_RECENT[scenario]

    start = CxoTime(start or '1999:001')
    stop_date = CxoTime(stop) if stop else '2099:001'

    if stop_date < cmds_recent['date'][0]:
        # Query does not overlap with recent commands, just use archive.
        logger.info('Getting commands from archive only')
        cmds = IDX_CMDS
    elif start < CxoTime(cmds_recent['date'][0]) + 1 * u.hr:
        # Query starts near beginning of recent commands and *might* need some
        # archive commands. The margin is set at 1 hour, but in reality it is
        # probably just the 3 minutes of typical overlaps between loads.
        cmds = _merge_cmds_archive_recent(start, scenario)
        logger.info('Getting commands from archive + recent')
    else:
        # Query is strictly within recent commands.
        cmds = cmds_recent
        logger.info('Getting commands from recent only')

    # Select the requested time range and make a copy. (Slicing is a view so
    # in theory bad things could happen without a copy).
    idx0 = cmds.find_date(start)
    idx1 = cmds.find_date(stop_date, side=('right' if inclusive_stop else 'left'))
    cmds = cmds[idx0:idx1].copy()

    if kwargs:
        # Specified extra filters on cmds search
        pars_dict = PARS_DICT.copy()
        # For any recent commands that have params as a dict, those will have
        # idx = -1. This doesn't work with _find, which is optimized to search
        # pars_dict for the matching search keys.
        # TODO: this step is only really required for kwargs that are not a column,
        # i.e. keys that are found only in params.
        for ii in np.flatnonzero(cmds['idx'] == -1):
            cmds[ii]['idx'] = get_par_idx_update_pars_dict(pars_dict, cmds[ii])
        cmds = _find(idx_cmds=cmds, pars_dict=pars_dict, **kwargs)

    cmds.rev_pars_dict = weakref.ref(REV_PARS_DICT)

    cmds.add_column(CxoTime(cmds['date'], format='date').secs, name='time', index=6)
    cmds['time'].info.format = '.3f'

    return cmds


def update_archive_and_get_cmds_recent(scenario=None, *, lookback=None, stop=None,
                                       cache=True):
    """Update local loads table and downloaded loads and return all recent cmds.

    This also caches the recent commands in the global CMDS_RECENT dict.

    This relies entirely on RLTT and load_events to assemble the commands.

    :param scenario: str, None
        Scenario name
    :param lookback: int, Quantity, None
        Lookback time from ``stop`` for recent loads. If None, use
        conf.default_lookback.
    :param stop: CxoTime-like, None
        Stop time for loads table (default is now + 21 days)
    :param cache: bool
        Cache the result in CMDS_RECENT dict.

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
        if rltt is not None:
            # Apply RLTT from this load to the current running loads (cmds_all).
            # Remove commands with date greater than the RLTT date. In most
            # cases this does not cut anything.
            for jj in range(0, ii):
                prev_cmds = cmds_list[jj]
                if prev_cmds['date'][-1] > rltt:
                    # See Boolean masks in
                    # https://occweb.cfa.harvard.edu/twiki/bin/view/Aspect/SkaPython#Ska_idioms_and_style
                    idx_rltt = prev_cmds.find_date(rltt, side='right')
                    logger.info(f'Removing {len(prev_cmds) - idx_rltt} '
                                f'cmds from {prev_cmds["source"][0]}')
                    prev_cmds.remove_rows(slice(idx_rltt, None))

        if len(cmds) > 0:
            logger.info(f'Adding {len(cmds)} commands from {cmds["source"][0]}')

    cmds_recent = vstack(cmds_list)
    cmds_recent.sort_in_backstop_order()

    if cache:
        # Cache recent commands so future requests for the same scenario are fast
        CMDS_RECENT[scenario] = cmds_recent

    return cmds_recent


def update_from_network_enabled(scenario):
    """Return True if updating from the network is enabled.

    Logic:
    - If scenario == 'flight' return False unless user is 'aca' and host is
      on HEAD network. This allows production updates of the flight scenario
      by cron job on HEAD, but otherwise do not try touching the archive files.
    - Else return conf.update_from_network.

    :param scenario: str, None
        Scenario name
    """
    if scenario == 'flight':
        # TODO: put these into configuration.
        if (os.getlogin() != 'aca'
                or not platform.node().endswith('.cfa.harvard.edu')):
            if conf.update_from_network:
                # Updating from network is enabled but this is turning it off.
                logger.info('Not updating flight scenario from network')
            return False
    else:
        return conf.update_from_network


def update_cmd_events(scenario=None):
    # If no network access allowed then just return the local file
    if not update_from_network_enabled(scenario):
        return get_cmd_events(scenario)

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
    cmd_events = Table.read(str(cmd_events_path), format='csv', fill_values=[])
    return cmd_events


def get_loads(scenario=None):
    loads_path = paths.LOADS_TABLE_PATH(scenario)
    logger.info(f'Reading loads file {loads_path}')
    loads = Table.read(str(loads_path), format='csv')
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
    # If no network access allowed then just return the local file
    if not update_from_network_enabled(scenario):
        return get_loads(scenario)

    if lookback is None:
        lookback = conf.default_lookback

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
    start = CxoTime(stop) - lookback * u.day
    if stop is None:
        stop = CxoTime.now() + dt
    else:
        stop = CxoTime(stop)
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
        try:
            contents = occweb.get_occweb_dir(dir_year_month)
        except requests.exceptions.HTTPError:
            continue

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

    if conf.clean_loads_dir:
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


def get_load_cmds_from_occweb_or_local(dir_year_month=None, load_name=None, use_ska_dir=False):
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

    if use_ska_dir:
        ska_dir = ska_load_dir(load_name)
        for filename in ska_dir.glob('CR????????.backstop'):
            backstop_text = filename.read_text()
            logger.info(f'Got backstop from {filename}')
            cmds = get_cmds_from_backstop(backstop_text.splitlines())
            logger.info(f'Saving {cmds_filename}')
            with gzip.open(cmds_filename, 'wb') as fh:
                pickle.dump(cmds, fh)
            return cmds
        else:
            raise ValueError(f'No backstop file found in {ska_dir}')

    load_dir_contents = occweb.get_occweb_dir(dir_year_month / load_name)
    for filename in load_dir_contents['Name']:
        if re.match(r'CR\d{3}.\d{4}\.backstop', filename):

            # Download the backstop file from OCCweb
            logger.info(f'Getting {dir_year_month / load_name / filename}')
            backstop_text = occweb.get_occweb_page(dir_year_month / load_name / filename,
                                                   cache=conf.cache_loads_in_astropy_cache)
            backstop_lines = backstop_text.splitlines()
            cmds = get_cmds_from_backstop(backstop_lines)

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
                        scenario=None, data_root='.', match_prev_cmds=True):
    """Update cmds2.h5 and cmds2.pkl archive files.

    This updates the archive though ``stop`` date, where is required that the
    ``stop`` date is within ``lookback`` days of existing data in the archive.

    :param lookback: int, None
        Number of days to look back to get recent load commands from OCCweb.
        Default is ``conf.default_lookback`` (currently 30).
    :param stop: CxoTime-like, None
        Stop date to update the archive to. Default is NOW + 21 days.
    :param log_level: int
        Logging level. Default is ``logging.INFO``.
    :param scenario: str, None
        Scenario name for loads and command events
    :param data_root: str, Path
        Root directory where cmds2.h5 and cmds2.pkl are stored. Default is '.'.
    :param match_prev_cmds: bool
        One-time use flag set to True to update the cmds archive near the v1/v2
        transition of APR1420B. See ``utils/migrate_cmds_to_cmds2.py`` for
        details.
    """
    # Local context manager for log_level and data_root
    kadi_logger = logging.getLogger('kadi')
    log_level_orig = kadi_logger.level
    with conf.set_temp('commands_dir', data_root):
        try:
            kadi_logger.setLevel(log_level)
            _update_cmds_archive(lookback, stop, match_prev_cmds, scenario, data_root)
        finally:
            kadi_logger.setLevel(log_level_orig)


def _update_cmds_archive(lookback, stop, match_prev_cmds, scenario, data_root):
    """Do the real work of updating the cmds archive"""
    idx_cmds_path = Path(data_root) / 'cmds2.h5'
    pars_dict_path = Path(data_root) / 'cmds2.pkl'

    if idx_cmds_path.exists():
        cmds_arch = load_idx_cmds(version=2, file=idx_cmds_path)
        pars_dict = load_pars_dict(version=2, file=pars_dict_path)
    else:
        # Make an empty cmds archive table and pars dict
        cmds_arch = CommandTable(names=list(CommandTable.COL_TYPES),
                                 dtype=list(CommandTable.COL_TYPES.values()))
        del cmds_arch['timeline_id']
        pars_dict = {}
        match_prev_cmds = False  # No matching of previous commands

    cmds_recent = update_archive_and_get_cmds_recent(
        scenario=scenario, stop=stop, lookback=lookback, cache=False)

    if match_prev_cmds:
        idx0_arch, idx0_recent = get_matching_block_idx(cmds_arch, cmds_recent)
    else:
        idx0_arch = len(cmds_arch)
        idx0_recent = 0

    # Convert from `params` col of dicts to index into same params in pars_dict.
    for cmd in cmds_recent:
        cmd['idx'] = get_par_idx_update_pars_dict(pars_dict, cmd)

    # If the length of the updated table will be the same as the existing table.
    # For the command below the no-op logic should be clear:
    # cmds_arch = vstack([cmds_arch[:idx0_arch], cmds_recent[idx0_recent:]])
    if idx0_arch == len(cmds_arch) and idx0_recent == len(cmds_recent):
        logger.info(f'No new commands found, skipping writing {idx_cmds_path}')
        return

    # Merge the recent commands with the existing archive.
    logger.info(f'Appending {len(cmds_recent) - idx0_recent} new commands after '
                f'removing {len(cmds_arch) - idx0_arch} from existing archive')
    logger.info(f' starting with cmds_arch[:{idx0_arch}] and adding '
                f'cmds_recent[{idx0_recent}:{len(cmds_recent)}]')

    # Remove params column before stacking and saving
    del cmds_recent['params']
    del cmds_arch['params']

    # Save the updated archive and pars_dict.
    cmds_arch_new = vstack([cmds_arch[:idx0_arch], cmds_recent[idx0_recent:]],
                           join_type='exact')
    logger.info(f'Writing {len(cmds_arch_new)} commands to {idx_cmds_path}')
    cmds_arch_new.write(str(idx_cmds_path), path='data', format='hdf5', overwrite=True)

    logger.info(f'Writing updated pars_dict to {pars_dict_path}')
    pickle.dump(pars_dict, open(pars_dict_path, 'wb'))


def get_matching_block_idx(cmds_arch, cmds_recent):
    # Find place in archive where the recent commands start.
    idx_arch_recent = cmds_arch.find_date(cmds_recent['date'][0])
    logger.info('Selecting commands from cmds_arch[{}:]'.format(idx_arch_recent))
    cmds_arch_recent = cmds_arch[idx_arch_recent:]

    # Define the column names that specify a complete and unique row
    key_names = ('date', 'type', 'tlmsid', 'scs', 'step', 'source', 'vcdu')

    recent_vals = [tuple(
        row[x].decode('ascii') if isinstance(row[x], bytes) else str(row[x])
        for x in key_names)
        for row in cmds_arch_recent]
    arch_vals = [tuple(
        row[x].decode('ascii') if isinstance(row[x], bytes) else str(row[x])
        for x in key_names)
        for row in cmds_recent]

    diff = difflib.SequenceMatcher(a=recent_vals, b=arch_vals, autojunk=False)

    matching_blocks = diff.get_matching_blocks()
    logger.info('Matching blocks for (a) recent commands and (b) existing HDF5')
    for block in matching_blocks:
        logger.info('  {}'.format(block))
    opcodes = diff.get_opcodes()
    logger.info('Diffs between (a) recent commands and (b) existing HDF5')
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
    idx0_recent = block.b + block.size
    idx0_arch = idx_arch_recent + block.a + block.size

    return idx0_arch, idx0_recent
