# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import calendar
import re
import gzip
import pickle
import itertools
import functools
import operator

import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
import requests

from kadi.commands import get_cmds_from_backstop
from kadi.commands.core import load_idx_cmds, load_pars_dict, LazyVal
from kadi.command_sets import get_cmds_from_event
from kadi import occweb
from kadi import paths
from cxotime import CxoTime
import pyyaks.logger


# TODO configuration options:
# - cache loads (backstop) downloads from OCCweb (useful for development but
#   likely to just take disk space for users)
# - commands_dir ?
# - commands_version (v1, v2)
# - default_lookback_time
# - update_from_network (default True)
# - remove_old_loads (default True) Remove older loads from local directory.

APPROVED_LOADS_OCCWEB_DIR = 'FOT/mission_planning/PRODUCTS/APPR_LOADS'

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

# TODO: make it easier to set the log level (e.g. add a set_level method() to
# logger object that sets all handlers to that level)
logger = pyyaks.logger.get_logger(name=__name__)


# Cached values of the full mission commands archive (cmds_v2.h5, cmds_v2.pkl).
# These are loaded on demand.
IDX_CMDS = LazyVal(functools.partial(load_idx_cmds, version=2))
PARS_DICT = LazyVal(functools.partial(load_pars_dict, version=2))
REV_PARS_DICT = LazyVal(lambda: {v: k for k, v in PARS_DICT.items()})

# Cache of recent commands keyed by scenario
CMDS_RECENT = {}


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
    if load['observing_stop'] is not np.ma.masked:
        bad |= ((cmds['date'] > load['observing_stop'])
                & (cmds['scs'] > 130))
    if load['vehicle_stop'] is not np.ma.masked:
        bad |= ((cmds['date'] > load['vehicle_stop'])
                & (cmds['scs'] < 131))
    if np.any(bad):
        logger.info(f'Cutting {bad.sum()} commands from {load["name"]}')
        cmds = cmds[~bad]
    return cmds


def merge_cmds_archive_recent(cmds_recent, start):
    idx0 = np.searchsorted(IDX_CMDS['date'], start.date)
    cmds_arch = IDX_CMDS[idx0:]
    key_names = ('date', 'type', 'tlmsid', 'scs', 'step', 'vcdu')
    idx1 = np.searchsorted(cmds_arch['date'], cmds_recent['date'][0])
    while True:
        for ii in range(5):
            if not np.all(cmds_arch[name][idx1 + ii] == cmds_recent[name][ii]
                          for name in key_names):
                break  # row didn't match, break from "for ii" loop, try next row
        else:
            break  # all 5 rows matched, break from "while True"

        idx1 += 1
        if cmds_arch['date'][idx1] != cmds_recent['date'][0]:
            raise ValueError(f'No matching commands block in archive found for recent_commands:\n'
                             f'{cmds_recent[:5]}')

    cmds = vstack([cmds_arch[:idx1], cmds_recent])
    return cmds


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
    :returns: CommandTable
    """
    if scenario not in CMDS_RECENT:
        cmds_recent = get_cmds_recent(scenario)
        CMDS_RECENT[scenario] = cmds_recent
    else:
        cmds_recent = CMDS_RECENT[scenario]

    start = CxoTime(start or '1999:001')
    stop = CxoTime(stop or '2099:001')

    if start < CxoTime(cmds_recent['date'][0]) + 1 * u.day:
        cmds = merge_cmds_archive_recent(cmds_recent, start)
    else:
        cmds = cmds_recent

    idx0 = np.searchsorted(cmds['date'], start.date)
    idx1 = np.searchsorted(cmds['date'], stop.date,
                           side=('right' if inclusive_stop else 'left'))
    cmds = cmds[idx0:idx1]

    return cmds


def get_cmds_recent(scenario=None):
    """Get commands using loads table, relying entirely on RLTT.

    :param scenario: str, None
        Scenario name

    :returns: CommandTable
    """
    cmds_list = []  # List of CommandTable objects from loads and cmd_events
    rltts = []  # Corresponding list of RLTTs, where cmd_events use None for RLTT

    # Update loads from OCCweb
    cmd_events = update_cmd_events(scenario)
    loads = update_loads(scenario, cmd_events=cmd_events)

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

    # Filter events outside the time interval, assuming command event cannot
    # last more than 2 weeks.
    start = CxoTime(min(loads['cmd_start']))
    stop = CxoTime(max(loads['cmd_stop']))
    bad = ((cmd_events['Date'] < (start - 14 * u.day).date)
           | (cmd_events['Date'] > stop.date))
    cmd_events = cmd_events[~bad]
    cmd_events_ids = [evt['Event'] + '-' + evt['Date'][:8] for evt in cmd_events]
    logger.info(f'Including cmd_events {", ".join(cmd_events_ids)}')

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

    out = vstack(cmds_list)
    out.sort(['date', 'step', 'scs'])

    return out


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


def update_loads(scenario=None, *, cmd_events=None, lookback=31, stop=None):
    """Update or create loads.csv and loads/ archive though ``lookback`` days

    CSV table file with column names in the first row and data in subsequent rows.

    - load_name: name of load products containing this load segment (e.g. "MAY2217B")
    - cmd_start: time of first command in load segment
    - cmd_stop: time of last command in load segment
    - rltt: running load termination time (terminates previous running loads).
    - schedule_stop_observing: activity end time for loads (propagation goes to this point).
    - schedule_stop_vehicle: activity end time for loads (propagation goes to this point).
    """
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
    dt = 14 * u.day
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

        dir_year_month = Path(APPROVED_LOADS_OCCWEB_DIR) / year / month_name
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
    # provides good visibility into what's going on.
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
                                                   cache=True)
            backstop_lines = backstop_text.splitlines()
            cmds = get_cmds_from_backstop(backstop_lines, remove_starcat=True)

            # Fix up the commands to be in the right format
            idx = cmds.colnames.index('timeline_id')
            cmds.add_column(load_name, index=idx, name='source')
            del cmds['timeline_id']

            logger.info(f'Saving {cmds_filename}')
            with gzip.open(cmds_filename, 'wb') as fh:
                pickle.dump(cmds, fh)
            return cmds
    else:
        raise ValueError(f'Could not find backstop file in {dir_year_month / load_name}')
