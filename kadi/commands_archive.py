# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path
import os
import calendar
import re
import gzip
import pickle
import itertools

import numpy as np
from astropy.table import Table, vstack
import astropy.units as u
import requests

from kadi.commands import get_cmds_from_backstop
from kadi.command_sets import get_cmds_from_event
from kadi import occweb
from kadi import paths
from cxotime import CxoTime
import Ska.DBI
import pyyaks.logger


SKA = Path(os.environ['SKA'])
CMD_STATES_PATH = SKA / 'data' / 'cmd_states' / 'cmd_states.db3'


APPROVED_LOADS_OCCWEB_DIR = 'FOT/mission_planning/PRODUCTS/APPR_LOADS'

# https://docs.google.com/spreadsheets/d/<document_id>/export?format=csv&gid=<sheet_id>
CMD_EVENTS_SHEET_ID = '19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ'
CMD_EVENTS_SHEET_URL = f'https://docs.google.com/spreadsheets/d/{CMD_EVENTS_SHEET_ID}/export?format=csv'  # noqa

RLTT_ERA_START = 'APR1420A'

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
logger = pyyaks.logger.get_logger('kadi')


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


# CSV table file with column names in the first row and data in subsequent rows.
#
# - load_name: name of load products containing this load segment (e.g. "MAY2217B")
# - cmd_start: time of first command in load segment
# - cmd_stop: time of last command in load segment
# - rltt: running load termination time (terminates previous running loads).
# - schedule_stop_observing: activity end time for loads (propagation goes to this point).
# - schedule_stop_vehicle: activity end time for loads (propagation goes to this point).
# - ~~approval_date: load approval date proxy using time-stamp of backstop file.~~
#


def get_loads_from_timelines(min_date=None):
    """Migrate the timelines and load_segments into a list of load dicts"""
    with Ska.DBI.DBI(dbi='sqlite', server=str(CMD_STATES_PATH)) as db:
        timelines = db.fetchall("""SELECT * from timelines""")
        load_segs = db.fetchall("""SELECT * from load_segments""")
    load_segs = Table(load_segs)
    load_segs.add_index('id')
    timelines = Table(timelines)

    # First assemble loads by load name
    loads = {}
    for timeline in timelines:
        if min_date is not None and timeline['datestart'] < min_date:
            continue

        # dir looks like /2002/JAN0702/oflsd/
        name = timeline['dir'][6:13] + timeline['dir'][-2].upper()
        loads.setdefault(name, {'timeline_start': '2099:001:00:00:00.000',
                                'observing_stop': '1999:001:00:00:00.000',
                                'vehicle_stop': '1999:001:00:00:000'})
        load = loads[name]

        load_seg = load_segs.loc[timeline['load_segment_id']]
        is_observing = load_seg['load_scs'] > 130

        # Set timeline_start as the min of the timeline start times and likewise
        # observing/vehicle stop as the max of the timeline stop times.
        if timeline['datestart'] < load['timeline_start']:
            load['timeline_start'] = timeline['datestart']
        if is_observing and timeline['datestop'] > load['observing_stop']:
            load['observing_stop'] = timeline['datestop']
        elif not is_observing and timeline['datestop'] > load['vehicle_stop']:
            load['vehicle_stop'] = timeline['datestop']

    # Turn the dict of loads into a list of loads sorted by start time
    out_loads = []
    for name in sorted(loads, key=lambda x: loads[x]['timeline_start']):
        load = loads[name]
        load['name'] = name
        del load['timeline_start']  # Not relevant (cmd_start is what matters)
        out_loads.append(loads[name])

    if min_date is not None:
        # First load may be incomplete when filtering on min_date, so remove it
        out_loads = out_loads[1:]

    return out_loads


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


def get_cmds(start=None, stop=None, cmds_dir=None, scenario=None,
             loads=None, cmd_events=None):
    """Get commands using loads table, relying entirely on RLTT.

    :param start: CxoTime-like
        Start time for cmds
    :param stop: CxoTime-like
        Stop time for cmds
    :param cmds_dir: str, Path, None
        Commands directory
    :param scenario: str, None
        Scenario name
    :param loads: Table, None
        Loads table (read from file if None)
    :param cmd_events: Table, None
        Command events table (read from file if None)
    :returns: CommandTable
    """
    cmds_list = []  # List of CommandTable objects from loads and cmd_events
    rltts = []  # Corresponding list of RLTTs, where cmd_events use None for RLTT

    # Start and stop filters, using lexical string comparisons
    start = CxoTime('1999:001') if start is None else CxoTime(start)
    stop = CxoTime('2099:001') if stop is None else CxoTime(stop)

    # First get command tables from each applicable load set
    if loads is None:
        loads = get_loads(scenario=scenario)

    bad = (loads['cmd_stop'] < start.date) | (loads['cmd_start'] > stop.date)
    loads = loads[~bad]
    logger.info(f'Including loads {", ".join(loads["name"])}')

    for load in loads:
        loads_backstop_path = paths.LOADS_BACKSTOP_PATH(cmds_dir, load['name'])
        with gzip.open(loads_backstop_path, 'rb') as fh:
            cmds = pickle.load(fh)

        # Apply load interrupts (SCS-107, NSM) from the loads table to this
        # command load. This assumes that loads.csv has been updated
        # appropriately from load_events.csv (which might have come from the
        # Load Events sheet).
        cmds = interrupt_load_commands(load, cmds)
        if len(cmds) > 0:
            logger.info(f'Load {load["name"]} has {len(cmds)} commands')
            cmds_list.append(cmds)
            rltts.append(load['rltt'])

    # Second get command tables from each event in cmd_events
    if cmd_events is None:
        cmd_events = get_cmd_events(cmds_dir, scenario)

    # Filter events outside the time interval, assuming command event cannot
    # last more than 2 weeks.
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
    ok = (out['date'] >= start.date) & (out['date'] < stop.date)
    out = out[ok]
    out.sort(['date', 'step', 'scs'])

    return out


def update_cmd_events(*, cmds_dir=None, scenario=None):
    if scenario is not None:
        # Ensure the scenario directory exists
        scenario_dir = paths.SCENARIO_DIR(scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)

    cmd_events_path = paths.CMD_EVENTS_PATH(cmds_dir, scenario)
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


def get_cmd_events(cmds_dir=None, scenario=None):
    cmd_events_path = paths.CMD_EVENTS_PATH(cmds_dir, scenario)
    logger.info(f'Reading command events {cmd_events_path}')
    cmd_events = Table.read(cmd_events_path, format='csv', fill_values=[])
    return cmd_events


def get_loads(cmds_dir=None, scenario=None):
    loads_path = paths.LOADS_TABLE_PATH(cmds_dir, scenario)
    loads = Table.read(loads_path, format='csv')
    return loads


def update_loads(cmds_dir=None, scenario=None, *, lookback=31, stop=None):
    """Update or create loads.csv and loads/ archive though ``lookback`` days"""
    if scenario is not None:
        # Ensure the scenario directory exists
        scenario_dir = paths.SCENARIO_DIR(scenario)
        scenario_dir.mkdir(parents=True, exist_ok=True)

    cmd_events = get_cmd_events(cmds_dir, scenario)

    # TODO for performance when we have decent testing:
    # Read in the existing loads table and grab the RLTT and scheduled stop
    # dates. Those are the only reason we need to read an existing set of
    # commands that are already locally available, but they are fixed and cannot
    # change. The only thing that might change is an interrupt time, e.g. if
    # the SCS time gets updated.
    # So maybe read in the loads table and make a dict of rltts and ssts keyed
    # by load name and use those to avoid reading in the cmds table.
    # For now just get things working reliably.

    loads_table_path = paths.LOADS_TABLE_PATH(cmds_dir, scenario)
    loads_rows = []

    # Probably too complicated, but this bit of code generates a list of dates
    # that are guaranteed to sample all the months in the lookback period with
    # two weeks of margin on either side.
    dt = 14 * u.day
    stop = CxoTime(stop) + dt
    start = stop - lookback * u.day - dt
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
                cmds = get_load_cmds_from_occweb_or_local(cmds_dir, dir_year_month, load_name)
                load = get_load_dict_from_cmds(load_name, cmds, cmd_events)
                loads_rows.append(load)

    # Finally, save the table to file
    loads_table = Table(loads_rows)
    logger.info(f'Saving {len(loads_table)} loads to {loads_table_path}')
    loads_table.sort('cmd_start')
    loads_table.write(loads_table_path, format='csv', overwrite=True)
    loads_table.write(loads_table_path.with_suffix('.dat'), format='ascii.fixed_width',
                      overwrite=True)


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


def get_load_cmds_from_occweb_or_local(cmds_dir, dir_year_month, load_name):
    """Get the load cmds (backstop) for ``load_name`` within ``dir_year_month``

    If the backstop file is already available locally, use that. Otherwise, the
    file is downloaded from OCCweb and is then parsed and saved as a gzipped
    pickle file of the corresponding CommandTable object.

    :param cmds_dir: str, Path, None
        Directory where the backstop files are saved.
    :param dir_year_month: Path
        Path to the directory containing the ``load_name`` directory.
    :param load_name: str
        Load name in the usual format e.g. JAN0521A.
    :returns: CommandTable
        Backstop commands for the load.
    """
    # Determine output file name and make directory if necessary.
    loads_dir = paths.LOADS_ARCHIVE_DIR(cmds_dir, load_name)
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
