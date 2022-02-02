# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections import defaultdict
import math
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
from astropy.table import Table
import astropy.units as u
import requests
from cxotime import CxoTime
from testr.test_helper import has_internet
from Chandra.Maneuver import NSM_attitude


from kadi.commands import get_cmds_from_backstop, conf
from kadi.commands.core import (decode_starcat_params, load_idx_cmds, load_pars_dict, LazyVal,
                                get_par_idx_update_pars_dict, _find, vstack_exact,
                                ska_load_dir, CommandTable, load_name_to_cxotime)
from kadi.commands.command_sets import get_cmds_from_event
from kadi import occweb, paths

# TODO configuration options, but use DEFAULT_* in the mean time
# - commands_version (v1, v2)

MATCHING_BLOCK_SIZE = 500

# TODO: cache translation from cmd_events to CommandTable's  [Probably not]

APPROVED_LOADS_OCCWEB_DIR = Path('FOT/mission_planning/PRODUCTS/APPR_LOADS')

# URL to download google sheets `doc_id`
CMD_EVENTS_SHEET_URL = 'https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv'

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

HAS_INTERNET = has_internet()

logger = logging.getLogger(__name__)


def clear_caches():
    """Clear all commands caches.

    This is useful for testing and in case upstream products like the Command
    Events sheet have changed during a session.
    """
    CMDS_RECENT.clear()
    MATCHING_BLOCKS.clear()
    for var in [IDX_CMDS, PARS_DICT, REV_PARS_DICT]:
        try:
            del IDX_CMDS._val
        except AttributeError:
            pass


def interrupt_load_commands(load, cmds):
    """Cut commands beyond observing or vehicle stop times.

    Orbit point commands are NOT cut so that in the case of a load stop the
    orbit points are still available. This takes advantage of additional code
    that de-duplicates orbit points.
    """
    bad = np.zeros(len(cmds), dtype=bool)
    if load['observing_stop'] != '':
        bad |= ((cmds['date'] > load['observing_stop'])
                & (cmds['scs'] > 130))
    if load['vehicle_stop'] != '':
        bad |= ((cmds['date'] > load['vehicle_stop'])
                & (cmds['scs'] < 131)
                & (cmds['type'] != 'ORBPOINT'))
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

    logger.info(f'Merging cmds_recent with archive commands from {start}')

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
    cmds = vstack_exact([cmds_arch, cmds_recent[recent_block_end:]])

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
    scenario = os.environ.get('KADI_SCENARIO', scenario)
    start = CxoTime(start or '1999:001')
    stop_date = CxoTime(stop).date if stop else '2099:001'

    # Default stop is either now (typically) or set by env var
    default_stop = CxoTime(os.environ.get('KADI_COMMANDS_DEFAULT_STOP'))

    # For flight scenario or no internet or if the query stop time is guaranteed
    # to not require recent commands then just use the archive.
    before_recent_cmds = stop_date < (default_stop - conf.default_lookback * u.day).date
    if scenario == 'flight' or not HAS_INTERNET or before_recent_cmds:
        cmds = IDX_CMDS
        logger.info('Getting commands from archive only')
    else:
        if scenario not in CMDS_RECENT:
            cmds_recent = update_archive_and_get_cmds_recent(
                scenario, cache=True,
                pars_dict=PARS_DICT, rev_pars_dict=REV_PARS_DICT)
        else:
            cmds_recent = CMDS_RECENT[scenario]

        # Get `cmds` as correct mix of recent and archive commands that contains
        # the requested date range.
        if stop_date < cmds_recent['date'][0]:
            # Query does not overlap with recent commands, just use archive.
            logger.info('Getting commands from archive only')
            cmds = IDX_CMDS
        elif start < CxoTime(cmds_recent['date'][0]) + 3 * u.day:
            # Query starts near beginning of recent commands and *might* need some
            # archive commands. The margin is set at 3 days to ensure that OBS
            # command continuity is maintained (there is at least one maneuver).
            cmds = _merge_cmds_archive_recent(start, scenario)
            logger.info(f'Getting commands from archive + recent {scenario=}')
        else:
            # Query is strictly within recent commands.
            cmds = cmds_recent
            logger.info(f'Getting commands from recent only {scenario=}')

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

    cmds['time'].info.format = '.3f'

    return cmds


def update_archive_and_get_cmds_recent(scenario=None, *, lookback=None, stop=None,
                                       cache=True,
                                       pars_dict=None, rev_pars_dict=None):
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
    # Allow for variations in input format of date
    dates = np.array([CxoTime(date).date for date in cmd_events['Date']])
    bad = ((dates < (start - 14 * u.day).date)
           | (dates > stop.date))
    cmd_events = cmd_events[~bad]
    cmd_events_ids = [evt['Event'] + ' at ' + evt['Date'] for evt in cmd_events]
    if len(cmd_events) > 0:
        logger.info('Including cmd_events:\n  {}'.format("\n  ".join(cmd_events_ids)))
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
                # First check for any overlap since prev_cmds is sorted by date.
                if prev_cmds['date'][-1] > rltt:
                    # Cut commands EXCEPT for ORBPOINT ones, which we leave as a
                    # special case to ensure availability of orbit events from
                    # commands. Otherwise RLTT can cut ORBPOINT commands that
                    # are not replaced by the subsquent loads.
                    bad = (prev_cmds['date'] > rltt) & (prev_cmds['type'] != 'ORBPOINT')
                    if np.any(bad):
                        n_bad = np.count_nonzero(bad)
                        logger.info(f'Removing {n_bad} cmds from {prev_cmds["source"][0]}')
                    cmds_list[jj] = prev_cmds[~bad]

        if len(cmds) > 0:
            logger.info(f'Adding {len(cmds)} commands from {cmds["source"][0]}')

    cmds_recent = vstack_exact(cmds_list)
    cmds_recent.sort_in_backstop_order()
    cmds_recent.deduplicate_orbit_cmds()
    cmds_recent.remove_not_run_cmds()
    cmds_recent = add_obs_cmds(cmds_recent, pars_dict, rev_pars_dict)

    if cache:
        # Cache recent commands so future requests for the same scenario are fast
        CMDS_RECENT[scenario] = cmds_recent

    return cmds_recent


def add_obs_cmds(cmds, pars_dict, rev_pars_dict):
    """Add LOAD_EVENT OBS commands with info about observations.

    This command includes the following:

    - manvr_start: date of maneuver start
    - npnt_enab: auto-transition to NPNT enabled
    - obs_start: as the command date
    - obs_stop: date of subsequent AONMMODE or AONSMSAF command)
    - obsid: observation ID
    - starcat_idx: index of starcat in reversed params dict

    :param cmds_recent: CommandTable
    :returns: CommandTable with added OBS commands
    """
    # Last command in cmds is the schedule stop time (i.e. obs_stop for the
    # last observation).
    schedule_stop_time = cmds['date'][-1]

    # Get the subset of commands needed to determine the state for OBS cmds like
    # maneuver commanding, obsid, starcat, etc.
    cmds_state = get_state_cmds(cmds)

    # Get a table of OBS cmds corresponding to the end of each maneuver. For the
    # first pass this only has maneuver info that is known at the point of
    # starting the maneuver.
    cmds_obs = get_cmds_obs_from_manvrs(cmds_state)

    # Put the OBS cmds into the state cmds table and then do the second pass to
    # determine obsid, starcat, etc at the end of the maneuver (i.e. at the
    # start of the obs). This returns just the obs commands and updates
    # `pars_dict` and `rev_pars_dict` in place.
    cmds_state_obs = cmds_state.add_cmds(cmds_obs)
    cmds_obs = get_cmds_obs_final(cmds_state_obs, pars_dict, rev_pars_dict,
                                  schedule_stop_time)

    # Finally add the OBS cmds to the recent cmds table.
    cmds_out = cmds.add_cmds(cmds_obs)
    return cmds_out


def get_state_cmds(cmds):
    """Get the state-changing commands need to create LOAD_EVENT OBS commands.

    :param cmds: CommandTable of input commands.
    :returns: CommandTable of state-changing commands.
    """
    state_tlmsids = [
        'AOSTRCAT',
        'COAOSQID',
        'AOMANUVR',
        'AONMMODE',
        'AONSMSAF',
        'AOUPTARQ',
        'AONM2NPE',
        'AONM2NPD']

    vals = [val.encode('ascii') for val in state_tlmsids]
    ok = np.isin(cmds['tlmsid'], vals)
    ok |= cmds['type'] == 'SIMTRANS'
    cmds = cmds[ok]
    cmds.sort(['date', 'scs'])
    return cmds


def get_cmds_obs_from_manvrs(cmds):
    """Get OBS commands corresponding to the end of each maneuver.

    This is the first pass of getting OBS commands, keying off each maneuver.
    These OBS commands are then re-inserted into the commands table at the point
    of the maneuver END for the second pass, where at the point in time of the
    maneuver end (obs start), all the state-changing commands have occurred.

    :param cmds: CommandTable of state-changing commands.
    :returns: CommandTable of OBS commands.
    """
    prev_att = None
    targ_att = None
    npnt_enab = False

    cmds_obs = []
    times = CxoTime(cmds['date']).secs

    for time, cmd in zip(times, cmds):
        tlmsid = cmd['tlmsid']
        if tlmsid == 'AOUPTARQ':
            pars = cmd['params']
            targ_att = (pars['q1'], pars['q2'], pars['q3'], pars['q4'])
        elif tlmsid == 'AONM2NPE':
            npnt_enab = True
        elif tlmsid == 'AONM2NPD':
            npnt_enab = False
        elif tlmsid in ('AOMANUVR', 'AONSMSAF'):
            if tlmsid == 'AONSMSAF':
                targ_att = tuple(NSM_attitude(prev_att or (0, 0, 0, 1), cmd['date']).q.tolist())
                npnt_enab = False
            if targ_att is None:
                # No target attitude is unexpected since we got to a MANVR cmd.
                logger.warning(f'WARNING: no target attitude for {cmd["date"]}')
                log_context_obs(cmds, cmd)
                continue
            if prev_att is None:
                # No previous attitude happens always for first manvr at the
                # beginning of loads, and normally we just push on to the next
                # OBS. But if we already have OBS command this is a problem.
                if cmds_obs:
                    logger.warning(f'WARNING: No previous attitude for {cmd["date"]}')
                    log_context_obs(cmds, cmd)
                prev_att = targ_att
                continue
            dur = manvr_duration(prev_att, targ_att)
            params = {'manvr_start': cmd['date'],
                      'prev_att': prev_att,
                      'targ_att': targ_att,
                      'npnt_enab': npnt_enab}
            cmd_obs = {'idx': -1,
                       'type': 'LOAD_EVENT',
                       'tlmsid': 'OBS',
                       'scs': 0,
                       'step': 0,
                       'source': cmd['source'],
                       'vcdu': -1,
                       'time': time + dur,
                       'params': params,
                       }
            cmds_obs.append(cmd_obs)
            prev_att = targ_att
            targ_att = None

    cmds_obs = CommandTable(rows=cmds_obs)
    # NB: much faster to do this vectorized than when defining cmd_obs above.
    cmds_obs.add_column(CxoTime(cmds_obs['time']).date, name='date', index=0)

    # If an NSM occurs within a maneuver then remove that obs
    nsms = cmds['tlmsid'] == 'AONSMSAF'
    if np.any(nsms):
        bad_idxs = []
        for nsm_date in cmds['date'][nsms]:
            for ii, cmd in enumerate(cmds_obs):
                if nsm_date > cmd['params']['manvr_start'] and nsm_date <= cmd['date']:
                    logger.info(f'NSM at {nsm_date} happened during maneuver preceding obs\n{cmd}')
                    log_context_obs(cmds, cmd)
                    bad_idxs.append(ii)
        if bad_idxs:
            logger.info(f'Removing obss at {bad_idxs}')
            cmds_obs.remove_rows(bad_idxs)

    # This is known to happen for cmds in AUG0103A, JAN2204B, JUL2604D due to
    # non-load maneuvers that are not captured correctly. Run the v1-v2
    # migration script to see. The prev_att is wrong and the incorrect maneuver
    # timing results in overlaps. It should not happen for modern loads.
    for cmd0, cmd1 in zip(cmds_obs[:-1], cmds_obs[1:]):
        if cmd1['params']['manvr_start'] <= cmd0['date'] < cmd1['date']:
            logger.warning(f'WARNING: fixing overlapping OBS cmds: \n{cmd0} \n{cmd1}')
            date0 = CxoTime(cmd1['params']['manvr_start']) - 15 * u.s
            cmd0['date'] = date0.date
            cmd0['time'] = date0.secs

    return cmds_obs


# parameters from characteristics_Hardware for normal maneuvers
MANVR_DELTA = 60.0
MANVR_ALPHAMAX = 2.18166e-006  # AKA alpha
MANVR_VMAX = 0.001309  # AKA omega


def manvr_duration(q1, q2):
    """Calculate the duration of a maneuver from two quaternions

    This is basically the same as the function in Chandra.Maneuver but
    optimized to work on a single 4-vector quaterion.

    :param q1: list of 4 quaternion elements
    :param q2: list of 4 quaternion elements
    :returns: duration of maneuver in seconds
    """
    # Compute 4th element of delta quaternion q_manvr = q2 / q1
    q_manvr_3 = abs(-q2[0] * q1[0]
                    - q2[1] * q1[1]
                    - q2[2] * q1[2]
                    + q2[3] * -q1[3])

    # 4th component is cos(theta/2)
    if (q_manvr_3 > 1):
        q_manvr_3 = 1
    phimax = 2 * math.acos(q_manvr_3)

    epsmax = MANVR_VMAX / MANVR_ALPHAMAX - MANVR_DELTA
    tau = phimax / MANVR_VMAX - epsmax - 2 * MANVR_DELTA
    if (tau >= 0):
        eps = epsmax
    else:
        tau = 0
        eps = np.sqrt(MANVR_DELTA ** 2 + 4 * phimax / MANVR_ALPHAMAX) / 2 - 1.5 * MANVR_DELTA
        if (eps < 0):
            eps = 0

    Tm = 4 * MANVR_DELTA + 2 * eps + tau
    return Tm


def get_cmds_obs_final(cmds, pars_dict, rev_pars_dict, schedule_stop_time):
    """Fill in the rest of params for each OBS command.

    Second pass of processing which implements a state machine to fill in the
    other parameters like obsid, starcat, simpos etc. for each OBS command.

    The observation state is completed by encountering the next transition to
    NMM or NSM.

    :param cmds: CommandTable of state-changing + OBS commands.
    :param pars_dict: dict of parameters for commands
    :param rev_pars_dict: reversed dict of parameters for commands
    :param schedule_stop_time: date of last command

    :returns: CommandTable of OBS commands with all parameters filled in.
    """
    # Initialize state variables. Note that the first OBS command may end up
    # with None values for some of these, but this is OK because of the command
    # merging with the archive commands which are always correct. Nevertheless
    # use values that are not None to avoid errors. For `sim_pos`, if the SIM
    # has not been commanded in a long time then it will be at -99616.
    obsid = -1
    starcat_idx = -1
    sim_pos = -99616
    obs_params = None
    cmd_obs_extras = []

    for cmd in cmds:
        tlmsid = cmd['tlmsid']
        if tlmsid == 'AOSTRCAT':
            if cmd['idx'] == -1:
                # OBS command only stores the index of the starcat params, so at
                # this point we need to put those params into the pars_dict and
                # rev_pars_dict and get the index.
                starcat_idx = get_par_idx_update_pars_dict(
                    pars_dict, cmd, rev_pars_dict=rev_pars_dict)
            else:
                starcat_idx = cmd['idx']
        elif tlmsid == 'COAOSQID':
            obsid = cmd['params']['id']
            # Look for obsid change within obs, likely an undercover
            # (target="cold blank ECS"). First stop the initial obs at the time
            # of the obsid command.
            if obs_params is not None:
                obs_params['obs_stop'] = cmd['date']
                # Then start a new obs at the time of the next obsid command.
                obs_params = obs_params.copy()
                obs_params['obsid'] = obsid
                obs_params['obs_start'] = cmd['date']
                # Collect these extra obs to a list to be added to cmds table later.
                cmd_obs = {'idx': -1,
                           'date': cmd['date'],
                           'type': 'LOAD_EVENT',
                           'tlmsid': 'OBS',
                           'scs': 0,
                           'step': 0,
                           'time': CxoTime(cmd['date']).secs,
                           'source': cmd['source'],
                           'vcdu': -1,
                           'params': obs_params,
                           }
                cmd_obs_extras.append(cmd_obs)
        elif tlmsid == 'OBS':
            obs_params = cmd['params']
            obs_params['obsid'] = obsid
            obs_params['simpos'] = sim_pos  # matches states 'simpos'
            obs_params['obs_start'] = cmd['date']
            if obs_params['npnt_enab']:
                obs_params['starcat_idx'] = starcat_idx

        elif (tlmsid in ('AONMMODE', 'AONSMSAF')
              or (tlmsid == 'AONM2NPE' and cmd['vcdu'] == -1)):
            # Stop current obs at next maneuver. In some v1 commands (for
            # migration) there are cases of non-load maneuvers without an
            # AONMMODE command and only the AONM2NPE and AOMANUVR commands.
            # In those cases just use ANOM2NPE as a proxy. In v2 commands this
            # is OK since AONM2NPE always comes after the AONMMODE command and
            # obs_params will be None at that point.
            if obs_params is not None:
                obs_params['obs_stop'] = cmd['date']
            # This closes out the observation
            obs_params = None
        elif cmd['type'] == 'SIMTRANS':
            sim_pos = cmd['params']['pos']

    # Filter down to just the observation commands
    cmds_obs = cmds[cmds['tlmsid'] == 'OBS']

    # Potentially add extra obs commands (typically undercovers) to cmds table
    if cmd_obs_extras:
        cmds_obs = cmds_obs.add_cmds(CommandTable(cmd_obs_extras))

    # The last OBS command will be missing obs_stop since there is no
    # subsequent maneuver. Fix that for the expected case but warn if an obs
    # before the end is missing obs_stop.
    for cmd in cmds_obs:
        if 'obs_stop' not in cmd['params']:
            cmd['params']['obs_stop'] = schedule_stop_time
            if cmd.index != len(cmds_obs) - 1:
                logger.warning(f'OBS command missing obs_stop\n{cmd}')
                log_context_obs(cmds, cmd)
        cmd['idx'] = get_par_idx_update_pars_dict(
            pars_dict, cmd, rev_pars_dict=rev_pars_dict)

    return cmds_obs


def log_context_obs(cmds, cmd, before=3600, after=3600):
    """Log commands before and after ``cmd``"""
    date_before = (CxoTime(cmd['date']) - before * u.s).date
    date_after = (CxoTime(cmd['date']) + after * u.s).date
    ok = (cmds['date'] >= date_before) & (cmds['date'] <= date_after)
    logger.warning(f'\n{cmds[ok]}')


def is_google_id(scenario):
    """Return True if scenario appears to be a Google ID.

    :param scenario: str, None
    :returns: bool
    """
    # Something better??
    return scenario is not None and len(scenario) > 35


def update_cmd_events(scenario=None):
    """Update local cmd_events.csv from Google Sheets for ``scenario``.

    :param scenario: str, None
        Scenario name
    :returns: Table
        Command events table
    """
    # Named scenarios with a name that isn't "flight" and does not look like a
    # google sheet ID are local files.
    use_local = scenario not in (None, 'flight') and not is_google_id(scenario)
    if use_local or not HAS_INTERNET:
        return get_cmd_events(scenario)

    # Ensure the scenario directory exists
    paths.SCENARIO_DIR(scenario).mkdir(parents=True, exist_ok=True)

    cmd_events_path = paths.CMD_EVENTS_PATH(scenario)

    doc_id = scenario if is_google_id(scenario) else conf.cmd_events_flight_id
    url = CMD_EVENTS_SHEET_URL.format(doc_id=doc_id)
    logger.info(f'Getting cmd_events from {url}')
    req = requests.get(url, timeout=30)
    if req.status_code != 200:
        raise ValueError(f'Failed to get cmd events sheet: {req.status_code}')

    cmd_events = Table.read(req.text, format='csv')
    ok = np.isin(cmd_events['State'], ('Predictive', 'Definitive'))
    cmd_events = cmd_events[ok]

    logger.info(f'Writing {len(cmd_events)} cmd_events to {cmd_events_path}')
    cmd_events.write(cmd_events_path, format='csv', overwrite=True)
    return cmd_events


def get_cmd_events(scenario=None):
    """Get local cmd_events.csv for ``scenario``.

    :param scenario: str, None
        Scenario name
    :returns: Table
        Command events table
    """
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
    # For testing allow override of default `stop` value
    if stop is None:
        stop = os.environ.get('KADI_COMMANDS_DEFAULT_STOP')

    if lookback is None:
        lookback = conf.default_lookback

    # Ensure the scenario directory exists
    paths.SCENARIO_DIR(scenario).mkdir(parents=True, exist_ok=True)

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
    vehicle_stop_events = ('NSM', 'Safe mode', 'Bright star hold')
    observing_stop_events = vehicle_stop_events + ('SCS-107',)

    load = {'name': load_name,
            'cmd_start': cmds['date'][0],
            'cmd_stop': cmds['date'][-1],
            'observing_stop': '',
            'vehicle_stop': ''}

    load['rltt'] = cmds.get_rltt()
    load['scheduled_stop_time'] = cmds.get_scheduled_stop_time()

    # CHANGE THIS to use LOAD_EVENT entries in commands. Or NOT?? Probably
    # provides good visibility into what's going on. But this hard-coding is
    # annoying.
    for cmd_event in cmd_events:
        cmd_event_date = cmd_event['Date']

        if (cmd_event_date >= load['cmd_start']
                and cmd_event_date <= load['cmd_stop']
                and cmd_event['Event'] in observing_stop_events):
            logger.info(f'{cmd_event["Event"]} at {cmd_event_date} found for {load_name}')
            load['observing_stop'] = cmd_event['Date']

            if cmd_event['Event'] in vehicle_stop_events:
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
            cmds = parse_backstop_and_write(load_name, cmds_filename, backstop_text)
            break
        else:
            raise ValueError(f'No backstop file found in {ska_dir}')

    else:  # use OCCweb
        load_dir_contents = occweb.get_occweb_dir(dir_year_month / load_name)
        for filename in load_dir_contents['Name']:
            if re.match(r'CR\d{3}.\d{4}\.backstop', filename):

                # Download the backstop file from OCCweb
                backstop_text = occweb.get_occweb_page(
                    dir_year_month / load_name / filename,
                    cache=conf.cache_loads_in_astropy_cache)
                cmds = parse_backstop_and_write(load_name, cmds_filename, backstop_text)
                break
        else:
            raise ValueError(f'Could not find backstop file in {dir_year_month / load_name}')

    return cmds


def parse_backstop_and_write(load_name, cmds_filename, backstop_text):
    """Parse ``backstop_text`` and write to ``cmds_filename`` (gzipped pickle).

    This sets the ``source`` column to ``load_name`` and removes the
    ``timeline_id`` column.
    """
    backstop_lines = backstop_text.splitlines()
    cmds = get_cmds_from_backstop(backstop_lines)

    # Fix up the commands to be in the right format
    idx = cmds.colnames.index('timeline_id')
    cmds.add_column(load_name, index=idx, name='source')
    del cmds['timeline_id']

    logger.info(f'Saving {cmds_filename}')
    with gzip.open(cmds_filename, 'wb') as fh:
        pickle.dump(cmds, fh)
    return cmds


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
    # For testing allow override of default `stop` value
    if stop is None:
        stop = os.environ.get('KADI_COMMANDS_DEFAULT_STOP')

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
        scenario=scenario, stop=stop, lookback=lookback, cache=False,
        pars_dict=pars_dict)

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
    cmds_arch_new = vstack_exact([cmds_arch[:idx0_arch], cmds_recent[idx0_recent:]])
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


TYPE_MAP = ['ACQ', 'GUI', 'BOT', 'FID', 'MON']
IMGSZ_MAP = ['4x4', '6x6', '8x8']
PAR_MAPS = [
    ('imnum', 'slot', int),
    ('type', 'type', lambda n: TYPE_MAP[n]),
    ('imgsz', 'sz', lambda n: IMGSZ_MAP[n]),
    ('maxmag', 'maxmag', float),
    ('yang', 'yang', lambda x: np.degrees(x) * 3600),
    ('zang', 'zang', lambda x: np.degrees(x) * 3600),
    ('dimdts', 'dim', int),
    ('restrk', 'res', int),
]


def convert_aostrcat_to_acatable(params):
    """Convert dict of AOSTRCAT parameters to an ACATable.

    The dict looks like::

       2009:032:11:13:42.800 | 8023994 0 | MP_STARCAT | TLMSID= AOSTRCAT, CMDS=
       49, IMNUM1= 0, YANG1= -3.74826751e-03, ZANG1= -8.44541515e-03, MAXMAG1=
       8.00000000e+00, MINMAG1= 5.79687500e+00, DIMDTS1= 1, RESTRK1= 1, IMGSZ1=2,
       TYPE1= 3, ...

       IMNUM: slot
       RESTRK: box size resolution, always 1 (high) in loads for non-MON slot
       DIMDTS: (halfwidth - 20) / 5  # assumes AQRES='H'
       TYPE: 4 => mon, 3 => fid, 2 => bot, 1 => gui, 0 => acq
       YANG: yang in radians
       ZANG: zang in radians
       IMGSZ: 0=4x4, 1=6x6, 2=8x8
       MINMAG = min mag
       MAXMAG = max mag

    :param params: dict of AOSTRCAT parameters
    :returns: ACATable
    """
    from proseco.catalog import ACATable

    for idx in range(1, 17):
        if params[f'minmag{idx}'] == params[f'maxmag{idx}'] == 0:
            break

    max_idx = idx
    cols = defaultdict(list)
    for par_name, col_name, func in PAR_MAPS:
        for idx in range(1, max_idx):
            cols[col_name].append(func(params[par_name + str(idx)]))

    out = ACATable(cols)
    out.add_column(np.arange(1, max_idx), index=1, name='idx')

    return out


def get_starcats(obsid=None, start=None, stop=None, scenario=None):
    """Get star catalogs corresponding to input parameters.

    The ``obsid``, ``start``, and ``stop`` parameters serve as matching filters
    on the list of star catalogs that is returned.

    There are numerous instances of multiple observations with the same obsid,
    so this function always returns a list of star catalogs even when ``obsid``
    is specified. In most cases you can just use the first element.

    Example::

        >>> from kadi.commands import get_starcats
        >>> cat = get_starcats(obsid=8008)[0]
        >>> cat
        <ACATable length=11>
        slot  idx  type  sz   maxmag   yang     zang    dim   res
        int64 int64 str3 str3 float64 float64  float64  int64 int64
        ----- ----- ---- ---- ------- -------- -------- ----- -----
            0     1  FID  8x8    8.00   937.71  -829.17     1     1
            1     2  FID  8x8    8.00 -1810.42  1068.87     1     1
            2     3  FID  8x8    8.00   403.68  1712.93     1     1
            3     4  BOT  6x6   10.86  -318.22  1202.41    20     1
            4     5  BOT  6x6   11.20  -932.79  -354.55    20     1
            5     6  BOT  6x6   10.97  2026.85  1399.61    20     1
            6     7  BOT  6x6   10.14   890.71 -1600.39    20     1
            7     8  BOT  6x6   10.66  2023.08 -2021.72    13     1
            0     9  ACQ  6x6   10.64    54.04   754.79    20     1
            1    10  ACQ  6x6   11.70   562.06  -186.39    20     1
            2    11  ACQ  6x6   11.30  1612.28  -428.24    20     1

    :param obsid: int, None
        ObsID
    :param start: CxoTime-like, None
        Start time (default=beginning of commands)
    :param stop: CxoTime-like, None
        Stop time (default=end of commands)
    :param scenario: str, None
        Scenario
    :returns: list of ACATable
        List star catalogs for matching observations.
    """
    obss = get_observations(obsid=obsid, start=start, stop=stop, scenario=scenario)
    starcats = []
    for obs in obss:
        if (idx := obs.get('starcat_idx')) is None:
            continue
        params = REV_PARS_DICT[idx]
        if isinstance(params, bytes):
            params = decode_starcat_params(params)
        starcat = convert_aostrcat_to_acatable(params)
        starcats.append(starcat)
    return starcats


OBSERVATIONS = {}


def get_observations(obsid=None, start=None, stop=None, scenario=None):
    """Get observations corresponding to input parameters.

    The ``obsid``, ``start``, and ``stop`` parameters serve as matching filters
    on the list of observations that is returned.

    There are numerous instances of multiple observations with the same obsid,
    so this function always returns a list of observation parameters even when
    ``obsid`` is specified. In most cases you can just use the first element.

    Examples::

        >>> from kadi.commands import get_observations
        >>> obs = get_observations(obsid=8008)[0]
        >>> obs
        {'obsid': 8008,
        'simpos': 92904,
        'obs_stop': '2007:002:18:04:28.965',
        'manvr_start': '2007:002:04:31:48.216',
        'targ_att': (0.149614271, 0.490896707, 0.831470649, 0.21282047),
        'npnt_enab': True,
        'obs_start': '2007:002:04:46:58.056',
        'prev_att': (0.319214732, 0.535685207, 0.766039803, 0.155969017),
        'starcat_idx': 144398}

        >>> obs_all = get_observations()  # All observations in commands archive

        # Might be convenient to handle this as a Table
        >>> from astropy.table import Table
        >>> obs_all = Table(obs_all)

    :param obsid: int, None
        ObsID
    :param start: CxoTime-like, None
        Start time (default=beginning of commands)
    :param stop: CxoTime-like, None
        Stop time (default=end of commands)
    :param scenario: str, None
        Scenario
    :returns: list of dict
        Observation parameters for matching observations.
    """
    if start is None:
        start = '1999:001'
    if stop is None:
        # Commands never extend more than 60 days in the future
        stop = (CxoTime.now() + 60 * u.day).date

    if scenario not in OBSERVATIONS:
        cmds = get_cmds(scenario=scenario)
        cmds_obs = cmds[cmds['tlmsid'] == 'OBS']
        OBSERVATIONS[scenario] = cmds_obs
    else:
        cmds_obs = OBSERVATIONS[scenario]

    i0, i1 = cmds_obs.find_date([start, stop])
    cmds_obs = cmds_obs[i0:i1]

    if obsid is not None:
        cmds_obs = cmds_obs[cmds_obs['obsid'] == obsid]
        if len(cmds_obs) == 0:
            raise ValueError(f'No matching observations for {obsid=}')

    obss = [cmd['params'] for cmd in cmds_obs]

    return obss
