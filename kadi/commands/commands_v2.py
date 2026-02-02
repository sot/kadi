# Licensed under a 3-clause BSD style license - see LICENSE.rst
import calendar
import collections
import difflib
import functools
import gzip
import logging
import math
import os
import pickle
import re
import weakref
from collections import defaultdict
from pathlib import Path
from typing import Callable, List

import astropy.units as u
import numpy as np
import requests
from astropy.table import Table
from cxotime import CxoTime
from parse_cm.paths import ParseLoadNameError, parse_load_name
from ska_helpers.retry import retry_func
from ska_helpers.utils import temp_env_var
from ska_sun import get_nsm_attitude
from testr.test_helper import has_internet

from kadi import occweb, paths
from kadi.commands.command_sets import get_cmds_from_event
from kadi.commands.core import (
    CommandTable,
    LazyVal,
    _find,
    filter_cmd_events_date_stop,
    filter_cmd_events_state,
    get_cmds_from_backstop,
    get_cxotime_now,
    get_par_idx_update_pars_dict,
    kadi_cmds_version,
    load_idx_cmds,
    load_name_to_cxotime,
    load_pars_dict,
    vstack_exact,
)
from kadi.config import conf

__all__ = ["clear_caches", "get_cmds"]

# TODO: cache translation from cmd_events to CommandTable's  [Probably not]

# Formally approved load products
APPROVED_LOADS_OCCWEB_DIR = Path("FOT/mission_planning/PRODUCTS/APPR_LOADS")

# Products that are currently in-review or in-work.
BACKSTOP_LOADS_OCCWEB_DIR = Path("FOT/mission_planning/Backstop")

# URL to download google sheets `doc_id`
# See https://stackoverflow.com/questions/33713084 (original question).
# See also kadi.commands.cmds_validate for the long-form URL.
CMD_EVENTS_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv"
)

# Cached values of the full mission commands archive (cmds_v2.h5, cmds_v2.pkl).
# These are loaded on demand.
IDX_CMDS = LazyVal(functools.partial(load_idx_cmds))
PARS_DICT = LazyVal(functools.partial(load_pars_dict))
REV_PARS_DICT = LazyVal(lambda: {v: k for k, v in PARS_DICT.items()})

# Cache of recent commands keyed by scenario
CMDS_RECENT = {}
MATCHING_BLOCKS = {}

# APR1420B was the first load set to have RLTT (backstop 6.9)
RLTT_ERA_START = CxoTime("2020-04-14")
RLTT_ERA_START_LOAD = "APR1420B"

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
            del var._val
        except AttributeError:
            pass

    from kadi.commands.observations import OBSERVATIONS

    OBSERVATIONS.clear()


def _merge_cmds_archive_recent(start, cache_key, cmds_recent):
    """Merge cmds archive from ``start`` onward with recent cmds for ``scenario``

    This assumes:
    - Recent commands overlap the cmds archive

    Parameters
    ----------
    start : CxoTime-like
        Start time for returned commands
    scenario : str
        Scenario name

    Returns
    -------
    CommandTable
        Commands from cmds archive and all recent commands
    """

    logger.info(f"Merging cmds_recent with archive commands from {start}")

    if cache_key not in MATCHING_BLOCKS:
        # Get index for start of cmds_recent within the cmds archive
        i0_arch_recent = IDX_CMDS.find_date(cmds_recent["date"][0])

        # Find the end of the first large (MATCHING_BLOCK_SIZE) block of
        # cmds_recent that overlap with archive cmds. Look for the matching
        # block in a subset of archive cmds that starts at the start of
        # cmds_recent. `arch_recent_offset` is the offset from `i0_arch_recent`
        # to the end of the matching block. `i0_recent` is the end of the
        # matching block in recent commands.
        arch_recent_offset, recent_block_end = get_matching_block_idx(
            IDX_CMDS[i0_arch_recent:], cmds_recent
        )
        arch_block_end = i0_arch_recent + arch_recent_offset
        MATCHING_BLOCKS[cache_key] = arch_block_end, recent_block_end, i0_arch_recent
    else:
        arch_block_end, recent_block_end, i0_arch_recent = MATCHING_BLOCKS[cache_key]

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
    date0 = cmds_recent["date"][0]
    i0_arch = cmds_arch.find_date(date0)
    key_names = ("date", "type", "tlmsid", "scs", "step", "vcdu")

    # Find block of commands in cmd_arch that match first min_match of
    # cmds_recent. Special case is min_match=0, which means we just want to
    # append the cmds_recent to the end of cmds_arch. This is the case for
    # the transition from pre-RLTT (APR1420B) to post, for the one-time
    # migration from version 1 to version 2.
    while min_match > 0:
        if all(
            np.all(
                cmds_arch[name][i0_arch : i0_arch + min_match]
                == cmds_recent[name][:min_match]
            )
            for name in key_names
        ):
            break
        # No joy, step forward and make sure date still matches
        i0_arch += 1
        if cmds_arch["date"][i0_arch] != date0:
            raise ValueError(
                "No matching commands block in archive found for recent_commands"
            )

    logger.info(f"Found matching commands block in archive at {i0_arch}")
    return i0_arch


def get_cmds(
    start=None,
    stop=None,
    *,
    inclusive_stop=False,
    scenario=None,
    event_filter: Callable | list[Callable] | None = None,
    **kwargs,
) -> CommandTable:
    """Get Chandra commands that ran on-board or are approved to run.

    Examples
    --------
    Get first few commands in 2013:

    >>> from kadi.commands import get_cmds
    >>> cmds = get_cmds('2013:001:00:00:00', '2013:001:00:56:10')
    >>> print(cmds)
            date            type      tlmsid   scs step      time      source    vcdu  params
    --------------------- ---------- ---------- --- ---- ------------- -------- ------- ------
    2013:001:00:37:37.653   ORBPOINT       None   0    0 473387924.837 DEC2412B 5533112    N/A
    2013:001:00:53:07.181 COMMAND_SW   AOACRSTD 129 1524 473388854.365 DEC2412B 5584176    N/A
    2013:001:00:54:07.181 COMMAND_SW   AOFUNCDS 129 1526 473388914.365 DEC2412B 5584410    N/A
    2013:001:00:55:07.181 COMMAND_SW   AOFUNCDS 129 1528 473388974.365 DEC2412B 5584644    N/A
    2013:001:00:56:07.181 COMMAND_SW   AONMMODE 129 1530 473389034.365 DEC2412B 5584878    N/A
    2013:001:00:56:07.181    ACISPKT AA00000000 132 1620 473389034.365 DEC2412B 5584878    N/A
    2013:001:00:56:07.181   SIMTRANS       None 132 1623 473389034.365 DEC2412B 5584878    N/A
    2013:001:00:56:07.438 COMMAND_SW   AONM2NPE 129 1532 473389034.622 DEC2412B 5584879    N/A

    Get all the SIM translation commands in the JAN2025A loads and print the first five
    like backstop:

    >>> cmds = get_cmds(source="JAN2025A", type="SIMTRANS")
    >>> cmds[:5].pprint_like_backstop)
    2025:020:08:27:26.428 | SIMTRANS         | None       | JAN2025A | pos=92904, scs=131
    2025:020:12:12:04.292 | SIMTRANS         | None       | JAN2025A | pos=75624, scs=131
    2025:020:15:56:41.704 | SIMTRANS         | None       | JAN2025A | pos=92904, scs=131
    2025:020:20:56:21.582 | SIMTRANS         | None       | JAN2025A | pos=75624, scs=131
    2025:021:01:29:20.281 | SIMTRANS         | None       | JAN2025A | pos=-99616, scs=131

    Parameters
    ----------
    start : CxoTime-like, optional
        Start time for cmds. Default is 1999:001.
    stop : CxoTime-like, optional
        Stop time for cmds. Default is current time + 1 year.
    scenario : str, optional
        Name of commands archive scenario to use instead of default.
    inclusive_stop : bool
        Include commands at exactly ``stop`` if True.
    event_filter : callable, list of callable, None
        Callable function or list of callable functions that takes an Event Table as
        input and returns a boolean mask with same length as Table. This is used to
        select rows from the Table. If None, no filtering is done.
    **kwargs : dict
        key=val keyword argument pairs for filtering.

    Returns
    -------
    CommandTable
    """
    logger.info(
        f"Getting commands from {CxoTime(start).date} to {CxoTime(stop).date} for {scenario=}"
    )
    scenario = os.environ.get("KADI_SCENARIO", scenario)
    start = CxoTime("1999:001" if start is None else start)
    stop = (CxoTime.now() + 1 * u.year) if stop is None else CxoTime(stop)
    lookback = conf.default_lookback

    # Get current "now" time, which might be mocked via CXOTIME_NOW.
    # get_cxotime_now() returns None if the env var is not set.
    cxotime_now = get_cxotime_now()

    # Cache key used for CMDS_RECENT and MATCHING_BLOCKS. These are all the relevant
    # kwargs in update_archive_and_get_cmds_recent().
    cache_key = scenario, cxotime_now, lookback, event_filter
    logger.info(f"Cache key: {cache_key}")

    # For flight scenario or no internet or if the query stop time is guaranteed
    # to not require recent commands then just use the archive.
    before_recent_cmds = stop < CxoTime(cxotime_now) - lookback * u.day
    if scenario == "flight" or not HAS_INTERNET or before_recent_cmds:
        cmds = IDX_CMDS
        logger.info(
            "Getting commands from archive only because of:"
            f" {scenario=} {before_recent_cmds=} {HAS_INTERNET=}"
        )
    else:
        if cache_key not in CMDS_RECENT:
            logger.info(
                "Recent commands not in cache: updating local cmd_events, loads "
                "and getting recent commands"
            )
            cmds_recent = update_cmd_events_and_loads_and_get_cmds_recent(
                scenario,
                lookback=lookback,
                stop_loads=cxotime_now,
                pars_dict=PARS_DICT,
                rev_pars_dict=REV_PARS_DICT,
                event_filter=event_filter,
            )
            CMDS_RECENT[cache_key] = cmds_recent
        else:
            logger.info(f"Getting recent commands from cache {cache_key=}")
            cmds_recent = CMDS_RECENT[cache_key]

        # Get `cmds` as correct mix of recent and archive commands that contains
        # the requested date range.
        if stop.date < cmds_recent["date"][0]:
            # Query does not overlap with recent commands, just use archive.
            logger.info(
                "Getting commands from archive only: stop < first commands recent for"
                " {scenario=}"
            )
            cmds = IDX_CMDS
        elif start < CxoTime(cmds_recent.meta["loads_start"]) + 3 * u.day:
            # Query starts near beginning of recent commands and *might* need some
            # archive commands. The margin is set at 3 days to ensure that OBS
            # command continuity is maintained (there is at least one maneuver).
            # See also the DESIGN commentary after the function.
            cmds = _merge_cmds_archive_recent(start, cache_key, cmds_recent)
            logger.info(
                "Getting commands from archive + recent: start < recent loads start +"
                f" 3 days for {scenario=}"
            )
        else:
            # Query is strictly within recent commands.
            cmds = cmds_recent
            logger.info(
                "Getting commands from recent only: start and stop are within recent"
                f" commands for {scenario=}"
            )

    # Select the requested time range and make a copy. (Slicing is a view so
    # in theory bad things could happen without a copy).
    idx0 = cmds.find_date(start)
    idx1 = cmds.find_date(stop, side=("right" if inclusive_stop else "left"))
    cmds = cmds[idx0:idx1].copy()

    if kwargs:
        # Specified extra filters on cmds search
        pars_dict = PARS_DICT.copy()
        # For any recent commands that have params as a dict, those will have
        # idx = -1. This doesn't work with _find, which is optimized to search
        # pars_dict for the matching search keys.
        # TODO: this step is only really required for kwargs that are not a column,
        # i.e. keys that are found only in params.
        for ii in np.flatnonzero(cmds["idx"] == -1):
            cmds[ii]["idx"] = get_par_idx_update_pars_dict(pars_dict, cmds[ii])
        cmds = _find(idx_cmds=cmds, pars_dict=pars_dict, **kwargs)

    cmds.rev_pars_dict = weakref.ref(REV_PARS_DICT)

    cmds["time"].info.format = ".3f"

    return cmds


# DESIGN commentary on this line in the above code::
#    start < CxoTime(cmds_recent.meta["loads_start"]) + 3 * u.day
#
# The question being asked here is "are there any applicable archive load
# commands which are NOT available from the recent load commands (i.e. the last
# 4 weeks of weekly loads)?".  Imagine these inputs:
#
#   start = Apr-02-2022
#   Loads = APR0122 (Never run due to anomaly), APR0622 (replan), APR0822,  APR1522,
#           APR2222
#   cmds_recent.meta["loads_start"] = Apr-01-2022 (first command in APR0122 approx)
#
# So what we care about is the first command in the applicable loads in the time
# span, not the first actually run load command in the span. The reason is that
# the archive is not going to provide additional commands here because the
# APR0122 commands will also not be in the archive.


def update_cmds_list_for_loads_in_backstop(
    cmd_events: Table,
    cmds_list: List[CommandTable],
) -> None:
    """Update ``cmds_list`` in place to add loads that are in backstop and not approved.

    This checks ``cmd_events`` for "Load in backstop" events and then reads those
    commands and appends them to ``cmds_list``.

    The ``Params`` for each such event must correspond to a load directory in
    https://occweb.cfa.harvard.edu/occweb/FOT/mission_planning/Backstop/ which has a
    backstop file. This can be a load name

    Parameters
    ----------
    cmd_events : Table
        Command events table.
    cmds_list : list of CommandTable
        List of CommandTable objects to update in place.
    """
    # Find loads that are "in backstop". Params is a single string with the load name.
    ok = cmd_events["Event"] == "Load in backstop"
    load_paths = cmd_events["Params"][ok]

    if len(load_paths) == 0:
        return

    for load_path in load_paths:
        # Load path like SEP1025A then it lives in BACKSTOP_LOADS_OCCWEB_DIR, otherwise
        # assume it is a relative path on OCCweb.
        try:
            parse_load_name(load_path)
        except ParseLoadNameError:
            parts = Path(load_path).parts
            load_dir = Path(*parts[:-1])
            load_name = parts[-1]
        else:
            load_dir = BACKSTOP_LOADS_OCCWEB_DIR
            load_name = load_path

        # Read backstop commands but do not write them to the local kadi "loads"
        # archive. This is reserved for approved loads.
        cmds = get_load_cmds_from_occweb_or_local(
            load_dir,
            load_name,
            in_work=True,
        )
        cmd_rltt = cmds.get_rltt_cmd()
        cmd_rltt["params"]["load_path"] = load_path
        cmds.meta["rltt"] = cmd_rltt["date"]
        logger.info(
            f"Adding {len(cmds)} commands from backstop load {load_path} "
            f"with RLTT={cmds.meta['rltt']}"
        )
        cmds_list.append(cmds)


def update_cmd_events_and_loads_and_get_cmds_recent(
    scenario=None,
    *,
    lookback=None,
    stop_loads=None,
    pars_dict=None,
    rev_pars_dict=None,
    event_filter: Callable | list[Callable] | None = None,
):
    """Update local loads table and downloaded loads and return all recent cmds.

    This is the main entry point for getting recent commands and for updating the
    archive HDF5 file.

    This also caches the recent commands in the global CMDS_RECENT dict.

    This relies entirely on RLTT and load_events to assemble the commands.

    Parameters
    ----------
    scenario : str, None
        Scenario name
    lookback : int, None
        Lookback time from ``stop`` for recent loads in days. If None, use
        conf.default_lookback.
    stop_loads : CxoTime-like[str], None
        Stop time for recent loads. If None, use get_cxotime_now().
    cache : bool
        Cache the result in CMDS_RECENT dict.
    pars_dict : dict, None
        Dictionary of parameters from the command table.
    rev_pars_dict : dict, None
        Inverse mapping of ``pars_dict``.
    event_filter : callable, list of callable, None
        Callable function or list of callable functions that takes an Event Table as
        input and returns a boolean mask with same length as Table. This is used to
        select rows from the Table. If None, no filtering is done.

    Returns
    -------
    CommandTable
    """
    # List of CommandTable objects from loads and cmd_events
    cmds_list: List[CommandTable] = []

    # Update local cmds_events.csv from Google Sheets
    cmd_events = update_cmd_events(scenario, event_filter)

    # Get load names that were not run at all or where observing was not run.
    # E.g. an SCS-107 near end of loads where next week vehicle loads only were
    # uplinked.
    not_run_loads = {}
    for not_run_type in ("Load", "Observing", "HRC"):
        not_run_loads[not_run_type] = {
            cmd_event["Params"]: cmd_event["Date"]
            for cmd_event in cmd_events
            if cmd_event["Event"] == f"{not_run_type} not run"
        }

    # Update loads table and download/archive backstop files from OCCweb
    loads = update_loads(scenario, lookback=lookback, stop_loads=stop_loads)
    logger.info(f"Including loads {', '.join(loads['name'])}")

    for load in loads:
        load_name = load["name"]
        loads_backstop_path = paths.LOADS_BACKSTOP_PATH(load_name)
        with gzip.open(loads_backstop_path, "rb") as fh:
            cmds: CommandTable = pickle.load(fh)
            if kadi_cmds_version() >= 3:
                cmds = add_scheduled_obsid_cmds(cmds)

        # Filter commands if loads (vehicle and/or observing) were approved but
        # never uplinked
        if load_name in not_run_loads["Load"]:
            # Keep only the orbit points
            cmds = cmds[cmds["type"] == "ORBPOINT"]
        elif load_name in not_run_loads["Observing"]:
            # Cut observing commands
            bad = np.isin(cmds["scs"], [131, 132, 133])
            cmds = cmds[~bad]
        elif load_name in not_run_loads["HRC"]:
            # Cut HRC state-changing commands
            date = not_run_loads["HRC"][load_name]
            bad = (
                (cmds["tlmsid"] == "224PCAON")
                | ((cmds["tlmsid"] == "COACTSX") & (cmds["coacts1"] == 134))
                | ((cmds["tlmsid"] == "COENASX") & (cmds["coenas1"] == 89))
                | ((cmds["tlmsid"] == "COENASX") & (cmds["coenas1"] == 90))
            ) & (cmds["date"] > date)
            cmds = cmds[~bad]

        if len(cmds) > 0:
            rltt = cmds.meta["rltt"] = cmds.get_rltt()
            if rltt is None and load_name not in not_run_loads["Load"]:
                # This is unexpected but press ahead anyway
                logger.error(f"No RLTT for {load_name=}")
            logger.info(f"Load {load_name} has {len(cmds)} commands with RLTT={rltt}")
            cmds_list.append(cmds)
        else:
            logger.info(f"Load {load_name} has no commands, skipping")

    # Filter events outside the time interval, assuming command event cannot
    # last more than 2 weeks.
    start_cmds = CxoTime(min(loads["cmd_start"]))
    stop_cmds = CxoTime(max(loads["cmd_stop"]))
    # Allow for variations in input format of date
    dates = np.array([CxoTime(date).date for date in cmd_events["Date"]], dtype=str)
    bad = (dates < (start_cmds - 14 * u.day).date) | (dates > stop_cmds.date)
    cmd_events = cmd_events[~bad]
    cmd_events_ids = [evt["Event"] + " at " + evt["Date"] for evt in cmd_events]
    if len(cmd_events) > 0:
        logger.info("Including cmd_events:\n  {}".format("\n  ".join(cmd_events_ids)))
    else:
        logger.info("No cmd_events to include")

    update_cmds_list_for_loads_in_backstop(cmd_events, cmds_list)

    for cmd_event in cmd_events:
        cmds = get_cmds_from_event(
            cmd_event["Date"], cmd_event["Event"], cmd_event["Params"]
        )

        # Events that do not generate commands return None from
        # get_cmds_from_event.
        if cmds is not None and len(cmds) > 0:
            cmds_list.append(cmds)

    # Sort cmds_list and rltts by the date of the first cmd in each cmds Table
    cmds_starts = np.array([cmds["date"][0] for cmds in cmds_list])
    idx_sort = np.argsort(cmds_starts)
    cmds_list = [cmds_list[ii] for ii in idx_sort]

    # Apply RLTT and any END SCS commands (CODISAXS) to loads. RLTT is applied
    # by stopping SCS's 128-133.
    for ii, cmds in enumerate(cmds_list):
        cmds_source = cmds["source"][0]
        if cmds_source == "CMD_EVT":
            cmds_source = f"CMD_EVT {cmds[0]['event']} at {cmds[0]['date']}"
        logger.info(f"Processing {cmds_source} with {len(cmds)} commands")
        end_scs = collections.defaultdict(list)
        if rltt := cmds.meta.get("rltt"):
            source = f"RLTT in {cmds['source'][0]}"
            end_scs[rltt, source].extend([128, 129, 130, 131, 132, 133])

        # Explicit END SCS commands. Most commonly these come from command events
        # like SCS-107, but can also be in weekly loads e.g. disabling an ACIS
        # ECS measurement in SCS-135.
        ok = cmds["tlmsid"] == "CODISASX"
        if np.any(ok):
            for cmd in cmds[ok]:
                if (scs := cmd["params"]["codisas1"]) >= 128:
                    source = f"DISABLE SCS in {cmd['source']} at {cmd['date']}"
                    end_scs[cmd["date"], source].append(scs)

        for (date_end, source), scss in end_scs.items():
            # Apply end SCS from these commands to the current running loads.
            # Remove commands with date greater than end SCS date. In most
            # cases this does not cut anything.
            for jj in range(ii):
                prev_cmds = cmds_list[jj]
                # First check for any overlap since prev_cmds is sorted by date.
                if len(prev_cmds) > 0 and prev_cmds["date"][-1] > date_end:
                    bad = (prev_cmds["date"] > date_end) & np.isin(
                        prev_cmds["scs"], scss
                    )
                    if np.any(bad):
                        n_bad = np.count_nonzero(bad)
                        logger.info(
                            f"Removing {n_bad} cmds in SCS slots {scss} from "
                            f"{prev_cmds['source'][0]} due to {source}"
                        )
                    cmds_list[jj] = prev_cmds[~bad]

        # If this command table has an RLTT then clip the SCHEDULED_STOP_TIME of all
        # previous command tables to RLTT.
        if rltt:
            for jj in range(ii):
                sched_stop_cmd = cmds_list[jj].get_scheduled_stop_time_cmd()
                if sched_stop_cmd and sched_stop_cmd["date"] > rltt:
                    sched_stop_cmd["params"]["scheduled_stop_time_orig"] = (
                        sched_stop_cmd["date"]
                    )
                    sched_stop_cmd["params"]["interrupt_load"] = cmds["source"][0]
                    sched_stop_cmd["date"] = rltt
                    sched_stop_cmd["time"] = CxoTime(rltt).secs

        if len(cmds) > 0:
            logger.info(f"Adding {len(cmds)} commands from {cmds_source}")

    cmds_recent: CommandTable = vstack_exact(cmds_list)
    cmds_recent.sort_in_backstop_order()
    cmds_recent.deduplicate_orbit_cmds()
    cmds_recent.remove_not_run_cmds()
    cmds_recent = add_obs_cmds(cmds_recent, pars_dict, rev_pars_dict)
    cmds_recent.meta["loads_start"] = start_cmds.date

    return cmds_recent


def add_obs_cmds(
    cmds: CommandTable,
    pars_dict,
    rev_pars_dict,
    *,
    prev_att=None,
    prev_obsid=None,
    prev_simpos=None,
):
    """Add 'type=LOAD_EVENT tlmsid=OBS' commands with info about observations.

    These "OBS" commands include the following params:
    - manvr_start: date of maneuver start
    - npnt_enab: auto-transition to NPNT enabled
    - obs_start: start date of NPNT corresponding to the command date
    - obs_stop: stop date of NPNT (subsequent AONMMODE or AONSMSAF command)
    - obsid: observation ID
    - prev_att: previous attitude (4-tuple of quaternion components)
    - simpos: SIM position
    - source: source of the observation (e.g. load name)
    - starcat_date: date of starcat command (if available)
    - starcat_idx: index of starcat in reversed params dict (if available)
    - targ_att: target attitude (4-tuple of quaternion components)

    Parameters
    ----------
    cmds_recent
        CommandTable
    pars_dict : dict
        Dictionary of parameters from the command table.
    rev_pars_dict : dict
        Dictionary of parameters from the command table, with keys reversed.
    prev_att : tuple
        Continuity attitude. If not supplied the first obs is dropped.
    prev_obsid : int, optional
        Previous obsid, default is -1.
    prev_simpos : int, optional
        Previous SIM position, default is -99616.

    Returns
    -------
    CommandTable
        Command table with added OBS commands
    """
    # Last command in cmds is the schedule stop time (i.e. obs_stop for the
    # last observation).
    schedule_stop_time = cmds["date"][-1]

    # Get the subset of commands needed to determine the state for OBS cmds like
    # maneuver commanding, obsid, starcat, etc.
    cmds_state = get_state_cmds(cmds)

    # Get a table of OBS cmds corresponding to the end of each maneuver. For the
    # first pass this only has maneuver info that is known at the point of
    # starting the maneuver.
    cmds_obs = get_cmds_obs_from_manvrs(cmds_state, prev_att)

    # Put the OBS cmds into the state cmds table and then do the second pass to
    # determine obsid, starcat, etc at the end of the maneuver (i.e. at the
    # start of the obs). This returns just the obs commands and updates
    # `pars_dict` and `rev_pars_dict` in place.
    cmds_state_obs = cmds_state.add_cmds(cmds_obs)
    cmds_obs = get_cmds_obs_final(
        cmds_state_obs,
        pars_dict,
        rev_pars_dict,
        schedule_stop_time,
        prev_obsid=prev_obsid,
        prev_simpos=prev_simpos,
    )

    # Finally add the OBS cmds to the recent cmds table.
    cmds_out = cmds.add_cmds(cmds_obs)
    return cmds_out


def get_state_cmds(cmds):
    """Get the state-changing commands need to create LOAD_EVENT OBS commands.

    Parameters
    ----------
    cmds
        CommandTable of input commands.

    Returns
    -------
    CommandTable of state-changing commands.
    """
    state_tlmsids = [
        "AOSTRCAT",
        "COAOSQID",
        "AOMANUVR",
        "AONMMODE",
        "AONSMSAF",
        "AOUPTARQ",
        "AONM2NPE",
        "AONM2NPD",
        "OBSID_SCH",
    ]

    if cmds["tlmsid"].dtype.kind == "S":
        vals = [val.encode("ascii") for val in state_tlmsids]
    ok = np.isin(cmds["tlmsid"], vals)
    ok |= cmds["type"] == "SIMTRANS"
    cmds = cmds[ok]
    cmds.sort(["date", "scs", "tlmsid"])

    # Note about sorting by 'tlmsid': this relates to an issue where the obsid
    # command COAOSQID is at exactly the same time as the AONMMODE commands
    # instead of just after it (e.g. in a null maneuver). In this case we need
    # to ensure that the AONMMODE commands are before the COAOSQID command in
    # the table order. The order of other commands is not important, so we rely
    # on the lucky fact that AONMMODE is alphabetically before COAOSQID to just
    # include `tlmsid`` in the lexical sort.
    #
    #  idx            date             type     tlmsid
    # ------ --------------------- ----------- --------
    #      6 2005:177:09:14:38.119  COMMAND_SW AOMANUVR
    #  10620 2005:177:09:17:27.868    MP_OBSID COAOSQID
    #  10621 2005:177:10:04:15.740    MP_OBSID COAOSQID
    #      1 2005:177:10:04:15.740  COMMAND_SW AONMMODE
    #      2 2005:177:10:04:15.997  COMMAND_SW AONM2NPE

    return cmds


def get_cmds_obs_from_manvrs(cmds, prev_att=None):
    """Get OBS commands corresponding to the end of each maneuver.

    This is the first pass of getting OBS commands, keying off each maneuver.
    These OBS commands are then re-inserted into the commands table at the point
    of the maneuver END for the second pass, where at the point in time of the
    maneuver end (obs start), all the state-changing commands have occurred.

    Parameters
    ----------
    cmds : CommandTable
        CommandTable of state-changing commands.
    prev_att : tuple
        Previous attitude (q1, q2, q3, q4)

    Returns
    -------
    CommandTable
        OBS commands.
    """
    targ_att = None
    npnt_enab = False

    cmds_obs = []
    times = CxoTime(cmds["date"]).secs

    for time, cmd in zip(times, cmds):
        tlmsid = cmd["tlmsid"]
        if tlmsid == "AOUPTARQ":
            pars = cmd["params"]
            targ_att = (pars["q1"], pars["q2"], pars["q3"], pars["q4"])
        elif tlmsid == "AONM2NPE":
            npnt_enab = True
        elif tlmsid == "AONM2NPD":
            npnt_enab = False
        elif tlmsid in ("AOMANUVR", "AONSMSAF"):
            if tlmsid == "AONSMSAF":
                targ_att = tuple(
                    get_nsm_attitude(prev_att or (0, 0, 0, 1), cmd["date"]).q.tolist()
                )
                npnt_enab = False
            if targ_att is None:
                # No target attitude is unexpected since we got to a MANVR cmd.
                logger.warning(f"WARNING: no target attitude for {cmd['date']}")
                log_context_obs(cmds, cmd)
                continue
            if prev_att is None:
                # No previous attitude happens always for first manvr at the
                # beginning of loads, and normally we just push on to the next
                # OBS. But if we already have OBS command this is a problem.
                if cmds_obs:
                    logger.warning(f"WARNING: No previous attitude for {cmd['date']}")
                    log_context_obs(cmds, cmd)
                prev_att = targ_att
                continue
            dur = manvr_duration(prev_att, targ_att)
            params = {
                "manvr_start": cmd["date"],
                "prev_att": prev_att,
                "targ_att": targ_att,
                "npnt_enab": npnt_enab,
            }
            cmd_obs = {
                "idx": -1,
                "type": "LOAD_EVENT",
                "tlmsid": "OBS",
                "scs": 0,
                "step": 0,
                "source": cmd["source"],
                "vcdu": -1,
                "time": time + dur,
                "params": params,
            }
            cmds_obs.append(cmd_obs)
            prev_att = targ_att
            targ_att = None

    cmds_obs = CommandTable(rows=cmds_obs)
    # NB: much faster to do this vectorized than when defining cmd_obs above.
    cmds_obs.add_column(CxoTime(cmds_obs["time"]).date, name="date", index=0)

    # If an NSM occurs within a maneuver then remove that obs
    nsms = cmds["tlmsid"] == "AONSMSAF"
    if np.any(nsms):
        bad_idxs = []
        for nsm_date in cmds["date"][nsms]:
            for ii, cmd in enumerate(cmds_obs):
                if nsm_date > cmd["params"]["manvr_start"] and nsm_date <= cmd["date"]:
                    logger.info(
                        f"NSM at {nsm_date} happened during maneuver preceding"
                        f" obs\n{cmd}"
                    )
                    log_context_obs(cmds, cmd, log_level="info")
                    bad_idxs.append(ii)
        if bad_idxs:
            logger.info(f"Removing obss at {bad_idxs}")
            cmds_obs.remove_rows(bad_idxs)

    # This is known to happen for cmds in AUG0103A, JAN2204B, JUL2604D due to
    # non-load maneuvers that are not captured correctly. Run the v1-v2
    # migration script to see. The prev_att is wrong and the incorrect maneuver
    # timing results in overlaps. It should not happen for modern loads.
    for cmd0, cmd1 in zip(cmds_obs[:-1], cmds_obs[1:]):
        if cmd1["params"]["manvr_start"] <= cmd0["date"] < cmd1["date"]:
            logger.warning(f"WARNING: fixing overlapping OBS cmds: \n{cmd0} \n{cmd1}")
            date0 = CxoTime(cmd1["params"]["manvr_start"]) - 15 * u.s
            cmd0["date"] = date0.date
            cmd0["time"] = date0.secs

    return cmds_obs


# parameters from characteristics_Hardware for normal maneuvers
MANVR_DELTA = 60.0
MANVR_ALPHAMAX = 2.18166e-006  # AKA alpha
MANVR_VMAX = 0.001309  # AKA omega


def manvr_duration(q1, q2):
    """Calculate the duration of a maneuver from two quaternions

    This is basically the same as the function in chandra_maneuver but
    optimized to work on a single 4-vector quaterion.

    Parameters
    ----------
    q1
        list of 4 quaternion elements
    q2
        list of 4 quaternion elements

    Returns
    -------
    duration of maneuver in seconds
    """
    # Compute 4th element of delta quaternion q_manvr = q2 / q1
    q_manvr_3 = abs(-q2[0] * q1[0] - q2[1] * q1[1] - q2[2] * q1[2] + q2[3] * -q1[3])

    # 4th component is cos(theta/2)
    if q_manvr_3 > 1:  # noqa: PLR1730
        q_manvr_3 = 1
    phimax = 2 * math.acos(q_manvr_3)

    epsmax = MANVR_VMAX / MANVR_ALPHAMAX - MANVR_DELTA
    tau = phimax / MANVR_VMAX - epsmax - 2 * MANVR_DELTA
    if tau >= 0:
        eps = epsmax
    else:
        tau = 0
        eps = (
            np.sqrt(MANVR_DELTA**2 + 4 * phimax / MANVR_ALPHAMAX) / 2
            - 1.5 * MANVR_DELTA
        )
        if eps < 0:  # noqa: PLR1730
            eps = 0

    tm = 4 * MANVR_DELTA + 2 * eps + tau
    return tm


def get_cmds_obs_final(
    cmds,
    pars_dict,
    rev_pars_dict,
    schedule_stop_time,
    *,
    prev_obsid=None,
    prev_simpos=None,
):
    """Fill in the rest of params for each OBS command.

    Second pass of processing which implements a state machine to fill in the
    other parameters like obsid, starcat, simpos etc. for each OBS command.

    The observation state is completed by encountering the next transition to
    NMM or NSM.

    Parameters
    ----------
    cmds
        CommandTable of state-changing + OBS commands.
    pars_dict
        dict of parameters for commands
    rev_pars_dict
        reversed dict of parameters for commands
    schedule_stop_time
        date of last command
    prev_obsid : int, optional
        Previous obsid, default is -1.
    prev_simpos : int, optional
        Previous SIM position, default is -99616.

    Returns
    -------
    CommandTable
        OBS commands with all parameters filled in.
    """
    # Initialize state variables. Note that the first OBS command may end up
    # with bogus values for some of these, but this is OK because of the command
    # merging with the archive commands which are always correct. Nevertheless
    # use values that are not None to avoid errors. For `sim_pos`, if the SIM
    # has not been commanded in a long time then it will be at -99616.
    obsid = -1 if prev_obsid is None else prev_obsid
    obsid_sched = -1 if prev_obsid is None else prev_obsid
    starcat_idx = None
    starcat_date = None
    sim_pos = -99616 if prev_simpos is None else prev_simpos
    obs_params = None
    cmd_obs_extras = []

    for cmd in cmds:
        tlmsid = cmd["tlmsid"]
        if tlmsid == "AOSTRCAT":
            starcat_date = cmd["date"]
            if cmd["idx"] == -1:
                # OBS command only stores the index of the starcat params, so at
                # this point we need to put those params into the pars_dict and
                # rev_pars_dict and get the index.
                starcat_idx = get_par_idx_update_pars_dict(
                    pars_dict, cmd, rev_pars_dict=rev_pars_dict
                )
            else:
                starcat_idx = cmd["idx"]

        elif tlmsid == "OBSID_SCH":
            obsid_sched = cmd["params"]["id"]

        elif tlmsid == "COAOSQID":
            obsid = cmd["params"]["id"]
            # Look for obsid change within obs, likely an undercover
            # (target="cold blank ECS"). First stop the initial obs at the time
            # of the obsid command.
            if obs_params is not None:
                obs_params["obs_stop"] = cmd["date"]
                # Then start a new obs at the time of the next obsid command.
                obs_params = obs_params.copy()
                obs_params["obsid"] = obsid
                obs_params["obs_start"] = cmd["date"]
                # Collect these extra obs to a list to be added to cmds table later.
                cmd_obs = {
                    "idx": -1,
                    "date": cmd["date"],
                    "type": "LOAD_EVENT",
                    "tlmsid": "OBS",
                    "scs": 0,
                    "step": 0,
                    "time": CxoTime(cmd["date"]).secs,
                    "source": cmd["source"],
                    "vcdu": -1,
                    "params": obs_params,
                }
                cmd_obs_extras.append(cmd_obs)
        elif tlmsid == "OBS":
            obs_params = cmd["params"]
            obs_params["obsid"] = obsid
            if kadi_cmds_version() >= 3:
                obs_params["obsid_sched"] = obsid_sched
            obs_params["simpos"] = sim_pos  # matches states 'simpos'
            obs_params["obs_start"] = cmd["date"]
            if obs_params["npnt_enab"]:
                if starcat_idx is not None and starcat_date is not None:
                    obs_params["starcat_idx"] = starcat_idx
                    obs_params["starcat_date"] = starcat_date
                else:
                    logger.info(
                        f"No starcat for obsid {obsid} at {cmd['date']} "
                        "even though npnt_enab is True"
                    )

        elif tlmsid in ("AONMMODE", "AONSMSAF") or (
            tlmsid == "AONM2NPE" and cmd["vcdu"] == -1
        ):
            # Stop current obs at next maneuver. In some v1 commands (for
            # migration) there are cases of non-load maneuvers without an
            # AONMMODE command and only the AONM2NPE and AOMANUVR commands.
            # In those cases just use ANOM2NPE as a proxy. In v2 commands this
            # is OK since AONM2NPE always comes after the AONMMODE command and
            # obs_params will be None at that point.
            if obs_params is not None:
                obs_params["obs_stop"] = cmd["date"]
            # This closes out the observation. Reset the star catalog state
            # variables since a valid star catalog is always uniquely associated
            # with an pointing observation.
            obs_params = None
            starcat_idx = None
            starcat_date = None

        elif cmd["type"] == "SIMTRANS":
            sim_pos = cmd["params"]["pos"]

    # Filter down to just the observation commands
    cmds_obs = cmds[cmds["tlmsid"] == "OBS"]

    # Potentially add extra obs commands (typically undercovers) to cmds table
    if cmd_obs_extras:
        cmds_obs = cmds_obs.add_cmds(CommandTable(cmd_obs_extras))

    # The last OBS command will be missing obs_stop since there is no
    # subsequent maneuver. Fix that for the expected case but warn if an obs
    # before the end is missing obs_stop.
    for cmd in cmds_obs:
        if "obs_stop" not in cmd["params"]:
            cmd["params"]["obs_stop"] = schedule_stop_time
            if cmd.index != len(cmds_obs) - 1:
                logger.warning(f"OBS command missing obs_stop\n{cmd}")
                log_context_obs(cmds, cmd)
        cmd["idx"] = get_par_idx_update_pars_dict(
            pars_dict, cmd, rev_pars_dict=rev_pars_dict
        )

    return cmds_obs


def log_context_obs(cmds, cmd, before=3600, after=3600, log_level="warning"):
    """Log commands before and after ``cmd``"""
    date_before = (CxoTime(cmd["date"]) - before * u.s).date
    date_after = (CxoTime(cmd["date"]) + after * u.s).date
    ok = (cmds["date"] >= date_before) & (cmds["date"] <= date_after)
    log_func = getattr(logger, log_level)
    log_func(f"\n{cmds[ok]}")


def is_google_id(scenario):
    """Return True if scenario appears to be a Google ID.

    Parameters
    ----------
    scenario
        str, None

    Returns
    -------
    bool
    """
    # Something better??
    return scenario is not None and len(scenario) > 35


def update_cmd_events(
    scenario=None,
    event_filter: Callable | list[Callable] | None = None,
) -> Table:
    """Update local cmd_events.csv from Google Sheets and read events for ``scenario``.

    For local scenarios, the cmd_events.csv file is read directly from the CSV file in
    ``paths.CMD_EVENTS_PATH(scenario)``. For the "flight" or None scenario, the
    cmd_events are read from the Google Sheet and then written to that local file.

    No event filtering is done prior to writing events from a sheet to scenario file.

    After reading the events, the events are filtered based on the "State" column. By
    default, only events with a "State" of "Predictive" or "Definitive" are included. If
    the configuration parameter ``conf.include_in_work_command_events`` is True, then
    events with a "State" of "In-work" are also included.

    Then the events are filtered based on the ``CXOTIME_NOW`` environment variable. If
    this environment variable is set, then events with a date after the value of
    ``CXOTIME_NOW`` are filtered out.

    Finally, events are filtered by `event_filter` if it is provided.

    Parameters
    ----------
    scenario : str, None
        Scenario name.
    event_filter : Callable, list of callable, None
        Callable function or list of callable functions that takes an Event Table as
        input and returns a boolean mask with same length as Table. This is used to
        select rows from the Table. If None, no filtering is done.

    Returns
    -------
    Table
        Filtered command events table for scenario.
    """
    if event_filter is None:
        event_filters = []
    elif callable(event_filter):
        event_filters = [event_filter]
    else:
        event_filters = list(event_filter)

    # Get sheet doc ids for scenario, or [] if not applicable
    doc_ids = get_sheet_doc_ids_for_scenario(scenario)
    if not doc_ids or not HAS_INTERNET:
        cmd_events = get_cmd_events_from_local(scenario)
    else:
        cmd_events = get_cmd_events_from_sheet(scenario, doc_ids)

    # Filter table based on State column (if available)
    if "State" in cmd_events.colnames:
        event_filters.append(filter_cmd_events_state)

    # If CXOTIME_NOW is set, filter out events after that date.
    if cxotime_now := get_cxotime_now():
        event_filters.append(filter_cmd_events_date_stop(cxotime_now))

    if event_filters:
        ok = np.ones(len(cmd_events), dtype=bool)
        for event_filter_func in event_filters:
            ok &= event_filter_func(cmd_events)
        cmd_events = cmd_events[ok]

    return cmd_events


def get_cmd_events_from_sheet(scenario: str | None, doc_ids: list[str]) -> Table:
    """
    Fetch command events from Google Sheet for ``scenario`` and write to a CSV file.

    Parameters
    ----------
    scenario : str, None
        Scenario identifier, which can be a Google Sheet ID or one of the special
        scenarios ("flight", "custom", "flight+custom", or None).
    doc_ids : list[str]
        Google sheet doc ids from which to read events, corresponding to the scenario.

    Returns
    -------
    Table
        Command events Table.

    Raises
    ------
    ValueError
        If the request to fetch the Google Sheet fails.
    """
    cmd_events_list = []
    for doc_id in doc_ids:
        # Fetch the command events from the Google Sheet URL(s).
        cmd_events = read_cmd_events_from_sheet(doc_id)
        cmd_events_list.append(cmd_events)

    # Combine command events from multiple documents if necessary.
    cmd_events = (
        cmd_events_list[0]
        if len(cmd_events_list) == 1
        else vstack_exact(cmd_events_list)
    )

    # Ensure the scenario directory exists and write file
    paths.SCENARIO_DIR(scenario).mkdir(parents=True, exist_ok=True)
    cmd_events_path = paths.CMD_EVENTS_PATH(scenario)
    logger.info(f"Writing {len(cmd_events)} cmd_events to {cmd_events_path}")
    cmd_events.write(cmd_events_path, format="csv", overwrite=True)

    return cmd_events


def get_sheet_doc_ids_for_scenario(scenario) -> list[str]:
    """Get Google sheet doc ids for a given scenario.

    Parameters
    ----------
    scenario : str, None
        Scenario name.

    Returns
    -------
    list[str]
        List of Google sheet document IDs. This will be an empty list if no
        Google sheet(s) are associated with the scenario.
    """
    if scenario in ("flight", None):
        doc_ids = [conf.cmd_events_flight_id]
    elif scenario == "custom":
        doc_ids = [conf.cmd_events_custom_id]
    elif scenario == "flight+custom":
        doc_ids = [conf.cmd_events_flight_id, conf.cmd_events_custom_id]
    elif is_google_id(scenario):
        doc_ids = [scenario]
    else:
        doc_ids = []
    return doc_ids


def read_cmd_events_from_sheet(doc_id):
    url = CMD_EVENTS_SHEET_URL.format(doc_id=doc_id)
    logger.info(f"Getting cmd_events from {url}")
    req = retry_func(requests.get)(url, timeout=5)
    if req.status_code != 200:
        raise ValueError(f"Failed to get cmd events sheet: {req.status_code} for {url}")

    cmd_events = Table.read(
        req.text, format="csv", fill_values=[], converters={"Params": str}
    )
    # Remove blank rows from Google sheet export
    ok = [cmd_event["Date"].strip() != "" for cmd_event in cmd_events]
    cmd_events = cmd_events[ok]
    return cmd_events


def get_cmd_events_from_local(scenario=None):
    """Get local cmd_events.csv for ``scenario``.

    Parameters
    ----------
    scenario : str, None
        Scenario name

    Returns
    -------
    Table
        Command events table
    """
    cmd_events_path = paths.CMD_EVENTS_PATH(scenario)
    logger.info(f"Reading command events {cmd_events_path}")
    cmd_events = Table.read(
        str(cmd_events_path), format="csv", fill_values=[], converters={"Params": str}
    )
    return cmd_events


def update_loads(scenario=None, *, lookback=None, stop_loads=None) -> Table:
    """Update local copy of approved command loads though ``lookback`` days."""
    if lookback is None:
        lookback = conf.default_lookback

    dt = 21 * u.day
    cxotime_now = get_cxotime_now()
    # This is either the true current time or else the mock time from CXOTIME_NOW.
    stop = CxoTime(stop_loads or cxotime_now)
    start = stop - lookback * u.day
    # Find loads out to 21 days into the future by default (in the case where
    # stop is the current time).
    if stop_loads is None and cxotime_now is None:
        stop += dt

    # Ensure the scenario directory exists
    paths.SCENARIO_DIR(scenario).mkdir(parents=True, exist_ok=True)

    # TODO for performance when we have decent testing:
    # Read in the existing loads table and grab the RLTT and scheduled stop
    # dates. Those are the only reason we need to read an existing set of
    # commands that are already locally available, but they are fixed and cannot
    # change. The only thing that might change is an interrupt time, e.g. if
    # the SCS time gets updated.
    # So maybe read in the loads table and make a dict of rltts and ssts keyed
    # by load name and use those to avoid reading in the cmds table.
    # For now just get things working reliably.

    loads_rows = []

    # Probably too complicated, but this bit of code generates a list of dates
    # that are guaranteed to sample all the months in the lookback period with
    # two weeks of margin on the tail end.
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
            contents = occweb.get_occweb_dir(dir_year_month, timeout=5)
        except requests.exceptions.HTTPError as exc:
            if str(exc).startswith("404"):
                logger.debug(f"No OCCweb directory for {dir_year_month}")
                continue
            else:
                raise

        # Find each valid load name in the directory listing and process:
        # - Find and download the backstop file
        # - Parse the backstop file and save to the loads/ archive as a gzipped
        #   pickle file
        # - Add the load to the table
        for content in contents:
            if re.match(r"[A-Z]{3}\d{4}[A-Z]/", content["Name"]):
                load_name = content["Name"][:8]  # chop the /
                load_date = load_name_to_cxotime(load_name)
                if load_date < RLTT_ERA_START:
                    logger.warning(
                        f"Skipping {load_name} which is before "
                        f"{RLTT_ERA_START} start of RLTT era"
                    )
                    continue
                if load_date >= start and load_date <= stop:
                    cmds = get_load_cmds_from_occweb_or_local(dir_year_month, load_name)
                    load = {
                        "name": load_name,
                        "cmd_start": cmds["date"][0],
                        "cmd_stop": cmds["date"][-1],
                    }
                    loads_rows.append(load)

    if not loads_rows:
        raise ValueError(f"No loads found in {lookback} days")

    # Finally, save the table to file
    loads_table = Table(loads_rows)

    if conf.clean_loads_dir:
        clean_loads_dir(loads_table)

    return loads_table


def clean_loads_dir(loads):
    """Remove load-like files from loads directory if not in ``loads``"""
    for file in Path(paths.LOADS_ARCHIVE_DIR()).glob("*.pkl.gz"):
        if (
            re.match(r"[A-Z]{3}\d{4}[A-Z]\.pkl\.gz", file.name)
            and file.name[:8] not in loads["name"]
        ):
            logger.info(f"Removing load file {file}")
            file.unlink()


def get_load_cmds_from_occweb_or_local(
    dir_year_month=None,
    load_name=None,
    *,
    in_work=False,
) -> CommandTable:
    """Get the load cmds (backstop) for ``load_name`` within ``dir_year_month``

    If the backstop file is already available locally, use that. Otherwise, the file is
    downloaded from OCCweb and is then parsed and saved as a gzipped pickle file of the
    corresponding CommandTable object.

    Parameters
    ----------
    dir_year_month : Path
        Path to the directory containing the ``load_name`` directory.
    load_name : str
        Load name in the usual format e.g. JAN0521A.
    in_work : bool
        If True then the backstop file is in-work and should not be saved in the command
        loads archive (~/.kadi/loads by default).

    Returns
    -------
    CommandTable
        Backstop commands for the load.
    """
    # Determine archived local gzip file name
    cmds_filename = paths.LOADS_ARCHIVE_DIR() / f"{load_name}.pkl.gz"

    if not in_work and cmds_filename.exists():
        # If the output file already exists, read the commands and return them.
        logger.info(f"Already have {cmds_filename}")
        with gzip.open(cmds_filename, "rb") as fh:
            cmds = pickle.load(fh)
    else:
        # Find the backstop file on OCCweb
        load_dir_contents = occweb.get_occweb_dir(dir_year_month / load_name, timeout=5)
        for filename in load_dir_contents["Name"]:
            if re.match(r"CR\d{3}.\d{4}\.backstop", filename):
                break
        else:
            raise ValueError(
                f"Could not find backstop file in {dir_year_month / load_name}"
            )
        # Download the backstop file from OCCweb
        backstop_text = occweb.get_occweb_page(
            dir_year_month / load_name / filename,
            cache=conf.cache_loads_in_astropy_cache,
            timeout=10,
        )
        cmds = parse_backstop(load_name, backstop_text)
        if not in_work:
            write_backstop(cmds, cmds_filename)

    if kadi_cmds_version() >= 3:
        cmds = add_scheduled_obsid_cmds(cmds)

    return cmds


def parse_backstop(load_name: str, backstop_text: str):
    """Parse ``backstop_text`.

    This sets the ``source`` column to ``load_name`` and removes the
    ``timeline_id`` column.
    """
    backstop_lines = backstop_text.splitlines()
    cmds = get_cmds_from_backstop(backstop_lines)

    # Fix up the commands to be in the right format
    idx = cmds.colnames.index("timeline_id")
    cmds.add_column(load_name, index=idx, name="source")
    del cmds["timeline_id"]

    return cmds


def add_scheduled_obsid_cmds(cmds: CommandTable) -> CommandTable:
    """Add OBSID commands in vehicle loads corresponding to COAOSQID commands.

    This allows tracking of the scheduled OBSID in the event of an SCS-107 where the
    original COAOSQID commands in the observing loads are dropped. The load event cmds
    are placed in the vehicle loads so they get stopped if vehicle loads are stopped
    by NSM etc.

    Parameters
    ----------
    cmds : CommandTable
        CommandTable of load commands

    Returns
    -------
    CommandTable
        New CommandTable with original table and added OBSID load event commands.
    """
    cmds_obsid = cmds[cmds["tlmsid"] == "COAOSQID"]
    cmds_obsid["type"] = "LOAD_EVENT"
    cmds_obsid["tlmsid"] = "OBSID_SCH"
    cmds_obsid["scs"] -= 3
    logger.info(f"Adding {len(cmds_obsid)} OBSID load event commands")

    return cmds.add_cmds(cmds_obsid)


def write_backstop(cmds: CommandTable, cmds_filename: str | Path):
    """Write CommandTable to a gzipped pickle file.

    This function saves a CommandTable object to disk as a compressed pickle file.
    ``cmds_filename`` is normally the kadi loads archive directory.

    It also creates the parent directories as needed.

    Parameters
    ----------
    cmds : CommandTable
        CommandTable object containing the backstop commands to be saved.
    cmds_filename : str or Path
        Path to the output file where the CommandTable will be saved. The file
        will be created with gzip compression.

    """
    logger.info(f"Saving {cmds_filename}")
    Path(cmds_filename).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cmds_filename, "wb") as fh:
        pickle.dump(cmds, fh)


def update_cmds_archive(
    *,
    lookback=None,
    stop=None,
    log_level=logging.INFO,
    scenario=None,
    data_root=".",
):
    """Update cmds2.h5 and cmds2.pkl archive files.

    This updates the archive though ``stop`` date, where is required that the
    ``stop`` date is within ``lookback`` days of existing data in the archive.

    By default this updates the latest version of the archive. This can be changed by
    setting the KADI_CMDS_VERSION environment variable. Note that the version is
    cached via kadi.commands.core.kadi_cmds_version().

    Parameters
    ----------
    lookback : int, None
        Number of days to look back to get recent load commands from OCCweb.
        Default is ``conf.default_lookback`` (currently 30).
    stop : CxoTime-like, None
        Stop date to update the archive to. Default is that future loads and command
        events are included.
    log_level : int
        Logging level. Default is ``logging.INFO``.
    scenario : str, None
        Scenario name for loads and command events
    data_root : str, Path
        Root directory where cmds2.h5 and cmds2.pkl are stored. Default is '.'.
    match_prev_cmds : bool
        One-time use flag set to True to update the cmds archive near the v1/v2
        transition of APR1420B. See ``utils/migrate_cmds_to_cmds2.py`` for
        details.
    """
    # For testing allow override of default `stop` value
    if stop is None:
        stop = get_cxotime_now()

    # Local context manager for log_level and data_root
    kadi_logger = logging.getLogger("kadi")
    log_level_orig = kadi_logger.level
    with conf.set_temp("commands_dir", data_root), temp_env_var("KADI", data_root):
        try:
            kadi_logger.setLevel(log_level)
            _update_cmds_archive(lookback, stop, scenario, data_root)
        finally:
            kadi_logger.setLevel(log_level_orig)


def _update_cmds_archive(lookback, stop_loads, scenario, data_root):
    """Do the real work of updating the cmds archive"""
    # Either no-match-prev-cmds or matching RLTT start disables matching previous cmds
    match_prev_cmds = not (conf.no_match_prev_cmds or conf.match_from_rltt_start)

    idx_cmds_path = Path(data_root) / "cmds2.h5"
    pars_dict_path = Path(data_root) / "cmds2.pkl"

    if idx_cmds_path.exists():
        cmds_arch = load_idx_cmds(version=2, file=idx_cmds_path)
        pars_dict = load_pars_dict(version=2, file=pars_dict_path)
    else:
        # Make an empty cmds archive table and pars dict
        cmds_arch = CommandTable(
            names=list(CommandTable.COL_TYPES),
            dtype=list(CommandTable.COL_TYPES.values()),
        )
        del cmds_arch["timeline_id"]
        pars_dict = {}
        match_prev_cmds = False  # No match of previous commands since there are none.

    cmds_recent = update_cmd_events_and_loads_and_get_cmds_recent(
        scenario=scenario,
        stop_loads=stop_loads,
        lookback=lookback,
        pars_dict=pars_dict,
    )

    if match_prev_cmds:
        idx0_arch, idx0_recent = get_matching_block_idx(cmds_arch, cmds_recent)
    else:
        idx0_recent = 0
        if conf.match_from_rltt_start:
            # Special case for reprocessing the commands archive starting at the RLTT
            # era from load RLTT_ERA_START_LOAD (APR1420B). Find the index of the first
            # command in these loads. Note np.argmax() returns the first matching index
            # in the case of multiple matches.
            idx0_arch = np.argmax(cmds_arch["source"] == RLTT_ERA_START_LOAD)
            logger.info(
                f"Matching from RLTT start load {RLTT_ERA_START_LOAD}, "
                f"idx0_arch={idx0_arch}, idx0_recent={idx0_recent}"
            )
        else:
            # Append to end of existing cmds archive file
            idx0_arch = len(cmds_arch)

    # Convert from `params` col of dicts to index into same params in pars_dict.
    for cmd in cmds_recent:
        cmd["idx"] = get_par_idx_update_pars_dict(pars_dict, cmd)

    # If the length of the updated table will be the same as the existing table.
    # For the command below the no-op logic should be clear:
    # cmds_arch = vstack([cmds_arch[:idx0_arch], cmds_recent[idx0_recent:]])
    if idx0_arch == len(cmds_arch) and idx0_recent == len(cmds_recent):
        logger.info(
            f"No new commands found, skipping writing {idx_cmds_path.absolute()}"
        )
        return

    # Merge the recent commands with the existing archive.
    logger.info(
        f"Appending {len(cmds_recent) - idx0_recent} new commands after "
        f"removing {len(cmds_arch) - idx0_arch} from existing archive"
    )
    logger.info(
        f" starting with cmds_arch[:{idx0_arch}] and adding "
        f"cmds_recent[{idx0_recent}:{len(cmds_recent)}]"
    )

    # Remove params column before stacking and saving
    del cmds_recent["params"]
    del cmds_arch["params"]

    # Save the updated archive and pars_dict.
    cmds_arch_new = vstack_exact([cmds_arch[:idx0_arch], cmds_recent[idx0_recent:]])
    logger.info(f"Writing {len(cmds_arch_new)} commands to {idx_cmds_path.absolute()}")
    cmds_arch_new.write(
        str(idx_cmds_path.absolute()), path="data", format="hdf5", overwrite=True
    )

    logger.info(f"Writing updated pars_dict to {pars_dict_path.absolute()}")
    pickle.dump(pars_dict, open(pars_dict_path.absolute(), "wb"))


def get_matching_block_idx(cmds_arch, cmds_recent):
    # Find place in archive where the recent commands start.
    idx_arch_recent = cmds_arch.find_date(cmds_recent["date"][0])
    cmds_arch_recent = cmds_arch[idx_arch_recent:]
    cmds_arch_recent.rev_pars_dict = weakref.ref(REV_PARS_DICT)
    logger.info("Selecting commands from cmds_arch[{}:]".format(idx_arch_recent))

    arch_vals = get_list_for_matching(cmds_arch_recent)
    recent_vals = get_list_for_matching(cmds_recent)

    diff = difflib.SequenceMatcher(a=arch_vals, b=recent_vals, autojunk=False)

    matching_blocks = diff.get_matching_blocks()
    opcodes = diff.get_opcodes()
    # Find the first matching block that is sufficiently long
    logger.info("Matching blocks for (a) recent commands and (b) existing HDF5")
    for block in matching_blocks:
        logger.info("  {}".format(block))
        if block.size > conf.matching_block_size:
            logger.info(
                f"Found matching block of size {block.size} > {conf.matching_block_size}"
            )
            break
    else:
        logger.info("Diffs between (a) recent commands and (b) existing HDF5")
        for opcode in opcodes:
            logger.info("  {}".format(opcode))

        raise ValueError(
            f"No matching blocks at least {conf.matching_block_size} long. This most likely "
            "means that you have not recently synced your local Ska data using `ska_sync`."
        )

    # Index into idx_cmds at the end of the large matching block.  block.b is the
    # beginning of the match.
    idx0_recent = block.b + block.size
    idx0_arch = idx_arch_recent + block.a + block.size

    return idx0_arch, idx0_recent


def get_list_for_matching(cmds: CommandTable) -> list[tuple]:
    # Define the column names that specify a complete and unique row
    keys = ("date", "type", "tlmsid", "scs", "step", "source", "vcdu")
    rows = []
    for cmd in cmds:
        row = tuple(
            cmd[key].decode("ascii") if isinstance(cmd[key], bytes) else str(cmd[key])
            for key in keys
        )
        # Special case for OBS command. The obs_stop param can change during anomaly
        # recovery when there is a CMD_EVT maneuver but no subsequent maneuver to define
        # obs_stop. In theory we could include these params in the match tuple for every
        # command in the matching, but it turns out that (for reasons I don't fully
        # understand) the AOSTRCAT yang and zang params are very slightly different at
        # floating point precision level, so the matches fail. Since all the other
        # commands are not mutable in this way we just apply this for OBS commands.
        if cmd["tlmsid"] == "OBS":
            row_params = tuple(
                (key, cmd["params"][key]) for key in sorted(cmd["params"])
            )
            row += row_params
        rows.append(row)

    return rows


TYPE_MAP = ["ACQ", "GUI", "BOT", "FID", "MON"]
IMGSZ_MAP = ["4x4", "6x6", "8x8"]
PAR_MAPS = [
    ("imnum", "slot", int),
    ("type", "type", lambda n: TYPE_MAP[n]),
    ("imgsz", "sz", lambda n: IMGSZ_MAP[n]),
    ("maxmag", "maxmag", float),
    ("yang", "yang", lambda x: np.degrees(x) * 3600),
    ("zang", "zang", lambda x: np.degrees(x) * 3600),
    ("dimdts", "dim", int),
    ("restrk", "res", int),
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

    Parameters
    ----------
    params
        dict of AOSTRCAT parameters

    Returns
    -------
    ACATable
    """
    from proseco.catalog import ACATable

    for idx in range(1, 17):
        if params[f"minmag{idx}"] == params[f"maxmag{idx}"] == 0:
            break

    max_idx = idx
    cols = defaultdict(list)
    for par_name, col_name, func in PAR_MAPS:
        for idx in range(1, max_idx):
            cols[col_name].append(func(params[par_name + str(idx)]))

    out = ACATable(cols)
    out.add_column(np.arange(1, max_idx), index=1, name="idx")

    return out
