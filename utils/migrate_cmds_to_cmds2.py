# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Functions to migrate from the V1 cmds archive to V2.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import ska_dbi
from astropy.table import Column, Table, vstack
from cxotime import CxoTime

from kadi import logger
from kadi.commands.commands_v2 import (
    add_obs_cmds,
    get_load_cmds_from_occweb_or_local,
    update_cmds_archive,
)
from kadi.commands.core import (
    get_cmds_from_backstop,
    get_par_idx_update_pars_dict,
    ska_load_dir,
)

logger.setLevel(1)


SKA = Path(os.environ["SKA"])
CMD_STATES_PATH = SKA / "data" / "cmd_states" / "cmd_states.db3"


CMDS_V2_START = "APR2020A"


def make_cmds2(start=None, stop=None, step=100):
    """Make initial cmds2.h5 and cmds2.pkl between ``start`` and ``stop``.

    This first converts the v1 archive to v2 format up through CMDS_V2_START.
    Then it does v2 update_cmds_archive every ``step`` days through ``stop``.

    Running with the default step of one year is efficient. For testing it can
    be useful to run with a step size of 7 days to simulate weekly updates.

    Example in ipython::

      # Optional setup for speed if doing this repeatedly
      >>> from kadi.commands import conf
      >>> conf.cache_loads_in_astropy_cache = True

      >>> %run -i utils/migrate_cmds_to_cmds2.py
      >>> make_cmds2()
    """
    migrate_cmds1_to_cmds2(start)

    # Start the V2 updates a week and a day after CMDS_V2_START
    date = CxoTime("2020-04-28")
    stop = CxoTime(stop)
    while date < stop:
        logger.info("*" * 80)
        logger.info(f"Updating cmds2 to {date}")
        logger.info("*" * 80)
        update_cmds_archive(stop=date, lookback=step + 30)
        date += step

    # Final catchup to `stop`
    update_cmds_archive(stop=stop, lookback=step + 30)


def migrate_cmds1_to_cmds2(start=None):
    """Migrate the legacy cmds.h5 through APR1320A to the new cmds2.h5 format.

    Key updates:
    - Migrating from timeline_id to source, which is either the load
      name or "CMD_EVT" for commands from the event table.
    - Add star catalog AOSTRCAT params in bytes-encoded form to pars dict.

    This create ``cmds2.h5`` and ``cmds2.pkl`` in the current directory.

    This includes commands prior to APR1420B, which is the first load of the
    RLTT era that includes LOAD_EVENT commands. This function should mostly be
    used with make_cmds2(), but for example after generating cmds2.h5 and
    cmds2.pkl with this, run::

       >>> %run utils/migrate_cmds_to_cmds2.py
       >>> migrate_cmds_to_cmds2()
       >>> from kadi.commands.commands_v2 import update_cmds_archive
       >>> update_cmds_archive(stop='2020-04-28', match_prev_cmds=True)

    After this running the ``update_cmds_archive`` command as normal will work.

    :param start: CxoTime-like, None
        Start date in existing loads to start at. Used in debugging.
    """
    # Load V1 cmds, being explicit about the file in case KADI is set for testing.
    if "KADI" in os.environ:
        raise ValueError("Cannot have KADI environment variable set")
    from kadi.commands import commands_v1

    cmds = commands_v1.get_cmds(start)

    # Make a local copy of cmds params dicts since the processing here updates
    # them in place. Only `pars_dict` gets written but both are used.
    pars_dict = commands_v1.PARS_DICT._val.copy()
    rev_pars_dict = commands_v1.REV_PARS_DICT._val.copy()

    # This code is to get the load name ("source") for each cmd
    with ska_dbi.DBI(dbi="sqlite", server=str(CMD_STATES_PATH)) as db:
        timelines = db.fetchall("""SELECT * from timelines""")
    timelines = Table(timelines)

    # Make a dict to translate from each timeline_id to the load name
    timeline_id_to_load_name = {0: "CMD_EVT"}
    for timeline in timelines:
        # dir looks like /2002/JAN0702/oflsd/
        load_name = timeline["dir"][6:13] + timeline["dir"][-2].upper()
        timeline_id_to_load_name[timeline["id"]] = load_name

    # Collect the sources (load names) represented in the cmds.h5, and read
    # each of the backstop files for each load
    sources = []
    starcat_cmds = {}
    for cmd in cmds:
        load_name = timeline_id_to_load_name[cmd["timeline_id"]]
        if load_name != "CMD_EVT" and load_name not in starcat_cmds:
            load_cmds = get_load_cmds_from_occweb_or_local(
                load_name=load_name, use_ska_dir=True
            )
            ok = load_cmds["tlmsid"] == "AOSTRCAT"
            starcat_cmds[load_name] = load_cmds[ok]
        sources.append(load_name)

    # For V1 provenance was provided by timeline_id, replace with source for V2.
    col_index = cmds.colnames.index("timeline_id")
    cmds.add_column(Column(sources, name="source", dtype="S8"), index=col_index)
    del cmds["timeline_id"]

    # Fix AONSMSAF in V1 at 2008:225:10:00:00.000 which was actually at
    # 2008:225.10:07:13.600. This was from a CTU reset during the maneuver but
    # the maneuver finished and NSM was because the ACA CCD warmed up (PEA
    # reset). This makes a difference since the actual time is just after the
    # maneuver end.
    print("Fixing AONSMSAF at 2008:225:10:00:00.000")
    idxs = np.where(cmds["date"] == "2008:225:10:00:00.000")[0]
    if len(idxs) == 1:
        cmd = cmds[idxs[0]]
        cmd["date"] = "2008:225:10:07:13.600"

    # Fix incorrect interrupt time for OCT1606B. Commands after 295:18:59:00 are
    # superceded by OCT2206A.
    bad = (cmds["source"] == "OCT1606B") & (cmds["date"] > "2006:295:18:59:00")
    print(f"Removing {np.count_nonzero(bad)} bad commands from OCT1606B")
    cmds = cmds[~bad]

    # Assign params for every AOSTRCAT command
    for ii, idx in enumerate(np.flatnonzero(cmds["tlmsid"] == "AOSTRCAT")):
        if ii % 1000 == 0:
            print(f"Processing star catalog {ii}")
        cmd = cmds[idx]
        load_starcat_cmds = starcat_cmds[cmd["source"]]
        ok = load_starcat_cmds["date"] == cmd["date"]
        if np.count_nonzero(ok) == 1:
            params = load_starcat_cmds["params"][ok][0]
            # Get new integer index for this starcat command. This also encodes
            # `params` into a bytes string which is what gets stored in
            # pars_dict.
            cmd["idx"] = get_par_idx_update_pars_dict(pars_dict, cmd, params)
        else:
            raise ValueError(f"Expected 1 AOSTRCAT cmd for {cmd}")

    idx_stop = np.flatnonzero(cmds["source"] == CMDS_V2_START)[0]
    cmds = cmds[:idx_stop]

    print("Adding obsid commands")
    cmds = add_obs_cmds(cmds, pars_dict, rev_pars_dict)

    del cmds["params"]
    print(f"Writing {len(cmds)} cmds to cmds2.h5")
    cmds.write("cmds2.h5", path="data", overwrite=True)
    print(f"Writing {len(pars_dict)} pars dict entries to cmds2.pkl")
    pickle.dump(pars_dict, open("cmds2.pkl", "wb"))


###############################################################################
# Stuff after here was used in initial testing / dev of the commands v2 code.
# Probably not useful going forward.
###############################################################################
def get_backstop_cmds_from_load_legacy(load):
    """This also updates the load cmd_start and cmd_stop as a side effect."""
    # THIS WILL BE MADE FASTER by using pre-generated gzipped CommandTable files
    load_name = load if isinstance(load, str) else load["name"]
    load_dir = ska_load_dir(load_name)
    backstop_files = list(load_dir.glob("CR*.backstop"))
    if len(backstop_files) != 1:
        raise ValueError(f"Expected 1 backstop file for {load_name}")
    bs = get_cmds_from_backstop(backstop_files[0], remove_starcat=True)
    return bs


def fix_load_based_on_backstop_legacy(load, bs):
    # Get the first and last cmds for the load which are not the RLTT and
    # scheduled_stop pseudo-cmds.
    for cmd in bs:
        if cmd["type"] != "LOAD_EVENT":
            load["cmd_start"] = cmd["date"]
            break
    for cmd in bs[::-1]:
        if cmd["type"] != "LOAD_EVENT":
            load["cmd_stop"] = cmd["date"]
            break
    for cmd in bs:
        if (
            cmd["type"] == "LOAD_EVENT"
            and cmd["params"]["event_type"] == "RUNNING_LOAD_TERMINATION_TIME"
        ):
            load["rltt"] = cmd["date"]
            break
    for cmd in bs[::-1]:
        if (
            cmd["type"] == "LOAD_EVENT"
            and cmd["params"]["event_type"] == "SCHEDULED_STOP_TIME"
        ):
            load["scheduled_stop_time"] = cmd["date"]
            break

    if load["observing_stop"] == load["cmd_stop"]:
        del load["observing_stop"]
    if load["vehicle_stop"] == load["cmd_stop"]:
        del load["vehicle_stop"]


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
    bs_cmds.sort(["date", "step", "scs"])
    return bs_cmds


def interrupt_load_commands_legacy(load, cmds):
    # Cut commands beyond stop times
    bad = np.zeros(len(cmds), dtype=bool)
    if "observing_stop" in load:
        bad |= (cmds["date"] > load["observing_stop"]) & (cmds["scs"] > 130)
    if "vehicle_stop" in load:
        bad |= (cmds["date"] > load["vehicle_stop"]) & (cmds["scs"] < 131)
    if np.any(bad):
        cmds = cmds[~bad]
    return cmds
