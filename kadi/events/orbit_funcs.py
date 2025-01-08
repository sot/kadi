# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import division

import logging
import os
import re
from pathlib import Path

import numpy as np
from chandra_time import DateTime


class NotFoundError(Exception):
    pass


ORBIT_POINTS_DTYPE = [
    ("date", "U21"),
    ("name", "U8"),
    ("orbit_num", "i4"),
    ("descr", "U50"),
]

ORBITS_DTYPE = [
    ("orbit_num", "i4"),
    ("start", "U21"),
    ("stop", "U21"),
    ("tstart", "f8"),
    ("tstop", "f8"),
    ("dur", "f4"),
    ("perigee", "U21"),
    ("t_perigee", "f8"),
    ("apogee", "U21"),
    ("start_radzone", "U21"),
    ("stop_radzone", "U21"),
    ("dt_start_radzone", "f4"),
    ("dt_stop_radzone", "f4"),
]

logger = logging.getLogger("events")

MPLOGS_DIR = Path(os.environ["SKA"], "data", "mpcrit1", "mplogs")

# Just for reference, all name=descr pairs between 2000 to 2013:001
NAMES = {
    "EALT0": "ALTITUDE ZONE ENTRY0",
    "EALT1": "ALTITUDE ZONE ENTRY 1",
    "EALT2": "ALTITUDE ZONE ENTRY2",
    "EALT3": "ALTITUDE ZONE ENTRY3",
    "EAPOGEE": "ORBIT APOGEE",
    "EASCNCR": "ORBIT ASCENDING NODE CROSSING",
    "EE1RADZ0": "ELECTRON1 RADIATION ENTRY0",
    "EE2RADZ0": "ELECTRON2 RADIATION ENTRY0",
    "EEF1000": "ELECTRON 1 RADIATION ENTRY 0",
    "EODAY": "EARTH SHADOW (UMBRA) EXIT",
    "EONIGHT": "EARTH SHADOW (UMBRA) ENTRY",
    "EP1RADZ0": "PROTON1 RADIATION ENTRY0",
    "EP2RADZ0": "PROTON2 RADIATION ENTRY0",
    "EPERIGEE": "ORBIT PERIGEE",
    "EPF1000": "PROTON 1 RADIATION ENTRY 0",
    "EQF003M": "PROTON FLUX ENTRY FOR ENERGY 0 LEVEL 0 KP 3 MEAN",
    "EQF013M": "PROTON FLUX ENTRY FOR ENERGY 0 LEVEL 1 KP 3 MEAN",
    "LSDAY": "LUNAR SHADOW (UMBRA) EXIT",
    "LSNIGHT": "LUNAR SHADOW (UMBRA) ENTRY",
    "LSPENTRY": "LUNAR SHADOW (PENUMBRA) ENTRY",
    "LSPEXIT": "LUNAR SHADOW (PENUMBRA) EXIT",
    "OORMPDS": "RADMON DISABLE",
    "OORMPEN": "RADMON ENABLE",
    "PENTRY": "EARTH SHADOW (PENUMBRA) ENTRY",
    "PEXIT": "EARTH SHADOW (PENUMBRA) EXIT",
    "XALT0": "ALTITUDE ZONE EXIT 0",
    "XALT1": "ALTITUDE ZONE EXIT 1",
    "XALT2": "ALTITUDE ZONE EXIT2",
    "XALT3": "ALTITUDE ZONE EXIT3",
    "XE1RADZ0": "ELECTRON1 RADIATION EXIT0",
    "XE2RADZ0": "ELECTRON2 RADIATION EXIT0",
    "XEF1000": "ELECTRON 1 RADIATION EXIT 0",
    "XP1RADZ0": "PROTON1 RADIATION EXIT0",
    "XP2RADZ0": "PROTON2 RADIATION EXIT0",
    "XPF1000": "PROTON 1 RADIATION EXIT 0",
    "XQF003M": "PROTON FLUX EXIT FOR ENERGY 0 LEVEL 0 KP 3 MEAN",
    "XQF013M": "PROTON FLUX EXIT FOR ENERGY 0 LEVEL 1 KP 3 MEAN",
}


def prune_dirs(dirs, regex):
    """
    Prune directories (in-place) that do not match ``regex``.
    """
    prunes = [x for x in dirs if not re.match(regex, x)]
    for prune in prunes:
        dirs.remove(prune)


# get_tlr_files is slow, so cache results (mostly for testing)
get_tlr_files_cache = {}


def get_tlr_files(mpdir=""):
    """
    Get all timeline report files within the specified SOT MP directory
    ``mpdir`` relative to the root of /data/mpcrit1/mplogs.

    Returns a list of dicts [{name, date},..]
    """
    rootdir = (MPLOGS_DIR / mpdir).absolute()
    try:
        return get_tlr_files_cache[rootdir]
    except KeyError:
        pass

    logger.info("Looking for TLR files in {}".format(rootdir))

    tlrfiles = []
    for root, dirs, files in os.walk(rootdir):
        root = root.rstrip("/")  # noqa: PLW2901
        depth = len(Path(root).parts) - len(MPLOGS_DIR.parts)
        logger.debug(f"get_trl_files: root={root} {depth} {rootdir}")
        if depth == 0:
            prune_dirs(dirs, r"\d{4}$")
        elif depth == 1:
            prune_dirs(dirs, r"[A-Z]{3}\d{4}$")
        elif depth == 2:
            prune_dirs(dirs, r"ofls[a-z]$")
        elif depth > 2:
            tlrs = [x for x in files if re.match(r".+\.tlr$", x)]
            if len(tlrs) == 0:
                logger.info("NO tlr file found in {}".format(root))
            else:
                logger.info("Located TLR file {}".format(os.path.join(root, tlrs[0])))
                tlrfiles.append(os.path.join(root, tlrs[0]))
            while dirs:
                dirs.pop()

    files = []
    for tlrfile in tlrfiles:
        monddyy, oflsv = tlrfile.split("/")[-3:-1]
        mon = monddyy[:3].capitalize()
        dd = monddyy[3:5]
        yy = int(monddyy[5:7])
        yyyy = 1900 + yy if yy > 95 else 2000 + yy
        caldate = "{}{}{} at 12:00:00.000".format(yyyy, mon, dd)
        files.append(
            (tlrfile, DateTime(caldate).date[:8] + oflsv, DateTime(caldate).date)
        )

    files = sorted(files, key=lambda x: x[1])
    out = [{"name": x[0], "date": x[2]} for x in files]
    get_tlr_files_cache[rootdir] = out

    return out


def prune_a_loads(tlrfiles):
    """
    When there are B or later products, take out the A loads.  This is where
    most mistakes are removed.  (CURRENTLY THIS FUNCTION IS NOT USED).
    """
    outs = []
    last_monddyy = None
    for tlrfile in reversed(tlrfiles):
        monddyy, oflsv = tlrfile.split("/")[-3:-1]
        if monddyy == last_monddyy and oflsv == "oflsa":
            continue
        else:
            outs.append(tlrfile)
            last_monddyy = monddyy

    return list(reversed(outs))


def filter_known_bad(orbit_points):
    """
    Filter some commands that are known to be incorrect.
    """
    ops = orbit_points
    bad = np.zeros(len(orbit_points), dtype=bool)
    bad |= (ops["name"] == "OORMPEN") & (ops["date"] == "2002:253:10:08:52.239")
    bad |= (ops["name"] == "OORMPEN") & (ops["date"] == "2004:010:10:00:00.000")
    return orbit_points[~bad]


def get_orbit_points(tlrfiles):
    """
    Get all orbit points from the timeline reports within the specified mission planning
    path '' (all) or 'YYYY' (year) or YYYY/MONDDYY (load).
    """
    orbit_points = []

    # tlrfiles = prune_a_loads(tlrfiles)

    for tlrfile in tlrfiles:
        # Parse thing like this:
        #  2012:025:21:22:21.732 EQF013M     1722   PROTON FLUX ENTRY FOR ENERGY 0 LEVEL ...
        # 012345678901234567890123456789012345678901234567890123456789
        logger.info("Getting points from {}".format(tlrfile))
        try:
            fh = open(tlrfile, "r", encoding="ascii", errors="ignore")
        except IOError as err:
            logger.warning(err)
            continue
        for line in fh:
            if len(line) < 30 or line[:2] != " 2":
                continue
            try:
                date, name, orbit_num, descr = line.split(None, 3)
            except ValueError:
                continue

            if name.startswith("OORMP"):
                orbit_num = -1
                descr = "RADMON {}ABLE".format("EN" if name.endswith("EN") else "DIS")
            elif line[23] in " -":
                continue

            if "DSS-" in name:
                continue
            if not re.match(r"\d{4}:\d{3}:\d{2}:\d{2}:\d{2}\.\d{3}", date):
                logger.info('Failed for date: "{}"'.format(date))
                continue
            if not re.match(r"[A-Z]+", name):
                logger.info('Failed for name: "{}"'.format(name))
                continue

            try:
                orbit_num = int(orbit_num)
            except TypeError:
                logger.info("Failed for orbit_num: {}".format(orbit_num))
                continue

            descr = descr.strip()
            orbit_points.append((date, name, orbit_num, descr))

    orbit_points = sorted(set(orbit_points), key=lambda x: x[0])
    return orbit_points


def get_nearest_orbit_num(orbit_nums, idx, d_idx):
    """
    Get the orbit number nearest to ``orbit_nums[idx]`` in direction ``d_idx``,
    skipping values of -1 (from radmon commanding).
    """
    while True:
        idx += d_idx
        if idx < 0 or idx >= len(orbit_nums):
            raise NotFoundError("No nearest orbit num found")
        if orbit_nums[idx] != -1:
            break
    return orbit_nums[idx], idx


def interpolate_orbit_points(orbit_points, name):
    """
    Linearly interpolate across any gaps for ``name`` orbit_points.
    """
    if len(orbit_points) == 0:
        return []

    ok = orbit_points["name"] == name
    ops = orbit_points[ok]
    # Get the indexes of missing orbits
    idxs = np.flatnonzero(np.diff(ops["orbit_num"]) > 1)
    new_orbit_points = []
    for idx in idxs:
        op0 = ops[idx]
        op1 = ops[idx + 1]
        orb_num0 = op0["orbit_num"]
        orb_num1 = op1["orbit_num"]
        time0 = DateTime(op0["date"]).secs
        time1 = DateTime(op1["date"]).secs
        for orb_num in range(orb_num0 + 1, orb_num1):
            time = time0 + (orb_num - orb_num0) / (orb_num1 - orb_num0) * (
                time1 - time0
            )
            date = DateTime(time).date
            new_orbit_point = (date, name, orb_num, op0["descr"])
            logger.info("Adding new orbit point {}".format(new_orbit_point))
            new_orbit_points.append(new_orbit_point)

    return new_orbit_points


def process_orbit_points(orbit_points):
    """
    Take the raw orbit points (list of tuples) and do some processing:

    - Remove duplicate events within 30 seconds of each other
    - Fill in orbit number for RADMON enable / disable points
    - Convert to a number structured array

    Returns a numpy array with processed orbit points::

      ORBIT_POINTS_DTYPE = [('date', 'U21'), ('name', 'U8'),
                            ('orbit_num', 'i4'), ('descr', 'U50')]
    """
    # Find neighboring pairs of orbit points that are identical except for date.
    # If the dates are then within 180 seconds of each other then toss the first
    # of the pair.
    if len(orbit_points) == 0:
        return np.array([], dtype=ORBIT_POINTS_DTYPE)

    uniq_orbit_points = []
    for op0, op1 in zip(orbit_points[:-1], orbit_points[1:]):
        if op0[1:4] == op1[1:4]:
            dt = (DateTime(op1[0]) - DateTime(op0[0])) * 86400
            if dt < 180:
                # logger.info('Removing duplicate orbit points:\n  {}\n  {}'
                #      .format(str(op0), str(op1)))
                continue
        uniq_orbit_points.append(op1)
    uniq_orbit_points.append(orbit_points[-1])
    orbit_points = uniq_orbit_points

    # Convert to a numpy structured array
    orbit_points = np.array(orbit_points, dtype=ORBIT_POINTS_DTYPE)

    # Filter known bad points
    orbit_points = filter_known_bad(orbit_points)

    # For key orbit points linearly interpolate across gaps in orbit coverage.
    new_ops = []
    for name in ("EPERIGEE", "EAPOGEE", "EASCNCR"):
        new_ops.extend(interpolate_orbit_points(orbit_points, name))

    # Add a new orbit point for the ascending node EXIT which is the end of each orbit.
    # This simplifies bookkeeping later.
    for op in orbit_points[orbit_points["name"] == "EASCNCR"]:
        new_ops.append(  # noqa: PERF401
            (op["date"], "XASCNCR", op["orbit_num"] - 1, op["descr"] + " EXIT")
        )

    # Add corresponding XASCNCR for any new EASCNCR points
    for op in new_ops:
        if op[1] == "EASCNCR":
            new_ops.append((op[0], "XASCNCR", op[2] - 1, op[3] + " EXIT"))  # noqa: PERF401

    logger.info("Adding {} new orbit points".format(len(new_ops)))
    new_ops = np.array(new_ops, dtype=ORBIT_POINTS_DTYPE)
    orbit_points = np.concatenate([orbit_points, new_ops])
    orbit_points.sort(order=["date", "orbit_num"])

    # Fill in orbit number for RADMON enable / disable points
    radmon_idxs = np.flatnonzero(orbit_points["orbit_num"] == -1)
    orbit_nums = orbit_points["orbit_num"]
    for idx in radmon_idxs:
        try:
            prev_num, prev_idx = get_nearest_orbit_num(orbit_nums, idx, -1)
            next_num, next_idx = get_nearest_orbit_num(orbit_nums, idx, +1)
        except NotFoundError:
            logger.info(
                "No nearest orbit point for orbit_points[{}] (len={})".format(
                    idx, len(orbit_points)
                )
            )
        else:
            if prev_num == next_num:
                orbit_nums[idx] = next_num
            else:
                logger.info(
                    "Unable to assign orbit num idx={} prev={} next={}".format(
                        idx, prev_num, next_num
                    )
                )
                logger.info("  {} {}".format(prev_idx, orbit_points[prev_idx]))
                logger.info(" * {} {}".format(idx, orbit_points[idx]))
                logger.info("  {} {}".format(next_idx, orbit_points[next_idx]))

    return orbit_points


def get_orbits(orbit_points):
    """
    Collate the orbit points into full orbits, with dates corresponding to start (ORBIT
    ASCENDING NODE CROSSING), stop, apogee, perigee, radzone start and radzone stop.
    Radzone is defined as the time covering perigee when radmon is disabled by command.
    This corresponds to the planned values and may differ from actual in the case of
    events that run SCS107 and prematurely disable RADMON.

    Returns a numpy structured array::

      ORBITS_DTYPE = [('orbit_num', 'i4'),
                      ('start', 'U21'), ('stop', 'U21'),
                      ('tstart', 'f8'), ('tstop', 'f8'), ('dur', 'f4'),
                      ('perigee', 'U21'), ('t_perigee', 'f8'), ('apogee', 'U21'),
                      ('start_radzone', 'U21'), ('stop_radzone', 'U21'),
                      ('dt_start_radzone', 'f4'), ('dt_stop_radzone', 'f4')]
    """

    def get_idx(ops, name):
        ok = ops["name"] == name
        if np.sum(ok) != 1:
            raise NotFoundError(
                "Expected one match for {} but found {} in orbit {}\n{}".format(
                    name, np.sum(ok), orbit_num, ops
                )
            )
        return np.flatnonzero(ok)[0]

    def get_date(ops, name):
        idx = get_idx(ops, name)
        return ops["date"][idx]

    def get_nearest_orbit_point(name, idx, d_idx):
        while True:
            idx += d_idx
            if idx < 0 or idx >= len(orbit_points):
                raise NotFoundError(
                    "Skipping orbit {}: no nearest orbit point {} found".format(
                        orbit_num, name
                    )
                )
            if orbit_points["name"][idx] == name:
                break
        return orbit_points[idx]

    def find_radzone(idx_perigee):
        """
        Find the extent of the radiation zone, defined as the last time before perigee
        that RADMON is enabled until the first time after perigee that RADMON is enabled.
        """
        idx = idx_perigee
        start_radzone = None
        while True:
            idx -= 1
            if idx < 0:
                raise NotFoundError(
                    "Did not find RADMON enable prior to {}".format(
                        orbit_points[idx_perigee]
                    )
                )
            if orbit_points["name"][idx] == "OORMPDS":
                start_radzone = orbit_points["date"][idx]
            if orbit_points["name"][idx] == "OORMPEN":
                if start_radzone is None:
                    raise NotFoundError(
                        "Found radmon enable before first disable at idx {}".format(idx)
                    )
                break

        idx = idx_perigee
        while True:
            idx += 1
            if idx >= len(orbit_points):
                raise NotFoundError(
                    "Did not find RADMON enable after to {}".format(
                        str(orbit_points[idx_perigee])
                    )
                )
            if orbit_points["name"][idx] == "OORMPEN":
                stop_radzone = orbit_points["date"][idx]
                break

        return start_radzone, stop_radzone

    # Copy orbit points and sort by orbit_num then date.  This allows using
    # search_sorted to select orbit_points corresponding to each orbit.  In
    # very rare cases (orbit 1448 I think), there are orbit_points that cross
    # orbit boundaries by a few seconds.  This is related to the technique of
    # reading in every TLR to get maximal coverage of orbit points.
    orbit_points = orbit_points.copy()
    orbit_points.sort(order=["orbit_num", "date"])

    orbit_nums = orbit_points["orbit_num"]
    uniq_orbit_nums = sorted(set(orbit_nums[orbit_nums > 0]))

    orbits = []
    for orbit_num in uniq_orbit_nums:
        i0 = np.searchsorted(orbit_nums, orbit_num, side="left")
        i1 = np.searchsorted(orbit_nums, orbit_num, side="right")
        ops = orbit_points[i0:i1]

        try:
            if "EASCNCR" not in ops["name"] or "XASCNCR" not in ops["name"]:
                raise NotFoundError("Skipping orbit {} incomplete".format(orbit_num))

            start = get_date(ops, "EASCNCR")
            stop = get_date(ops, "XASCNCR")
            date_apogee = get_date(ops, "EAPOGEE")
            date_perigee = get_date(ops, "EPERIGEE")

            idx_perigee = get_idx(ops, "EPERIGEE") + i0
            start_radzone, stop_radzone = find_radzone(idx_perigee)
        except NotFoundError as err:
            logger.info(err)
            continue
        else:
            dt_radzones = [
                (DateTime(date) - DateTime(date_perigee)) * 86400.0
                for date in (start_radzone, stop_radzone)
            ]
            tstart = DateTime(start).secs
            tstop = DateTime(stop).secs
            orbit = (
                orbit_num,
                start,
                stop,
                tstart,
                tstop,
                tstop - tstart,
                date_perigee,
                DateTime(date_perigee).secs,
                date_apogee,
                start_radzone,
                stop_radzone,
                dt_radzones[0],
                dt_radzones[1],
            )
            logger.info(
                "get_orbits: Adding orbit {} {} {}".format(orbit_num, start, stop)
            )
            orbits.append(orbit)

    orbits = np.array(orbits, dtype=ORBITS_DTYPE)
    return orbits


def get_radzone_from_orbit(orbit):
    """
    Extract the RadZone fields from an orbit descriptor (which is one row
    of the orbits structured array).
    """
    start_radzone = DateTime(orbit["start_radzone"], format="date")
    stop_radzone = DateTime(orbit["stop_radzone"], format="date")
    tstart = start_radzone.secs
    tstop = stop_radzone.secs
    dur = tstop - tstart
    radzone = {
        "start": start_radzone.date,
        "stop": stop_radzone.date,
        "tstart": tstart,
        "tstop": tstop,
        "dur": dur,
        "orbit_num": orbit["orbit_num"],
        "perigee": orbit["perigee"],
    }

    return radzone
