# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
from collections import defaultdict
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.table import unique as table_unique
from cxotime import CxoTime

# kadi.commands.commands_v2 is also a dependency but encapsulated in the
# functions to reduce top-level imports for v1-compatibility

logger = logging.getLogger(__name__)


__all__ = ["get_starcats", "get_observations", "get_starcats_as_table"]

AGASC_FILE = Path(os.environ["SKA"], "data", "agasc", "proseco_agasc_1p7.h5")

TYPE_MAP = ["ACQ", "GUI", "BOT", "FID", "MON"]
IMGSZ_MAP = ["4x4", "6x6", "8x8"]
RAD_TO_DEG = 180 / np.pi * 3600
PAR_MAPS = [
    ("imnum", "slot", int),
    ("type", "type", lambda n: TYPE_MAP[n]),
    ("imgsz", "sz", lambda n: IMGSZ_MAP[n]),
    ("maxmag", "maxmag", float),
    ("yang", "yang", lambda x: x * RAD_TO_DEG),
    ("zang", "zang", lambda x: x * RAD_TO_DEG),
    ("dimdts", "dim", int),
    ("restrk", "res", int),
]

# Cache of observations by scenario
OBSERVATIONS = {}

# Cache of important columns in proseco_agasc_1p7.h5
STARS_AGASC = None


def set_detector_and_sim_offset(aca, simpos):
    """Get detector from SIM position.

    Finds the detector with nominal SIM position closest to ``simpos``.
    Taken from fot_matlab_tools/MissionScheduling/axafutil/getSIFromPosition.m.

    :param aca: ACATable
        Input star catalog
    :param simpos: int, SIM position (steps)
    """
    from proseco import characteristics_fid as FID

    sim_offsets = [simpos - simpos_nom for simpos_nom in FID.simpos.values()]
    idx = np.argmin(np.abs(sim_offsets))
    detector = list(FID.simpos.keys())[idx]
    sim_offset = sim_offsets[idx]
    aca.detector = detector
    aca.sim_offset = sim_offset


def set_fid_ids(aca):
    from proseco.fid import get_fid_positions

    from kadi.commands import conf

    fid_yangs, fid_zangs = get_fid_positions(
        detector=aca.detector, focus_offset=0, sim_offset=aca.sim_offset
    )

    idxs_aca = np.where(aca["type"] == "FID")[0]
    for idx_aca in idxs_aca:
        yang = aca["yang"][idx_aca]
        zang = aca["zang"][idx_aca]
        dys = np.abs(yang - fid_yangs)
        dzs = np.abs(zang - fid_zangs)
        # Match fid positions in a box. Default is a very loose tolerance of 40
        # arcsec. The fids are widely spaced and there was a ~20 arcsec change
        # in fid positions in 2007.
        halfw = conf.fid_id_match_halfwidth
        idxs_fid = np.where((dys < halfw) & (dzs < halfw))[0]
        n_idxs_fid = len(idxs_fid)
        if n_idxs_fid == 1:
            aca[idx_aca]["id"] = idxs_fid[0] + 1
            aca[idx_aca]["mag"] = 7.0
        elif n_idxs_fid > 1:
            logger.warning(
                f"WARNING: found {n_idxs_fid} fids in obsid {aca.obsid} at "
                f"{aca.date}\n{aca[idx_aca]}"
            )
        # Note that no fids found happens normally for post SCS-107 observations
        # because the SIM is translated so don't warn in this case.


def set_star_ids(aca):
    """Find the star ID for each star in the ACA.

    This set the ID in-place to the brightest star within 1.5 arcsec of the
    commanded position.

    :param aca: ACATable
        Input star catalog
    """
    from chandra_aca.transform import radec_to_yagzag
    from Quaternion import Quat

    from kadi.commands import conf

    q_att = Quat(aca.att)
    stars = get_agasc_cone_fast(q_att.ra, q_att.dec, radius=1.2, date=aca.date)
    yang_stars, zang_stars = radec_to_yagzag(
        stars["RA_PMCORR"], stars["DEC_PMCORR"], q_att
    )
    idxs_aca = np.where(np.isin(aca["type"], ("ACQ", "GUI", "BOT")))[0]
    for idx_aca in idxs_aca:
        yang = aca["yang"][idx_aca]
        zang = aca["zang"][idx_aca]
        dys = np.abs(yang - yang_stars)
        dzs = np.abs(zang - zang_stars)

        # Get the brightest star within a box (default = 1.5 arcsec halfwidth)
        halfw = conf.star_id_match_halfwidth
        ok = (dys < halfw) & (dzs < halfw)
        if np.any(ok):
            idx = np.argmin(stars["MAG_ACA"][ok])
            aca[idx_aca]["id"] = stars["AGASC_ID"][ok][idx]
            aca[idx_aca]["mag"] = stars["MAG_ACA"][ok][idx]
        else:
            logger.info(
                f"WARNING: star idx {idx_aca + 1} not found in obsid {aca.obsid} at "
                f"{aca.date}"
            )


def convert_aostrcat_to_acatable(obs, params, as_dict=False):
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

    :param obs: dict of observation (OBS command) parameters
    :param params: dict of AOSTRCAT parameters
    :returns: ACATable
    """
    from Chandra.Time import date2secs
    from proseco.acq import AcqTable
    from proseco.catalog import ACATable
    from proseco.guide import GuideTable

    for idx in range(1, 17):
        if params[f"minmag{idx}"] == params[f"maxmag{idx}"] == 0:
            break

    max_idx = idx
    cols = defaultdict(list)
    for par_name, col_name, func in PAR_MAPS:
        for idx in range(1, max_idx):
            cols[col_name].append(func(params[par_name + str(idx)]))

    aca = ACATable(cols)
    aca.acqs = AcqTable()
    aca.guides = GuideTable()
    aca.add_column(np.arange(1, max_idx), index=1, name="idx")

    aca.obsid = obs["obsid"]
    aca.att = obs["targ_att"]
    aca.date = obs["starcat_date"]
    aca.duration = date2secs(obs["obs_stop"]) - date2secs(obs["obs_start"])

    # Make the catalog more complete and provide stuff temps needed for plot()
    aca.t_ccd = -20.0
    aca.acqs.t_ccd = -20.0
    aca.guides.t_ccd = -20.0

    halfws = 20 + np.where(aca["res"], 5, 40) * aca["dim"]
    halfws[aca["type"] == "MON"] = 25
    aca["halfw"] = halfws

    return aca


def get_starcats_as_table(
    start=None,
    stop=None,
    *,
    obsid=None,
    unique=False,
    scenario=None,
    cmds=None,
    starcat_date=None,
):
    """Get a single table of star catalog entries corresponding to input parameters.

    This function calls ``get_starcats`` with the same parameters and then
    concatenates the results into a single table for convenience. In addition
    to the usual star catalog columns, the ``obsid`` and ``starcat_date`` are
    included.

    The ``unique`` parameter can be set to ``True`` to only return unique star
    catalog entries. There are numerous instances of a single commanded star
    catalogs that is associated with two ObsIDs, for instance ACIS undercover
    observations. To get only the first one, set ``unique=True``.

    In the following example we get every unique commanded guide star in 2020
    and then join that with the corresponding observation information::

      >>> from kadi.commands import get_starcats_as_table, get_observations
      >>> from astropy import table
      >>> start='2020:001'
      >>> stop='2021:001'
      >>> aces = get_starcats_as_table(start, stop, unique=True)
      >>> ok = np.isin(aces['type'], ['GUI', 'BOT'])
      >>> guides = aces[ok]
      >>> obss = table.Table(get_observations(start, stop))
      >>> obss = obss[~obss['starcat_date'].mask]  # keep only obs with starcat
      >>> guides = table.join(guides, obss, keys=['starcat_date', 'obsid'])

    :param start: CxoTime-like, None Start time (default=beginning of commands)
    :param stop: CxoTime-like, None Stop time (default=end of commands)
    :param obsid: int, None ObsID
    :param unique: bool, if True return remove duplicate entries
    :param scenario: str, None Scenario
    :param cmds: CommandTable, None Use this command table instead of querying
        the archive.
    :param starcat_date: CxoTime-like, None
        Date of the observation's star catalog
    :returns: astropy ``Table`` star catalog entries for matching observations.

    """
    starcats = get_starcats(
        obsid=obsid,
        start=start,
        stop=stop,
        scenario=scenario,
        cmds=cmds,
        starcat_date=starcat_date,
        as_dict=True,
    )
    out = defaultdict(list)
    for starcat in starcats:
        n_cat = len(starcat["slot"])
        out["obsid"].append([starcat["meta"]["obsid"]] * n_cat)
        out["starcat_date"].append([starcat["meta"]["date"]] * n_cat)
        for name, vals in starcat.items():
            if name != "meta":
                out[name].append(vals)

    for name in out:
        out[name] = np.concatenate(out[name])

    out = Table(out)
    if unique:
        out = table_unique(out, keys=["starcat_date", "idx"])

    return out


def get_starcats(
    start=None,
    stop=None,
    *,
    obsid=None,
    scenario=None,
    cmds=None,
    as_dict=False,
    starcat_date=None,
):
    """Get a list of star catalogs corresponding to input parameters.

    The ``start``, ``stop`` and ``obsid`` parameters serve as matching filters
    on the list of star catalogs that is returned.

    By default the result is a list of ``ACATable`` objects similar to the
    output of ``proseco.get_aca_catalog``. If ``as_dict`` is ``True`` then the
    the result is a list of dictionaries with the same keys as the table columns.
    This is substantially faster than the default.

    There are numerous instances of multiple observations with the same obsid,
    so this function always returns a list of star catalogs even when ``obsid``
    is specified. In most cases you can just use the first element.

    The ``mag`` column corresponds to the AGASC magnitude *without* the AGASC
    supplement.

    Star ID's are determined by finding the brightest AGASC star within a search
    box centered at the catalog location. The search box is 1.5 arcsec halfwidth
    in size, but it can be changed by setting the ``star_id_match_halfwidth``
    configuration parameter. Fid ID's are determined similarly by computing fid
    locations given the commanded SIM-Z position. The default box size is 40
    arcsec halfwidth, but it can be changed by setting the
    ``fid_id_match_halfwidth`` configuration parameter.

    The first time each particular star catalog is fetched, the star and fid
    ID's are computed which is relatively slow. The resulting star catalog is
    (by default) cached in the ``~/.kadi/starcats.db`` file. Subsequent calls
    are significantly faster.

    Example::

        >>> from kadi.commands import get_starcats
        >>> cat = get_starcats(obsid=8008)[0]
        >>> cat
        [<ACATable length=11>
        slot  idx     id    type  sz    mag    maxmag   yang     zang    dim   res
        int64 int64  int64   str3 str3 float64 float64 float64  float64  int64 int64
        ----- ----- -------- ---- ---- ------- ------- -------- -------- ----- -----
            0     1        1  FID  8x8    7.00    8.00   937.71  -829.17     1     1
            1     2        5  FID  8x8    7.00    8.00 -1810.42  1068.87     1     1
            2     3        6  FID  8x8    7.00    8.00   403.68  1712.93     1     1
            3     4 31075128  BOT  6x6    9.35   10.86  -318.22  1202.41    20     1
            4     5 31076560  BOT  6x6    9.70   11.20  -932.79  -354.55    20     1
            5     6 31463496  BOT  6x6    9.46   10.97  2026.85  1399.61    20     1
            6     7 31983336  BOT  6x6    8.64   10.14   890.71 -1600.39    20     1
            7     8 32374896  BOT  6x6    9.17   10.66  2023.08 -2021.72    13     1
            0     9 31075368  ACQ  6x6    9.13   10.64    54.04   754.79    20     1
            1    10 31982136  ACQ  6x6   10.19   11.70   562.06  -186.39    20     1
            2    11 32375384  ACQ  6x6    9.79   11.30  1612.28  -428.24    20     1]

    :param start: CxoTime-like, None Start time (default=beginning of commands)
    :param stop: CxoTime-like, None Stop time (default=end of commands)
    :param obsid: int, None ObsID
    :param scenario: str, None Scenario
    :param cmds: CommandTable, None Use this command table instead of querying
        the archive.
    :param as_dict: bool, False Return a list of dictionaries instead of a list
        of ACATable objects.
    :param starcat_date: CxoTime-like, None
        Date of the observation's star catalog
    :returns: list of ACATable List star catalogs for matching observations.
    """
    import shelve
    from contextlib import ExitStack

    from proseco.acq import AcqTable
    from proseco.catalog import ACATable
    from proseco.guide import GuideTable

    from kadi.commands import conf
    from kadi.commands.commands_v2 import REV_PARS_DICT
    from kadi.commands.core import decode_starcat_params
    from kadi.paths import STARCATS_CACHE_PATH

    obss = get_observations(
        obsid=obsid,
        start=start,
        stop=stop,
        scenario=scenario,
        cmds=cmds,
        starcat_date=starcat_date,
    )
    starcats = []
    rev_pars_dict = REV_PARS_DICT if cmds is None else cmds.rev_pars_dict()

    with ExitStack() as context_stack:
        if conf.cache_starcats:
            starcats_db = context_stack.enter_context(
                shelve.open(str(STARCATS_CACHE_PATH()))
            )
        else:
            # Defining as a dict provides the same interface as shelve but
            # precludes caching.
            starcats_db = {}

        for obs in obss:
            if (idx := obs.get("starcat_idx")) is None:
                continue

            db_key = "{}-{}-{:05d}".format(
                obs["starcat_date"], obs["source"], obs["obsid"]
            )
            if db_key in starcats_db:
                # From the cache
                starcat_dict = starcats_db[db_key]
                if as_dict:
                    starcat = starcat_dict
                else:
                    meta = starcat_dict.pop("meta")
                    starcat = ACATable(starcat_dict)
                    starcat.meta = meta
                    starcat.acqs = AcqTable()
                    starcat.guides = GuideTable()
            else:
                # From the commands archive, building ACATable from backstop params
                params = rev_pars_dict[idx]
                if isinstance(params, bytes):
                    params = decode_starcat_params(params)
                starcat = convert_aostrcat_to_acatable(obs, params)
                set_detector_and_sim_offset(starcat, obs["simpos"])
                starcat.add_column(-999, index=2, name="id")
                starcat.add_column(-999.0, index=5, name="mag")
                set_fid_ids(starcat)
                set_star_ids(starcat)

                starcat_dict = {
                    name: starcat[name].tolist() for name in starcat.colnames
                }
                starcat_dict["meta"] = starcat.meta
                del starcat_dict["meta"]["acqs"]
                del starcat_dict["meta"]["guides"]

                if as_dict:
                    starcat = starcat_dict

                starcats_db[db_key] = starcat_dict

            starcats.append(starcat)

    return starcats


def get_observations(
    start=None, stop=None, *, obsid=None, scenario=None, cmds=None, starcat_date=None
):
    """Get observations corresponding to input parameters.

    The ``start``, ``stop``, ``starcat_date`` and ``obsid`` parameters serve as matching filters
    on the list of observations that is returned.

    Over the mission there are thousands of instances of multiple observations
    with the same obsid, so this function always returns a list of observation
    parameters even when ``obsid`` is specified. This most frequently occurs
    after any unexpected stoppage of the observng loads (SCS-107) which
    therefore cancels subsequent obsid commanding. In many cases you can just
    use the first element.

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
        'starcat_idx': 144398,
        'source': 'DEC2506C'}

        >>> obs_all = get_observations()  # All observations in commands archive

        # Might be convenient to handle this as a Table
        >>> from astropy.table import Table
        >>> obs_all = Table(obs_all)

        >>> from kadi.commands import get_observations
        >>> get_observations(starcat_date='2022:001:17:00:58.521')
        [{'obsid': 23800,
        'simpos': 75624,
        'obs_stop': '2022:002:01:24:53.004',
        'manvr_start': '2022:001:17:01:02.772',
        'targ_att': (0.177875061, 0.452625075, 0.827436517, 0.280784286),
        'npnt_enab': True,
        'obs_start': '2022:001:17:33:53.255',
        'prev_att': (0.116555575, -0.407948573, -0.759717367, 0.492770009),
        'starcat_date': '2022:001:17:00:58.521',
        'starcat_idx': 171677,
        'source': 'DEC3021A'}]

    :param start: CxoTime-like, None
        Start time (default=beginning of commands)
    :param stop: CxoTime-like, None
        Stop time (default=end of commands)
    :param obsid: int, None
        ObsID
    :param scenario: str, None
        Scenario
    :param cmds: CommandTable, None
        Use this command table instead of querying the archive.
    :param starcat_date: CxoTime-like, None
        Date of the observation's star catalog
    :returns: list of dict
        Observation parameters for matching observations.
    """
    from kadi.commands.commands_v2 import get_cmds

    if starcat_date is not None:
        start = starcat_date if start is None else start
        stop = CxoTime(starcat_date) + 7 * u.day if stop is None else stop
    start = CxoTime("1999:001" if start is None else start)
    stop = (CxoTime.now() + 1 * u.year) if stop is None else CxoTime(stop)

    if cmds is None:
        if scenario not in OBSERVATIONS:
            cmds = get_cmds(scenario=scenario)
            cmds_obs = cmds[cmds["tlmsid"] == "OBS"]
            obsids = []
            for cmd in cmds_obs:
                if cmd["params"] is None:
                    _obsid = cmd["obsid"]
                else:
                    _obsid = cmd["params"]["obsid"]
                obsids.append(_obsid)

            cmds_obs["obsid"] = obsids
            OBSERVATIONS[scenario] = cmds_obs
        else:
            cmds_obs = OBSERVATIONS[scenario]
    else:
        cmds_obs = cmds[cmds["tlmsid"] == "OBS"]

    # Get observations in date range with padding
    # _some_ padding is necessary because start/stop are used to filter observations based on
    # obs_start, which can be more than 30 minutes after starcat_date. I was generous with padding.
    i0, i1 = cmds_obs.find_date([(start - 7 * u.day).date, (stop + 7 * u.day).date])
    cmds_obs = cmds_obs[i0:i1]

    if starcat_date is not None:
        cmds_obs = cmds_obs[cmds_obs["starcat_date"] == starcat_date]
        if len(cmds_obs) == 0:
            raise ValueError(f"No matching observations for {starcat_date=}")

    if obsid is not None:
        cmds_obs = cmds_obs[cmds_obs["obsid"] == obsid]
        if len(cmds_obs) == 0:
            raise ValueError(f"No matching observations for {obsid=}")

    obss = [cmd["params"].copy() for cmd in cmds_obs]
    for obs, cmd_obs in zip(obss, cmds_obs):
        obs["source"] = cmd_obs["source"]

    # Filter observations by date to include any observation that intersects
    # the date range.
    datestart = start.date
    datestop = stop.date
    obss = [
        obs
        for obs in obss
        if obs["obs_start"] <= datestop and obs["obs_stop"] >= datestart
    ]

    return obss


def get_agasc_cone_fast(ra, dec, radius=1.5, date=None):
    """
    Get AGASC catalog entries within ``radius`` degrees of ``ra``, ``dec``.

    This is a fast version of agasc.get_agasc_cone() that keeps the key columns
    in memory instead of accessing the H5 file each time.

    :param dec: Declination (deg)
    :param radius: Cone search radius (deg)
    :param date: Date for proper motion (default=Now)

    :returns: astropy Table of AGASC entries
    """
    global STARS_AGASC

    agasc_file = AGASC_FILE
    import tables
    from agasc.agasc import add_pmcorr_columns, get_ra_decs, sphere_dist

    ra_decs = get_ra_decs(agasc_file)

    if STARS_AGASC is None:
        with tables.open_file(agasc_file, "r") as h5:
            dat = h5.root.data[:]
            cols = {
                "AGASC_ID": dat["AGASC_ID"],
                "RA": dat["RA"],
                "DEC": dat["DEC"],
                "PM_RA": dat["PM_RA"],
                "PM_DEC": dat["PM_DEC"],
                "EPOCH": dat["EPOCH"],
                "MAG_ACA": dat["MAG_ACA"],
            }
            STARS_AGASC = Table(cols)
            del dat  # Explicitly delete to free memory (?)

    idx0, idx1 = np.searchsorted(ra_decs.dec, [dec - radius, dec + radius])

    dists = sphere_dist(ra, dec, ra_decs.ra[idx0:idx1], ra_decs.dec[idx0:idx1])
    ok = dists <= radius
    stars = STARS_AGASC[idx0:idx1][ok]

    add_pmcorr_columns(stars, date)

    return stars
