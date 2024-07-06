# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import os
from collections import defaultdict
from pathlib import Path

import agasc
import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.table import unique as table_unique
from cxotime import CxoTime, date2secs

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

# Standard column order for ACATable
STARCAT_NAMES = [
    "slot",
    "idx",
    "id",
    "type",
    "sz",
    "mag",
    "maxmag",
    "yang",
    "zang",
    "dim",
    "res",
    "halfw",
]


def get_detector_and_sim_offset(simpos):
    """Get detector from SIM position.

    Finds the detector with nominal SIM position closest to ``simpos``.
    Taken from fot_matlab_tools/MissionScheduling/axafutil/getSIFromPosition.m.

    Parameters
    ----------
    aca : ACATable
        Input star catalog
    simpos
        int, SIM position (steps)

    Returns
    -------
    detector : str
        Detector name
    sim_offset : int
        SIM offset (steps)
    """
    from proseco import characteristics_fid as FID

    sim_offsets = [simpos - simpos_nom for simpos_nom in FID.simpos.values()]
    idx = np.argmin(np.abs(sim_offsets))
    detector = list(FID.simpos.keys())[idx]
    sim_offset = sim_offsets[idx]
    return detector, sim_offset


def set_fid_ids(aca: dict) -> None:
    """Find the FID ID for each FID in the ACA.

    ``aca`` is a dict of list with starcat values along with a ``meta`` key containing
    relevant observation info. This is from ``convert_aostrcat_to_starcat_dict()``.

    This function sets the ``id`` and ``mag`` in-place to the closest FID.
    """
    from proseco.fid import get_fid_positions

    from kadi.commands import conf

    obs = aca["meta"]
    fid_yangs, fid_zangs = get_fid_positions(
        detector=obs["detector"], focus_offset=0, sim_offset=obs["sim_offset"]
    )
    n_idx = len(aca["idx"])
    idxs_aca = [idx for idx in range(n_idx) if aca["type"][idx] == "FID"]
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
            aca["id"][idx_aca] = int(idxs_fid[0]) + 1  # Cast to pure Python
            aca["mag"][idx_aca] = 7.0
        elif n_idxs_fid > 1:
            logger.warning(
                f"WARNING: found {n_idxs_fid} fids in obsid {obs['obsid']} at "
                f"{obs['date']}\n{aca[idx_aca]}"
            )
        # Note that no fids found happens normally for post SCS-107 observations
        # because the SIM is translated so don't warn in this case.


class StarIdentificationFailed(Exception):
    """Exception raised when star identification fails."""


def set_star_ids(aca: dict) -> None:
    """Find the star ID for each star in the ACA.

    ``aca`` is a dict of list with starcat values along with a ``meta`` key containing
    relevant observation info. This is from ``convert_aostrcat_to_starcat_dict()``.

    This set the ``id`` and ``mag`` in-place to the brightest star within 1.5 arcsec of
    the commanded position.

    This function uses AGASC 1.7 or 1.8, depending on the observation date. For dates
    before 2024-Jul-21, AGASC 1.7 is used. Between 2024-Jul-21 and 2024-Aug-19, both
    versions are tried (1.8 then 1.7). After 2024-Aug-19, only 1.8 is used.

    Parameters
    ----------
    aca : dict
        Input star catalog
    """
    date = aca["meta"]["date"]
    if date < "2024:203":
        # Always 1p7 before 2024-July-21 (before JUL2224 loads)
        versions = ["1p7"]
    elif date < "2024:233":
        # Could be 1p8 or 1p7 within 30 days later (uncertainty in promotion date)
        versions = ["1p8", "1p7"]
    else:
        # Always 1p8 after 30 days after JUL2224
        versions = ["1p8"]

    # Try allowed versions and stop on first success. If no success then issue warning.
    # Be aware that _set_star_ids works in place so the try/except is not atomic so the
    # ``aca`` dict can be partially updated. This is not expected to be an issue in
    # practice, and a warning is issue in any case.
    err_star_id = None
    for version in versions:
        agasc_file = agasc.get_agasc_file(version)
        try:
            _set_star_ids(aca, agasc_file)
        except StarIdentificationFailed as err:
            err_star_id = err
        else:
            break
    else:
        # All versions failed, issue warning
        logger.warning(str(err_star_id))


def _set_star_ids(aca: dict, agasc_file: str) -> None:
    """Work function to find the star ID for each star in the ACA.

    This function does the real work for ``set_star_ids`` but it allows for trying
    AGASC 1.8 and falling back to 1.7 in case of failure.

    ``aca`` is a dict of list with starcat values along with a ``meta`` key containing
    relevant observation info. This is from ``convert_aostrcat_to_starcat_dict()``.

    This set the ``id`` and ``mag`` in-place to the brightest star within 1.5 arcsec of the
    commanded position.

    Parameters
    ----------
    aca : dict
        Input star catalog
    agasc_file : str
        AGASC file name
    """
    from chandra_aca.transform import radec_to_yagzag
    from Quaternion import Quat

    from kadi.commands import conf

    obs = aca["meta"]
    q_att = Quat(obs["att"])
    stars = get_agasc_cone_fast(
        q_att.ra,
        q_att.dec,
        radius=1.2,
        date=obs["date"],
        matlab_pm_bug=True,
        agasc_file=agasc_file,
    )
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
            aca["id"][idx_aca] = int(stars["AGASC_ID"][ok][idx])
            aca["mag"][idx_aca] = float(stars["MAG_ACA"][ok][idx])
        else:
            raise StarIdentificationFailed(
                f"WARNING: star idx {idx_aca + 1} not found in obsid {obs['obsid']} at "
                f"{obs['date']}"
            )


def convert_starcat_dict_to_acatable(starcat_dict: dict):
    """Convert star catalog dict to an ACATable, including obs metadata.

    Parameters
    ----------
    starcat_dict
        dict of list with starcat values

    Returns
    -------
    ACATable
    """
    from proseco.acq import AcqTable
    from proseco.catalog import ACATable
    from proseco.guide import GuideTable

    meta = starcat_dict.pop("meta")
    aca = ACATable(starcat_dict)
    starcat_dict["meta"] = meta

    aca.acqs = AcqTable()
    aca.guides = GuideTable()

    for attr in ("obsid", "att", "date", "duration", "detector", "sim_offset"):
        setattr(aca, attr, meta[attr])

    # Make the catalog more complete and provide stuff temps needed for plot()
    aca.t_ccd = -20.0
    aca.acqs.t_ccd = -20.0
    aca.guides.t_ccd = -20.0

    return aca


def convert_aostrcat_to_starcat_dict(params: dict) -> dict[str, list]:
    """Convert dict of AOSTRCAT parameters to a dict of list for each attribute.

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
    obs
        dict of observation (OBS command) parameters
    params
        dict of AOSTRCAT parameters

    Returns
    -------
    dict of list
        Dict of list keyed on each catalog attribute
    """
    for idx in range(1, 17):
        if params[f"minmag{idx}"] == params[f"maxmag{idx}"] == 0:
            break

    max_idx = idx
    aca = {}
    aca["idx"] = np.arange(1, max_idx)
    for par_name, col_name, func in PAR_MAPS:
        aca[col_name] = [func(params[par_name + str(idx)]) for idx in range(1, max_idx)]
    ress = aca["res"]
    dims = aca["dim"]
    halfws = [20 + (5 if ress[idx] else 40) * dims[idx] for idx in range(max_idx - 1)]
    for idx, typ in enumerate(aca["type"]):
        if typ == "MON":
            halfws[idx] = 25
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

    Parameters
    ----------
    start : CxoTime-like, None
        Start time (default=beginning of commands)
    stop : CxoTime-like, None
        Stop time (default=end of commands)
    obsid : int, None
        ObsID
    unique : bool
        If True return remove duplicate entries
    scenario : str, None
        Scenario
    cmds : CommandTable, None
        Use this command table instead of querying the archive.
    starcat_date : CxoTime-like, None
        Date of the observation's star catalog

    Returns
    -------
    Table
        Star catalog entries for matching observations.
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
    show_progress=False,
):
    """Get a list of star catalogs corresponding to input parameters.

    The ``start``, ``stop`` and ``obsid`` parameters serve as matching filters
    on the list of star catalogs that is returned.

    By default the result is a list of ``ACATable`` objects similar to the
    output of ``proseco.get_aca_catalog``.

    If ``as_dict`` is ``True`` then the the result is a list of dictionaries
    with the same keys as the table columns plus a special "meta" key. The
    "meta" value is a dict with relevant metadata including the obsid, att,
    date, duration, sim_offset, and detctor. This method is substantially faster
    than the default.

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

    Parameters
    ----------
    start : CxoTime-like, None
        Start time (default=beginning of commands)
    stop : CxoTime-like, None
        Stop time (default=end of commands)
    obsid : int, None
        ObsID
    scenario : str, None
        Scenario
    cmds : CommandTable, None
        Use this command table instead of querying the archive.
    as_dict : bool, False
        Return a list of dict instead of a list of ACATable objects.
    starcat_date : CxoTime-like, None
        Date of the observation's star catalog
    show_progress : bool
        Show progress bar for long queries (default=False)

    Returns
    -------
    list
        List of star catalogs (ACATable or dict) for matching observations.
    """
    import shelve
    from contextlib import ExitStack

    from tqdm import tqdm

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
        if conf.cache_starcats and cmds is None:
            # Using shelve provides a persistent cache of star catalogs. Only do this
            # when using the global commands archive, not when `cmds` are provided.
            starcats_db = context_stack.enter_context(
                shelve.open(str(STARCATS_CACHE_PATH()))
            )
        else:
            # Defining as a dict provides the same interface as shelve but
            # precludes caching.
            starcats_db = {}

        obss_iter = tqdm(obss) if show_progress else obss
        for obs in obss_iter:
            if (idx := obs.get("starcat_idx")) is None:
                continue

            db_key = "{}-{}-{:05d}".format(
                obs["starcat_date"], obs["source"], obs["obsid"]
            )
            if db_key in starcats_db:
                starcat_ids, starcat_mags = starcats_db[db_key]
            else:
                starcat_ids = None
                starcat_mags = None

            # From the commands archive, building ACA catalog dict from backstop
            # params
            params = rev_pars_dict[idx]
            if isinstance(params, bytes):
                params = decode_starcat_params(params)
            starcat_dict = convert_aostrcat_to_starcat_dict(params)
            n_idx = len(starcat_dict["idx"])
            meta = dict(
                obsid=obs["obsid"],
                att=obs["targ_att"],
                date=obs["starcat_date"],
                duration=date2secs(obs["obs_stop"]) - date2secs(obs["obs_start"]),
            )
            meta["detector"], meta["sim_offset"] = get_detector_and_sim_offset(
                obs["simpos"]
            )
            starcat_dict["meta"] = meta

            if starcat_ids is None or starcat_mags is None:
                starcat_dict["id"] = [-999] * n_idx
                starcat_dict["mag"] = [-999.0] * n_idx
                set_fid_ids(starcat_dict)
                set_star_ids(starcat_dict)
                starcats_db[db_key] = (starcat_dict["id"], starcat_dict["mag"])
            else:
                starcat_dict["id"] = starcat_ids
                starcat_dict["mag"] = starcat_mags

            starcat_dict = {key: starcat_dict[key] for key in STARCAT_NAMES + ["meta"]}

            starcats.append(
                starcat_dict
                if as_dict
                else convert_starcat_dict_to_acatable(starcat_dict)
            )

    return starcats


def get_observations(
    start=None, stop=None, *, obsid=None, scenario=None, cmds=None, starcat_date=None
):
    """Get observations corresponding to input parameters.

    The ``start``, ``stop``, ``starcat_date`` and ``obsid`` parameters serve as
    matching filters on the list of observations that is returned.

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
        'starcat_date': '2007:002:04:31:43.965',
        'starcat_idx': 147908,
        'source': 'DEC2506C'}

        >>> obs_all = get_observations()  # All observations in commands archive

        # Might be convenient to handle this as a Table >>> from astropy.table
        import Table >>> obs_all = Table(obs_all)

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

    Parameters
    ----------
    start : CxoTime-like, None
        Start time (default=beginning of commands)
    stop : CxoTime-like, None
        Stop time (default=end of commands)
    obsid : int, None
        ObsID
    scenario : str, None
        Scenario
    cmds : CommandTable, None
        Use this command table instead of querying the archive
    starcat_date : CxoTime-like, None
        Date of the observation's star catalog

    Returns
    -------
    list of dict
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

    # Get observations in date range with padding. _some_ padding is necessary
    # because start/stop are used to filter observations based on obs_start,
    # which can be more than 30 minutes after starcat_date. I was generous with
    # padding.
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


def get_agasc_cone_fast(
    ra, dec, radius=1.5, date=None, matlab_pm_bug=False, agasc_file=None
):
    """
    Get AGASC catalog entries within ``radius`` degrees of ``ra``, ``dec``.

    This is a thin wrapper around of agasc.get_agasc_cone() that returns a subset of
    proseco_agasc columns: AGASC_ID, RA, DEC, PM_RA, PM_DEC, EPOCH, MAG_ACA, RA_PMCORR,
    DEC_PMCORR. The full catalog for those columns is cached in memory for speed.

    Parameters
    ----------
    ra : float
        Right ascension (deg)
    dec : float
        Declination (deg)
    radius : float
        Cone search radius (deg)
    date : CxoTime-like, None
        Date for proper motion (default=Now)
    matlab_pm_bug : bool
        Apply MATLAB proper motion bug prior to the MAY2118A loads (default=False)
    agasc_file : str, None
        AGASC file name (default=None)

    Returns
    -------
    Table
        Table of AGASC entries with AGASC_ID, RA, DEC, PM_RA, PM_DEC, EPOCH, MAG_ACA,
        RA_PMCORR, DEC_PMCORR columns.
    """
    import agasc

    columns = (
        "AGASC_ID",
        "RA",
        "DEC",
        "PM_RA",
        "PM_DEC",
        "EPOCH",
        "MAG_ACA",
    )
    stars = agasc.get_agasc_cone(
        ra,
        dec,
        radius=radius,
        date=date,
        columns=columns,
        cache=True,
        matlab_pm_bug=matlab_pm_bug,
        agasc_file=agasc_file,
    )
    return stars
