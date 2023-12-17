# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import re
from pathlib import Path
from typing import Optional


class ParseLoadNameError(ValueError):
    ...


def _version_str(version):
    """Legacy version string"""
    return "" if version in (None, 1, "1") else str(version)


def SKA_DATA():
    return Path(os.environ.get("SKA", "/proj/sot/ska"), "data/kadi")


def DATA_DIR():
    return Path(os.environ.get("KADI", SKA_DATA())).absolute()


def EVENTS_DB_PATH():
    return DATA_DIR() / "events3.db3"


def IDX_CMDS_PATH(version=None):
    version = _version_str(version)
    return DATA_DIR() / f"cmds{version}.h5"


def PARS_DICT_PATH(version=None):
    version = _version_str(version)
    return DATA_DIR() / f"cmds{version}.pkl"


def CMDS_DIR(scenario=None):
    from kadi.commands import conf

    # Special case for "flight" scenario. This is hardwired to $SKA/data/kadi
    # and is intended for use in load review tools run on HEAD.
    if scenario == "flight":
        cmds_dir = SKA_DATA()
    else:
        cmds_dir = Path(conf.commands_dir).expanduser()

    return cmds_dir.absolute()


def LOADS_ARCHIVE_DIR():
    out = CMDS_DIR() / "loads"
    return out


def LOADS_BACKSTOP_PATH(load_name):
    out = LOADS_ARCHIVE_DIR() / f"{load_name}.pkl.gz"
    return out


def SCENARIO_DIR(scenario=None):
    scenario_dir = CMDS_DIR(scenario)
    if scenario is None:
        scenario = os.environ.get("KADI_SCENARIO")
    if scenario:
        scenario_dir = scenario_dir / scenario
    return scenario_dir


def LOADS_TABLE_PATH(scenario=None):
    return SCENARIO_DIR(scenario) / "loads.csv"


def CMD_EVENTS_PATH(scenario=None):
    return SCENARIO_DIR(scenario) / "cmd_events.csv"


def STARCATS_CACHE_PATH():
    from kadi.commands import conf

    return Path(conf.commands_dir).expanduser() / "starcats_cache"


def parse_load_name(load_name: str) -> bool:
    """Parse a load name like "DEC1123A" into its components.

    Returns a tuple with (mon, day, yr, rev, year).

    Parameters
    ----------
    load_name : str
        Load name like "DEC1123A"

    Returns
    -------
    mon : str
        Month like "DEC"
    day : str
        Day like "11"
    yr : str
        Year like "23"
    rev : str
        Revision like "A"
    year : int
        Year like 2023

    Raises
    ------
    ParseLoadNameError
        If load_name is not a valid load name like "DEC1123A"
    """
    re_load_name = r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{4}[A-Z]$"
    is_load_name = isinstance(load_name, str) and re.match(re_load_name, load_name)
    if not is_load_name:
        raise ParseLoadNameError(
            f"load_name {load_name} is not a valid str load name like 'DEC1123A'"
        )

    mon = load_name[:3]
    day = load_name[3:5]
    if int(day) > 31:
        raise ParseLoadNameError(f"load_name {load_name} has invalid day {day}")
    yr = load_name[5:7]
    rev = load_name[7]
    if yr == "99":
        year = 1999
    else:
        year = 2000 + int(yr)

    return mon, day, yr, rev, year


def ska_load_dir(load_name: str, root: Optional[str | Path] = None) -> Path:
    """Return a load directory path in $SKA/data from a load name like "DEC1123B".

    Parameters
    ----------
    load_name : str
        Load name like "DEC1123B"
    root : str or pathlib.Path, optional
        Root dir to prepend to load directory (default ``$SKA/data/mpcrit1/mplogs``).

    Returns
    -------
    load_dir : pathlib.Path
        Path to load directory like
        "/proj/sot/ska/data/mpcrit1/mplogs/2023/DEC1123/oflsb"
    """
    mon, day, yr, rev, year = parse_load_name(load_name)

    if root is None:
        root = Path(os.environ["SKA"]) / "data" / "mpcrit1" / "mplogs"
    load_dir = root / str(year) / f"{mon}{day}{yr}" / f"ofls{rev.lower()}"

    return load_dir


def backstop_path(load_dir: str | Path) -> list[dict]:
    """Get path to from backstop file in load directory.

    Parameters
    ----------
    load_dir : str or Path
        Directory with load review files (like "<root>/DEC1123/oflsb") or load name like
        "DEC1123B". If supplied as a load name then the load directory is found in
        ``$SKA/data/mpcrit1/mplogs`` using ``ska_load_dir()``.

    Returns
    -------
    backstop_path : pathlib.Path
        Path to backstop file ``CR*.backstop`` in load directory.
    """
    try:
        backstop_dir = ska_load_dir(load_dir)
    except ParseLoadNameError:
        backstop_dir = Path(load_dir)

    backstop_paths = list(backstop_dir.glob("CR*.backstop"))
    if len(backstop_paths) == 0:
        raise FileNotFoundError(f"No backstop files found in {backstop_dir}")
    elif len(backstop_paths) > 1:
        raise FileNotFoundError(
            f"Multiple backstop files found in {backstop_dir}: {backstop_paths}"
        )

    return backstop_paths[0]
