import functools
import os
from pathlib import Path

import numpy as np
import pytest
import ska_helpers.utils
from astropy import units as u
from astropy.io import ascii
from astropy.table import Table
from chandra_time import DateTime
from cheta import fetch
from cxotime import CxoTime, CxoTimeLike

from kadi import commands  # noqa: E402
from kadi.commands import states  # noqa: E402
from kadi.commands.tests.test_commands import get_cmds_from_cmd_evts_text

try:
    fetch.get_time_range("dp_pitch")
    HAS_PITCH = True
except Exception:
    HAS_PITCH = False

# Canonical state0 giving spacecraft state at beginning of timelines
# 2002:007:13 fetch --start 2002:007:13:00:00 --stop 2002:007:13:02:00 aoattqt1
# aoattqt2 aoattqt3 aoattqt4 cobsrqid aopcadmd tscpos
STATE0 = {
    "ccd_count": 5,
    "clocking": 0,
    "datestart": "2002:007:13:00:00.000",
    "datestop": "2099:001:00:00:00.000",
    "dec": -11.500,
    "fep_count": 0,
    "hetg": "RETR",
    "letg": "RETR",
    "obsid": 61358,
    "pcad_mode": "NPNT",
    "pitch": 61.37,
    "power_cmd": "AA00000000",
    "q1": -0.568062,
    "q2": 0.121674,
    "q3": 0.00114141,
    "q4": 0.813941,
    "ra": 352.000,
    "roll": 289.37,
    "si_mode": "undef",
    "simfa_pos": -468,
    "simpos": -99616,
    "trans_keys": "undef",
    "tstart": 127020624.552,
    "tstop": 3187296066.184,
    "vid_board": 0,
    "dither": "None",
}


def assert_all_close_states(states_regr, states_test, **kwargs):
    """
    Compare all columns of the commanded states tables.

    Parameters
    ----------
    states_regr : astropy.table.Table
        Regression states table which defines the expected columns and values.
    states_test : astropy.table.Table
        Test states table to compare against the regression data.
    kwargs : dict
        Additional keyword arguments to pass to `np.testing.assert_allclose`.
    """
    assert len(states_regr) == len(states_test), "State tables have different lengths"

    for key in states_regr.colnames:
        for sts in states_test, states_regr:
            if sts[key].dtype.kind == "O":
                # Convert object column to string
                sts[key] = [str(val) for val in sts[key]]

        col_test = states_test[key]
        col_regr = states_regr[key]
        assert (kind := col_test.dtype.kind) == col_regr.dtype.kind

        if kind == "f":
            np.testing.assert_allclose(col_regr, col_test, **kwargs)
        else:
            np.testing.assert_equal(col_regr, col_test)


def get_regression_pathdir():
    """Get the directory path for regression data."""
    pathdir = Path(__file__).parent / "data" / "regression"
    pathdir.mkdir(exist_ok=True)
    return pathdir


def get_regression_path(
    start: CxoTime,
    stop: CxoTime,
    state_keys: tuple[str],
) -> Path:
    """Get file path for regression data based on start, stop, and state keys.

    This is within the `kadi/commands/tests/data` directory.
    """
    start_date = start.date[:8].replace(":", "")
    stop_date = stop.date[:8].replace(":", "")
    dates_keys_str = "-".join(state_keys + (start_date, stop_date))
    path = get_regression_pathdir() / f"{dates_keys_str}.ecsv"
    return path


def write_one_regression_data(
    start: CxoTimeLike,
    stop: CxoTimeLike,
    state_keys: tuple[str],
) -> None:
    """Write regression data to an ECSV file.

    This is used to generate regression data for a specific time range and combination
    of state keys.
    """
    start = CxoTime(start)
    stop = CxoTime(stop)

    # Need to set CXOTIME_NOW and lookback so that commands and states are regenerated
    # from scratch instead of grabbing from the commands archive. This accounts for
    # potential changes in command sets and all the other states machinery. An example
    # is the change to Safe Mode handling (2023:044) which used to explicitly include
    # a NSM transition for the maneuver, but no longer does so.
    lookback = (stop - start).jd + 1
    with (
        commands.conf.set_temp("default_lookback", lookback),
        ska_helpers.utils.temp_env_var("CXOTIME_NOW", (stop + 12 * u.hr).date),
    ):
        sts = states.get_states(
            start, stop, state_keys=state_keys, merge_identical=False
        )

    sts.remove_columns(["tstart", "tstop", "trans_keys"])
    for col in sts.itercols():
        if col.dtype.kind == "O":
            # Convert object columns to string
            sts[col.name] = [str(val) for val in col]

    path = get_regression_path(start, stop, state_keys)
    print(f"Writing regression data to {path.relative_to(Path.cwd())}")
    sts.meta["start"] = start.date
    sts.meta["stop"] = stop.date
    sts.meta["state_keys"] = state_keys
    sts.write(path, format="ascii.ecsv", overwrite=True)

    # Provide info on coverage of state keys
    if len(sts) == 1:
        print(" - Only one state found")
    else:
        for col in sts.itercols():
            if len(np.unique(col)) == 1:
                # Remove columns with only one unique value
                print(f" - {col.name} has only one unique value")


def write_regression_data(
    start: CxoTimeLike,
    stop: CxoTimeLike,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
):
    """
    Write regression data for all combinations of state keys for start and stop time.

    This is used to manually generate regression data for all state keys. A combination
    of state keys is a list of state keys that are updated together in the same
    transition or transition classes. An example is ['letg', 'hetg', 'grating']. This
    corresponds to the state keys listed here:

    https://sot.github.io/kadi/commands_states/index.html#state-keys

    When running this function, it is useful to examine the output to ensure that
    transitions are generated for all state keys.

    Example usage for baseline regression data. The period from 2023:030 to 2023:060
    includes a safe mode and NSM transition, and is used for attitude state key combos.
    The period from 2024:360 to 2025:030 is more normal but also has an SCS-107, the
    LETG anomaly and coverage of HRC commanding. For efficiency do not include the
    attitude states in the latter interval.

    >>> from kadi.commands.tests.test_states import write_regression_data
    >>> att_keys = ["pitch", "ra", "q1"]
    >>> write_regression_data("2023:030", "2023:060", include=att_keys)
    >>> write_regression_data("2024:360", "2025:030", exclude=att_keys)

    Parameters
    ----------
    start : CxoTimeLike
        Start time for the regression data.
    stop : CxoTimeLike
        Stop time for the regression data.
    include : list of str, optional
        List of state keys to include in the regression data.
    exclude : list of str, optional
        List of state keys to exclude from the regression data (applied after
        ``include``).
    """
    all_state_keys = states.get_state_keys_transition_classes()
    for state_keys in all_state_keys:
        fn = "-".join(state_keys)
        if include and not any(key in state_keys for key in include):
            print(f"Skipping {fn} because it is not in include list")
            continue
        if exclude and any(key in state_keys for key in exclude):
            print(f"Skipping {fn} because it is in exclude list")
            continue

        # NOTE: the commands caching machinery notices CXOTIME_NOW and uses that to decide
        # if there are cached commands available. The first time through this is going to
        # fetch all the backstop commands for the time range and then after cache them.

        # Get states from current test code using command states data and write to file
        write_one_regression_data(start, stop, state_keys)


def get_regression_filenames():
    for path in get_regression_pathdir().glob("*.ecsv"):
        yield str(path.relative_to(Path.cwd()))


@pytest.mark.parametrize("filename", list(get_regression_filenames()))
def test_regression(filename, monkeypatch):
    """Test current code vs. regression data in tests/data/regression"""
    # Read regression data
    sts_regr = Table.read(filename, format="ascii.ecsv")
    start = CxoTime(sts_regr.meta["start"])
    stop = CxoTime(sts_regr.meta["stop"])
    state_keys = sts_regr.meta["state_keys"]
    lookback = (stop - start).jd + 1

    # NOTE: the commands caching machinery notices CXOTIME_NOW and uses that to decide
    # if there are cached commands available. The first time through this is going to
    # fetch all the backstop commands for the time range and then after cache them.

    # Get states from current test code using command states data
    with (
        commands.conf.set_temp("default_lookback", lookback),
        ska_helpers.utils.temp_env_var("CXOTIME_NOW", (stop + 12 * u.hr).date),
    ):
        sts_test = states.get_states(
            start, stop, state_keys=state_keys, merge_identical=False
        )

    assert_all_close_states(sts_regr, sts_test)


def test_acis_simode():
    # Test that the simode state for HRC observations is correct
    # Before Nov 2022
    sts = states.get_states(
        "2019:140:01:04:00",
        "2019:146:23:52:00",
        merge_identical=True,
        state_keys=[
            "si_mode",
            "simpos",
            "clocking",
            "radmon",
            "obsid",
            "fep_count",
            "format",
        ],
    )
    idxs = (
        (sts["radmon"] == "ENAB") & (sts["clocking"] == 1) & (sts["format"] == "FMT1")
    )
    eidxs = idxs & (sts["obsid"] % 2 == 0)
    oidxs = idxs & (sts["obsid"] % 2 != 0)
    sidxs = sts["simpos"] < -90000
    iidxs = (sts["simpos"] > -60000) & (sts["simpos"] < -40000)
    chps4 = sts["fep_count"] == 4
    chps5 = sts["fep_count"] == 5
    assert np.all(sts["si_mode"][eidxs & iidxs & chps5] == "HIE_0002")
    assert np.all(sts["si_mode"][eidxs & iidxs & chps4] == "HIE_0003")
    assert np.all(sts["si_mode"][oidxs & iidxs] == "HIO_0002")
    assert np.all(sts["si_mode"][oidxs & sidxs] == "HSO_0002")
    # Between Nov 2022 and Feb 2024
    sts = states.get_states(
        "2023:060:02:00:00",
        "2023:260:02:00:00",
        merge_identical=True,
        state_keys=["si_mode", "simpos", "clocking", "radmon"],
    )
    idxs = (sts["radmon"] == "ENAB") & (sts["simpos"] < -50000) & (sts["clocking"] == 1)
    assert np.all(sts["si_mode"][idxs] == "H2C_0001")
    # After Feb 2024
    sts = states.get_states(
        "2024:040:02:00:00",
        "2024:050:02:00:00",
        merge_identical=True,
        state_keys=["si_mode", "simpos", "clocking", "radmon"],
    )
    idxs = (sts["radmon"] == "ENAB") & (sts["simpos"] < -50000) & (sts["clocking"] == 1)
    assert np.all(sts["si_mode"][idxs] == "H2C_0002")
    # Test bias and no-bias SIMODEs
    sts = states.get_states(
        "2024:028:05:53:56.363",
        "2024:028:17:17:52.589",
        merge_identical=True,
        state_keys=["si_mode", "obsid", "clocking"],
    )
    acis_run = sts["clocking"] == 1
    assert np.all(sts["si_mode"][acis_run & sts["obsid"] == 27148] == "TE_006E6B")
    assert np.all(sts["si_mode"][acis_run & sts["obsid"] == 27073] == "TE_006E6")
    assert np.all(sts["si_mode"][acis_run & sts["obsid"] == 29216] == "TE_006E6")

    # Minimal test for ACIS raw mode SI modes to verify that they are found in
    # a period of time known to have raw mode commanding.
    kstates = states.get_states(
        start="2017:189:12:00:00", stop="2017:197:12:00:00", state_keys=["si_mode"]
    )
    assert "TN_000B4" in kstates["si_mode"]
    assert "TN_000B6" in kstates["si_mode"]


def test_configurable_maneuver_rasl_step_time():
    cmd_evts_text = """\
State,Date,Event,Params,Author,Reviewer,Comment
Definitive,2024:024:01:00:00,NSM,,,,
Definitive,2024:024:02:00:00,Maneuver sun rasl,10,,,
"""
    cmds = get_cmds_from_cmd_evts_text(cmd_evts_text)
    with commands.conf.set_temp("maneuver_rasl_step_time", 55.0):
        sts = states.get_states(
            "2024:024:01:00:00",
            "2024:024:03:00:00",
            cmds=cmds,
            state_keys=["pcad_mode", "rasl"],
        )
    # Actual RASL step time is 50 seconds just below the configured value of 55.
    exp_text = """
      datestart              datestop       pcad_mode   rasl  trans_keys
2024:024:01:00:00.000 2024:024:01:10:29.075      NSUN 312.750  pcad_mode
2024:024:01:10:29.075 2024:024:02:00:00.000      NSUN 312.732       rasl
2024:024:02:00:00.000 2024:024:02:00:50.000      NSUN 312.726       rasl
2024:024:02:00:50.000 2024:024:02:01:40.000      NSUN 313.976       rasl
2024:024:02:01:40.000 2024:024:02:02:30.000      NSUN 315.226       rasl
2024:024:02:02:30.000 2024:024:02:03:20.000      NSUN 316.475       rasl
2024:024:02:03:20.000 2024:024:02:04:10.000      NSUN 317.725       rasl
2024:024:02:04:10.000 2024:024:02:05:00.000      NSUN 318.975       rasl
2024:024:02:05:00.000 2024:024:02:05:50.000      NSUN 320.225       rasl
2024:024:02:05:50.000 2024:024:02:06:40.000      NSUN 321.475       rasl
2024:024:02:06:40.000 2024:024:03:00:00.000      NSUN 322.718       rasl
"""
    exp = Table.read(exp_text, format="ascii.basic", guess=False)
    sts.pprint_exclude_names = ["tstart", "tstop"]
    sts["rasl"].format = ".3f"
    assert_all_close_states(exp, sts, atol=1e-2, rtol=0)


def test_configurable_maneuver_nman_step_time():
    # Maneuver from 200.278  46.089  86.615 to 200, 36, 86
    # with a step time of 120 seconds. Confirm output steps are about 2 minutes apart.
    cmd_evts_text = """\
State,Date,Event,Params,Author,Reviewer,Comment
Definitive,2024:024:00:00:01,Maneuver, 0.10993576 0.67800962 0.64839541 -0.32832983,,,
"""
    cmds = get_cmds_from_cmd_evts_text(cmd_evts_text)
    with commands.conf.set_temp("maneuver_nman_step_time", 120.0):
        sts = states.get_states(
            "2024:024:00:00:00",
            "2024:024:01:00:00",
            cmds=cmds,
            state_keys=["pcad_mode", "ra", "dec", "roll"],
        )
    sts.pprint_exclude_names = ["tstart", "tstop"]
    for name in ["ra", "dec", "roll"]:
        sts[name].format = ".3f"
    exp_text = """
      datestart              datestop       pcad_mode    ra     dec     roll   trans_keys
2024:024:00:00:00.000 2024:024:00:00:01.000      NPNT 200.278  46.089  86.615    ""
2024:024:00:00:01.000 2024:024:00:00:11.250      NMAN 200.278  46.089  86.615   pcad_mode
2024:024:00:00:11.250 2024:024:00:02:06.211      NMAN 200.276  46.023  86.611 dec,ra,roll
2024:024:00:02:06.211 2024:024:00:04:01.173      NMAN 200.248  44.806  86.535 dec,ra,roll
2024:024:00:04:01.173 2024:024:00:05:56.134      NMAN 200.176  41.953  86.359 dec,ra,roll
2024:024:00:05:56.134 2024:024:00:07:51.095      NMAN 200.078  38.508  86.150 dec,ra,roll
2024:024:00:07:51.095 2024:024:00:09:46.056      NMAN 200.015  36.468  86.028 dec,ra,roll
2024:024:00:09:46.056 2024:024:00:10:43.537      NMAN 200.000  36.000  86.000 dec,ra,roll
2024:024:00:10:43.537 2024:024:01:00:00.000      NPNT 200.000  36.000  86.000   pcad_mode
"""
    exp = Table.read(exp_text, format="ascii.basic", guess=False)
    assert_all_close_states(exp, sts, atol=1e-3, rtol=0)


def test_cmd_line_interface(tmpdir):
    """
    Test command line interface
    """
    filename = os.path.join(str(tmpdir), "out.txt")
    states.get_chandra_states(
        [
            "--outfile",
            filename,
            "--start",
            "2017:001:21:00:00",
            "--stop",
            "2017:002:11:30:00",
            "--state-keys",
            "obsid,si_mode,pcad_mode",
        ]
    )
    with open(filename) as fh:
        out = fh.read()

    # Work around bug in astropy 3.0 where text table output has `\r\r\n` line endings
    # due to putting os.linesep \r\n at the end of each line in universal newline mode,
    # at which point python turns the \n into \r\n.
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines == [
        "             datestart               datestop  obsid    si_mode  pcad_mode ",
        " 2017:001:21:00:00.000  2017:001:21:02:06.467  18140  TE_008FCB       NPNT ",
        " 2017:001:21:02:06.467  2017:001:21:05:06.467  18140  TE_008FCB       NMAN ",
        " 2017:001:21:05:06.467  2017:001:21:06:41.467  19973  TE_008FCB       NMAN ",
        " 2017:001:21:06:41.467  2017:001:21:23:02.282  19973  TE_00A58B       NMAN ",
        " 2017:001:21:23:02.282  2017:002:11:23:43.185  19973  TE_00A58B       NPNT ",
        " 2017:002:11:23:43.185  2017:002:11:26:43.185  19973  TE_00A58B       NMAN ",
        " 2017:002:11:26:43.185  2017:002:11:29:29.870  50432  TE_00A58B       NMAN ",
        " 2017:002:11:29:29.870  2017:002:11:30:00.000  50432  TE_00A58B       NPNT ",
    ]


@pytest.mark.skipif("not HAS_PITCH")
def test_sun_vec_versus_telemetry():
    """
    Test sun vector values `pitch` and `off_nominal_roll` versus flight telem.  Include
    Load maneuver at 2017:349:20:52:37.719 in DEC1117 with large pitch and
    off_nominal_roll change (from around zero to -17 deg).

    State values are within 1.5 degrees of telemetry.
    """

    state_keys = ["pitch", "off_nom_roll"]
    start, stop = "2017:349:10:00:00", "2017:350:10:00:00"
    cmds = commands.get_cmds(start, stop)
    rk = states.get_states(state_keys=state_keys, cmds=cmds, merge_identical=True)[
        -20:-1
    ]

    tstart = DateTime(rk["datestart"]).secs
    tstop = DateTime(rk["datestop"]).secs
    tmid = (tstart + tstop) / 2

    # Pitch from telemetry
    dat = fetch.Msid("pitch", tstart[0] - 100, tstop[-1] + 100)
    dat.interpolate(times=tmid)
    delta = np.abs(dat.vals - rk["pitch"])
    assert np.max(rk["pitch"]) - np.min(rk["pitch"]) > 75  # Big maneuver
    assert np.all(delta < 1.5)

    # Off nominal roll (not roll from ra,dec,roll) from telemetry
    dat = fetch.Msid("roll", tstart[0] - 100, tstop[-1] + 100)
    dat.interpolate(times=tmid)
    delta = np.abs(dat.vals - rk["off_nom_roll"])
    assert np.max(rk["off_nom_roll"]) - np.min(rk["off_nom_roll"]) > 20  # Large range
    assert np.all(delta < 1.5)


def test_dither():
    """Values look reasonable given load commands"""
    cmds = commands.get_cmds("2017:341:21:40:05", "2017:350:00:00:00")
    rk = states.get_states(
        state_keys=[
            "dither_phase_pitch",
            "dither_phase_yaw",
            "dither_ampl_pitch",
            "dither_ampl_yaw",
            "dither_period_pitch",
            "dither_period_yaw",
        ],
        cmds=cmds,
    )

    assert np.all(
        rk["datestart"]
        == [
            "2017:341:21:40:05.265",
            "2017:342:08:26:34.023",
            "2017:345:02:45:50.318",
            "2017:345:09:02:21.704",
            "2017:345:17:18:08.893",
            "2017:346:04:35:44.546",
            "2017:349:21:28:42.426",
        ]
    )
    assert np.all(rk["dither_phase_pitch"] == 0.0)
    assert np.all(rk["dither_phase_yaw"] == 0.0)
    ampls = [
        20.00149628163879,
        7.9989482580707696,
        20.00149628163879,
        7.9989482580707696,
        20.00149628163879,
        7.9989482580707696,
        20.00149628163879,
    ]
    assert np.allclose(rk["dither_ampl_pitch"], ampls, atol=1e-6, rtol=0)
    assert np.allclose(rk["dither_ampl_yaw"], ampls, atol=1e-6, rtol=0)
    assert np.allclose(
        rk["dither_period_pitch"],
        [
            768.5740994184024,
            707.13038356352104,
            768.5740994184024,
            707.13038356352104,
            768.5740994184024,
            707.13038356352104,
            768.5740994184024,
        ],
        atol=1e-6,
        rtol=0,
    )

    assert np.allclose(
        rk["dither_period_yaw"],
        [
            1086.9567572759399,
            999.99938521341483,
            1086.9567572759399,
            999.99938521341483,
            1086.9567572759399,
            999.99938521341483,
            1086.9567572759399,
        ],
        atol=1e-6,
        rtol=0,
    )


def test_fids_state():
    kstates = states.get_states("2023:001", "2023:002", state_keys=["fids"])
    exp = [
        "      datestart              datestop          fids   trans_keys",
        "--------------------- --------------------- --------- ----------",
        "2023:001:00:00:00.000 2023:001:03:23:59.954 {3, 4, 5}           ",
        "2023:001:03:23:59.954 2023:001:13:47:14.830     set()       fids",
        "2023:001:13:47:14.830 2023:001:13:47:15.830     set()       fids",
        "2023:001:13:47:15.830 2023:001:13:47:16.086       {2}       fids",
        "2023:001:13:47:16.086 2023:001:13:47:16.342    {2, 4}       fids",
        "2023:001:13:47:16.342 2023:001:18:09:53.020 {2, 4, 5}       fids",
        "2023:001:18:09:53.020 2023:001:18:09:54.020     set()       fids",
        "2023:001:18:09:54.020 2023:001:18:09:54.276       {1}       fids",
        "2023:001:18:09:54.276 2023:001:18:09:54.532    {1, 2}       fids",
        "2023:001:18:09:54.532 2023:001:23:10:51.222 {1, 2, 5}       fids",
        "2023:001:23:10:51.222 2023:001:23:10:52.222     set()       fids",
        "2023:001:23:10:52.222 2023:001:23:10:52.478       {2}       fids",
        "2023:001:23:10:52.478 2023:001:23:10:52.734    {2, 4}       fids",
        "2023:001:23:10:52.734 2023:002:00:00:00.000 {2, 4, 5}       fids",
    ]
    out = kstates["datestart", "datestop", "fids", "trans_keys"].pformat()
    assert out == exp


def test_get_continuity_regress(fast_sun_position_method):
    """Regression test against values produced by get_continuity during development.
    Correctness not validated for all values.
    The particular time of 2018:001:12:00:00 happens during a maneuver, so this
    tests a bug fix where maneuver transitions were leaking past the stop time.
    It also tests that all continuity times are before the stop time.
    """
    expected = {
        "ccd_count": 3,
        "clocking": 1,
        "dec": 29.541207067565384,
        "dither": "ENAB",
        "fep_count": 3,
        "hetg": "RETR",
        "letg": "RETR",
        "obsid": 20392,
        "off_nom_roll": -1.7094393884391508,
        "pcad_mode": "NMAN",
        "pitch": 132.28726587630078,
        "power_cmd": "XTZ0000005",
        "q1": -0.3244449073149954,
        "q2": -0.5849791307822495,
        "q3": -0.7432701449727902,
        "q4": 0.009187477373013092,
        "ra": 155.127026108128,
        "roll": 83.06347122856376,
        "si_mode": "TE_005C6B",
        "simfa_pos": -468,
        "simpos": 75624,
        "targ_q1": 0.304190361,
        "targ_q2": 0.445053899,
        "targ_q3": 0.787398757,
        "targ_q4": 0.298995734,
        "vid_board": 1,
    }

    dates = {
        "ccd_count": "2018:001:11:58:21.735",
        "clocking": "2018:001:11:59:28.735",
        "dec": "2018:001:11:57:10.940",
        "dither": "2017:364:11:51:48.955",
        "fep_count": "2018:001:11:58:21.735",
        "hetg": "2018:001:02:58:48.143",
        "letg": "2017:364:10:50:43.995",
        "obsid": "2018:001:11:55:05.818",
        "off_nom_roll": "2018:001:11:57:10.940",
        "pcad_mode": "2018:001:11:52:05.818",
        "pitch": "2018:001:11:57:10.940",
        "power_cmd": "2018:001:11:59:28.735",
        "q1": "2018:001:11:57:10.940",
        "q2": "2018:001:11:57:10.940",
        "q3": "2018:001:11:57:10.940",
        "q4": "2018:001:11:57:10.940",
        "ra": "2018:001:11:57:10.940",
        "roll": "2018:001:11:57:10.940",
        "si_mode": "2018:001:11:59:24.735",
        "simfa_pos": "2017:364:11:39:00.159",
        "simpos": "2018:001:02:55:13.804",
        "targ_q1": "2018:001:11:52:10.175",
        "targ_q2": "2018:001:11:52:10.175",
        "targ_q3": "2018:001:11:52:10.175",
        "targ_q4": "2018:001:11:52:10.175",
        "vid_board": "2018:001:11:58:21.735",
    }

    with states.disable_grating_move_duration():
        continuity = states.get_continuity("2018:001:12:00:00")

    for key, val in expected.items():
        if isinstance(val, (int, str)):
            assert continuity[key] == val
        else:
            assert np.isclose(continuity[key], val, rtol=0, atol=1e-7)
        assert continuity["__dates__"][key] == dates[key]
        assert continuity["__dates__"][key] < "2018:001:12:00:00.000"
        # Transitions with no spacecraft command (instead from injected maneuver
        # state breaks)
        manvr_keys = (
            "pitch",
            "off_nom_roll",
            "ra",
            "dec",
            "roll",
            "q1",
            "q2",
            "q3",
            "q4",
        )
        if key not in manvr_keys:
            cmds = commands.get_cmds(date=continuity["__dates__"][key])
            assert len(cmds) > 0


def test_get_continuity_vs_states():
    """
    Functional test: continuity for a certain date should be the same as the last states
    when fed in cmds up through that date.
    """
    date0 = "2017:014:12:00:00"
    # Get last state up through `date0`.  Hardwire the lookback here to 21 days.
    cmds = commands.get_cmds("2016:360:12:00:00", date0)
    sts = states.get_states(cmds=cmds, stop=date0)
    sts0 = sts[-1]

    continuity = states.get_continuity(date0)
    del continuity["__dates__"]

    for key, val in continuity.items():
        if isinstance(val, (int, str)):
            assert sts0[key] == val
        else:
            assert np.isclose(sts0[key], val, rtol=0, atol=1e-7)


def test_get_states_with_cmds_and_start_stop():
    """Test using get_states with supplied commands along with start and
    stop."""
    # Get 6 commands from 2020:001:02:55:00.000 to 2020:001:02:55:01.285
    # (just comm setup commanding)
    cmds = commands.get_cmds("2020:001:02:00:00", "2020:001:03:00:00")

    sts = states.get_states(cmds=cmds, state_keys=["fep_count"])
    assert len(sts) == 1
    assert sts["datestart"][0] == "2020:001:02:55:00.000"
    assert sts["datestop"][-1] == "2020:001:02:55:01.285"
    assert np.all(sts["fep_count"] == 4)

    sts = states.get_states(
        cmds=cmds,
        state_keys=["fep_count"],
        start="2020:001:00:00:00",
        stop="2020:002:00:00:00",
    )
    assert len(sts) == 1
    assert sts["datestart"][0] == "2020:001:00:00:00.000"
    assert sts["datestop"][-1] == "2020:002:00:00:00.000"
    assert np.all(sts["fep_count"] == 4)


def test_get_continuity_keys():
    """Test that output has only the desired state keys. Also test that one can
    provide a string instead of list of state keys"""
    continuity = states.get_continuity("2017:014:12:00:00", "clocking")
    assert set(continuity) == {"clocking", "__dates__"}


def test_get_continuity_fail():
    with pytest.raises(ValueError) as err:  # noqa: PT011
        states.get_continuity("2017:014:12:00:00", "letg", lookbacks=[3])
    assert "did not find transitions" in str(err)


@pytest.mark.parametrize("all_keys", [True, False])
def test_reduce_states_merge_identical(all_keys):
    tstart = np.arange(0, 5)
    tstop = np.arange(1, 6)
    datestart = DateTime(tstart).date
    datestop = DateTime(tstop).date
    dat0 = Table(
        [datestart, datestop, tstart, tstop],
        names=["datestart", "datestop", "tstart", "tstop"],
    )
    reduce_states = functools.partial(states.reduce_states, all_keys=all_keys)

    # Table with something that changes every time
    dat = dat0.copy()
    dat["vals"] = np.arange(5)
    dat["val1"] = 1
    dat["val_not_key"] = 2  # Not part of the key
    dr = reduce_states(dat, ["vals", "val1"], merge_identical=True)
    reduce_names = ["datestart", "datestop", "tstart", "tstop", "vals", "val1"]
    if all_keys:
        # All the original cols + trans_keys
        assert dr.colnames == dat.colnames + ["trans_keys"]
        assert np.all(dr[dat.colnames] == dat)
    else:
        # No `val_not_key` column
        assert dr.colnames == reduce_names + ["trans_keys"]
        assert np.all(dr[reduce_names] == dat[reduce_names])

    # Table with nothing that changes
    dat = dat0.copy()
    dat["vals"] = 1
    dat["val1"] = 1
    dr = reduce_states(dat, ["vals", "val1"], merge_identical=True)
    assert len(dr) == 1
    assert dr["datestart"][0] == dat["datestart"][0]
    assert dr["datestop"][0] == dat["datestop"][-1]

    # Table with edge changes
    dat = dat0.copy()
    dat["vals"] = [1, 0, 0, 0, 1]
    dr = reduce_states(dat, ["vals"], merge_identical=True)
    assert len(dr) == 3
    assert np.all(dr["datestart"] == dat["datestart"][[0, 1, 4]])
    assert np.all(dr["datestop"] == dat["datestop"][[0, 3, 4]])
    assert np.all(dr["vals"] == [1, 0, 1])

    # Table with multiple changes
    dat = dat0.copy()
    dat["val1"] = [1, 0, 1, 1, 1]
    dat["val2"] = [1, 1, 1, 1, 0]
    dr = reduce_states(dat, ["val1", "val2"], merge_identical=True)
    assert len(dr) == 4
    assert np.all(dr["datestart"] == dat["datestart"][[0, 1, 2, 4]])
    assert np.all(dr["datestop"] == dat["datestop"][[0, 1, 3, 4]])
    assert np.all(dr["val1"] == [1, 0, 1, 1])
    assert np.all(dr["val2"] == [1, 1, 1, 0])
    assert str(dr["trans_keys"][0]) == "val1,val2"
    assert dr["trans_keys"][0] == {"val1", "val2"}
    assert dr["trans_keys"][1] == {"val1"}
    assert dr["trans_keys"][2] == {"val1"}
    assert dr["trans_keys"][3] == {"val2"}


###########################################################################
# Comparing to backstop history files or regression outputs
###########################################################################


def compare_backstop_history(history, state_key, *, compare_val=True):
    hist = ascii.read(
        history,
        guess=False,
        format="no_header",
        converters={"col1": [ascii.convert_numpy(str)]},
    )
    start = DateTime(hist["col1"][0], format="greta") - 1 / 86400.0
    stop = DateTime(hist["col1"][-1], format="greta") + 1 / 86400.0
    sts = states.get_states(start=start, stop=stop, state_keys=state_key)
    sts = sts[1:]  # Drop the first state (which is continuity at start time)
    assert len(sts) == len(hist)
    assert np.all(DateTime(sts["datestart"]).greta == hist["col1"])
    if compare_val:
        assert np.all(sts[state_key] == hist["col3"])


def test_backstop_format():
    """
    Check format state against backstop history for a period including
    formats 1, 2, 4, 6.  Note: the backstop history file uses values
    like CSELFMT6 while kadi uses FMT6.  So this history file data has
    been modified by hand.
    """
    history = """
2018175.023217769 | FMT4
2018175.023635569 | FMT2
2018175.112341049 | FMT4
2018175.112758849 | FMT2
2018175.145145543 | FMT4
2018175.145603343 | FMT1
2018176.042725446 | FMT4
2018176.043143246 | FMT2
2018180.025446998 | FMT6
2018180.025918998 | FMT2
2018182.084127442 | FMT4
2018182.084545242 | FMT2
2018183.014029332 | FMT4
2018183.014447132 | FMT2
2018183.165708250 | FMT1
2018183.202141577 | FMT2
2018183.224415139 | FMT4
2018183.224832939 | FMT2
2018184.073919399 | FMT4
2018184.074337199 | FMT2
2018184.155253648 | FMT4
2018184.155711448 | FMT2
2018185.011524641 | FMT4
2018185.011942441 | FMT2
2018185.174452818 | FMT4
2018185.174910618 | FMT2
2018186.023832494 | FMT4
2018186.024250294 | FMT2
2018186.164746947 | FMT4
2018186.165204747 | FMT2
2018186.225430494 | FMT4
2018186.225848294 | FMT2
2018187.044624742 | FMT4
2018187.045042542 | FMT2
2018188.084459620 | FMT4
2018188.084917420 | FMT2
2018189.020632813 | FMT4
2018189.021050613 | FMT2
2018189.145628609 | FMT4
2018189.150046409 | FMT2
2018191.084412307 | FMT1
2018191.121427923 | FMT4
2018191.121845723 | FMT2
2018191.204552786 | FMT4
2018191.205010586 | FMT2
2018192.070720386 | FMT4
2018192.071138186 | FMT2
2018194.021855327 | FMT4
2018194.022658127 | FMT2
2018194.053425927 | FMT1
2018194.082940727 | FMT2
2018194.113213927 | FMT1
2018194.142728727 | FMT2
2018194.173001927 | FMT1
2018194.202516727 | FMT4
2018194.203319527 | FMT2
2018195.232506128 | FMT4
2018195.232923928 | FMT2
2018196.233532760 | FMT1
2018197.111225660 | FMT2
2018204.140820832 | FMT4
2018204.141238632 | FMT2
2018204.230033777 | FMT4
2018204.230451577 | FMT2
2018209.131913154 | FMT4
2018209.132330954 | FMT1
2018209.191502370 | FMT4
2018209.191920170 | FMT2"""
    compare_backstop_history(history, "format")


def test_backstop_subformat():
    """
    Check subformat state against backstop history.
    """
    history = """
2018174.171002313 | NORM
2018175.053002313 | NORM
2018175.170002313 | NORM
2018175.234002313 | NORM
2018176.035502313 | NORM
2018176.131002313 | NORM
2018176.212002313 | NORM
2018177.055602570 | NORM
2018177.084602313 | NORM
2018177.210502313 | NORM
2018178.051002313 | NORM
2018178.165002313 | NORM
2018178.234002313 | NORM
2018179.025502313 | NORM
2018179.125502313 | NORM
2018179.164502313 | NORM
2018180.020000000 | EPS
2018180.020201000 | NORM
2018180.054002313 | NORM
2018180.162502313 | NORM
2018180.210002313 | NORM
2018181.040502313 | NORM
2018181.130002313 | NORM
2018181.191002313 | NORM
2018182.025502313 | NORM
2018182.155702313 | NORM
2018182.210502313 | NORM
2018183.040002313 | NORM
2018183.060000000 | EPS
2018183.060201000 | NORM
2018183.162502313 | NORM
2018183.212502313 | NORM
2018184.052502313 | NORM
2018184.191502313 | NORM
2018185.011502313 | NORM
2018185.075302313 | NORM
2018185.103135601 | EPS
2018185.103338601 | NORM
2018185.202002313 | NORM
2018186.035502313 | NORM
2018186.151002313 | NORM
2018186.200502313 | NORM
2018187.043502313 | NORM
2018187.120002313 | NORM
2018188.003002313 | NORM
2018188.053002313 | NORM
2018188.151502313 | NORM
2018188.211502313 | NORM
2018189.044502313 | NORM
2018189.141002313 | NORM
2018189.214502313 | NORM
2018190.021502313 | NORM
2018190.181502313 | NORM
2018191.033002313 | NORM
2018191.135002313 | NORM
2018191.161502313 | NORM
2018192.030002313 | NORM
2018192.150002313 | NORM
2018193.025502570 | NORM
2018193.134002313 | NORM
2018193.213002313 | NORM
2018194.053002313 | NORM
2018194.153502313 | NORM
2018194.192502313 | NORM
2018195.053002313 | NORM
2018195.142002313 | NORM
2018195.225002313 | NORM
2018196.062502313 | NORM
2018196.135002313 | NORM
2018196.214002313 | NORM
2018197.024002313 | NORM
2018197.134002313 | NORM
2018197.204002313 | NORM
2018198.060002313 | NORM
2018198.123002313 | NORM
2018198.194002313 | NORM
2018199.024502313 | NORM
2018199.130502313 | NORM
2018199.203002313 | NORM
2018200.023502313 | NORM
2018200.151002313 | NORM
2018200.224002313 | NORM
2018201.014502570 | NORM
2018201.131502313 | NORM
2018201.192502313 | NORM
2018202.023002313 | NORM
2018202.123002313 | NORM
2018202.223502313 | NORM
2018203.053002313 | NORM
2018203.123502313 | NORM
2018203.223002313 | NORM
2018204.044502313 | NORM
2018204.144002313 | NORM
2018204.183502313 | NORM
2018205.053002313 | NORM
2018205.124502313 | NORM
2018205.232002313 | NORM
2018206.034002313 | NORM
2018206.083102570 | NORM
2018206.190502313 | NORM
2018207.032502313 | NORM
2018207.132502313 | NORM
2018207.190502313 | NORM
2018208.053002313 | NORM
2018208.125502313 | NORM
2018208.232502570 | NORM
2018209.040502313 | NORM
2018209.134002313 | NORM
2018209.201502313 | NORM
2018210.033002313 | NORM
2018210.141502313 | NORM
2018210.210002313 | NORM
2018211.031002313 | NORM
"""
    compare_backstop_history(history, "subformat")


def test_backstop_sun_pos_mon():
    """
    Check sun_pos_mon state against backstop history for a period including
    eclipses (DEC2517 loads).
    """
    history = """
2017358.213409495 | DISA
2017359.070917145 | ENAB
2017359.110537952 | DISA
2017359.133903984 | ENAB
2017359.135819633 | DISA
2017359.152229633 | ENAB
2017359.163011934 | DISA
2017359.182223865 | ENAB
2017359.204011816 | DISA
2017360.031705497 | ENAB
2017360.153229134 | DISA
2017361.002600047 | ENAB
2017361.133347996 | DISA
2017362.050738157 | ENAB
2017362.052653806 | DISA
2017362.065643806 | ENAB
2017362.072949107 | DISA
2017362.093438150 | ENAB
2017362.102906100 | DISA
2017362.150855879 | ENAB
2017363.030442589 | DISA
2017363.123151176 | ENAB
2017363.180119126 | DISA
2017364.030554007 | ENAB
2017364.105021957 | DISA
2017364.184204432 | ENAB
2017364.205923508 | DISA
2017364.222743508 | ENAB
2017364.223855809 | DISA
2017365.041756837 | ENAB
2017365.071023324 | DISA
2017365.100753744 | ENAB
2018001.115218119 | DISA
2018001.204528158 | ENAB
2018001.215636109 | DISA
2018002.010801542 | ENAB
2018002.123611436 | DISA
2018002.135651436 | ENAB
2018002.140743737 | DISA
2018002.161832404 | ENAB
2018002.165620354 | DISA
2018003.000920504 | ENAB
2018003.005818020 | DISA
2018003.044025204 | ENAB
2018004.053611930 | DISA
2018004.095807597 | ENAB
2018004.110915547 | DISA
2018005.020152908 | ENAB
2018005.041826278 | DISA
2018005.052326278 | ENAB
2018005.053238579 | DISA
2018005.072207093 | ENAB
2018005.123435064 | DISA
2018005.153821753 | ENAB
2018006.072924150 | DISA
2018006.222954700 | ENAB
2018007.024422649 | DISA
"""
    compare_backstop_history(history, "sun_pos_mon")


def test_backstop_sun_pos_mon_lunar():
    """
    Test that sun_pos_mon is correct for a lunar eclipse:
    2017:087:07:49:55.838 ORBPOINT    scs=0 step=0 timeline_id=426102503 event_type=LSPENTRY
    2017:087:08:10:35.838 ORBPOINT    scs=0 step=0 timeline_id=426102503 event_type=LSPEXIT

    We expect SPM enable at the LSPEXIT time + 11 minutes = 2017:087:08:21:35.838
    """
    history = """
      datestart              datestop       sun_pos_mon
2017:087:03:04:02.242 2017:087:07:21:40.189        DISA
2017:087:07:21:40.189 2017:087:07:44:55.838        ENAB
2017:087:07:44:55.838 2017:087:08:21:35.838        DISA
# Here is the relevant entry:
2017:087:08:21:35.838 2017:087:08:30:50.891        ENAB
#
2017:087:08:30:50.891 2017:087:17:18:18.571        DISA
2017:087:17:18:18.571 2017:087:23:59:21.087        ENAB
2017:087:23:59:21.087 2099:365:00:00:00.000        DISA
"""
    spm = ascii.read(history, guess=False)
    cmds = commands.get_cmds("2017:087:03:04:02", "2017:088:00:00:00")
    sts = states.get_states(state_keys=["sun_pos_mon"], cmds=cmds)

    assert len(sts) == len(spm)
    assert np.all(sts["datestart"] == spm["datestart"])
    assert np.all(sts["sun_pos_mon"] == spm["sun_pos_mon"])


def test_backstop_ephem_update():
    history = """
2017107.051708675 | EPHEMERIS
2017109.204526240 | EPHEMERIS
2017112.121320933 | EPHEMERIS
2017115.034040060 | EPHEMERIS
2017117.190746809 | EPHEMERIS
"""
    compare_backstop_history(history, "ephem_update", compare_val=False)


def test_backstop_radmon_no_scs107():
    """
    Test radmon states for nominal loads.  The current non-load commands from
    chandra_cmd_states does not have the RADMON disable associated with SCS107
    runs.  This test does have a TOO interrupt.
    """
    history = """
2017288.180259346 | ENAB OORMPEN
2017290.180310015 | DISA OORMPDS
2017291.092509015 | ENAB OORMPEN
2017293.102122372 | DISA OORMPDS
2017294.002721372 | ENAB OORMPEN
2017296.011402807 | DISA OORMPDS
#*****< COMBINED WEEKLY LOAD OCT2317B START @ 2017296.023600000 >*****
2017296.163001807 | ENAB OORMPEN
#*****< TOO INTERRUPT @ 2017298.035018327 >*****
#*****< COMBINED WEEKLY LOAD OCT2517A START @ 2017298.035018327 >*****
2017298.163305593 | DISA OORMPDS
2017299.074904593 | ENAB OORMPEN
2017301.074557816 | DISA OORMPDS
2017301.225605168 | ENAB OORMPEN
#*****< COMBINED WEEKLY LOAD OCT3017A START @ 2017302.203127168 >*****
2017303.223102296 | DISA OORMPDS
2017304.145230146 | ENAB OORMPEN
2017306.141547552 | DISA OORMPDS
2017307.060715402 | ENAB OORMPEN
2017309.054704675 | DISA OORMPDS
2017309.212423675 | ENAB OORMPEN
#*****< COMBINED WEEKLY LOAD NOV0617A START @ 2017310.083948275 >*****
2017311.204210370 | DISA OORMPDS
2017312.131224228 | ENAB OORMPEN
2017314.122303220 | DISA OORMPDS
"""
    compare_backstop_history(history, "radmon")


@pytest.mark.xfail()
def test_backstop_radmon_with_scs107():
    """
    Test radmon states for nominal loads.  The current non-load commands from
    chandra_cmd_states does not have the RADMON disable associated with SCS107
    runs.  This test will fail until that is corrected.
    """
    history = """
#*****< COMBINED WEEKLY LOAD FEB2717B START @ 2017058.030722450 >*****
2017058.142355572 | ENAB OORMPEN
2017060.163220418 | DISA OORMPDS
2017061.063059418 | ENAB OORMPEN
2017063.063213151 | DISA OORMPDS
2017063.220905151 | ENAB OORMPEN
#*****< COMBINED WEEKLY LOAD MAR0617A START @ 2017065.032756098 >*****
2017065.234830643 | DISA OORMPDS
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017066.002421000 >*****
2017066.002421000 | DISA OORMPDS
#*****< COMBINED WEEKLY LOAD MAR0817B START @ 2017067.045700000 >*****
2017067.045759000 | ENAB OORMPEN
2017068.132618302 | DISA OORMPDS
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017068.170500000 >*****
2017068.170500000 | DISA OORMPDS
#*****< COMBINED WEEKLY LOAD MAR1117A START @ 2017070.034200000 >*****
2017070.034302000 | ENAB OORMPEN
2017071.050459069 | DISA OORMPDS
2017071.202623069 | ENAB OORMPEN
2017073.211120865 | DISA OORMPDS
2017074.112119865 | ENAB OORMPEN
#*****< GENERAL REPLAN @ 2017075.025717891 >*****
#*****< COMBINED WEEKLY LOAD MAR1517B START @ 2017075.025717891 >*****
2017076.114752248 | DISA OORMPDS
2017077.032151248 | ENAB OORMPEN
2017079.033650539 | DISA OORMPDS
#*****< COMBINED WEEKLY LOAD MAR2017E START @ 2017079.062200000 >*****
2017079.181742105 | ENAB OORMPEN
2017081.193300753 | DISA OORMPDS
2017082.095459753 | ENAB OORMPEN
2017084.101011889 | DISA OORMPDS
2017085.014410889 | ENAB OORMPEN
#*****< COMBINED WEEKLY LOAD MAR2717B START @ 2017086.112306039 >*****
2017087.020909786 | DISA OORMPDS
2017087.171226786 | ENAB OORMPEN
2017089.174813242 | DISA OORMPDS
2017090.082412242 | ENAB OORMPEN
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017090.190157000 >*****
2017090.190157000 | DISA OORMPDS
2017091.012000000 | ENAB OORMPEN
2017092.012000000 | DISA OORMPDS
#*****< COMBINED WEEKLY LOAD APR0217B START @ 2017092.015641495 >*****
2017092.015740495 | ENAB OORMPEN
2017092.083131454 | DISA OORMPDS
2017093.000530454 | ENAB OORMPEN
2017095.003842982 | DISA OORMPDS
2017095.145241982 | ENAB OORMPEN
2017097.160524315 | DISA OORMPDS
2017098.065323315 | ENAB OORMPEN
"""
    compare_backstop_history(history, "radmon")


def test_backstop_scs98():
    history = """
#*****< COMBINED WEEKLY LOAD MAR0617A START @ 2017065.032756098 >*****
2017065.033109424 | ENAB
2017065.172454796 | DISA
2017065.235443439 | ENAB
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017066.002421000 >*****
#*****< COMBINED WEEKLY LOAD MAR0817B START @ 2017067.045700000 >*****
2017067.050013326 | ENAB
2017068.060730723 | DISA
2017068.133216572 | ENAB
2017068.165938365 | DISA
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017068.170500000 >*****
#*****< COMBINED WEEKLY LOAD MAR1117A START @ 2017070.034200000 >*****
2017070.122206153 | DISA
2017070.201316153 | ENAB
2017070.235735378 | DISA
2017071.153104906 | ENAB
2017072.064807993 | DISA
2017072.124449411 | ENAB
2017072.232046570 | DISA
2017073.222841760 | ENAB
2017074.003843326 | DISA
2017074.105425944 | ENAB
#*****< GENERAL REPLAN @ 2017075.025717891 >*****
#*****< COMBINED WEEKLY LOAD MAR1517B START @ 2017075.025717891 >*****
2017075.043355842 | DISA
2017075.181053442 | ENAB
2017076.005534622 | DISA
2017076.115452631 | ENAB
2017076.125543326 | DISA
2017076.191617099 | ENAB
2017077.010443326 | DISA
2017077.025510711 | ENAB
2017077.192403829 | DISA
2017078.102835409 | ENAB
2017078.172100604 | DISA
"""
    compare_backstop_history(history, "scs98")


def test_backstop_scs84():
    """
    SCS 84 has not changed from DISA since 2006 due to a change in operations.
    """
    sts = states.get_states(
        start="2017:001:12:00:00", stop="2017:300:12:00:00", state_keys=["scs84"]
    )
    assert len(sts) == 1
    assert sts[0]["scs84"] == "DISA"
    assert sts[0]["datestart"] == "2017:001:12:00:00.000"
    assert sts[0]["datestop"] == "2017:300:12:00:00.000"


def test_backstop_simpos():
    """
    This changes the SCS times by up to a second to match kadi non-load commands.
    """
    history = """
#*****< COMBINED WEEKLY LOAD MAR0617A START @ 2017065.032756098 >*****
2017065.033056098 | 92904
2017065.234230433 | -99616
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017066.002421000 >*****
#
# <<< HAND-EDIT to match kadi non-load commands >>>
#
2017066.002422025 | -99616
#*****< COMBINED WEEKLY LOAD MAR0817B START @ 2017067.045700000 >*****
2017067.050000000 | 75624
2017067.150632506 | 92904
2017067.215047348 | 75624
2017068.054245410 | 92904
2017068.132018092 | -99616
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017068.170500000 >*****
#
# <<< HAND-EDIT to match kadi non-load commands >>>
#
2017068.170043025 | -99616
#*****< COMBINED WEEKLY LOAD MAR1117A START @ 2017070.034200000 >*****
2017070.034500000 | 75624
2017070.201302827 | 92904
2017071.045858859 | -99616
2017071.202624069 | 92904
2017072.062719085 | 75624
2017072.124436085 | 92904
2017072.225359330 | -99616
2017073.082839602 | 75624
2017073.210544431 | -99616
2017074.112120865 | 92904
"""
    compare_backstop_history(history, "simpos")


def test_backstop_simfa_pos():
    history = """
2017064.080620766 | -468
2017064.172912782 | -991
2017064.221612657 | -418
2017064.221740517 | -468
#*****< COMBINED WEEKLY LOAD MAR0617A START @ 2017065.032756098 >*****
2017065.033255274 | -536
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017066.002421000 >*****
#*****< COMBINED WEEKLY LOAD MAR0817B START @ 2017067.045700000 >*****
2017067.050536434 | -418
2017067.050655194 | -468
2017067.150831682 | -536
2017067.215246524 | -418
2017067.215405284 | -468
2017068.054444586 | -536
#*****< LOSS OF ATTITUDE REF WITH SCS 107 EXECUTION @ 2017068.170500000 >*****
#*****< COMBINED WEEKLY LOAD MAR1117A START @ 2017070.034200000 >*****
2017070.035036434 | -418
2017070.035155194 | -468
2017070.201502003 | -536
2017072.062918261 | -418
2017072.063037021 | -468
2017072.124635261 | -536
2017072.225959540 | -991
2017073.083416036 | -418
2017073.083543896 | -468
2017074.112721075 | -536
#*****< GENERAL REPLAN @ 2017075.025717891 >*****
#*****< COMBINED WEEKLY LOAD MAR1517B START @ 2017075.025717891 >*****
2017075.041328208 | -991
2017075.150028950 | -418
"""
    compare_backstop_history(history, "simfa_pos")


def test_backstop_grating():
    history = """
2017177.220854816 | HETG HETGIN
2017178.061645416 | NONE HETGRE
2017179.000620925 | LETG LETGIN
2017180.214634037 | NONE LETGRE
2017180.215004808 | HETG HETGIN
2017181.144236212 | NONE HETGRE
2017181.144606983 | LETG LETGIN
2017182.110550792 | NONE LETGRE
#*****< COMBINED WEEKLY LOAD JUL0317B START @ 2017184.073335870 >*****
2017184.074010209 | HETG HETGIN
2017184.131300809 | NONE HETGRE
2017185.235853273 | HETG HETGIN
2017186.171507322 | NONE HETGRE
2017187.045154922 | HETG HETGIN
2017188.010601539 | NONE HETGRE
2017189.103922638 | HETG HETGIN
#*****< COMBINED WEEKLY LOAD JUL1017A START @ 2017191.062621188 >*****
2017191.063255527 | NONE HETGRE
"""
    compare_backstop_history(history, "grating")


def test_backstop_eclipse_entry():
    history = """
2017172.170405996 | 1639 EOECLETO
2017175.082556900 | 1171 EOECLETO
2017191.064050486 | 1873 EOECLETO
2017193.083930486 | 1171 EOECLETO
2017354.160001000 | 1639 EOECLETO
2017357.080001000 | 1171 EOECLETO
2018003.040001000 | 1405 EOECLETO
2018005.071227278 | 1171 EOECLETO
"""
    compare_backstop_history(history, "eclipse_timer")


def compare_regress_output(regress, state_key):
    regr = ascii.read(regress, format="fixed_width_two_line", fill_values=None)
    start = regr["datestart"][0]
    stop = regr["datestop"][-1]
    sts = states.get_states(start, stop, state_keys=[state_key])
    for colname in ("datestart", "datestop", state_key):
        assert np.all(regr[colname] == sts[colname])


def test_regress_eclipse():
    """
    Regression check for day 2017:087 lunar eclipse and the 2017 winter
    eclipse season.  Values match fairly closely (within 20 seconds) the backstop values
    (presumably those were produced before loads?)

    Start Time (UTCJFOUR)    Stop Time (UTCJFOUR)    Duration (sec)    Current Condition
    ---------------------    --------------------    --------------    -----------------
     356/2017 22:40:24.71    356/2017 22:44:31.35           246.636    Penumbra
     356/2017 22:44:31.35    356/2017 23:28:03.07          2611.725    Umbra
     356/2017 23:28:03.07    356/2017 23:31:39.57           216.503    Penumbra

     359/2017 14:03:19.99    359/2017 14:06:17.01           177.018    Penumbra
     359/2017 14:06:17.01    359/2017 15:08:46.48          3749.467    Umbra
     359/2017 15:08:46.48    359/2017 15:11:14.26           147.781    Penumbra

     362/2017 05:31:49.63    362/2017 05:34:27.02           157.393    Penumbra
     362/2017 05:34:27.02    362/2017 06:43:19.69          4132.669    Umbra
     362/2017 06:43:19.69    362/2017 06:45:28.73           129.045    Penumbra

     364/2017 21:04:13.31    364/2017 21:06:47.45           154.137    Penumbra
     364/2017 21:06:47.45    364/2017 22:14:19.84          4052.392    Umbra
     364/2017 22:14:19.84    364/2017 22:16:26.52           126.682    Penumbra

     002/2018 12:40:55.68    002/2018 12:43:40.19           164.507    Penumbra
     002/2018 12:43:40.19    002/2018 13:43:08.52          3568.326    Umbra
     002/2018 13:43:08.52    002/2018 13:45:26.46           137.942    Penumbra

     005/2018 04:23:10.67    005/2018 04:26:38.91           208.237    Penumbra
     005/2018 04:26:38.91    005/2018 05:09:03.73          2544.820    Umbra
     005/2018 05:09:03.73    005/2018 05:12:06.26           182.528    Penumbra

    """
    regress = """
      datestart              datestop       eclipse  trans_keys
--------------------- --------------------- -------- ----------
2017:080:12:00:00.000 2017:087:07:49:55.838      DAY
2017:087:07:49:55.838 2017:087:08:10:35.838 PENUMBRA    eclipse
2017:087:08:10:35.838 2017:090:12:00:00.000      DAY    eclipse
"""
    compare_regress_output(regress, "eclipse")

    regress = """
      datestart              datestop       eclipse  trans_keys
--------------------- --------------------- -------- ----------
2017:355:12:00:00.000 2017:356:22:40:15.030      DAY
2017:356:22:40:15.030 2017:356:22:44:25.030 PENUMBRA    eclipse
2017:356:22:44:25.030 2017:356:23:28:15.030    UMBRA    eclipse
2017:356:23:28:15.030 2017:356:23:31:45.030 PENUMBRA    eclipse

2017:356:23:31:45.030 2017:359:14:03:19.633      DAY    eclipse
2017:359:14:03:19.633 2017:359:14:06:19.633 PENUMBRA    eclipse
2017:359:14:06:19.633 2017:359:15:08:59.633    UMBRA    eclipse
2017:359:15:08:59.633 2017:359:15:11:29.633 PENUMBRA    eclipse

2017:359:15:11:29.633 2017:362:05:31:53.806      DAY    eclipse
2017:362:05:31:53.806 2017:362:05:34:33.806 PENUMBRA    eclipse
2017:362:05:34:33.806 2017:362:06:43:33.806    UMBRA    eclipse
2017:362:06:43:33.806 2017:362:06:45:43.806 PENUMBRA    eclipse

2017:362:06:45:43.806 2017:364:21:04:23.508      DAY    eclipse
2017:364:21:04:23.508 2017:364:21:06:53.508 PENUMBRA    eclipse
2017:364:21:06:53.508 2017:364:22:14:43.508    UMBRA    eclipse
2017:364:22:14:43.508 2017:364:22:16:43.508 PENUMBRA    eclipse

2017:364:22:16:43.508 2018:002:12:41:11.436      DAY    eclipse
2018:002:12:41:11.436 2018:002:12:43:51.436 PENUMBRA    eclipse
2018:002:12:43:51.436 2018:002:13:43:31.436    UMBRA    eclipse
2018:002:13:43:31.436 2018:002:13:45:51.436 PENUMBRA    eclipse

2018:002:13:45:51.436 2018:005:04:23:26.278      DAY    eclipse
2018:005:04:23:26.278 2018:005:04:26:56.278 PENUMBRA    eclipse
2018:005:04:26:56.278 2018:005:05:09:26.278    UMBRA    eclipse
2018:005:05:09:26.278 2018:005:05:12:26.278 PENUMBRA    eclipse
2018:005:05:12:26.278 2018:006:12:00:00.000      DAY    eclipse
"""
    compare_regress_output(regress, "eclipse")


def test_continuity_just_after_command():
    """
    Fix issue where continuity required having at least one other
    command after the relevant continuity command.
    """
    cont = states.get_continuity("2018:001:12:00:00", "targ_q1")
    assert cont["__dates__"]["targ_q1"] == "2018:001:11:52:10.175"

    # 1 msec later
    cont = states.get_continuity("2018:001:11:52:10.176", "targ_q1")
    assert cont["__dates__"]["targ_q1"] == "2018:001:11:52:10.175"


def test_continuity_far_future():
    """
    Test that using a continuity date in the future (well past commamds)
    works.
    """
    # Now + 150 days.  Don't know the answer but just make sure this
    # runs to completion.  The lookbacks of 7 and 30 days should fail
    # but 180 days will get something.
    cont = states.get_continuity(DateTime() + 150, "obsid")
    assert "obsid" in cont
    assert "obsid" in cont["__dates__"]


def test_acis_power_cmds():
    start = "2017:359:13:37:50"
    stop = "2017:360:00:46:00"
    state_keys = ["power_cmd", "ccd_count", "fep_count", "vid_board"]
    cmds = commands.get_cmds(start, stop)
    continuity = states.get_continuity(start, state_keys)
    test_states = states.get_states(
        cmds=cmds, continuity=continuity, state_keys=state_keys, reduce=False
    )
    vid_dn = np.where(test_states["power_cmd"] == "WSVIDALLDN")[0]
    assert (test_states["ccd_count"][vid_dn] == 0).all()
    assert (
        test_states["fep_count"][vid_dn] == test_states["fep_count"][vid_dn[0] - 1]
    ).all()
    pow_zero = np.where(test_states["power_cmd"] == "WSPOW00000")[0]
    assert (test_states["ccd_count"][pow_zero] == 0).all()
    assert (test_states["fep_count"][pow_zero] == 0).all()


def test_continuity_with_transitions_SPM():  # noqa: N802
    """
    Test that continuity returns a dict that has the __transitions__ key set
    to a list of transitions that are needed for correct continuity.  Part of
    fix for #125.

    Start time is one minute before auto-re-enable of SPM during eclipse handling
    This is in the middle of the transitions generated by SPMEnableTransition.
    The test here is seeing the ENAB transition that is one minute after the
    continuity start time.  This will be used by get_states() to get things
    right using this continuity dict.
    """
    start = "2017:087:08:20:35.838"
    stop = "2017:087:10:20:35.838"
    cont = states.get_continuity(start, state_keys=["sun_pos_mon"])
    assert cont == {
        "__dates__": {"sun_pos_mon": "2017:087:07:44:55.838"},
        "__transitions__": [
            states.Transition("2017:087:08:21:35.838", {"sun_pos_mon": "ENAB"})
        ],
        "sun_pos_mon": "DISA",
    }

    # fmt: off
    exp = [
        "      datestart              datestop           tstart        tstop     "
        "sun_pos_mon  trans_keys",
        "--------------------- --------------------- ------------- ------------- "
        "----------- -----------",
        "2017:087:08:20:35.838 2017:087:08:21:35.838 607076505.022 "
        "607076565.022        DISA            ",
        "2017:087:08:21:35.838 2017:087:08:30:50.891 607076565.022 "
        "607077120.075        ENAB sun_pos_mon",
        "2017:087:08:30:50.891 2017:087:10:20:35.838 607077120.075 "
        "607083705.022        DISA sun_pos_mon",
    ]
    # fmt: on
    sts = states.get_states(start, stop, state_keys=["sun_pos_mon"])
    assert sts.pformat(max_lines=-1, max_width=-1) == exp


def test_continuity_with_no_transitions_SPM():  # noqa: N802
    """Test that continuity returns a dict that does NOT have the __transitions__
    key set if not needed.  Part of fix for #125.

    """
    cont = states.get_continuity("2017:001:12:00:00", state_keys=["sun_pos_mon"])
    assert cont == {
        "sun_pos_mon": "DISA",
        "__dates__": {"sun_pos_mon": "2017:001:04:23:55.764"},
    }


def test_get_pitch_from_mid_maneuver(fast_sun_position_method):
    """Regression test for the fix for #125.  Mostly the same as the test above, but for
    the Maneuver transition class.

    """
    start = "2019:039:14:16:25.002"  # Mid-maneuver
    stop = "2019:039:16:00:00.000"
    exp = [
        "      datestart              datestop             pitch        pcad_mode",
        "--------------------- --------------------- ------------------ ---------",
        "2019:039:14:16:25.002 2019:039:14:21:00.332  79.18318279740022      NMAN",
        "2019:039:14:21:00.332 2019:039:14:25:47.295 100.68142145564228      NMAN",
        "2019:039:14:25:47.295 2019:039:14:30:34.258 122.18389895580964      NMAN",
        "2019:039:14:30:34.258 2019:039:14:35:21.221  143.4709296459047      NMAN",
        "2019:039:14:35:21.221 2019:039:14:40:08.184 157.81546233001538      NMAN",
        "2019:039:14:40:08.184 2019:039:14:42:31.665 161.95150302398088      NMAN",
        "2019:039:14:42:31.665 2019:039:16:00:00.000 161.93209261077413      NPNT",
    ]
    exp = Table.read(exp, format="ascii")

    sts = states.get_states(start, stop, state_keys=["pitch", "pcad_mode"])

    assert np.all(exp["datestart"] == sts["datestart"])
    assert np.all(exp["datestop"] == sts["datestop"])
    assert np.all(exp["pcad_mode"] == sts["pcad_mode"])
    assert np.all(np.isclose(exp["pitch"], sts["pitch"], rtol=0, atol=1e-8))


def test_get_states_start_between_aouptarg_aomanuvr_cmds():
    """Test fix for #198

    Relevant commands::

      2021:025:13:54:50.097  COMMAND_SW AOFUNCDS   130
      2021:025:13:55:50.097  COMMAND_SW AONMMODE   130
      2021:025:13:55:50.354  COMMAND_SW AONM2NPE   130
      2021:025:13:55:54.454 MP_TARGQUAT AOUPTARQ   130  **
      2021:025:13:55:56.097  MP_STARCAT AOSTRCAT   130
      2021:025:13:56:00.348  COMMAND_HW    CNOOP   130
      2021:025:13:56:00.348  COMMAND_SW AOMANUVR   130  **
      2021:025:13:56:01.373  COMMAND_SW AOACRSTE   130

    """
    # This fails prior to the fix with ValueError: cannot convert float NaN to
    # integer in chandra_maneuver.
    sts = states.get_states(
        "2021:025:13:56:00.000",
        "2021:026:00:00:00",
        state_keys=("q1", "pcad_mode"),
        continuity={},
    )
    exp = [
        "      datestart              datestop            q1     pcad_mode",
        "--------------------- --------------------- ----------- ---------",
        "2021:025:13:56:00.000 2021:025:23:43:14.731        None      None",
        "2021:025:23:43:14.731 2021:025:23:43:24.982        None      NMAN",
        "2021:025:23:43:24.982 2021:025:23:46:04.982 0.129401742      NMAN",
        "2021:025:23:46:04.982 2021:025:23:47:24.982 0.129399482      NMAN",
        "2021:025:23:47:24.982 2021:026:00:00:00.000 0.129399482      NPNT",
    ]

    exp = Table.read(exp, format="ascii", fill_values=[("None", "0")])
    for name in ("datestart", "datestop", "pcad_mode"):
        assert np.all(sts[name] == exp[name])
    assert np.allclose(sts["q1"][2:].astype(float), exp["q1"][2:], atol=1e-5)
    assert sts["q1"][0] is None
    assert sts["q1"][1] is None
    assert sts["pcad_mode"][0] is None

    # Failure example from #198
    cont = states.get_continuity("2021:032:13:56:00.000")
    assert cont["__dates__"]["q1"] == "2021:032:12:47:31.814"


def test_get_continuity_and_pitch_from_mid_maneuver(fast_sun_position_method):
    """Test for bug in continuity first noted at:
    https://github.com/acisops/acis_thermal_check/pull/30#issuecomment-665240053

    Continuity during mid-maneuver was incorrect.
    """
    start = "2017:207:23:35:00"
    stop = "2017:208:00:00:00"
    sts = states.get_states(start, stop, state_keys=["pitch", "pcad_mode"])
    exp = [
        "      datestart              datestop             pitch        pcad_mode",
        "--------------------- --------------------- ------------------ ---------",
        "2017:207:23:35:00.000 2017:207:23:35:27.066   74.2497570112975      NMAN",
        "2017:207:23:35:27.066 2017:207:23:40:06.264  60.94464535476736      NMAN",
        "2017:207:23:40:06.264 2017:207:23:42:25.863 57.135312591551276      NMAN",
        "2017:207:23:42:25.863 2017:207:23:45:30.816  57.13353523338555      NPNT",
        "2017:207:23:45:30.816 2017:208:00:00:00.000  57.12776241937949      NPNT",
    ]
    exp = Table.read(exp, format="ascii")

    assert np.all(exp["datestart"] == sts["datestart"])
    assert np.all(exp["datestop"] == sts["datestop"])
    assert np.all(exp["pcad_mode"] == sts["pcad_mode"])
    assert np.all(np.isclose(exp["pitch"], sts["pitch"], rtol=0, atol=1e-8))

    cont = states.get_continuity(start, state_keys=["pitch", "pcad_mode"])
    assert np.isclose(cont["pitch"], sts["pitch"][0], rtol=0, atol=1e-2)
    assert cont["pcad_mode"] == sts["pcad_mode"][0]


def test_acisfp_setpoint_state():
    sts = states.get_states(
        "1999-01-01 12:00:00", "2004-01-01 12:00:00", state_keys="acisfp_setpoint"
    )
    del sts["tstart"]
    del sts["tstop"]

    assert repr(sts).splitlines() == [
        "<Table length=5>",
        "      datestart              datestop       acisfp_setpoint    trans_keys  ",
        "        str21                 str21             float64          object    ",
        "--------------------- --------------------- --------------- ---------------",
        "1999:001:12:00:00.000 2003:130:05:07:28.341          -121.0                ",
        "2003:130:05:07:28.341 2003:130:19:09:28.930          -130.0 acisfp_setpoint",
        "2003:130:19:09:28.930 2003:132:14:22:33.782          -121.0 acisfp_setpoint",
        "2003:132:14:22:33.782 2003:133:22:04:22.425          -130.0 acisfp_setpoint",
        "2003:133:22:04:22.425 2004:001:12:00:00.000          -121.0 acisfp_setpoint",
    ]

    sts = states.get_states(
        "2018-01-01 12:00:00", "2020-03-01 12:00:00", state_keys="acisfp_setpoint"
    )
    del sts["tstart"]
    del sts["tstop"]
    assert repr(sts).splitlines() == [
        "<Table length=6>",
        "      datestart              datestop       acisfp_setpoint    trans_keys  ",
        "        str21                 str21             float64          object    ",
        "--------------------- --------------------- --------------- ---------------",
        "2018:001:12:00:00.000 2018:249:20:16:04.603          -121.0                ",
        "2018:249:20:16:04.603 2018:250:07:19:51.657          -126.0 acisfp_setpoint",
        "2018:250:07:19:51.657 2018:294:22:29:00.000          -121.0 acisfp_setpoint",
        "2018:294:22:29:00.000 2020:048:20:59:22.304          -121.0 acisfp_setpoint",
        "2020:048:20:59:22.304 2020:049:13:05:52.537          -126.0 acisfp_setpoint",
        "2020:049:13:05:52.537 2020:061:12:00:00.000          -121.0 acisfp_setpoint",
    ]


def test_grating_motion_states():
    sts = states.get_states(
        "2021:227:12:00:00", "2021:230:12:00:00", state_keys=["letg", "hetg", "grating"]
    )
    del sts["tstart"]
    del sts["tstop"]
    # fmt: off
    exp = [
        "      datestart              datestop          letg      hetg   grating  trans_keys ",
        "--------------------- --------------------- --------- --------- ------- ------------",
        "2021:227:12:00:00.000 2021:227:23:06:03.276      RETR      RETR    NONE             ",
        "2021:227:23:06:03.276 2021:227:23:08:40.276      RETR INSR_MOVE    HETG grating,hetg",
        "2021:227:23:08:40.276 2021:228:08:15:00.722      RETR      INSR    HETG         hetg",
        "2021:228:08:15:00.722 2021:228:08:17:33.722      RETR RETR_MOVE    NONE grating,hetg",
        "2021:228:08:17:33.722 2021:229:17:41:45.525      RETR      RETR    NONE         hetg",
        "2021:229:17:41:45.525 2021:229:17:45:08.525 INSR_MOVE      RETR    LETG grating,letg",
        "2021:229:17:45:08.525 2021:230:00:37:56.002      INSR      RETR    LETG         letg",
        "2021:230:00:37:56.002 2021:230:00:41:19.002 RETR_MOVE      RETR    NONE grating,letg",
        "2021:230:00:41:19.002 2021:230:12:00:00.000      RETR      RETR    NONE         letg",
    ]
    # fmt: on
    assert sts.pformat() == exp


def test_hrc_states():
    sts = states.get_states(
        start="2022:140",
        stop="2022:280",
        state_keys=["hrc_15v", "hrc_24v", "hrc_i", "hrc_s"],
        merge_identical=True,
        continuity={"hrc_15v": "OFF", "hrc_24v": "OFF", "hrc_i": "OFF", "hrc_s": "OFF"},
    )
    del sts["tstart"]
    del sts["tstop"]
    # fmt: off
    exp = [
        "      datestart              datestop       hrc_15v hrc_24v hrc_i hrc_s trans_keys",
        "--------------------- --------------------- ------- ------- ----- ----- ----------",
        "2022:140:00:00:00.000 2022:237:18:50:01.000     OFF     OFF   OFF   OFF           ",
        "2022:237:18:50:01.000 2022:237:18:50:13.000      ON     OFF   OFF   OFF    hrc_15v",
        "2022:237:18:50:13.000 2022:237:18:51:46.000      ON      ON   OFF   OFF    hrc_24v",
        "2022:237:18:51:46.000 2022:237:18:52:49.000      ON     OFF   OFF   OFF    hrc_24v",
        "2022:237:18:52:49.000 2022:237:21:45:41.000      ON     OFF    ON   OFF      hrc_i",
        "2022:237:21:45:41.000 2022:237:21:45:53.000      ON     OFF   OFF   OFF      hrc_i",
        "2022:237:21:45:53.000 2022:263:17:20:01.000     OFF     OFF   OFF   OFF    hrc_15v",
        "2022:263:17:20:01.000 2022:263:17:20:13.000      ON     OFF   OFF   OFF    hrc_15v",
        "2022:263:17:20:13.000 2022:263:17:21:46.000      ON      ON   OFF   OFF    hrc_24v",
        "2022:263:17:21:46.000 2022:263:17:22:42.000      ON     OFF   OFF   OFF    hrc_24v",
        "2022:263:17:22:42.000 2022:263:21:35:53.000      ON     OFF   OFF    ON      hrc_s",
        "2022:263:21:35:53.000 2022:263:21:36:06.000      ON     OFF   OFF   OFF      hrc_s",
        "2022:263:21:36:06.000 2022:280:00:00:00.000     OFF     OFF   OFF   OFF    hrc_15v"
    ]
    # fmt: on
    assert sts.pformat() == exp


def test_hrc_states_with_scs_commanding():
    """Test that SCS commanding is included in the HRC state transitions"""
    # Extracted from FEB0623T test loads, but with a bogus hardware 215PCAON command at
    # 2023:038:00:00:00.000 added to ensure that redundant commanding is handled
    # correctly.
    backstop = """\
2023:037:17:43:00.636 |  2535285 0 | COMMAND_SW       | HEX= 8408600, MSID= COACTSX, COACTS1=134 , COACTS2=0 , SCS= 131, STEP= 822, TLMSID= COACTSX
2023:038:00:00:00.000 |  2630289 0 | COMMAND_HW       | HEX= 6420000, MSID= 215PCAON, SCS= 131, STEP= 1090, TLMSID= 215PCAON
2023:038:00:28:45.739 |  2630290 0 | COMMAND_HW       | HEX= 6420000, MSID= 215PCAOF, SCS= 131, STEP= 1091, TLMSID= 215PCAOF
2023:038:20:54:55.267 |  2917391 0 | COMMAND_SW       | HEX= 8408600, MSID= COACTSX, COACTS1=134 , COACTS2=0 , SCS= 131, STEP= 1875, TLMSID= COACTSX
2023:039:01:06:24.025 |  2976274 0 | COMMAND_HW       | HEX= 6420000, MSID= 215PCAOF, SCS= 132, STEP= 99, TLMSID= 215PCAOF
2023:039:23:12:15.958 |  3286720 0 | COMMAND_SW       | HEX= 8408600, MSID= COACTSX, COACTS1=134 , COACTS2=0 , SCS= 132, STEP= 705, TLMSID= COACTSX
2023:040:04:55:45.483 |  3367148 0 | COMMAND_HW       | HEX= 6420000, MSID= 215PCAOF, SCS= 132, STEP= 1027, TLMSID= 215PCAOF
2023:042:04:02:57.978 |  4029128 0 | COMMAND_SW       | HEX= 8408600, MSID= COACTSX, COACTS1=134 , COACTS2=0 , SCS= 133, STEP= 314, TLMSID= COACTSX
2023:042:08:35:28.888 |  4092937 0 | COMMAND_HW       | HEX= 6420000, MSID= 215PCAOF, SCS= 133, STEP= 460, TLMSID= 215PCAOF"""
    cmds = commands.read_backstop(backstop.splitlines())
    sts = states.get_states(
        cmds=cmds,
        state_keys=["hrc_15v"],
        continuity={"hrc_15v": "OFF"},
        merge_identical=True,
    )
    del sts["tstart"]
    del sts["tstop"]
    exp = [
        "      datestart              datestop       hrc_15v trans_keys",
        "--------------------- --------------------- ------- ----------",
        "2023:037:17:43:00.636 2023:038:00:28:45.739      ON    hrc_15v",
        "2023:038:00:28:45.739 2023:038:20:54:55.267     OFF    hrc_15v",
        "2023:038:20:54:55.267 2023:039:01:06:24.025      ON    hrc_15v",
        "2023:039:01:06:24.025 2023:039:23:12:15.958     OFF    hrc_15v",
        "2023:039:23:12:15.958 2023:040:04:55:45.483      ON    hrc_15v",
        "2023:040:04:55:45.483 2023:042:04:02:57.978     OFF    hrc_15v",
        "2023:042:04:02:57.978 2023:042:08:35:28.888      ON    hrc_15v",
        "2023:042:08:35:28.888 2023:042:08:35:28.888     OFF    hrc_15v",
    ]
    assert sts.pformat() == exp


def test_early_start_exception():
    with pytest.raises(
        ValueError, match="no continuity found for start='2002:001:00:00:00.000'"
    ):
        states.get_states("2002:001", "2003:001", state_keys=["orbit_point"])


def test_nsm_continuity():
    """Test continuity when the 7-day lookback is before an NSM / safe sun
    where current state attitude is None. Tests fix for #258.
    """
    # Prior to the fix this raised an exception, so just check that it runs.
    states.get_continuity("2022:301:12:42:00", scenario="flight")


def test_sun_pos_mon_within_eclipse():
    """Test for #289 where sun_pos_mon was not being set correctly within eclipse.

    Relevant commands near an eclipse::

      2022:109:21:27:27.034 | COMMAND_SW       | EOESTECN   | APR1822A |
      2022:109:21:29:27.034 | ORBPOINT         | None       | APR1822A | PENTRY
      2022:109:22:03:07.034 | ORBPOINT         | None       | APR1822A | PEXIT

    Battery connect time is 2022:109:21:27:27.034
    Expected sun_pos_mon ENAB is at 2022:109:22:14:07.034.
    """
    starts = CxoTime(
        [
            "2022:109:21:24:00.000",  # Early
            "2022:109:21:27:27.033",  # 1 ms before battery connect
            "2022:109:21:27:27.035",  # 1 ms after battery connect
            "2022:109:21:29:27.033",  # 1 ms before pentry
            "2022:109:21:29:27.035",  # 1 ms after pentry
            "2022:109:22:03:07.033",  # 1 ms before pexit
            "2022:109:22:03:07.035",  # 1 ms after pexit
            "2022:109:22:14:07.033",  # 1 ms before OBC autonomous sun_pos_mon ENAB
        ]
    )

    stop = "2022:110:00:00:00.000"
    spm_state_keys = states.SPM_STATE_KEYS

    for start in starts:
        exp_start_date = max(start.date, "2022:109:21:27:27.034")
        # fmt: off
        exp = [
            "      datestart       sun_pos_mon    battery_connect    eclipse_enable_spm",
            "--------------------- ----------- --------------------- ------------------",
            f"{exp_start_date    }        DISA 2022:109:21:27:27.034               True",
            "2022:109:22:14:07.034        ENAB 2022:109:21:27:27.034               True",
        ]
        # fmt: on

        sts = states.get_states(
            start, stop, state_keys=spm_state_keys, merge_identical=True
        )

        names = ["datestart"] + spm_state_keys
        assert sts[names][-2:].pformat() == exp


def test_sun_pos_mon_within_eclipse_no_spm_enab(monkeypatch):
    """
    Test a case where battery connect is more than 125 sec before pentry.

    2005:014:15:31:36.410 | COMMAND_SW       | EOESTECN   | JAN1005B
    2005:014:15:33:49.164 | ORBPOINT         | None       | JAN1005B | PENTRY
    2005:014:16:38:09.164 | ORBPOINT         | None       | JAN1005B | PEXIT
    """
    # PR #323 changed the time threshold from 125 to 135 sec. Here we monkeypatch back
    # to 125 sec to test the case where battery connect is more than 125 sec
    # before pentry. From telemetry this case with a dt of around 133 sec actually did
    # end up with the SPM enabled.
    monkeypatch.setattr(states.EclipseEnableSPM, "BATTERY_CONNECT_MAX_DT", 125)

    sts = states.get_states(
        "2005:014:16:38:10",  # Just after pexit
        "2005:014:17:00:00",  # 22 min later
        state_keys=states.SPM_STATE_KEYS,
    )

    exp = [
        "      datestart       sun_pos_mon    battery_connect    eclipse_enable_spm",
        "--------------------- ----------- --------------------- ------------------",
        "2005:014:16:38:10.000        DISA 2005:014:15:31:36.410              False",
    ]
    names = ["datestart"] + states.SPM_STATE_KEYS
    assert sts[names].pformat() == exp


def test_get_continuity_acis_cmd_requires_obsid():
    """See https://github.com/sot/kadi/issues/325.

    These are times where the requested time - lookback lands before a particular ACIS
    power command that needs obsid but after the obsid command. Without the fix for #325
    this would raise an exception.
    """
    for time in ["2021:205:09:35:06.322", "2021:020:07:04:40.280"]:
        states.get_continuity(time)


def test_get_continuity_spm_eclipse():
    """Similar to above but for sun_pos_mon.

    This tests a corner case where the requested time - lookback is between the
    eclipse entry and battery connect. Previously this would call
    ``chandra_datetime.date2secs(None)`` which *should* fail but instead was giving a
    date in 1858. The end result was OK but this was a bit accidental.
    """
    cont = states.get_continuity(
        CxoTime("2017:087:07:47:59.838") + 7 * u.day, state_keys=states.SPM_STATE_KEYS
    )
    assert cont == {
        "sun_pos_mon": "DISA",
        "battery_connect": "2017:087:07:47:55.838",
        "eclipse_enable_spm": True,
        "__dates__": {
            "sun_pos_mon": "2017:093:07:47:03.821",
            "battery_connect": "2017:087:07:47:55.838",
            "eclipse_enable_spm": "2017:087:07:49:55.838",
        },
    }


def test_interpolate_states_extrapolate():
    """Test that the states are correctly interpolated and extrapolated."""
    # fmt: off
    sts =  states.get_states("2024:001:00:01:00", "2024:001:00:05:00", state_keys=["obsid"])
    times = ["2024:001:00:00:00", "2024:001:00:02:30", "2024:001:00:05:00", "2024:001:00:06:00"]
    sts_interp = states.interpolate_states(sts, times)
    assert np.all(sts_interp["obsid"] == [43839] * 4)
