import os
from pathlib import Path

# Use data file from parse_cm.test for get_cmds_from_backstop test.
# This package is a dependency
import agasc
import astropy.units as u
import numpy as np
import parse_cm.paths
import parse_cm.tests
import pytest
import ska_helpers.utils
import ska_sun
from astropy.table import Table, vstack
from chandra_time import secs2date
from cxotime import CxoTime
from Quaternion import Quat
from testr.test_helper import has_internet

import kadi
import kadi.commands.states as kcs
from kadi import commands
from kadi.commands import (
    CommandTable,
    commands_v2,
    conf,
    core,
    get_observations,
    get_starcats,
    get_starcats_as_table,
    read_backstop,
)
from kadi.commands.command_sets import get_cmds_from_event
from kadi.scripts import update_cmds_v2

HAS_MPDIR = Path(os.environ["SKA"], "data", "mpcrit1", "mplogs", "2020").exists()
HAS_INTERNET = has_internet()

try:
    agasc.get_agasc_filename(version="1p8")
    HAS_AGASC_1P8 = True
except FileNotFoundError:
    HAS_AGASC_1P8 = False


def get_cmds_from_cmd_evts_text(cmd_evts_text: str):
    """Get commands from a cmd_events text string.

    This helper function can make it easier to test scenarios without creating a
    scenario file.
    """
    cmd_evts = Table.read(cmd_evts_text, format="ascii.csv", fill_values=[])
    cmds_list = [
        get_cmds_from_event(cmd_evt["Date"], cmd_evt["Event"], cmd_evt["Params"])
        for cmd_evt in cmd_evts
    ]
    cmds: CommandTable = core.vstack_exact(cmds_list)
    cmds.sort_in_backstop_order()

    return cmds


def test_find():
    idx_cmds = commands_v2.IDX_CMDS
    pars_dict = commands_v2.PARS_DICT

    cs = core._find(
        "2012:029:12:00:00", "2012:030:12:00:00", idx_cmds=idx_cmds, pars_dict=pars_dict
    )
    assert isinstance(cs, Table)
    assert len(cs) == 151
    assert np.all(cs["source"][:10] == "JAN2612A")
    assert np.all(cs["source"][-10:] == "JAN3012C")
    assert cs["date"][0] == "2012:029:13:00:00.000"
    assert cs["date"][-1] == "2012:030:11:00:01.285"
    assert cs["tlmsid"][-1] == "CTXBON"

    cs = core._find(
        "2012:029:12:00:00",
        "2012:030:12:00:00",
        type="simtrans",
        idx_cmds=idx_cmds,
        pars_dict=pars_dict,
    )
    assert len(cs) == 2
    assert np.all(cs["date"] == ["2012:030:02:00:00.000", "2012:030:08:27:02.000"])

    cs = core._find(
        "2012:015:12:00:00",
        "2012:030:12:00:00",
        idx_cmds=idx_cmds,
        pars_dict=pars_dict,
        type="acispkt",
        tlmsid="wsvidalldn",
    )
    assert len(cs) == 3
    assert np.all(
        cs["date"]
        == ["2012:018:01:16:15.798", "2012:020:16:51:17.713", "2012:026:05:28:09.000"]
    )

    cs = core._find(
        "2011:001:12:00:00",
        "2014:001:12:00:00",
        msid="aflcrset",
        idx_cmds=idx_cmds,
        pars_dict=pars_dict,
    )
    assert len(cs) == 2494


def test_get_cmds():
    cs = commands.get_cmds("2012:029:12:00:00", "2012:030:12:00:00")
    assert isinstance(cs, commands.CommandTable)
    assert len(cs) == 151  # OBS commands in v2 only
    assert np.all(cs["source"][:10] == "JAN2612A")
    assert np.all(cs["source"][-10:] == "JAN3012C")
    assert cs["date"][0] == "2012:029:13:00:00.000"
    assert cs["date"][-1] == "2012:030:11:00:01.285"
    assert cs["tlmsid"][-1] == "CTXBON"

    cs = commands.get_cmds("2012:029:12:00:00", "2012:030:12:00:00", type="simtrans")
    assert len(cs) == 2
    assert np.all(cs["date"] == ["2012:030:02:00:00.000", "2012:030:08:27:02.000"])
    assert np.all(cs["pos"] == [75624, 73176])  # from params

    cmd = cs[1]

    assert repr(cmd).startswith("<Cmd 2012:030:08:27:02.000 SIMTRANS")
    assert str(cmd).startswith("2012:030:08:27:02.000 SIMTRANS")
    assert repr(cmd).endswith(
        "scs=133 step=161 source=JAN3012C vcdu=15639968 pos=73176>"
    )
    assert str(cmd).endswith("scs=133 step=161 source=JAN3012C vcdu=15639968 pos=73176")

    assert cmd["pos"] == 73176
    assert cmd["step"] == 161


def test_get_cmds_zero_length_result():
    cmds = commands.get_cmds(date="2017:001:12:00:00")
    assert len(cmds) == 0
    source_name = "source"
    assert cmds.colnames == [
        "idx",
        "date",
        "type",
        "tlmsid",
        "scs",
        "step",
        "time",
        source_name,
        "vcdu",
        "params",
    ]


def test_get_cmds_inclusive_stop():
    # get_cmds returns start <= date < stop for inclusive_stop=False (default)
    # or start <= date <= stop for inclusive_stop=True.
    # Query over a range that includes two commands at exactly start and stop.
    start, stop = "2020:001:15:50:00.000", "2020:001:15:50:00.257"
    cmds = commands.get_cmds(start, stop)
    assert np.all(cmds["date"] == [start])

    cmds = commands.get_cmds(start, stop, inclusive_stop=True)
    assert np.all(cmds["date"] == [start, stop])


def test_cmds_as_list_of_dict():
    cmds = commands.get_cmds("2020:140", "2020:141")
    cmds_list = cmds.as_list_of_dict()
    assert isinstance(cmds_list, list)
    assert isinstance(cmds_list[0], dict)
    cmds_rt = commands.CommandTable(cmds)
    assert set(cmds_rt.colnames) == set(cmds.colnames)
    for name in cmds.colnames:
        assert np.all(cmds_rt[name] == cmds[name])


def test_cmds_as_list_of_dict_ska_parsecm():
    """Test the ska_parsecm=True compatibility mode for list_of_dict"""
    cmds = commands.get_cmds("2020:140", "2020:141")
    cmds_list = cmds.as_list_of_dict(ska_parsecm=True)
    assert isinstance(cmds_list, list)
    assert isinstance(cmds_list[0], dict)
    exp = {
        "cmd": "COMMAND_HW",  # Cmd parameter exists and matches type
        "date": "2020:140:00:00:00.000",
        "idx": 21387,
        "params": {"HEX": "7C063C0", "MSID": "CIU1024T"},  # Keys are upper case
        "scs": 129,
        "step": 496,
        "time": 706233669.184,
        "tlmsid": "CIMODESL",
        "type": "COMMAND_HW",
        "vcdu": 12516929,
        "source": "MAY1820A",
    }

    assert cmds_list[0] == exp

    for cmd in cmds_list:
        assert cmd.get("cmd") == cmd.get("type")
        assert all(param.upper() == param for param in cmd["params"])


def test_get_cmds_from_backstop_and_add_cmds():
    bs_file = Path(parse_cm.tests.__file__).parent / "data" / "CR182_0803.backstop"
    bs_cmds = commands.get_cmds_from_backstop(bs_file, remove_starcat=True)

    cmds = commands.get_cmds(start="2018:182:00:00:00", stop="2018:182:08:00:00")

    assert len(bs_cmds) == 674
    assert len(cmds) == 57

    # Get rid of source and timeline_id columns which can vary between v1 and v2
    for cs in bs_cmds, cmds:
        if "source" in cs.colnames:
            del cs["source"]
        if "timeline_id" in cs.colnames:
            del cs["timeline_id"]

    assert bs_cmds.colnames == cmds.colnames
    for bs_col, col in zip(bs_cmds.itercols(), cmds.itercols()):
        assert bs_col.dtype == col.dtype

    assert np.all(secs2date(cmds["time"]) == cmds["date"])
    assert np.all(secs2date(bs_cmds["time"]) == bs_cmds["date"])

    new_cmds = cmds.add_cmds(bs_cmds)
    assert len(new_cmds) == len(cmds) + len(bs_cmds)

    # No MP_STARCAT command parameters by default
    ok = bs_cmds["type"] == "MP_STARCAT"
    assert np.count_nonzero(ok) == 15
    assert np.all(bs_cmds["params"][ok] == {})

    # Accept MP_STARCAT commands (also check read_backstop command)
    bs_cmds = commands.read_backstop(bs_file)
    ok = bs_cmds["type"] == "MP_STARCAT"
    assert np.count_nonzero(ok) == 15
    assert np.all(bs_cmds["params"][ok] != {})


@pytest.mark.skipif("not HAS_MPDIR")
@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_commands_create_archive_regress(
    tmpdir, fast_sun_position_method, disable_hrc_scs107_commanding
):
    """Create cmds archive from scratch and test that it matches flight

    This tests over an eventful month that includes IU reset/NSM, SCS-107
    (radiation), fast replan, loads approved but not uplinked, etc.
    """
    kadi_orig = os.environ.get("KADI")
    start = CxoTime("2021:290")
    stop = start + 30 * u.day
    cmds_flight = commands.get_cmds(start + 3 * u.day, stop - 3 * u.day)
    cmds_flight.fetch_params()

    sched_stop_flight: np.ndarray = (cmds_flight["type"] == "LOAD_EVENT") & (
        cmds_flight["event_type"] == "SCHEDULED_STOP_TIME"
    )

    with conf.set_temp("commands_dir", str(tmpdir)):
        try:
            os.environ["KADI"] = str(tmpdir)
            update_cmds_v2.main(
                (
                    f"--stop={stop.date}",
                    f"--data-root={tmpdir}",
                )
            )
            # Force reload of LazyVal
            del commands_v2.IDX_CMDS._val
            del commands_v2.PARS_DICT._val
            del commands_v2.REV_PARS_DICT._val

            # Make sure we are seeing the temporary cmds archive
            cmds_empty = commands.get_cmds(start - 60 * u.day, start - 50 * u.day)
            cmds_empty = commands.get_cmds(start - 60 * u.day, start - 50 * u.day)
            assert len(cmds_empty) == 0

            cmds_local = commands.get_cmds(start + 3 * u.day, stop - 3 * u.day)

            cmds_local.fetch_params()
            if len(cmds_flight) != len(cmds_local):
                # Code to debug problems, leave commented for production
                # out = "\n".join(cmds_flight.pformat_like_backstop())
                # Path("cmds_flight.txt").write_text(out)
                # out = "\n".join(cmds_local.pformat_like_backstop())
                # Path("cmds_local.txt").write_text(out)
                assert len(cmds_flight) == len(cmds_local)

            sched_stop_local: np.ndarray = (cmds_local["type"] == "LOAD_EVENT") & (
                cmds_local["event_type"] == "SCHEDULED_STOP_TIME"
            )

            # PR#364 changed the time stamp of SCHEDULED_STOP_TIME commands when there
            # is an interrupt, which in turn changes the order. First check that the
            # sources of such commands are the same (we have the same sched stop
            # commands but ignore timestamps).
            assert sorted(cmds_flight[sched_stop_flight]["source"]) == sorted(
                cmds_local[sched_stop_local]["source"]
            )
            # Now remove such commands from both for the rest of the comparison.
            cmds_flight = cmds_flight[~sched_stop_flight]
            cmds_local = cmds_local[~sched_stop_local]

            # Validate quaternions using numeric comparison and then remove
            # from the == comparison below.  This is both appropriate for these
            # numeric values and also necessary to deal with architecture
            # differences when run on fido vs kady for example.
            # In reality, the numeric differences only occur on "calculated"
            # quaternions such as the NSM quaternions.
            for idx in range(len(cmds_flight)):
                for att_name in ["targ_att", "prev_att"]:
                    if att_name not in cmds_flight[idx]["params"]:
                        continue
                    flight_q = Quat(q=cmds_flight[idx]["params"][att_name])
                    local_q = Quat(q=cmds_local[idx]["params"][att_name])
                    dq = flight_q.dq(local_q)
                    for attr in ("roll0", "pitch", "yaw"):
                        assert abs(getattr(dq, attr)) < 1e-6
                    del cmds_flight[idx]["params"][att_name]
                    del cmds_local[idx]["params"][att_name]

            # 'starcat_idx' param in OBS cmd does not match since the pickle files
            # are different, so remove it.
            for cmds in (cmds_local, cmds_flight):
                for cmd in cmds:
                    if cmd["tlmsid"] == "OBS" and "starcat_idx" in cmd["params"]:
                        del cmd["params"]["starcat_idx"]

            for attr in ("tlmsid", "date", "params"):
                assert np.all(cmds_flight[attr] == cmds_local[attr])

        finally:
            if kadi_orig is None:
                del os.environ["KADI"]
            else:
                os.environ["KADI"] = kadi_orig

            commands.clear_caches()


def stop_date_fixture_factory(stop_date):
    @pytest.fixture()
    def stop_date_fixture(monkeypatch):
        commands.clear_caches()
        monkeypatch.setenv("CXOTIME_NOW", stop_date)
        cmds_dir = Path(conf.commands_dir) / CxoTime(stop_date).iso[:9]
        with commands.conf.set_temp("commands_dir", str(cmds_dir)):
            yield
        commands.clear_caches()

    return stop_date_fixture


# 2021:297 0300z just after recovery maneuver following 2021:296 NSM
stop_date_2021_10_24 = stop_date_fixture_factory("2021-10-24T03:00:00")
stop_date_2020_12_03 = stop_date_fixture_factory("2020-12-03")
stop_date_2023_203 = stop_date_fixture_factory("2023:203")
stop_date_2025_10_25 = stop_date_fixture_factory("2025-10-25")


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_scheduled_stop_time_commands(stop_date_2025_10_25):
    """Test a time frame with two load interrupts"""
    cmds = commands.get_cmds("2025:290", "2025:300")
    cok = cmds[(cmds["type"] == "LOAD_EVENT") & (cmds["tlmsid"] == "None")]
    params_list = cok["params"].tolist()
    for date, params in zip(cok["date"], params_list):
        params["date"] = date
    assert params_list == [
        {"event_type": "SCHEDULED_STOP_TIME", "date": "2025:292:21:47:50.679"},
        {
            "event_type": "RUNNING_LOAD_TERMINATION_TIME",
            "date": "2025:292:21:47:50.679",
        },
        {
            "event_type": "SCHEDULED_STOP_TIME",
            "scheduled_stop_time_orig": "2025:299:22:52:08.292",
            "interrupt_load": "OCT2125A",
            "date": "2025:294:18:45:00.000",
        },
        {
            "event_type": "RUNNING_LOAD_TERMINATION_TIME",
            "date": "2025:294:18:45:00.000",
        },
        {
            "event_type": "SCHEDULED_STOP_TIME",
            "scheduled_stop_time_orig": "2025:299:22:52:08.282",
            "interrupt_load": "OCT2225A",
            "date": "2025:295:21:30:00.000",
        },
        {
            "event_type": "RUNNING_LOAD_TERMINATION_TIME",
            "date": "2025:295:21:30:00.000",
        },
        {"event_type": "SCHEDULED_STOP_TIME", "date": "2025:299:22:52:08.292"},
    ]


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_nsm_safe_mode_pitch_offsets_state_constraints(stop_date_2023_203):
    """Test NSM, Safe mode transitions with pitch offsets along with state constraints.

    State constraints testing means that an NMAN maneuver with auto-NPNT transition
    is properly interrupted by a NSM or Safe Mode. In this case the NMAN is interrupted
    and NPNT never happens.
    """
    scenario = "test-nsm-safe-mode"
    cmd_events_path = kadi.paths.CMD_EVENTS_PATH(scenario)
    cmd_events_path.parent.mkdir(parents=True, exist_ok=True)
    # Maneuver attitude: ska_sun.get_att_for_sun_pitch_yaw(pitch=170, yaw=0, time="2023:199")
    text = """
    State,Date,Event,Params,Author,Reviewer,Comment
    Definitive,2023:199:00:00:00.000,Safe mode,120,,,
    Definitive,2023:199:02:00:00.000,NSM,,,,
    Definitive,2023:199:03:00:00.000,Maneuver,-0.84752928 0.52176697 0.08279618 0.05097206,,,
    Definitive,2023:199:03:17:00.000,NSM,50,,,
    Definitive,2023:199:03:30:00.000,Safe mode,,,,
    Definitive,2023:199:04:30:00.000,NSM,100,,,
    """
    cmd_events_path.write_text(text)
    states = kcs.get_states(
        "2023:198:23:00:00",
        "2023:199:05:00:00",
        state_keys=["pitch", "pcad_mode"],
        scenario=scenario,
    )
    states["pitch"].info.format = ".1f"
    out = states["datestart", "pitch", "pcad_mode"].pformat()

    exp = [
        "      datestart       pitch pcad_mode",
        "--------------------- ----- ---------",
        "2023:198:23:00:00.000 144.6      NPNT",
        "2023:199:00:00:00.000 143.9      STBY",  # Safe mode to 120
        "2023:199:00:04:31.439 135.7      STBY",
        "2023:199:00:09:02.879 123.7      STBY",
        "2023:199:00:13:34.318 120.0      STBY",
        "2023:199:02:00:00.000 119.2      NSUN",  # NSM to default 90
        "2023:199:02:04:57.883 109.2      NSUN",
        "2023:199:02:09:55.766  94.5      NSUN",
        "2023:199:02:14:53.648  90.0      NSUN",
        "2023:199:03:00:00.000  90.0      NMAN",  # Maneuver to 170
        "2023:199:03:00:10.250  90.3      NMAN",
        "2023:199:03:05:02.010  93.9      NMAN",
        "2023:199:03:09:53.770 103.1      NMAN",
        "2023:199:03:14:45.530 115.6      NMAN",
        "2023:199:03:17:00.000 114.8      NSUN",  # NMAN interrupted by NSM to 50
        "2023:199:03:21:48.808 106.0      NSUN",
        "2023:199:03:26:37.617  87.8      NSUN",
        "2023:199:03:30:00.000  87.9      STBY",  # NSM to 50 interrupted by Safe mode
        "2023:199:03:35:06.588  89.3      STBY",  # to default 90
        "2023:199:03:40:13.175  90.0      STBY",
        "2023:199:04:30:00.000  90.6      NSUN",  # NSM to 100
        "2023:199:04:34:11.100  96.9      NSUN",
        "2023:199:04:38:22.200 100.0      NSUN",
    ]

    assert out == exp


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_cmds_v2_arch_only(stop_date_2020_12_03):  # noqa: ARG001
    cmds = commands.get_cmds(start="2020-01-01", stop="2020-01-02")
    cmds = cmds[cmds["tlmsid"] != "OBS"]
    assert len(cmds) == 153
    assert np.all(cmds["idx"] != -1)
    # Also do a zero-length query
    cmds = commands.get_cmds(start="2020-01-01", stop="2020-01-01")
    assert len(cmds) == 0
    commands.clear_caches()


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_cmds_v2_arch_recent(stop_date_2020_12_03):  # noqa: ARG001
    cmds = commands.get_cmds(start="2020-09-01", stop="2020-12-01")
    cmds = cmds[cmds["tlmsid"] != "OBS"]

    # Since recent matches arch in the past, even though the results are a mix
    # of arch and recent, they commands actually come from the arch because of
    # how the matching block is used (commands come from arch up through the end
    # of the matching block).
    assert np.all(cmds["idx"] != -1)
    # PR #248: made this change from 17640 to 17644
    assert 17640 <= len(cmds) <= 17644

    commands.clear_caches()


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_cmds_v2_recent_only(stop_date_2020_12_03):  # noqa: ARG001
    # This query stop is well beyond the default stop date, so it should get
    # only commands out to the end of the NOV3020A loads (~ Dec 7).
    cmds = commands.get_cmds(start="2020-12-01", stop="2021-01-01")
    cmds = cmds[cmds["tlmsid"] != "OBS"]
    assert len(cmds) == 1523
    assert np.all(cmds["idx"] == -1)
    # fmt: off
    assert cmds[:5].pformat_like_backstop() == [
        "2020:336:00:08:38.610 | COMMAND_HW       | CNOOP      | NOV3020A | hex=7E00000, msid=CNOOPLR, scs=128",
        "2020:336:00:08:39.635 | COMMAND_HW       | CNOOP      | NOV3020A | hex=7E00000, msid=CNOOPLR, scs=128",
        "2020:336:00:12:55.214 | ACISPKT          | AA00000000 | NOV3020A | cmds=3, words=3, scs=131",
        "2020:336:00:12:55.214 | ORBPOINT         | None       | NOV3020A | event_type=XEF1000, scs=0",
        "2020:336:00:12:59.214 | ACISPKT          | AA00000000 | NOV3020A | cmds=3, words=3, scs=131",
    ]
    assert cmds[-5:].pformat_like_backstop() == [
        "2020:342:03:15:02.313 | COMMAND_SW       | OFMTSNRM   | NOV3020A | hex=8010A00, msid=OFMTSNRM, scs=130",
        "2020:342:03:15:02.313 | COMMAND_SW       | COSCSEND   | NOV3020A | hex=C800000, msid=OBC_END_SCS, scs=130",
        "2020:342:06:04:34.287 | ACISPKT          | AA00000000 | NOV3020A | cmds=3, words=3, scs=133",
        "2020:342:06:04:34.287 | COMMAND_SW       | COSCSEND   | NOV3020A | hex=C800000, msid=OBC_END_SCS, scs=133",
        "2020:342:06:04:34.287 | LOAD_EVENT       | None       | NOV3020A | event_type=SCHEDULED_STOP_TIME, scs=0",
    ]
    # fmt: on
    # Same for no stop date
    cmds = commands.get_cmds(start="2020-12-01", stop=None)
    cmds = cmds[cmds["tlmsid"] != "OBS"]
    assert len(cmds) == 1523
    assert np.all(cmds["idx"] == -1)

    # zero-length query
    cmds = commands.get_cmds(start="2020-12-01", stop="2020-12-01")
    assert len(cmds) == 0
    commands.clear_caches()


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_cmds_nsm_2021(stop_date_2021_10_24, disable_hrc_scs107_commanding):
    """NSM at ~2021:296:10:41. This tests non-load commands from cmd_events."""
    cmds = commands.get_cmds("2021:296:10:35:00")  # , '2021:298:01:58:00')
    cmds = cmds[cmds["tlmsid"] != "OBS"]
    exp = [
        "2021:296:10:35:00.000 | COMMAND_HW       | CIMODESL   | OCT1821A | "
        "hex=7C067C0, msid=CIU1024X, scs=128",
        "2021:296:10:35:00.257 | COMMAND_HW       | CTXAOF     | OCT1821A | "
        "hex=780000C, msid=CTXAOF, scs=128",
        "2021:296:10:35:00.514 | COMMAND_HW       | CPAAOF     | OCT1821A | "
        "hex=780001E, msid=CPAAOF, scs=128",
        "2021:296:10:35:00.771 | COMMAND_HW       | CTXBOF     | OCT1821A | "
        "hex=780004C, msid=CTXBOF, scs=128",
        "2021:296:10:35:01.028 | COMMAND_HW       | CPABON     | OCT1821A | "
        "hex=7800056, msid=CPABON, scs=128",
        "2021:296:10:35:01.285 | COMMAND_HW       | CTXBON     | OCT1821A | "
        "hex=7800044, msid=CTXBON, scs=128",
        "2021:296:10:41:57.000 | LOAD_EVENT       | None       | CMD_EVT  | "
        "event=Load_not_run, event_date=2021:296:10:41:57, event_type=LOAD_NOT_RUN, "
        "load=OCT2521A, scs=0",
        "2021:296:10:41:57.000 | COMMAND_SW       | AONSMSAF   | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, scs=0",
        "2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codisas1=128 , scs=0",
        "2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codisas1=129 , scs=0",
        "2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codisas1=130 , scs=0",
        "2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codisas1=131 , scs=0",
        "2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codisas1=132 , scs=0",
        "2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codisas1=133 , scs=0",
        "2021:296:10:41:57.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, scs=0",
        "2021:296:10:41:58.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, msid=AFLCRSET, scs=0",
        "2021:296:10:41:58.025 | SIMTRANS         | None       | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, pos=-99616, scs=0",
        "2021:296:10:42:20.000 | MP_OBSID         | COAOSQID   | CMD_EVT  | "
        "event=Obsid, event_date=2021:296:10:42:20, id=0, scs=0",
        "2021:296:10:43:03.685 | ACISPKT          | AA00000000 | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, scs=0",
        "2021:296:10:43:04.710 | ACISPKT          | AA00000000 | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, scs=0",
        "2021:296:10:43:14.960 | ACISPKT          | WSPOW0002A | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, scs=0",
        "2021:296:10:43:14.960 | COMMAND_SW       | AODSDITH   | CMD_EVT  | "
        "event=NSM, event_date=2021:296:10:41:57, scs=0",
        "2021:297:01:41:01.000 | COMMAND_SW       | AONMMODE   | CMD_EVT  | "
        "event=Maneuver, event_date=2021:297:01:41:01, msid=AONMMODE, scs=0",
        "2021:297:01:41:01.256 | COMMAND_SW       | AONM2NPE   | CMD_EVT  | "
        "event=Maneuver, event_date=2021:297:01:41:01, msid=AONM2NPE, scs=0",
        "2021:297:01:41:05.356 | MP_TARGQUAT      | AOUPTARQ   | CMD_EVT  | "
        "event=Maneuver, event_date=2021:297:01:41:01, q1=7.05469070e-01, "
        "q2=3.29883070e-01, q3=5.34409010e-01, q4=3.28477660e-01, scs=0",
        "2021:297:01:41:11.250 | COMMAND_SW       | AOMANUVR   | CMD_EVT  | "
        "event=Maneuver, event_date=2021:297:01:41:01, msid=AOMANUVR, scs=0",
        "2021:297:02:12:42.886 | ORBPOINT         | None       | OCT1821A | "
        "event_type=EQF003M, scs=0",
        "2021:297:03:40:42.886 | ORBPOINT         | None       | OCT1821A | "
        "event_type=EQF005M, scs=0",
        "2021:297:03:40:42.886 | ORBPOINT         | None       | OCT1821A | "
        "event_type=EQF015M, scs=0",
        "2021:297:04:43:26.016 | ORBPOINT         | None       | OCT1821A | "
        "event_type=EALT1, scs=0",
        "2021:297:04:43:27.301 | ORBPOINT         | None       | OCT1821A | "
        "event_type=XALT1, scs=0",
        "2021:297:12:42:42.886 | ORBPOINT         | None       | OCT1821A | "
        "event_type=EQF013M, scs=0",
        "2021:297:13:59:39.602 | ORBPOINT         | None       | OCT1821A | "
        "event_type=EEF1000, scs=0",
        "2021:297:14:01:00.000 | LOAD_EVENT       | None       | OCT1821A | "
        "event_type=SCHEDULED_STOP_TIME, scs=0",
    ]

    assert cmds.pformat_like_backstop(max_params_width=200) == exp
    commands.clear_caches()


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_cmds_scenario(stop_date_2020_12_03):  # noqa: ARG001
    """Test custom scenario with a couple of ACIS commands"""
    # First make the cmd_events.csv file for the scenario
    scenario = "test_acis"
    cmds_dir = Path(commands.conf.commands_dir) / scenario
    cmds_dir.mkdir(exist_ok=True, parents=True)
    # Note variation in format of date, since this comes from humans.
    # This also does not have a State column, which tests code to put that in.
    cmd_evts_text = """\
Date,Event,Params,Author,Comment
2020-12-01T00:08:30,Command,ACISPKT | TLMSID=WSPOW00000",Tom Aldcroft,
2020-12-01 00:08:39,Command,"ACISPKT | TLMSID=WSVIDALLDN",Tom Aldcroft,
"""
    (cmds_dir / "cmd_events.csv").write_text(cmd_evts_text)

    # Now get commands in a time range that includes the new command events
    cmds = commands.get_cmds(
        "2020-12-01 00:08:00", "2020-12-01 00:09:00", scenario=scenario
    )
    cmds = cmds[cmds["tlmsid"] != "OBS"]
    exp = [
        "2020:336:00:08:30.000 | ACISPKT          | WSPOW00000 | CMD_EVT  |"
        " event=Command, event_date=2020:336:00:08:30, scs=0",
        "2020:336:00:08:38.610 | COMMAND_HW       | CNOOP      | NOV3020A |"
        " hex=7E00000, msid=CNOOPLR, scs=128",
        "2020:336:00:08:39.000 | ACISPKT          | WSVIDALLDN | CMD_EVT  |"
        " event=Command, event_date=2020:336:00:08:39, scs=0",
        "2020:336:00:08:39.635 | COMMAND_HW       | CNOOP      | NOV3020A |"
        " hex=7E00000, msid=CNOOPLR, scs=128",
    ]
    assert cmds.pformat_like_backstop() == exp
    commands.clear_caches()


stop_date_2024_01_30 = stop_date_fixture_factory("2024-01-30")


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
@pytest.mark.parametrize("d_rasl", [-12, 12])
def test_nsm_offset_pitch_rasl_with_rate_command_events(d_rasl, stop_date_2024_01_30):  # noqa: ARG001
    """Test custom scenario with NSM offset pitch load event command"""
    # First make the cmd_events.csv file for the scenario
    scenario = "test_nsm_offset_pitch"
    cmds_dir = Path(commands.conf.commands_dir) / scenario
    cmds_dir.mkdir(exist_ok=True, parents=True)
    # Note variation in format of date, since this comes from humans.
    cmd_evts_text = f"""\
State,Date,Event,Params,Author,Reviewer,Comment
Definitive,2024:024:08:00:00,NSM,120,,,
Definitive,2024:024:10:00:00,Maneuver sun rasl,{d_rasl} 0.02,,,
"""
    (cmds_dir / "cmd_events.csv").write_text(cmd_evts_text)

    # Now get commands in a time range that includes the new command events
    cmds = commands.get_cmds(
        "2024-01-24 09:00:00", "2024-01-25 12:00:00", scenario=scenario
    )
    cmds = cmds[(cmds["tlmsid"] != "OBS") & (cmds["type"] != "ORBPOINT")]
    exp = [
        f"2024:024:10:00:00.000 | LOAD_EVENT       | SUN_RASL   | CMD_EVT  | event=Maneuver_sun_rasl, event_date=2024:024:10:00:00, rasl={d_rasl}, rate=2.00000000e-02, scs=0",
    ]

    assert cmds.pformat_like_backstop(max_params_width=200) == exp

    # Now get states in a time range that includes the new command events
    states = kcs.get_states(
        "2024:024:10:00:00",
        "2024:024:11:00:00",
        state_keys=["pitch", "rasl"],
        scenario=scenario,
    )
    if d_rasl == 12:
        exp_text = """
      datestart              datestop        pitch    rasl
2024:024:10:00:00.000 2024:024:10:02:00.000 120.048 312.622
2024:024:10:02:00.000 2024:024:10:04:00.000 120.049 315.022
2024:024:10:04:00.000 2024:024:10:06:00.000 120.049 317.422
2024:024:10:06:00.000 2024:024:10:08:00.000 120.050 319.822
2024:024:10:08:00.000 2024:024:10:10:00.000 120.050 322.222
2024:024:10:10:00.000 2024:024:11:00:00.000 120.057 324.626
"""
    else:
        exp_text = """
      datestart              datestop         pitch     rasl
2024:024:10:00:00.000 2024:024:10:02:00.000 120.0476  312.621
2024:024:10:02:00.000 2024:024:10:04:00.000 120.0484  310.221
2024:024:10:04:00.000 2024:024:10:06:00.000 120.0492 307.8219
2024:024:10:06:00.000 2024:024:10:08:00.000 120.0501 305.4220
2024:024:10:08:00.000 2024:024:10:10:00.000 120.0511 303.0221
2024:024:10:10:00.000 2024:024:11:00:00.000 120.0644 300.6232
"""

    exp = Table.read(exp_text, format="ascii", guess=False, delimiter=" ")
    # Total RASL change is 12 degrees.
    rasls = states["rasl"]
    n_rasl = len(rasls)
    np.testing.assert_allclose(states["pitch"], exp["pitch"], atol=1e-2, rtol=0)
    np.testing.assert_allclose(rasls, exp["rasl"], atol=1e-2, rtol=0)
    np.testing.assert_allclose(rasls[-1] - rasls[0], d_rasl, atol=1e-2, rtol=0)
    np.testing.assert_allclose(np.diff(rasls), d_rasl / (n_rasl - 1), atol=1e-2, rtol=0)
    np.testing.assert_equal(states["datestart"], exp["datestart"])
    np.testing.assert_equal(states["datestop"], exp["datestop"])


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_nsm_offset_pitch_rasl_command_events(stop_date_2024_01_30):  # noqa: ARG001
    """Test custom scenario with NSM offset pitch load event command"""
    # First make the cmd_events.csv file for the scenario
    scenario = "test_nsm_offset_pitch"
    cmds_dir = Path(commands.conf.commands_dir) / scenario
    cmds_dir.mkdir(exist_ok=True, parents=True)
    # Note variation in format of date, since this comes from humans.
    cmd_evts_text = """\
State,Date,Event,Params,Author,Reviewer,Comment
Definitive,2024:025:04:00:00,Maneuver sun rasl,90,,,
Definitive,2024:025:00:00:00,Maneuver sun pitch,160,,,
Definitive,2024:024:09:44:06,NSM,,,,
"""
    (cmds_dir / "cmd_events.csv").write_text(cmd_evts_text)

    # Now get commands in a time range that includes the new command events
    cmds = commands.get_cmds(
        "2024-01-24 12:00:00", "2024-01-25 05:00:00", scenario=scenario
    )
    cmds = cmds[(cmds["tlmsid"] != "OBS") & (cmds["type"] != "ORBPOINT")]
    exp = [
        "2024:025:00:00:00.000 | LOAD_EVENT       | SUN_PITCH  | CMD_EVT  | event=Maneuver_sun_pitch, event_date=2024:025:00:00:00, pitch=160, scs=0",
        "2024:025:04:00:00.000 | LOAD_EVENT       | SUN_RASL   | CMD_EVT  | event=Maneuver_sun_rasl, event_date=2024:025:04:00:00, rasl=90, rate=2.50000000e-02, scs=0",
    ]

    assert cmds.pformat_like_backstop(max_params_width=200) == exp

    states = kcs.get_states(
        "2024:024:09:00:00",
        "2024:025:02:00:00",
        state_keys=["pitch", "pcad_mode"],
        scenario=scenario,
    )
    exp = [
        "      datestart       pitch pcad_mode",
        "--------------------- ----- ---------",
        "2024:024:09:00:00.000 172.7      NPNT",
        "2024:024:09:13:49.112 172.7      NMAN",
        "2024:024:09:13:59.363 172.1      NMAN",
        "2024:024:09:18:25.800 165.1      NMAN",
        "2024:024:09:22:52.238 149.7      NMAN",
        "2024:024:09:27:18.675 131.3      NMAN",
        "2024:024:09:31:45.112 113.7      NMAN",
        "2024:024:09:36:11.550 102.7      NMAN",
        "2024:024:09:40:37.987  99.6      NMAN",
        "2024:024:09:42:51.205  99.6      NPNT",
        "2024:024:09:44:06.000  99.0      NSUN",
        "2024:024:09:48:17.979  93.0      NSUN",
        "2024:024:09:52:29.957  90.2      NSUN",
        "2024:025:00:00:00.000  91.3      NSUN",
        "2024:025:00:04:48.673 100.6      NSUN",
        "2024:025:00:09:37.346 119.8      NSUN",
        "2024:025:00:14:26.019 141.3      NSUN",
        "2024:025:00:19:14.692 155.8      NSUN",
        "2024:025:00:24:03.365 160.0      NSUN",
    ]

    out = states["datestart", "pitch", "pcad_mode"]
    out["pitch"].format = ".1f"
    assert out.pformat() == exp

    states = kcs.get_states(
        "2024:024:09:00:00",
        "2024:025:08:00:00",
        state_keys=["q1", "q2", "q3", "q4"],
        scenario=scenario,
    )

    # Interpolate states at two times just after the pitch maneuver and just after the
    # roll about sun line (rasl) maneuver.
    dates = ["2024:025:04:00:00", "2024:025:05:00:01"]
    sts = kcs.interpolate_states(states, dates)
    q1 = Quat([sts["q1"][0], sts["q2"][0], sts["q3"][0], sts["q4"][0]])
    q2 = Quat([sts["q1"][1], sts["q2"][1], sts["q3"][1], sts["q4"][1]])
    pitch1, rasl1 = ska_sun.get_sun_pitch_yaw(q1.ra, q1.dec, dates[0])
    pitch2, rasl2 = ska_sun.get_sun_pitch_yaw(q2.ra, q2.dec, dates[1])
    assert np.isclose(pitch1, 160, atol=0.2)
    assert np.isclose(pitch2, 160, atol=0.5)
    assert np.isclose((rasl2 - rasl1) % 360, 90, atol=0.2)

    commands.clear_caches()


def test_command_set_bsh():
    cmds = get_cmds_from_event("2000:001", "Bright star hold", "")
    exp = """\
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=128 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=129 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=130 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=131 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=132 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=133 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, msid=AFLCRSET, scs=0
2000:001:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, pos=-99616, scs=0
2000:001:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:17.960 | COMMAND_HW       | 215PCAOF   | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:19.165 | COMMAND_HW       | 2IMHVOF    | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:20.190 | COMMAND_HW       | 2SPHVOF    | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:21.215 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:22.240 | COMMAND_HW       | 2S1STHV    | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:23.265 | COMMAND_HW       | 2S2HVOF    | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:24.290 | COMMAND_HW       | 2S1HVOF    | CMD_EVT  | event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0"""

    assert cmds.pformat_like_backstop(max_params_width=None) == exp.splitlines()
    commands.clear_caches()


def test_command_set_safe_mode():
    cmds = get_cmds_from_event("2000:001", "Safe mode", "")
    exp = """\
2000:001:00:00:00.000 | COMMAND_SW       | ACPCSFSU   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CSELFMT5   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=128 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=129 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=130 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=131 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=132 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, msid=CODISASX, codisas1=133 , scs=0
2000:001:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, msid=AFLCRSET, scs=0
2000:001:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, pos=-99616, scs=0
2000:001:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:17.960 | COMMAND_HW       | 215PCAOF   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:19.165 | COMMAND_HW       | 2IMHVOF    | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:20.190 | COMMAND_HW       | 2SPHVOF    | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:21.215 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:22.240 | COMMAND_HW       | 2S1STHV    | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:23.265 | COMMAND_HW       | 2S2HVOF    | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:24.290 | COMMAND_HW       | 2S1HVOF    | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0
2000:001:00:01:25.315 | COMMAND_SW       | AODSDITH   | CMD_EVT  | event=Safe_mode, event_date=2000:001:00:00:00, scs=0"""
    assert cmds.pformat_like_backstop(max_params_width=None) == exp.splitlines()
    commands.clear_caches()


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_bright_star_hold_event(
    cmds_dir, stop_date_2020_12_03, disable_hrc_scs107_commanding
):
    """Make a scenario with a bright star hold event.

    Confirm that this inserts expected commands and interrupts all load commands.
    """
    bsh_dir = Path(conf.commands_dir) / "bsh"
    bsh_dir.mkdir(parents=True, exist_ok=True)
    cmd_events_file = bsh_dir / "cmd_events.csv"
    cmd_events_file.write_text(
        """\
State,Date,Event,Params,Author,Comment
Definitive,2020:337:00:00:00,Bright star hold,,Tom Aldcroft,
"""
    )
    cmds = commands.get_cmds(start="2020:336:21:48:00", stop="2020:338", scenario="bsh")
    exp = [
        "2020:336:21:48:03.312 | LOAD_EVENT       | OBS        | NOV3020A | "
        "manvr_start=2020:336:21:09:24.361, prev_att=(-0.242373434, -0.348723922, "
        "0.42827",
        "2020:336:21:48:06.387 | COMMAND_SW       | CODISASX   | NOV3020A | "
        "hex=8456200, msid=CODISASX, codisas1=98 , scs=128",
        "2020:336:21:48:07.412 | COMMAND_SW       | AOFUNCEN   | NOV3020A | "
        "hex=803031E, msid=AOFUNCEN, aopcadse=30 , scs=128",
        "2020:336:21:54:23.061 | COMMAND_SW       | AOFUNCEN   | NOV3020A | "
        "hex=8030320, msid=AOFUNCEN, aopcadse=32 , scs=128",
        "2020:336:21:55:23.061 | COMMAND_SW       | AOFUNCEN   | NOV3020A | "
        "hex=8030315, msid=AOFUNCEN, aopcadse=21 , scs=128",
        # BSH interrupt at 2020:337
        "2020:337:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, msid=CODISASX, "
        "codisas1=12",
        "2020:337:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, msid=CODISASX, "
        "codisas1=12",
        "2020:337:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, msid=CODISASX, "
        "codisas1=13",
        "2020:337:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, msid=CODISASX, "
        "codisas1=13",
        "2020:337:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, msid=CODISASX, "
        "codisas1=13",
        "2020:337:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, msid=CODISASX, "
        "codisas1=13",
        "2020:337:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0",
        "2020:337:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, msid=AFLCRSET, scs=0",
        "2020:337:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, pos=-99616, scs=0",
        "2020:337:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0",
        "2020:337:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0",
        "2020:337:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  | "
        "event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0",
        # Only ORBPOINT from here on
        "2020:337:02:07:03.790 | ORBPOINT         | None       | NOV3020A | "
        "event_type=EAPOGEE, scs=0",
        "2020:337:21:15:45.455 | ORBPOINT         | None       | NOV3020A | "
        "event_type=EALT1, scs=0",
        "2020:337:21:15:46.227 | ORBPOINT         | None       | NOV3020A | "
        "event_type=XALT1, scs=0",
    ]
    assert cmds.pformat_like_backstop() == exp
    commands.clear_caches()


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_observations_by_obsid_single():
    obss = get_observations(obsid=8008)
    assert len(obss) == 1
    del obss[0]["starcat_idx"]
    assert obss == [
        {
            "obsid": 8008,
            "simpos": 92904,
            "obs_stop": "2007:002:18:04:28.965",
            "manvr_start": "2007:002:04:31:48.216",
            "targ_att": (0.149614271, 0.490896707, 0.831470649, 0.21282047),
            "npnt_enab": True,
            "obs_start": "2007:002:04:46:58.056",
            "prev_att": (0.319214732, 0.535685207, 0.766039803, 0.155969017),
            "starcat_date": "2007:002:04:31:43.965",
            "source": "DEC2506C",
        }
    ]


def test_get_observations_by_obsid_multi():
    # Following ACA high background NSM 2019:248
    obss = get_observations(obsid=47912, scenario="flight")
    # Don't compare starcat_idx because it might change with a repro
    for obs in obss:
        obs.pop("starcat_idx", None)

    assert obss == [
        {
            "obsid": 47912,
            "simpos": -99616,
            "obs_stop": "2019:248:16:51:18.000",
            "manvr_start": "2019:248:14:52:35.407",
            "targ_att": (-0.564950617, 0.252299958, -0.165669121, 0.767938327),
            "npnt_enab": True,
            "obs_start": "2019:248:15:27:35.289",
            "prev_att": (-0.218410783, 0.748632452, -0.580771797, 0.233560059),
            "starcat_date": "2019:248:14:52:31.156",
            "source": "SEP0219B",
        },
        {
            "obsid": 47912,
            "simpos": -99616,
            "obs_stop": "2019:249:01:59:00.000",
            "manvr_start": "2019:248:16:51:18.000",
            "targ_att": (
                -0.3594375808951632,
                0.6553454859043244,
                -0.4661410647781301,
                0.47332803366853643,
            ),
            "npnt_enab": False,
            "obs_start": "2019:248:17:18:17.732",
            "prev_att": (-0.564950617, 0.252299958, -0.165669121, 0.767938327),
            "source": "CMD_EVT",
        },
        {
            "obsid": 47912,
            "simpos": -99616,
            "obs_stop": "2019:249:23:30:00.000",
            "manvr_start": "2019:249:01:59:10.250",
            "targ_att": (-0.54577727, 0.27602874, -0.17407247, 0.77177334),
            "npnt_enab": True,
            "obs_start": "2019:249:02:25:31.907",
            "prev_att": (
                -0.3594375808951632,
                0.6553454859043244,
                -0.4661410647781301,
                0.47332803366853643,
            ),
            "source": "CMD_EVT",
        },
    ]


def test_get_observations_by_start_date():
    # Test observations from a 6 months ago onward
    obss = get_observations(start=CxoTime.now() - 180 * u.day, scenario="flight")
    assert len(obss) > 500
    # Latest obs should also be no less than 14 days old
    assert obss[-1]["obs_start"] > (CxoTime.now() - 14 * u.day).date


def test_get_observations_by_start_stop_date_with_scenario():
    # Test observations in a range and use the scenario keyword
    obss = get_observations(start="2022:001", stop="2022:002", scenario="flight")
    assert len(obss) == 7
    assert obss[1]["obsid"] == 45814
    assert obss[1]["obs_start"] == "2022:001:05:48:44.808"
    assert obss[-1]["obsid"] == 23800
    assert obss[-1]["obs_start"] == "2022:001:17:33:53.255"


def test_get_observations_no_match():
    with pytest.raises(ValueError, match="No matching observations for obsid=8008"):
        get_observations(
            obsid=8008, start="2022:001", stop="2022:002", scenario="flight"
        )


def test_get_observations_start_stop_inclusion():
    # Covers time from the middle of obsid 8008 to the middle of obsid 8009
    obss = get_observations("2007:002:05:00:00", "2007:002:20:00:01", scenario="flight")
    assert len(obss) == 2

    # One second in the middle of obsid 8008
    obss = get_observations("2007:002:05:00:00", "2007:002:05:00:01", scenario="flight")
    assert len(obss) == 1

    # During a maneuver
    obss = get_observations("2007:002:18:05:00", "2007:002:18:08:00", scenario="flight")
    assert len(obss) == 0


years = np.arange(2003, 2025)


@pytest.mark.parametrize("year", years)
def test_get_starcats_each_year(year):
    starcats = get_starcats(start=f"{year}:001", stop=f"{year}:004", scenario="flight")
    assert len(starcats) > 2
    for starcat in starcats:
        # Make sure fids and stars are all ID'd
        ok = starcat["type"] != "MON"
        assert np.all(starcat["id"][ok] != -999)


def test_get_starcat_only_agasc1p7():
    """
    For obsids 3829 and 2576, try AGASC 1.7 only and show successful star
    identification.
    """
    with (
        conf.set_temp("cache_starcats", False),
        conf.set_temp("date_start_agasc1p8", "2003:001"),
    ):
        starcat = get_starcats(
            "2002:365:18:00:00", "2002:365:19:00:00", scenario="flight"
        )[0]
        assert np.all(starcat["id"] != -999)
        assert np.all(starcat["mag"] != -999)


@pytest.mark.skipif(not HAS_AGASC_1P8, reason="AGASC 1.8 not available")
def test_get_starcat_only_agasc1p8():
    """For obsids 3829 and 2576, try AGASC 1.8 only

    For 3829 star identification should succeed, for 2576 it fails.
    """
    with (
        conf.set_temp("cache_starcats", False),
        conf.set_temp("date_start_agasc1p8", "1994:001"),
    ):
        # Force AGASC 1.8 and show that star identification fails
        with ska_helpers.utils.set_log_level(kadi.logger, "CRITICAL"):
            starcats = get_starcats(
                "2002:365:16:00:00", "2002:365:19:00:00", scenario="flight"
            )
        assert np.count_nonzero(starcats[0]["id"] == -999) == 0
        assert np.count_nonzero(starcats[0]["mag"] == -999) == 0
        assert np.count_nonzero(starcats[1]["id"] == -999) == 3
        assert np.count_nonzero(starcats[1]["mag"] == -999) == 3


def test_get_starcats_with_cmds():
    start, stop = "2021:365:19:00:00", "2022:002:01:25:00"
    cmds = commands.get_cmds(start, stop, scenario="flight")
    starcats0 = get_starcats(start, stop)
    starcats1 = get_starcats(cmds=cmds)
    assert len(starcats0) == len(starcats1)
    for starcat0, starcat1 in zip(starcats0, starcats1):
        eq = starcat0.values_equal(starcat1)
        for col in eq.itercols():
            assert np.all(col)


def test_get_starcats_obsid():
    from mica.starcheck import get_starcat

    sc_kadi = get_starcats(obsid=26330, scenario="flight")[0]
    sc_mica = get_starcat(26330)
    assert len(sc_kadi) == len(sc_mica)
    assert sc_kadi.colnames == [
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
    for name in sc_kadi.colnames:
        if name == "mag":
            continue  # kadi mag is latest from agasc, could change
        elif name == "maxmag":
            assert np.allclose(sc_kadi[name], sc_mica[name], atol=0.001, rtol=0)
        elif name in ("yang", "zang"):
            assert np.all(np.abs(sc_kadi[name] - sc_mica[name]) < 1)
        else:
            assert np.all(sc_kadi[name] == sc_mica[name])


def test_get_starcats_date():
    """Test that the starcat `date` is set to obs `starcat_date`.

    And that this matches the time of the corresponding MP_STARCAT AOSTRCAT
    command.

    Note: from https://icxc.harvard.edu//mp/mplogs/2006/DEC2506/oflsc/starcheck.html#obsid8008
    MP_STARCAT at 2007:002:04:31:43.965 (VCDU count = 7477935)
    """  # noqa: E501
    sc = get_starcats(obsid=8008, scenario="flight")[0]
    obs = get_observations(obsid=8008, scenario="flight")[0]
    assert sc.date == obs["starcat_date"] == "2007:002:04:31:43.965"
    cmds = commands.get_cmds("2007:002", "2007:003")
    sc_cmd = cmds[cmds["date"] == obs["starcat_date"]][0]
    assert sc_cmd["type"] == "MP_STARCAT"


def test_get_starcats_by_date():
    # Test that the getting a starcat using the starcat_date as argument
    # returns the same catalog as using the OBSID.
    sc = get_starcats(obsid=8008, scenario="flight")[0]
    sc_by_date = get_starcats(starcat_date="2007:002:04:31:43.965", scenario="flight")[
        0
    ]
    assert np.all(sc == sc_by_date)
    with pytest.raises(ValueError, match="No matching observations for starcat_date"):
        get_starcats(starcat_date="2007:002:04:31:43.966", scenario="flight")


def test_get_starcats_as_table():
    """Test that get_starcats_as_table returns the same as vstacked get_starcats"""
    start, stop = "2020:001", "2020:002"
    starcats = get_starcats(start, stop, scenario="flight")
    obsids = []
    dates = []
    for starcat in starcats:
        obsids.extend([starcat.obsid] * len(starcat))
        dates.extend([starcat.date] * len(starcat))
        # Meta causes warnings in vstack, just ignore here
        starcat.meta = {}
    aces = get_starcats_as_table(start, stop, scenario="flight")
    aces_from_starcats = vstack(starcats)
    assert np.all(aces["obsid"] == obsids)
    assert np.all(aces["starcat_date"] == dates)
    for name in aces_from_starcats.colnames:
        assert np.all(aces[name] == aces_from_starcats[name])


@pytest.mark.parametrize(
    "par_str",
    [
        "ACISPKT|  TLmSID= aa0000000, par1 = 1 ,   par2=-1.0",
        "AcisPKT|TLmSID=AA0000000 ,par1=1, par2=-1.0",
        "ACISPKT|  TLmSID = aa0000000 , par1  =1,    par2 =  -1.0",
    ],
)
def test_get_cmds_from_event_case(par_str):
    cmds = get_cmds_from_event("2022:001", "Command", par_str)
    assert len(cmds) == 1
    cmd = cmds[0]
    assert cmd["type"] == "ACISPKT"
    assert cmd["params"] == {
        "event": "Command",
        "event_date": "2022:001:00:00:00",
        "par1": 1,
        "par2": -1.0,
    }


cmd_events_all_text = """\
    Event,Params
    Observing not run,FEB1422A
    Load not run,OCT2521A
    Command,"ACISPKT | TLMSID= AA00000000, CMDS= 3, WORDS= 3, PACKET(40)= D80000300030603001300"
    Command not run,"COMMAND_SW | TLMSID=4OHETGIN, HEX= 8050300, MSID= 4OHETGIN"
    Obsid,65527
    Maneuver,0.70546907 0.32988307 0.53440901 0.32847766
    Safe mode,
    NSM,
    SCS-107,
    Bright star hold,
    Dither,ON
    """
cmd_events_all = Table.read(
    cmd_events_all_text, format="ascii.csv", fill_values=[], converters={"Params": str}
)
cmd_events_all_exps = [
    [
        "2020:001:00:00:00.000 | LOAD_EVENT       | None       | CMD_EVT  | event=Observing_not_run, event_date=2020:001:00:00:00, event_type=OBSERVING_NOT_RUN, load=FEB1422A, scs=0"
    ],
    [
        "2020:001:00:00:00.000 | LOAD_EVENT       | None       | CMD_EVT  | event=Load_not_run, event_date=2020:001:00:00:00, event_type=LOAD_NOT_RUN, load=OCT2521A, scs=0"
    ],
    [
        "2020:001:00:00:00.000 | ACISPKT          | AA00000000 | CMD_EVT  | event=Command, event_date=2020:001:00:00:00, cmds=3, words=3, scs=0"
    ],
    [
        "2020:001:00:00:00.000 | NOT_RUN          | 4OHETGIN   | CMD_EVT  | event=Command_not_run, event_date=2020:001:00:00:00, hex=8050300, msid=4OHETGIN, __type__=COMMAND_SW, scs=0"
    ],
    [
        "2020:001:00:00:00.000 | MP_OBSID         | COAOSQID   | CMD_EVT  | event=Obsid, event_date=2020:001:00:00:00, id=65527, scs=0"
    ],
    [
        "2020:001:00:00:00.000 | COMMAND_SW       | AONMMODE   | CMD_EVT  |"
        " event=Maneuver, event_date=2020:001:00:00:00, msid=AONMMODE, scs=0",
        "2020:001:00:00:00.256 | COMMAND_SW       | AONM2NPE   | CMD_EVT  |"
        " event=Maneuver, event_date=2020:001:00:00:00, msid=AONM2NPE, scs=0",
        "2020:001:00:00:04.356 | MP_TARGQUAT      | AOUPTARQ   | CMD_EVT  |"
        " event=Maneuver, event_date=2020:001:00:00:00, q1=7.05469070e-01,"
        " q2=3.29883070e-01, q3=5.34409010e-01, q4=3.28477660e-01, scs=0",
        "2020:001:00:00:10.250 | COMMAND_SW       | AOMANUVR   | CMD_EVT  |"
        " event=Maneuver, event_date=2020:001:00:00:00, msid=AOMANUVR, scs=0",
    ],
    [
        "2020:001:00:00:00.000 | COMMAND_SW       | ACPCSFSU   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CSELFMT5   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=128 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=129 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=130 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=131 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=132 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=133 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, msid=AFLCRSET, scs=0",
        "2020:001:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, pos=-99616, scs=0",
        "2020:001:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:17.960 | COMMAND_SW       | AODSDITH   | CMD_EVT  |"
        " event=Safe_mode, event_date=2020:001:00:00:00, scs=0",
    ],
    [
        "2020:001:00:00:00.000 | COMMAND_SW       | AONSMSAF   | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, msid=CODISASX, codisas1=128 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, msid=CODISASX, codisas1=129 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, msid=CODISASX, codisas1=130 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, msid=CODISASX, codisas1=131 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, msid=CODISASX, codisas1=132 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, msid=CODISASX, codisas1=133 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, msid=AFLCRSET, scs=0",
        "2020:001:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, pos=-99616, scs=0",
        "2020:001:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:17.960 | COMMAND_SW       | AODSDITH   | CMD_EVT  | event=NSM,"
        " event_date=2020:001:00:00:00, scs=0",
    ],
    [
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=131 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=132 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, msid=CODISASX, codisas1=133 ,"
        " scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, msid=AFLCRSET, scs=0",
        "2020:001:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, pos=-99616, scs=0",
        "2020:001:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  |"
        " event=SCS-107, event_date=2020:001:00:00:00, scs=0",
    ],
    [
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, msid=CODISASX,"
        " codisas1=128 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, msid=CODISASX,"
        " codisas1=129 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, msid=CODISASX,"
        " codisas1=130 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, msid=CODISASX,"
        " codisas1=131 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, msid=CODISASX,"
        " codisas1=132 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | CODISASX   | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, msid=CODISASX,"
        " codisas1=133 , scs=0",
        "2020:001:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, msid=AFLCRSET, scs=0",
        "2020:001:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, pos=-99616, scs=0",
        "2020:001:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, scs=0",
        "2020:001:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  |"
        " event=Bright_star_hold, event_date=2020:001:00:00:00, scs=0",
    ],
    [
        "2020:001:00:00:00.000 | COMMAND_SW       | AOENDITH   | CMD_EVT  | event=Dither, event_date=2020:001:00:00:00, scs=0"
    ],
]


@pytest.mark.parametrize("idx", range(len(cmd_events_all_exps)))
def test_get_cmds_from_event_all(idx, disable_hrc_scs107_commanding):
    """Test getting commands from every event type in the Command Events sheet"""
    cevt = cmd_events_all[idx]
    exp = cmd_events_all_exps[idx]
    cmds = get_cmds_from_event("2020:001:00:00:00", cevt["Event"], cevt["Params"])
    if cmds is not None:
        cmds = cmds.pformat_like_backstop(max_params_width=None)
    assert cmds == exp


cmd_events_rts_text = """\
    Event,Params
    RTS,"RTSLOAD,1_4_CTI,NUM_HOURS=39:00:00,SCS_NUM=135"
    """
cmd_events_rts = Table.read(
    cmd_events_rts_text, format="ascii.csv", fill_values=[], converters={"Params": str}
)
cmd_events_rts_exps = [
    [
        "2020:001:00:00:00.000 | COMMAND_SW       | OORMPEN    | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, msid=OORMPEN, scs=135",
        "2020:001:00:00:01.000 | ACISPKT          | WSVIDALLDN | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:001:00:00:02.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, 2s2sthv2=0 , msid=2S2STHV, scs=135",
        "2020:001:00:00:03.000 | COMMAND_HW       | 2S2HVON    | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, msid=2S2HVON, scs=135",
        "2020:001:00:00:13.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, 2s2sthv2=4 , msid=2S2STHV, scs=135",
        "2020:001:00:00:23.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, 2s2sthv2=8 , msid=2S2STHV, scs=135",
        "2020:001:00:00:24.000 | ACISPKT          | WSPOW08E1E | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:001:00:01:27.000 | ACISPKT          | WT00C62014 | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:001:00:01:31.000 | ACISPKT          | XTZ0000005 | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:001:00:01:35.000 | ACISPKT          | RS_0000001 | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:001:00:01:39.000 | ACISPKT          | RH_0000001 | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:002:15:01:39.000 | COMMAND_HW       | 2S2HVOF    | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, msid=2S2HVOF, scs=135",
        "2020:002:15:01:39.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, msid=OORMPDS, scs=135",
        "2020:002:15:01:40.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, 2s2sthv2=0 , msid=2S2STHV, scs=135",
        "2020:002:17:40:00.000 | ACISPKT          | AA00000000 | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:002:17:40:10.000 | ACISPKT          | AA00000000 | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:002:17:40:14.000 | ACISPKT          | WSPOW00000 | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
        "2020:002:17:40:18.000 | ACISPKT          | RS_0000001 | CMD_EVT  | event=RTS,"
        " event_date=2020:001:00:00:00, scs=135",
    ],
]


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
@pytest.mark.parametrize("idx", range(len(cmd_events_rts_exps)))
def test_get_cmds_from_event_rts(idx, disable_hrc_scs107_commanding):
    """Test getting commands from every event type in the Command Events sheet"""
    cevt = cmd_events_rts[idx]
    exp = cmd_events_rts_exps[idx]
    cmds = get_cmds_from_event("2020:001:00:00:00", cevt["Event"], cevt["Params"])
    if cmds is not None:
        cmds = cmds.pformat_like_backstop(max_params_width=None)
    assert cmds == exp


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_scenario_with_rts(monkeypatch, fast_sun_position_method):
    # Test a custom scenario with RTS. This is basically the same as the
    # example in the documentation.
    from kadi import paths

    monkeypatch.setenv("CXOTIME_NOW", "2021:299")

    # Ensure local cmd_events.csv is up to date by requesting "recent" commands
    # relative to the default stop.
    cmds = commands.get_cmds(start="2021:299")

    path_flight = paths.CMD_EVENTS_PATH()
    path_cti = paths.CMD_EVENTS_PATH(scenario="nsm-cti")
    path_cti.parent.mkdir(exist_ok=True, parents=True)

    # Make a new custom scenario from the flight version
    events_flight = Table.read(path_flight)
    cti_event = {
        "State": "Definitive",
        "Date": "2021:297:13:00:00",
        "Event": "RTS",
        "Params": "RTSLOAD,1_CTI06,NUM_HOURS=12:00:00,SCS_NUM=135",
        "Author": "Tom Aldcroft",
        "Reviewer": "John Scott",
        "Comment": "",
    }
    events_cti = events_flight.copy()
    events_cti.add_row(cti_event)
    events_cti.write(path_cti, overwrite=True)

    # Now read the commands from the custom scenario
    cmds = commands.get_cmds(
        "2021:296:10:35:00", "2021:298:01:58:00", scenario="nsm-cti"
    )
    cmds.fetch_params()
    for cmd in cmds:
        if "hex" in cmd["params"]:
            del cmd["params"]["hex"]

    exp = """\
2021:296:10:35:00.000 | COMMAND_HW       | CIMODESL   | OCT1821A | msid=CIU1024X, scs=128
2021:296:10:35:00.257 | COMMAND_HW       | CTXAOF     | OCT1821A | msid=CTXAOF, scs=128
2021:296:10:35:00.514 | COMMAND_HW       | CPAAOF     | OCT1821A | msid=CPAAOF, scs=128
2021:296:10:35:00.771 | COMMAND_HW       | CTXBOF     | OCT1821A | msid=CTXBOF, scs=128
2021:296:10:35:01.028 | COMMAND_HW       | CPABON     | OCT1821A | msid=CPABON, scs=128
2021:296:10:35:01.285 | COMMAND_HW       | CTXBON     | OCT1821A | msid=CTXBON, scs=128
2021:296:10:41:57.000 | LOAD_EVENT       | None       | CMD_EVT  | event=Load_not_run, event_date=2021:296:10:41:57, event_type
2021:296:10:41:57.000 | COMMAND_SW       | AONSMSAF   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codi
2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codi
2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codi
2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codi
2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codi
2021:296:10:41:57.000 | COMMAND_SW       | CODISASX   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, msid=CODISASX, codi
2021:296:10:41:57.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:41:58.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, msid=AFLCRSET, scs=
2021:296:10:41:58.025 | SIMTRANS         | None       | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, pos=-99616, scs=0
2021:296:10:42:20.000 | MP_OBSID         | COAOSQID   | CMD_EVT  | event=Obsid, event_date=2021:296:10:42:20, id=0, scs=0
2021:296:10:43:03.685 | ACISPKT          | AA00000000 | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:04.710 | ACISPKT          | AA00000000 | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:14.960 | ACISPKT          | WSPOW0002A | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:14.960 | COMMAND_HW       | 215PCAOF   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:16.165 | COMMAND_HW       | 2IMHVOF    | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:17.190 | COMMAND_HW       | 2SPHVOF    | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:18.215 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:19.240 | COMMAND_HW       | 2S1STHV    | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:20.265 | COMMAND_HW       | 2S2HVOF    | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:21.290 | COMMAND_HW       | 2S1HVOF    | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:10:43:22.315 | COMMAND_SW       | AODSDITH   | CMD_EVT  | event=NSM, event_date=2021:296:10:41:57, scs=0
2021:296:11:08:12.966 | LOAD_EVENT       | OBS        | CMD_EVT  | manvr_start=2021:296:10:41:57.000, prev_att=(0.594590732, 0.
2021:297:01:41:01.000 | COMMAND_SW       | AONMMODE   | CMD_EVT  | event=Maneuver, event_date=2021:297:01:41:01, msid=AONMMODE,
2021:297:01:41:01.256 | COMMAND_SW       | AONM2NPE   | CMD_EVT  | event=Maneuver, event_date=2021:297:01:41:01, msid=AONM2NPE,
2021:297:01:41:05.356 | MP_TARGQUAT      | AOUPTARQ   | CMD_EVT  | event=Maneuver, event_date=2021:297:01:41:01, q1=7.05469070e
2021:297:01:41:11.250 | COMMAND_SW       | AOMANUVR   | CMD_EVT  | event=Maneuver, event_date=2021:297:01:41:01, msid=AOMANUVR,
2021:297:02:05:11.042 | LOAD_EVENT       | OBS        | CMD_EVT  | manvr_start=2021:297:01:41:11.250, prev_att=(0.2854059718181
2021:297:02:12:42.886 | ORBPOINT         | None       | OCT1821A | event_type=EQF003M, scs=0
2021:297:03:40:42.886 | ORBPOINT         | None       | OCT1821A | event_type=EQF005M, scs=0
2021:297:03:40:42.886 | ORBPOINT         | None       | OCT1821A | event_type=EQF015M, scs=0
2021:297:04:43:26.016 | ORBPOINT         | None       | OCT1821A | event_type=EALT1, scs=0
2021:297:04:43:27.301 | ORBPOINT         | None       | OCT1821A | event_type=XALT1, scs=0
2021:297:12:42:42.886 | ORBPOINT         | None       | OCT1821A | event_type=EQF013M, scs=0
2021:297:13:00:00.000 | COMMAND_SW       | OORMPEN    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, msid=OORMPEN, scs=1
2021:297:13:00:01.000 | ACISPKT          | WSVIDALLDN | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
2021:297:13:00:02.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, 2s2sthv2=0 , msid=2
2021:297:13:00:03.000 | COMMAND_HW       | 2S2HVON    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, msid=2S2HVON, scs=1
2021:297:13:00:13.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, 2s2sthv2=4 , msid=2
2021:297:13:00:23.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, 2s2sthv2=8 , msid=2
2021:297:13:00:24.000 | ACISPKT          | WSPOW0CF3F | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
2021:297:13:01:27.000 | ACISPKT          | WT007AC024 | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
2021:297:13:01:31.000 | ACISPKT          | XTZ0000005 | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
2021:297:13:01:35.000 | ACISPKT          | RS_0000001 | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
2021:297:13:01:39.000 | ACISPKT          | RH_0000001 | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
2021:297:13:59:39.602 | ORBPOINT         | None       | OCT2521A | event_type=EEF1000, scs=0
2021:297:14:01:00.000 | LOAD_EVENT       | None       | OCT1821A | event_type=SCHEDULED_STOP_TIME, scs=0
2021:297:14:37:39.602 | ORBPOINT         | None       | OCT2521A | event_type=EPF1000, scs=0
2021:297:15:01:42.681 | ORBPOINT         | None       | OCT2521A | event_type=EALT0, scs=0
2021:297:15:01:43.574 | ORBPOINT         | None       | OCT2521A | event_type=XALT0, scs=0
2021:297:16:30:13.364 | ORBPOINT         | None       | OCT2521A | event_type=EPERIGEE, scs=0
2021:297:17:58:42.322 | ORBPOINT         | None       | OCT2521A | event_type=EALT0, scs=0
2021:297:17:58:43.505 | ORBPOINT         | None       | OCT2521A | event_type=XALT0, scs=0
2021:297:20:09:39.602 | ORBPOINT         | None       | OCT2521A | event_type=XPF1000, scs=0
2021:297:20:16:57.284 | ORBPOINT         | None       | OCT2521A | event_type=EASCNCR, scs=0
2021:297:21:37:39.602 | ORBPOINT         | None       | OCT2521A | event_type=XEF1000, scs=0
2021:297:22:32:42.886 | ORBPOINT         | None       | OCT2521A | event_type=XQF015M, scs=0
2021:297:23:00:42.886 | ORBPOINT         | None       | OCT2521A | event_type=XQF005M, scs=0
2021:298:00:12:42.886 | ORBPOINT         | None       | OCT2521A | event_type=XQF003M, scs=0
2021:298:00:12:42.886 | ORBPOINT         | None       | OCT2521A | event_type=XQF013M, scs=0
2021:298:01:01:39.000 | COMMAND_HW       | 2S2HVOF    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, msid=2S2HVOF, scs=1
2021:298:01:01:39.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, msid=OORMPDS, scs=1
2021:298:01:01:40.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, 2s2sthv2=0 , msid=2
2021:298:01:57:00.000 | LOAD_EVENT       | None       | OCT2521B | event_type=RUNNING_LOAD_TERMINATION_TIME, scs=0
2021:298:01:57:00.000 | COMMAND_SW       | AOACRSTD   | OCT2521B | msid=AOACRSTD, scs=128
2021:298:01:57:00.000 | ACISPKT          | AA00000000 | OCT2521B | cmds=3, words=3, scs=131
2021:298:01:57:03.000 | ACISPKT          | AA00000000 | OCT2521B | cmds=3, words=3, scs=131
2021:298:01:57:33.000 | COMMAND_SW       | CODISASX   | OCT2521B | msid=CODISASX, codisas1=135 , scs=131
2021:298:01:57:34.000 | COMMAND_SW       | COCLRSX    | OCT2521B | msid=COCLRSX, coclrs1=135 , scs=131"""

    out = "\n".join(cmds.pformat_like_backstop(max_params_width=60))
    assert out == exp

    # 11 RTS commands. Note that the ACIS stop science commands from the RTS
    # are NOT evident because they are cut by the RLTT at 2021:298:01:57:00.
    # TODO: (someday?) instead of the RLTT notice the disable SCS 135 command
    # CODISASX.
    ok = cmds["event"] == "RTS"
    assert np.count_nonzero(ok) == 14


stop_date_2022_236 = stop_date_fixture_factory("2022-08-23")


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_no_rltt_for_not_run_load(stop_date_2022_236):  # noqa: ARG001
    """The AUG2122A loads were never run but they contain an RLTT that had been
    stopping the 2022:232:03:09 ACIS ECS that was previously running. This tests
    the fix.
    """  # noqa: D205
    exp = [
        "         date           tlmsid   scs",
        "--------------------- ---------- ---",
        "2022:232:03:09:00.000 AA00000000 135",
        "2022:232:03:09:04.000 WSPOW00000 135",
        "2022:232:03:09:28.000 WSPOW08E1E 135",
        "2022:232:03:10:31.000 WT00C62014 135",
        "2022:232:03:10:35.000 XTZ0000005 135",
        "2022:232:03:10:39.000 RS_0000001 135",
        "2022:232:03:10:43.000 RH_0000001 135",
        "2022:233:18:10:43.000 AA00000000 135",  # <== After the AUG2122A RLTT
        "2022:233:18:10:53.000 AA00000000 135",
        "2022:233:18:10:57.000 WSPOW0002A 135",
        "2022:233:18:12:00.000 RS_0000001 135",
    ]
    cmds = commands.get_cmds("2022:232:03:00:00", "2022:233:18:30:00")
    cmds = cmds[cmds["type"] == "ACISPKT"]
    assert cmds["date", "tlmsid", "scs"].pformat() == exp


stop_date_2022_352 = stop_date_fixture_factory("2022-12-17")


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_30_day_lookback_issue(stop_date_2022_352):  # noqa: ARG001
    """Test for fix in PR #265 of somewhat obscure issue where a query
    within the default 30-day lookback could give zero commands. Prior to
    the fix the query below would give zero commands (with the default stop date
    set accordingly)."""  # noqa: D205, D209
    cmds = commands.get_cmds("2022:319", "2022:324")
    assert len(cmds) > 200

    # Hit the CMDS_RECENT cache as well
    cmds = commands.get_cmds("2022:319:00:00:01", "2022:324:00:00:01")
    assert len(cmds) > 200


def test_fill_gaps():
    from kadi.commands.utils import fill_gaps_with_nan

    times = [1, 20, 21, 200, 300]
    vals = [0, 1, 2, 3, 4]
    times_out, vals_out = fill_gaps_with_nan(times, vals, max_gap=2)
    times_exp = [1, 1.001, 19.999, 20, 21, 21.001, 199.999, 200, 200.001, 299.999, 300]
    assert np.allclose(times_out, times_exp)
    vals_exp = np.array(
        [0.0, np.nan, np.nan, 1.0, 2.0, np.nan, np.nan, 3.0, np.nan, np.nan, 4.0]
    )
    is_nan = np.isnan(vals_out)
    assert np.all(is_nan == np.isnan(vals_exp))
    assert np.all(vals_out[~is_nan] == vals_exp[~is_nan])


def test_get_rltt_scheduled_stop_time():
    """RLTT and scheduled stop time are both 2023:009:04:14:00.000."""
    cmds = commands.get_cmds("2023:009", "2023:010")
    rltt = cmds.get_rltt()
    assert rltt == "2023:009:04:14:00.000"

    stt = cmds.get_scheduled_stop_time()
    assert stt == "2023:009:04:14:00.000"

    cmds = commands.get_cmds("2023:009:12:00:00", "2023:010")
    assert cmds.get_rltt() is None
    assert cmds.get_scheduled_stop_time() is None


# For HRC not run testing
stop_date_2023200 = stop_date_fixture_factory("2023:200")


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_hrc_not_run_scenario(stop_date_2023200):  # noqa: ARG001
    """Test custom scenario with HRC not run"""
    from kadi.commands.states import get_states

    # Baseline states WITHOUT the HRC not run command events
    states_exp = [
        "      datestart       hrc_i hrc_s hrc_24v hrc_15v",
        "--------------------- ----- ----- ------- -------",
        "2023:183:00:00:00.000   OFF   OFF     OFF     OFF",
        "2023:184:18:42:15.224   OFF   OFF     OFF      ON",  # JUL0323A
        "2023:184:18:43:41.224    ON   OFF     OFF      ON",
        "2023:184:22:43:49.224   OFF   OFF     OFF      ON",
        "2023:184:22:44:01.224   OFF   OFF     OFF     OFF",
        "2023:190:23:39:43.615   OFF   OFF     OFF      ON",
        "2023:190:23:39:44.615   OFF   OFF      ON      ON",  # Note 24V transition
        "2023:190:23:41:17.615   OFF   OFF     OFF      ON",
        "2023:190:23:42:45.615   OFF    ON     OFF      ON",
        "2023:191:03:46:13.615   OFF   OFF     OFF      ON",
        "2023:191:03:46:26.615   OFF   OFF     OFF     OFF",
        "2023:194:20:03:50.666   OFF   OFF     OFF      ON",  # JUL1023A
        "2023:194:20:03:51.666   OFF   OFF      ON      ON",
        "2023:194:20:05:24.666   OFF   OFF     OFF      ON",
        "2023:194:20:06:52.666    ON   OFF     OFF      ON",
        "2023:194:23:13:40.666   OFF   OFF     OFF      ON",
        "2023:194:23:13:52.666   OFF   OFF     OFF     OFF",
    ]

    keys = ["hrc_i", "hrc_s", "hrc_24v", "hrc_15v"]
    states = get_states(
        start="2023:183",
        stop="2023:195",
        state_keys=keys,
        merge_identical=True,
    )
    states_out = states[["datestart"] + keys].pformat()
    assert states_out == states_exp

    # First make the cmd_events.csv file for the scenario where F_HRC_SAFING is run at
    # 2023:184:20:00:00.000.
    # Note that JUL0323A runs from 2023:183:16:39:00.000 to 2023:191:03:46:28.615.
    # We expect the HRC to be off from 2023:184 2000z until the first observation in the
    # JUL1023A loads which start at 2023:191:03:43:28.615

    scenario = "hrc_not_run"
    cmds_dir = Path(commands.conf.commands_dir) / scenario
    cmds_dir.mkdir(exist_ok=True, parents=True)
    # Note variation in format of date, since this comes from humans.
    cmd_evts_text = """\
State,Date,Event,Params,Author,Reviewer,Comment
Definitive,2023:184:20:00:00.000,HRC not run,JUL0323A,Tom,Jean,F_HRC_SAFING 2023:184:20:00:00
"""
    (cmds_dir / "cmd_events.csv").write_text(cmd_evts_text)

    # Now get states in same time range for the HRC not run scenario.
    keys = ["hrc_i", "hrc_s", "hrc_24v", "hrc_15v"]
    states = get_states(
        start="2023:183",
        stop="2023:195",
        state_keys=keys,
        merge_identical=True,
        scenario=scenario,
    )
    states_exp = [
        "      datestart       hrc_i hrc_s hrc_24v hrc_15v",
        "--------------------- ----- ----- ------- -------",
        "2023:183:00:00:00.000   OFF   OFF     OFF     OFF",
        "2023:184:18:42:15.224   OFF   OFF     OFF      ON",  # JUL0323A
        "2023:184:18:43:41.224    ON   OFF     OFF      ON",
        "2023:184:20:00:00.000   OFF   OFF     OFF     OFF",  # Shut off by HRC not run
        "2023:194:20:03:50.666   OFF   OFF     OFF      ON",  # JUL1023A
        "2023:194:20:03:51.666   OFF   OFF      ON      ON",
        "2023:194:20:05:24.666   OFF   OFF     OFF      ON",
        "2023:194:20:06:52.666    ON   OFF     OFF      ON",
        "2023:194:23:13:40.666   OFF   OFF     OFF      ON",
        "2023:194:23:13:52.666   OFF   OFF     OFF     OFF",
    ]

    states_out = states[["datestart"] + keys].pformat()
    assert states_out == states_exp

    commands.clear_caches()


test_command_not_run_cases = [
    {
        # Matches multiple commands
        "event": {
            "date": "2023:351:13:30:33.849",
            "event": "Command not run",
            "params_str": "COMMAND_SW | TLMSID= COACTSX",
        },
        "removed": [3, 4],
    },
    {
        # Matches one command with multiple criteria
        "event": {
            "date": "2023:351:13:30:33.849",
            "event": "Command not run",
            "params_str": (
                "COMMAND_SW | TLMSID= COACTSX, HEX= 840B100, "
                "MSID= COACTSX, COACTS1=177 , COACTS2=0 , SCS= 128, STEP= 690"
            ),
        },
        "removed": [3],
    },
    {
        # Wrong TLMSID
        "event": {
            "date": "2023:351:13:30:33.849",
            "event": "Command not run",
            "params_str": (
                "COMMAND_SW | TLMSID= XXXXXXX, HEX= 840B100, "
                "MSID= COACTSX, COACTS1=177 , COACTS2=0 , SCS= 128, STEP= 690"
            ),
        },
        "removed": [],
    },
    {
        # Wrong SCS
        "event": {
            "date": "2023:351:13:30:33.849",
            "event": "Command not run",
            "params_str": (
                "COMMAND_SW | TLMSID= XXXXXXX, HEX= 840B100, "
                "MSID= COACTSX, COACTS1=177 , COACTS2=0 , SCS= 133, STEP= 690"
            ),
        },
        "removed": [],
    },
    {
        # Wrong Step
        "event": {
            "date": "2023:351:13:30:33.849",
            "event": "Command not run",
            "params_str": (
                "COMMAND_SW | TLMSID= XXXXXXX, HEX= 840B100, "
                "MSID= COACTSX, COACTS1=177 , COACTS2=0 , SCS= 128, STEP= 111"
            ),
        },
        "removed": [],
    },
    {
        # No TLMSID
        "event": {
            "date": "2023:351:19:38:41.550",
            "event": "Command not run",
            "params_str": "SIMTRANS   | POS= 92904, SCS= 131, STEP= 1191",
        },
        "removed": [6],
    },
]


@pytest.mark.parametrize("case", test_command_not_run_cases)
def test_command_not_run(case):
    backstop_text = """
2023:351:13:30:32.824 | 0 0 | COMMAND_SW | TLMSID= AOMANUVR, HEX= 8034101, MSID= AOMANUVR, SCS= 128, STEP= 686
2023:351:13:30:33.849 | 1 0 | COMMAND_SW | TLMSID= AOACRSTE, HEX= 8032001, MSID= AOACRSTE, SCS= 128, STEP= 688
2023:351:13:30:33.849 | 2 0 | COMMAND_SW | TLMSID= COENASX, HEX= 844B100, MSID= COENASX, COENAS1=177 , SCS= 128, STEP= 689
2023:351:13:30:33.849 | 3 0 | COMMAND_SW | TLMSID= COACTSX, HEX= 840B100, MSID= COACTSX, COACTS1=177 , COACTS2=0 , SCS= 128, STEP= 690
2023:351:13:30:33.849 | 4 0 | COMMAND_SW | TLMSID= COACTSX, HEX= 8402600, MSID= COACTSX, COACTS1=38 , COACTS2=0 , SCS= 128, STEP= 691
2023:351:13:30:55.373 | 5 0 | COMMAND_HW | TLMSID= 4MC5AEN, HEX= 4800012, MSID= 4MC5AEN, SCS= 131, STEP= 892
2023:351:19:38:41.550 | 6 0 | SIMTRANS   | POS= 92904, SCS= 131, STEP= 1191
    """
    cmds = commands.read_backstop(backstop_text.strip().splitlines())
    cmds["source"] = "DEC1123A"
    cmds_exp = cmds.copy()
    cmds_exp.remove_rows(case["removed"])
    cmds_from_event = get_cmds_from_event(**case["event"])
    cmds_with_event = cmds.add_cmds(cmds_from_event)
    cmds_with_event.sort_in_backstop_order()
    cmds_with_event.remove_not_run_cmds()
    assert cmds_with_event.pformat_like_backstop() == cmds_exp.pformat_like_backstop()


def test_add_cmds():
    """Test add_cmds with and without RLTT"""
    cmds1_text = """
2023:344:23:30:00.000 | 16200421 0 | COMMAND_SW | TLMSID= CMD1, SCS= 128, STEP= 0
2023:344:23:32:59.999 | 16200421 0 | COMMAND_SW | TLMSID= CMD2, SCS= 128, STEP= 0
2023:344:23:33:00.000 | 16201123 0 | COMMAND_SW | TLMSID= CMD3, SCS= 128, STEP= 0
2023:344:23:33:00.000 | 16201123 0 | COMMAND_SW | TLMSID= CMD4, SCS= 128, STEP= 0
2023:344:23:33:00.001 | 16201124 0 | COMMAND_SW | TLMSID= CMD5, SCS= 128, STEP= 0
2023:344:23:34:00.000 | 16201124 0 | COMMAND_SW | TLMSID= CMD6, SCS= 128, STEP= 0
"""

    cmds2_text = """
2023:344:23:30:00.000 | 16200421 0 | COMMAND_SW | TLMSID= CMD1_2, SCS= 130, STEP= 0
2023:344:23:32:59.999 | 16200421 0 | COMMAND_SW | TLMSID= CMD2_2, SCS= 130, STEP= 0
2023:344:23:33:00.000 |        0 0 | LOAD_EVENT | TYPE= RUNNING_LOAD_TERMINATION_TIME, SCS=0, STEP=0
2023:344:23:33:00.000 | 16201123 0 | COMMAND_SW | TLMSID= CMD3_2, SCS= 130, STEP= 0
2023:344:23:33:00.000 | 16201123 0 | COMMAND_SW | TLMSID= CMD4_2, SCS= 130, STEP= 0
2023:344:23:33:00.001 | 16201124 0 | COMMAND_SW | TLMSID= CMD5_2, SCS= 130, STEP= 0
2023:344:23:34:00.000 | 16201124 0 | COMMAND_SW | TLMSID= CMD6_2, SCS= 130, STEP= 0
"""
    cmds1 = commands.read_backstop(cmds1_text.strip().splitlines())
    cmds2 = commands.read_backstop(cmds2_text.strip().splitlines())
    cmds12_rltt = cmds1.add_cmds(cmds2, rltt=cmds2.get_rltt())
    # Commands from cmds1 with date > 2023:344:23:33:00:000 (RLTT) are removed.
    exp_rltt = [
        "2023:344:23:30:00.000 | COMMAND_SW       | CMD1       |        0 | scs=128",
        "2023:344:23:30:00.000 | COMMAND_SW       | CMD1_2     |        0 | scs=130",
        "2023:344:23:32:59.999 | COMMAND_SW       | CMD2       |        0 | scs=128",
        "2023:344:23:32:59.999 | COMMAND_SW       | CMD2_2     |        0 | scs=130",
        "2023:344:23:33:00.000 | COMMAND_SW       | CMD3       |        0 | scs=128",
        "2023:344:23:33:00.000 | COMMAND_SW       | CMD4       |        0 | scs=128",
        "2023:344:23:33:00.000 | LOAD_EVENT       | None       |        0 | event_type=RUNNING_LOAD_TERMINATION_TIME, scs=0",
        "2023:344:23:33:00.000 | COMMAND_SW       | CMD3_2     |        0 | scs=130",
        "2023:344:23:33:00.000 | COMMAND_SW       | CMD4_2     |        0 | scs=130",
        "2023:344:23:33:00.001 | COMMAND_SW       | CMD5_2     |        0 | scs=130",
        "2023:344:23:34:00.000 | COMMAND_SW       | CMD6_2     |        0 | scs=130",
    ]
    assert cmds12_rltt.pformat_like_backstop() == exp_rltt

    cmds12_no_rltt = cmds1.add_cmds(cmds2)
    # All commands from both cmds1 and cmds2 are included.
    exp_no_rltt = [
        "2023:344:23:30:00.000 | COMMAND_SW       | CMD1       |        0 | scs=128",
        "2023:344:23:30:00.000 | COMMAND_SW       | CMD1_2     |        0 | scs=130",
        "2023:344:23:32:59.999 | COMMAND_SW       | CMD2       |        0 | scs=128",
        "2023:344:23:32:59.999 | COMMAND_SW       | CMD2_2     |        0 | scs=130",
        "2023:344:23:33:00.000 | COMMAND_SW       | CMD3       |        0 | scs=128",
        "2023:344:23:33:00.000 | COMMAND_SW       | CMD4       |        0 | scs=128",
        "2023:344:23:33:00.000 | LOAD_EVENT       | None       |        0 | event_type=RUNNING_LOAD_TERMINATION_TIME, scs=0",
        "2023:344:23:33:00.000 | COMMAND_SW       | CMD3_2     |        0 | scs=130",
        "2023:344:23:33:00.000 | COMMAND_SW       | CMD4_2     |        0 | scs=130",
        "2023:344:23:33:00.001 | COMMAND_SW       | CMD5       |        0 | scs=128",
        "2023:344:23:33:00.001 | COMMAND_SW       | CMD5_2     |        0 | scs=130",
        "2023:344:23:34:00.000 | COMMAND_SW       | CMD6       |        0 | scs=128",
        "2023:344:23:34:00.000 | COMMAND_SW       | CMD6_2     |        0 | scs=130",
    ]
    assert cmds12_no_rltt.pformat_like_backstop() == exp_no_rltt


def test_read_backstop_with_observations():
    """Test reading backstop with observations in it.

    This tests reading the DEC1123A loads and showing that the observations and
    star catalogs are exactly the same as in the flight commands archive.
    """
    try:
        path = parse_cm.paths.load_file_path("DEC1123A", "CR*.backstop", "backstop")
    except FileNotFoundError:
        pytest.skip("No backstop file found")

    cmds = read_backstop(path, add_observations=True)
    obss = get_observations(cmds=cmds)
    starcats = get_starcats(cmds=cmds)

    cmds_flight = commands.get_cmds(source="DEC1123A")
    obss_cmds_flight = get_observations(cmds=cmds_flight)
    starcats_cmds_flight = get_starcats(cmds=cmds_flight)

    assert len(obss) == len(obss_cmds_flight)
    assert len(starcats) == len(starcats_cmds_flight)

    for obs, obs_flight in zip(obss, obss_cmds_flight):
        # starcat idx will be different
        obs.pop("starcat_idx", None)
        obs_flight.pop("starcat_idx", None)
        assert obs == obs_flight

    for starcat, starcat_flight in zip(starcats, starcats_cmds_flight):
        assert starcat.pformat() == starcat_flight.pformat()
