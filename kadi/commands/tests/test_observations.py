import astropy.units as u
import numpy as np
import parse_cm.backstop
import pytest
import ska_helpers.utils
from astropy.table import Table, vstack
from cxotime import CxoTime
from testr.test_helper import has_internet

import kadi
import kadi.commands as kc
from kadi.commands import (
    conf,
    core,
    get_observation,
    get_observations,
    get_starcats,
    get_starcats_as_table,
)

HAS_INTERNET = has_internet()
KADI_CMDS_VERSION = core.kadi_cmds_version()

pytestmark = pytest.mark.skipif(
    KADI_CMDS_VERSION < 3, reason="requires KADI_CMDS_VERSION >= 3"
)


def test_get_observations_merge_manual_obsid_change_obs_splits():
    start, stop = "2025:001:00:00:00", "2025:001:14:00:00"
    with conf.set_temp("merge_manual_obsid_change_obs_splits", False):
        obss1 = Table(get_observations(start, stop, scenario="flight"))
    assert obss1["obsid", "obsid_sched", "obs_start", "obs_stop"].pformat() == [
        "obsid obsid_sched       obs_start              obs_stop      ",
        "----- ----------- --------------------- ---------------------",
        "30550       28365 2024:366:21:58:02.061 2025:001:08:44:01.810",
        "30550       29835 2025:001:09:10:23.636 2025:001:12:48:34.040",
        "65518       29835 2025:001:12:48:34.040 2025:001:13:08:03.384",
        "65518       25501 2025:001:13:44:13.046 2025:001:16:38:44.384",
    ]

    obss2 = Table(get_observations(start, stop, scenario="flight"))
    assert obss2["obsid", "obsid_sched", "obs_start", "obs_stop"].pformat() == [
        "obsid obsid_sched       obs_start              obs_stop      ",
        "----- ----------- --------------------- ---------------------",
        "30550       28365 2024:366:21:58:02.061 2025:001:08:44:01.810",
        "30550       29835 2025:001:09:10:23.636 2025:001:13:08:03.384",
        "65518       25501 2025:001:13:44:13.046 2025:001:16:38:44.384",
    ]


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


def test_get_observations_bad_filter_key():
    with pytest.raises(KeyError, match="invalid filter key"):
        get_observations(bad_key="value", scenario="flight")


def test_get_observations_and_starcats_filtering():
    # Observation impacted by SCS-107 around 2025:001 and rescheduled in MAR1025B
    obss = Table(get_observations(obsid_sched=28365, scenario="flight"))
    assert obss["obsid", "obsid_sched", "source", "obs_start"].pformat() == [
        "obsid obsid_sched  source        obs_start      ",
        "----- ----------- -------- ---------------------",
        "30550       28365 DEC2324B 2024:366:21:58:02.061",
        "28365       28365 MAR1025B 2025:072:23:55:14.230",
    ]
    starcats = get_starcats(obsid_sched=28365, scenario="flight")
    for obs, starcat in zip(obss, starcats, strict=True):
        assert starcat.date == obs["starcat_date"]

    obss = Table(
        get_observations(obsid_sched=28365, source="MAR1025B", scenario="flight")
    )
    assert obss["obsid", "obsid_sched", "source", "obs_start"].pformat() == [
        "obsid obsid_sched  source        obs_start      ",
        "----- ----------- -------- ---------------------",
        "28365       28365 MAR1025B 2025:072:23:55:14.230",
    ]
    starcats = get_starcats(obsid_sched=28365, source="MAR1025B", scenario="flight")
    for obs, starcat in zip(obss, starcats, strict=True):
        assert starcat.date == obs["starcat_date"]

    obss = Table(
        get_observations(manvr_start="2024:366:21:21:11.789", scenario="flight")
    )
    assert obss["obsid", "obsid_sched", "source", "manvr_start"].pformat() == [
        "obsid obsid_sched  source       manvr_start     ",
        "----- ----------- -------- ---------------------",
        "30550       28365 DEC2324B 2024:366:21:21:11.789",
    ]

    starcats = get_starcats(manvr_start="2024:366:21:21:11.789", scenario="flight")
    for obs, starcat in zip(obss, starcats, strict=True):
        assert starcat.date == obs["starcat_date"]


def test_get_starcats_as_table_with_filtering():
    starcats = get_starcats_as_table(source="MAR1025B", scenario="flight")
    assert len(starcats) == 551
    exp = [
        "slot     id     type",
        "---- ---------- ----",
        "   0 1026443544  BOT",
        "   1 1026557304  BOT",
        "   2 1026571288  BOT",
        "   3 1026573992  BOT",
        "   4 1026561016  BOT",
    ]
    assert starcats[:5]["slot", "id", "type"].pformat() == exp

    starcats = get_starcats_as_table(
        obsid_sched=28365, source="MAR1025B", scenario="flight"
    )
    exp = [
        "slot     id     type",
        "---- ---------- ----",
        "   0          2  FID",
        "   1          4  FID",
        "   2          5  FID",
        "   3 1006782456  BOT",
        "   4 1006774552  BOT",
        "   5 1006781496  BOT",
        "   6 1006778248  BOT",
        "   7 1006637024  BOT",
        "   0 1006779528  ACQ",
        "   1 1006784024  ACQ",
        "   2 1006767392  ACQ",
    ]
    assert starcats["slot", "id", "type"].pformat() == exp


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


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((8008,), {}),  # obsid supplied as a positional start time
        ((), {"start": 8008}),
        ((), {"stop": 65535}),
        ((), {"start": 0.0}),
    ],
)
def test_get_observations_obsid_as_time(args, kwargs):
    with pytest.raises(ValueError, match="is not a valid Chandra time"):
        get_observations(*args, scenario="flight", **kwargs)


def test_get_observations_numeric_time():
    """Numeric times after the mission start are still valid CXC seconds"""
    obss = get_observations(
        CxoTime("2022:001").secs, CxoTime("2022:002").secs, scenario="flight"
    )
    assert obss == get_observations("2022:001", "2022:002", scenario="flight")
    assert len(obss) > 0


def test_get_observation_success():
    # obsid_sched=8008 is a pre-APR1420B load (DEC2506C) where obsid_sched defaults to
    # obsid. There is exactly one matching observation so get_observation should succeed.
    obs = get_observation(obsid_sched=8008)
    obs.pop("starcat_idx", None)
    assert obs == {
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


def test_get_observation_multiple_matches():
    # obsid 47912 has two observations (split by an NSM) so get_observation should raise.
    with pytest.raises(ValueError, match="expected one observation"):
        get_observation(obsid=47912, scenario="flight")


def test_get_observation_no_match():
    # simpos=10000 does not match either observation for obsid 47912.
    with pytest.raises(ValueError, match="No matching observations"):
        get_observation(obsid=47912, simpos=10000, scenario="flight")


def test_get_observations_start_stop_inclusion():
    # Covers time from the middle of obsid 8008 to the middle of obsid 8009
    obss = get_observations("2007:002:05:00:00", "2007:002:20:00:01", scenario="flight")
    assert len(obss) == 2

    obs_8009 = obss[1]

    # One second in the middle of obsid 8008
    obss = get_observations("2007:002:05:00:00", "2007:002:05:00:01", scenario="flight")
    assert len(obss) == 1

    # During a maneuver
    obss = get_observations("2007:002:18:05:00", "2007:002:18:08:00", scenario="flight")
    assert len(obss) == 1

    # Exactly at obs 8009 stop: filtering is inclusive manvr_start <= date <= obs_stop
    date = obs_8009["manvr_start"]
    obss = get_observations(date, date, scenario="flight")
    assert len(obss) == 1
    assert obss[0]["obsid"] == 8009

    date = obs_8009["obs_stop"]
    obss = get_observations(date, date, scenario="flight")
    assert len(obss) == 1
    assert obss[0]["obsid"] == 8009

    # In the no-observation zone between 8008 and 8009, in the ~10 sec after transition
    # to NMM (previous obs_stop) but before maneuver starts (next manvr_start).
    date = CxoTime(obs_8009["manvr_start"]) - 1 * u.ms
    obss = get_observations(date, date, scenario="flight")
    assert len(obss) == 0


def test_set_starcat_ids_fail(monkeypatch):
    """Munge commanded position of a fid and star so they are not identified."""
    monkeypatch.setenv("AGASC_SUPPLEMENT_ENABLED", "False")
    monkeypatch.setenv("AGASC_HDF5_FILE", "proseco_agasc_1p8.h5")
    # Get AOSTRCAT cmd for obsid 28501
    obsid = 28501
    obs = kc.get_observations(obsid=obsid, scenario="flight")[0]
    sc = kc.get_starcats(obsid=obsid, scenario="flight")[0]

    sc.get_id(2)["zang"] = 0.0
    sc.get_id(239878784)["zang"] = 0.0
    cmds = kc.get_cmds("2023:346:01:00:00", obs["obs_stop"])
    cmds_bad = parse_cm.backstop.replace_starcat_backstop(cmds, {obsid: sc})

    sc_bad = kc.get_starcats(cmds=cmds_bad)[0]
    exp = [
        "slot idx     id    type  sz   mag   maxmag   yang     zang   dim res halfw",
        "---- --- --------- ---- --- ------- ------ -------- -------- --- --- -----",
        "   0   1      -999  FID 8x8 -999.00   8.50  -773.14     0.00   1   1    25",
        "   1   2         4  FID 8x8    7.00   8.50  2140.38   166.73   1   1    25",
        "   2   3         5  FID 8x8    7.00   8.50 -1826.24   160.26   1   1    25",
        "   3   4      -999  BOT 8x8 -999.00   7.69 -1599.55     0.00  28   1   160",
        "   4   5 239732072  BOT 8x8    6.31   7.81 -1863.14 -2051.75  28   1   160",
        "   5   6 167512680  BOT 8x8    8.48   9.98  1771.34   620.06  28   1   160",
        "   6   7 239734216  BOT 8x8    8.86  10.36  -588.55  1245.06  20   1   120",
        "   7   8 239736384  BOT 8x8    8.95  10.45 -2299.70    57.83  28   1   160",
        "   0   9 239733224  ACQ 8x8    8.98  10.48   580.73   572.45  16   1   100",
        "   1  10 167513136  ACQ 8x8    9.63  11.13  2152.55  -445.40  28   1   160",
        "   2  11 239736664  ACQ 8x8    9.42  10.92  -856.79   583.51  28   1   160",
    ]
    assert sc_bad.pformat() == exp


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
        assert np.count_nonzero(starcats[1]["id"] == -999) == 1
        assert np.count_nonzero(starcats[1]["mag"] == -999) == 1


def test_get_starcats_with_cmds():
    """Test getting star catalogs with commands"""
    # The start is the AOSTRCAT date for the first of 7 observations.
    start, stop = "2021:365:18:39:25.983", "2022:002:01:25:00"
    cmds = kc.get_cmds(start, stop, scenario="flight")
    starcats0 = get_starcats(start, stop, scenario="flight")
    starcats1 = get_starcats(cmds=cmds)
    assert len(starcats0) == len(starcats1)
    for starcat0, starcat1 in zip(starcats0, starcats1):
        eq = starcat0.values_equal(starcat1)
        for col in eq.itercols():
            assert np.all(col)


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_starcats_obsid():
    from mica.starcheck import get_starcat

    sc_kadi = get_starcats(obsid=26330, scenario="flight")[0]
    # get_starcat() requires internet - it calls get_observations() with no scenario
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
    cmds = kc.get_cmds("2007:002", "2007:003", scenario="flight")
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
