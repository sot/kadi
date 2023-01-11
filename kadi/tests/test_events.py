# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import sys
from copy import deepcopy

import numpy as np
from Chandra.Time import DateTime

from .. import events


def test_xdg_config_home_env_var():
    """
    Test that code near top of settings.py which optionally sets XDG_CONFIG_HOME does the
    right thing of NOT setting in a non-production environment, and setting in a web
    production environment.

    Test this both ways by setting PYTHONPATH=/proj/web-kadi, or not.  Use -v -s flags in
    pytest to confirm expect path.
    """
    ska_data_config = os.path.join(os.environ["SKA"], "data", "config")
    if any(pth.startswith("/proj/web-kadi") for pth in sys.path):
        # Apache WSGI config sets local PYTHONPATH to include /proj/web-kadi (regardless
        # of where kadi is actually getting imported).
        print("Checking production web path")
        assert os.environ["XDG_CONFIG_HOME"] == ska_data_config
        assert os.environ["XDG_CACHE_HOME"] == os.environ["XDG_CONFIG_HOME"]
    else:
        # Normal import, not part of web server
        print("Checking normal import path")
        assert not os.environ.get("XDG_CONFIG_HOME", "").startswith(ska_data_config)
        assert not os.environ.get("XDG_CACHE_HOME", "").startswith(ska_data_config)


def test_overlapping_intervals():
    """
    Intervals that overlap due to interval_pad get merged.
    """
    start = "2013:221:00:10:00.000"
    stop = "2013:221:00:20:00.000"
    fa_moves = deepcopy(events.fa_moves)
    fa_moves.interval_pad = 0.0
    assert fa_moves.intervals(start, stop) == [
        ("2013:221:00:11:33.100", "2013:221:00:12:05.900"),
        ("2013:221:00:12:38.700", "2013:221:00:13:11.500"),
    ]
    fa_moves.interval_pad = 300.0
    assert fa_moves.intervals(start, stop) == [
        ("2013:221:00:10:00.000", "2013:221:00:18:11.500")
    ]


def test_interval_pads():
    """
    Intervals pads.
    """
    start = "2013:221:00:10:00.000"
    stop = "2013:221:00:20:00.000"
    intervals = [
        ("2013:221:00:11:33.100", "2013:221:00:12:05.900"),
        ("2013:221:00:12:38.700", "2013:221:00:13:11.500"),
    ]

    assert events.fa_moves.intervals(start, stop) == intervals

    fa_moves = events.fa_moves()
    assert fa_moves.intervals(start, stop) == intervals

    fa_moves = events.fa_moves(pad=0)
    assert fa_moves.intervals(start, stop) == intervals

    fa_moves = events.fa_moves(pad=(0, 0))
    assert fa_moves.intervals(start, stop) == intervals

    # 5 seconds earlier and 10 seconds later
    fa_moves = events.fa_moves(pad=(5, 10))
    assert fa_moves.intervals(start, stop) == [
        ("2013:221:00:11:28.100", "2013:221:00:12:15.900"),
        ("2013:221:00:12:33.700", "2013:221:00:13:21.500"),
    ]

    fa_moves = events.fa_moves(pad=300)
    assert fa_moves.intervals(start, stop) == [
        ("2013:221:00:10:00.000", "2013:221:00:18:11.500")
    ]


def test_query_event_intervals():
    intervals = (events.manvrs & events.tsc_moves).intervals(
        "2012:001:12:00:00", "2012:002:12:00:00"
    )
    assert intervals == [
        ("2012:001:18:21:31.715", "2012:001:18:22:04.515"),
        ("2012:002:02:53:03.804", "2012:002:02:54:50.917"),
    ]


def test_basic_query():
    rad_zones = events.rad_zones.filter("2013:001:12:00:00", "2013:007:12:00:00")
    assert str(rad_zones).splitlines() == [
        "<RadZone: 1852 2013:003:16:19:36 2013:004:02:21:34 dur=36.1 ksec>",
        "<RadZone: 1853 2013:006:08:22:22 2013:006:17:58:48 dur=34.6 ksec>",
    ]
    rzt = rad_zones.table
    rzt["tstart"].format = ".3f"
    rzt["tstop"].format = ".3f"
    rzt["dur"].format = ".3f"
    # fmt: off
    assert rzt.pformat(max_width=-1) == [
        "        start                  stop             tstart        tstop        dur    orbit orbit_num        perigee       ",  # noqa
        "--------------------- --------------------- ------------- ------------- --------- ----- --------- ---------------------",  # noqa
        "2013:003:16:19:36.289 2013:004:02:21:34.289 473617243.473 473653361.473 36118.000  1852      1852 2013:003:22:29:59.302",  # noqa
        "2013:006:08:22:22.982 2013:006:17:58:48.982 473847810.166 473882396.166 34586.000  1853      1853 2013:006:13:58:21.389",  # noqa
    ]
    # fmt: on

    rad_zones = events.rad_zones.filter("2013:001:12:00:00", "2013:002:12:00:00")
    assert len(rad_zones) == 0
    assert len(rad_zones.table) == 0


def test_short_query():
    """
    Short duration queries that test that filter will return partially
    included intervals.
    """
    dwells = events.dwells.filter("2012:002:00:00:00", "2012:002:00:00:01")
    assert len(dwells) == 1
    dwells = events.dwells.filter("2012:001:18:40:00", "2012:001:18:42:00")
    assert len(dwells) == 1
    dwells = events.dwells.filter("2012:002:02:49:00", "2012:002:02:50:00")
    assert len(dwells) == 1


def test_get_obsid():
    """
    Test that the get_obsid() method gives the right obsid for all event models.
    """
    models = events.models.get_event_models()
    for model in models.values():
        if model.__name__ == "SafeSun":
            continue  # Doesn't work for SafeSun because of bad OBC telem
        model_obj = model.objects.filter(start__gte="2002:010:12:00:00")[0]
        obsid = model_obj.get_obsid()
        obsid_obj = events.obsids.filter(obsid__exact=obsid)[0]
        model_obj_start = DateTime(
            getattr(model_obj, model_obj._get_obsid_start_attr)
        ).date
        assert obsid_obj.start <= model_obj_start
        assert obsid_obj.stop > model_obj_start

        # Now test that searching for objects with the same obsid gets
        # some matching objects and that they all have the same obsid.
        if model_obj.model_name in ("major_event", "safe_sun"):
            continue  # Doesn't work for these
        query = getattr(events, model_obj.model_name + "s")
        query_events = query.filter(obsid=obsid)
        assert len(query_events) >= 1
        for query_event in query_events:
            assert query_event.get_obsid() == obsid


def test_intervals_filter():
    """
    Test setting filter keywords in the EventQuery object itself.
    """
    ltt_bads = events.ltt_bads
    start, stop = "2000:121:12:00:00", "2000:134:12:00:00"

    # 2000-04-30 00:00:00 | ELBI_LOW        | R
    # 2000-04-30 00:00:00 | EPOWER1         | R
    # 2000-05-01 00:00:00 | 3SDTSTSV        | Y
    # 2000-05-13 00:00:00 | 3SDP15V         | 1

    lines = sorted(
        str(ltt_bads().filter("2000:121:12:00:00", "2000:134:12:00:00")).splitlines()
    )
    assert lines == [
        "<LttBad: start=2000:121:00:00:00.000 msid=ELBI_LOW flag=R>",
        "<LttBad: start=2000:121:00:00:00.000 msid=EPOWER1 flag=R>",
        "<LttBad: start=2000:122:00:00:00.000 msid=3SDTSTSV flag=Y>",
        "<LttBad: start=2000:134:00:00:00.000 msid=3SDP15V flag=1>",
    ]

    # No filter
    assert ltt_bads.intervals(start, stop) == [
        ("2000:121:12:00:00.000", "2000:123:00:00:00.000"),
        ("2000:134:00:00:00.000", "2000:134:12:00:00.000"),
    ]

    assert ltt_bads(flag="Y").intervals(start, stop) == [
        ("2000:122:00:00:00.000", "2000:123:00:00:00.000")
    ]

    assert ltt_bads(msid="ELBI_LOW", flag="R").intervals(start, stop) == [
        ("2000:121:12:00:00.000", "2000:122:00:00:00.000")
    ]

    assert ltt_bads(msid="ELBI_LOW", flag="1").intervals(start, stop) == []


def test_get_overlaps():
    """
    Unit test QuerySet._get_full_overlaps and QuerySet._get_partial_overlaps
    """
    dates = [
        ("2000:001:00:00:00", "2000:002:00:00:00"),
        ("2000:003:00:00:00", "2000:004:00:00:00"),
        ("2000:005:00:00:00", "2000:006:00:00:00"),
    ]
    datestarts, datestops = zip(*dates)
    datestarts, datestops = np.array(datestarts), np.array(datestops)

    overlaps = dates
    overlap_starts, overlap_stops = zip(*overlaps)
    overlap_starts, overlap_stops = np.array(overlap_starts), np.array(overlap_stops)

    x = events.manvrs.filter("2000:001:12:00:00", "2000:002:12:00:00")
    indices = x._get_full_overlaps(overlap_starts, overlap_stops, datestarts, datestops)
    assert np.all(indices == [0, 1, 2])

    indices = x._get_partial_overlaps(
        overlap_starts, overlap_stops, datestarts, datestops
    )
    assert np.all(indices == [0, 1, 2])

    overlaps = [
        ("2000:001:00:00:00", "2000:001:12:00:00"),
        ("2000:001:13:00:00", "2000:002:00:00:00"),
        ("2000:005:00:00:01", "2000:006:00:00:00"),
    ]
    overlap_starts, overlap_stops = zip(*overlaps)
    overlap_starts, overlap_stops = np.array(overlap_starts), np.array(overlap_stops)
    indices = x._get_full_overlaps(overlap_starts, overlap_stops, datestarts, datestops)
    assert len(indices) == 0

    indices = x._get_partial_overlaps(
        overlap_starts, overlap_stops, datestarts, datestops
    )
    assert np.all(indices == [0, 2])

    overlap_starts, overlap_stops = [], []
    overlap_starts, overlap_stops = np.array(overlap_starts), np.array(overlap_stops)

    indices = x._get_full_overlaps(overlap_starts, overlap_stops, datestarts, datestops)
    assert len(indices) == 0

    indices = x._get_partial_overlaps(
        overlap_starts, overlap_stops, datestarts, datestops
    )
    assert len(indices) == 0


def test_select_overlapping():
    """
    Functional test of selecting overlapping events
    """
    manvrs = events.manvrs.filter("2001:001:00:00:00", "2001:003:00:00:00")

    manvrs_sel = manvrs.select_overlapping(~events.rad_zones)
    assert [repr(x) for x in manvrs_sel] == [
        "<Manvr: start=2001:001:07:48:35.843 dur=2073 n_dwell=2 template=nman_dwell>",
        "<Manvr: start=2001:002:20:50:34.523 dur=1185 n_dwell=1 template=normal>",
    ]

    manvrs_sel = manvrs.select_overlapping(events.rad_zones)
    assert [repr(x) for x in manvrs_sel] == [
        "<Manvr: start=2001:002:05:20:14.046 dur=1184 n_dwell=1 template=normal>",
        "<Manvr: start=2001:002:08:53:23.997 dur=241 n_dwell=1 template=normal>",
        "<Manvr: start=2001:002:14:55:24.773 dur=240 n_dwell=1 template=normal>",
    ]

    manvrs_sel = manvrs.select_overlapping(events.tsc_moves, allow_partial=False)
    assert [repr(x) for x in manvrs_sel] == []

    manvrs_sel = manvrs.select_overlapping(events.tsc_moves)
    assert [repr(x) for x in manvrs_sel] == [
        "<Manvr: start=2001:002:20:50:34.523 dur=1185 n_dwell=1 template=normal>"
    ]
