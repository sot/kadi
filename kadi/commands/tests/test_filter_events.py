# Use data file from parse_cm.test for get_cmds_from_backstop test.
# This package is a dependency
import astropy.table as apt
import numpy as np
import pytest
from testr.test_helper import has_internet

import kadi.commands as kc
import kadi.commands.commands_v2 as kcc2
import kadi.commands.observations as kco

HAS_INTERNET = has_internet()


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_filter_scs107_2024366(cmds_dir):
    """Filter the command events associated with SCS-107 on 2024:366"""
    start = "2024:365"
    stop = "2025:003"

    obss_as_run = apt.Table(kc.get_observations(start, stop))

    with kc.set_time_now(stop), kc.conf.set_temp("default_lookback", 20):
        obss_planned = apt.Table(
            kc.get_observations(start, stop, event_filter=kc.filter_scs107_events)
        )

    # As-run gets an extra observation because the manual obsid update at
    # 2025:001:12:48:34.040 breaks the observation into two.
    assert len(obss_as_run) == 34
    assert len(obss_planned) == 33

    # Now get rid of the 2nd of the observation that was split into two. The source of
    # CMD_EVT is from the Obsid command event at 2025:001:12:48:34.040.
    ok = obss_as_run["source"] != "CMD_EVT"
    obss_as_run = obss_as_run[ok]

    # Make sure that the source and starcat_date are the same.
    assert set(obss_as_run.colnames) == set(obss_planned.colnames)
    for name in obss_as_run.colnames:
        if name in ["obsid", "simpos"]:
            continue
    assert np.all(obss_as_run[name] == obss_planned[name])

    # Now filter out the two observations that have no star catalog. These are both
    # the gyro hold in the high-IR zone.
    ok = ~obss_as_run["starcat_date"].mask
    obss_as_run = obss_as_run[ok]
    obss_planned = obss_planned[ok]

    obss_run_planned = apt.join(obss_as_run, obss_planned, keys="starcat_date")
    lines = obss_run_planned[
        "obsid_1", "obsid_2", "simpos_1", "simpos_2", "starcat_date"
    ].pformat()

    assert lines == [
        "obsid_1 obsid_2 simpos_1 simpos_2      starcat_date    ",
        "------- ------- -------- -------- ---------------------",
        "  30693   30693    92904    92904 2024:364:21:02:54.559",
        "  30699   30699    92560    92560 2024:365:00:21:34.399",
        "  28631   28631    92904    92904 2024:365:03:49:22.699",
        "  30486   30486   -50504   -50504 2024:365:12:13:55.474",
        "  30497   30497    75624    75624 2024:365:16:49:40.474",
        "  42815   42815   -99616   -99616 2024:365:21:32:34.599",
        "  42814   42814   -99616   -99616 2024:365:23:24:58.901",
        "  42813   42813   -99616   -99616 2024:366:01:17:07.965",
        "  42812   42812   -99616   -99616 2024:366:02:58:28.076",
        "  42810   42810   -99616   -99616 2024:366:05:08:36.000",
        "  42809   42809   -99616   -99616 2024:366:05:48:15.632",
        "  30700   30700    92560    92560 2024:366:08:26:39.076",
        "  30690   30690    92904    92904 2024:366:11:55:08.999",
        "  30550   30550   -99616   -99616 2024:366:15:21:03.938",
        "  30550   28803   -99616    92904 2024:366:18:05:04.938",
        "  30550   28365   -99616    75624 2024:366:21:21:07.538",
        "  30550   29835   -99616    92560 2025:001:08:44:07.810",
        "  65518   25501   -99616   -50504 2025:001:13:08:09.384",
        "  65518   26975   -99616    92904 2025:001:16:38:50.384",
        "  65518   30500   -99616    75624 2025:001:20:54:41.492",
        "  65518   25505   -99616   -50504 2025:002:01:35:29.276",
        "  65518   29904   -99616    92904 2025:002:03:07:54.276",
        "  65518   30688   -99616    92904 2025:002:04:53:45.276",
        "  65518   30082   -99616    92904 2025:002:08:11:25.231",
        "  65518   42808   -99616   -99616 2025:002:12:59:42.571",
        "  65518   42807   -99616   -99616 2025:002:14:17:07.525",
        "  65518   42806   -99616   -99616 2025:002:17:17:25.479",
        "  65518   42805   -99616   -99616 2025:002:18:36:19.655",
        "  65518   42803   -99616   -99616 2025:002:20:37:36.000",
        "  65518   42802   -99616   -99616 2025:002:21:24:48.840",
        "  65518   42801   -99616   -99616 2025:002:23:18:02.848",
    ]

    keys = [
        (None, None, 30, None),
        (None, "2025:003", 20, kc.filter_scs107_events),
    ]
    assert list(kcc2.CMDS_RECENT) == keys
    assert list(kco.OBSERVATIONS) == keys
