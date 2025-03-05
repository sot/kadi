# Use data file from parse_cm.test for get_cmds_from_backstop test.
# This package is a dependency
import astropy.table as apt
import numpy as np
import pytest
from testr.test_helper import has_internet

import kadi.commands as kc
import kadi.commands.commands_v2 as kcc2
import kadi.commands.observations as kco
import kadi.commands.states as kcs

HAS_INTERNET = has_internet()

# Expected observation outputs for two test cases
exp_2024366 = [
    "obsid_1 obsid_2 simpos_1 simpos_2 source_1      starcat_date    ",
    "------- ------- -------- -------- -------- ---------------------",
    "  42814   42814   -99616   -99616 DEC2324B 2024:365:23:24:58.901",
    "  42813   42813   -99616   -99616 DEC2324B 2024:366:01:17:07.965",
    "  42812   42812   -99616   -99616 DEC2324B 2024:366:02:58:28.076",
    "  42810   42810   -99616   -99616 DEC2324B 2024:366:05:08:36.000",
    "  42809   42809   -99616   -99616 DEC2324B 2024:366:05:48:15.632",
    "  30700   30700    92560    92560 DEC2324B 2024:366:08:26:39.076",
    "  30690   30690    92904    92904 DEC2324B 2024:366:11:55:08.999",
    "  30550   30550   -99616   -99616 DEC2324B 2024:366:15:21:03.938",  # SCS-107
    "  30550   28803   -99616    92904 DEC2324B 2024:366:18:05:04.938",
    "  30550   28365   -99616    75624 DEC2324B 2024:366:21:21:07.538",
    "  30550   29835   -99616    92560 DEC2324B 2025:001:08:44:07.810",
    "  65518   25501   -99616   -50504 DEC2324B 2025:001:13:08:09.384",  # Obsid update
    "  65518   26975   -99616    92904 DEC2324B 2025:001:16:38:50.384",
    "  65518   30500   -99616    75624 DEC2324B 2025:001:20:54:41.492",
    "  65518   25505   -99616   -50504 DEC2324B 2025:002:01:35:29.276",
    "  65518   29904   -99616    92904 DEC2324B 2025:002:03:07:54.276",
    "  65518   30688   -99616    92904 DEC2324B 2025:002:04:53:45.276",
    "  65518   30082   -99616    92904 DEC2324B 2025:002:08:11:25.231",
    "  65518   42808   -99616   -99616 DEC2324B 2025:002:12:59:42.571",
    "  65518   42807   -99616   -99616 DEC2324B 2025:002:14:17:07.525",
    "  65518   42806   -99616   -99616 DEC2324B 2025:002:17:17:25.479",
    "  65518   42805   -99616   -99616 DEC2324B 2025:002:18:36:19.655",
    "  65518   42803   -99616   -99616 DEC2324B 2025:002:20:37:36.000",
    "  65518   42802   -99616   -99616 DEC2324B 2025:002:21:24:48.840",
    "  65518   42801   -99616   -99616 DEC2324B 2025:002:23:18:02.848",
    "  65518   30692   -99616    75624 DEC2324B 2025:003:01:18:38.220",
    "  30707   30707    92904    92904 JAN0325A 2025:003:03:30:50.971",  # Replan
    "  42808   42808    92904    92904 JAN0325A 2025:003:07:11:30.932",
    "  30552   30552   -99616   -99616 JAN0325A 2025:003:08:20:57.635",
    "  29750   29750    75624    75624 JAN0325A 2025:003:11:33:42.635",
    "  30708   30708    92904    92904 JAN0325A 2025:003:12:45:37.779",
    "  30689   30689    92904    92904 JAN0325A 2025:003:16:51:59.824",
    "  30083   30083    92904    92904 JAN0325A 2025:003:20:10:55.729",
]

exp_2025012 = [
    "obsid_1 obsid_2 simpos_1 simpos_2 source_1      starcat_date    ",
    "------- ------- -------- -------- -------- ---------------------",
    "  28808   28808    92904    92904 JAN0325A 2025:011:21:57:32.070",
    "  30727   30727   -99616   -99616 JAN0325A 2025:012:00:16:58.622",
    "  30375   30375    75624    75624 JAN0325A 2025:012:03:46:03.622",
    "  28203   28203    75624    75624 JAN0325A 2025:012:09:12:20.647",  # SCS-107
    "  28203   30713   -99616    75624 JAN0325A 2025:012:16:32:24.171",
    "  28203   28777   -99616   -50504 JAN0325A 2025:012:19:57:10.011",
    "  28203   30720   -99616    75624 JAN0325A 2025:012:23:49:31.011",
    "  28203   42779   -99616   -99616 JAN0325A 2025:013:03:02:14.487",
    "  28203   42778   -99616   -99616 JAN1325A 2025:013:04:44:06.000",  # Observing not run
    "  28203   42777   -99616   -99616 JAN1325A 2025:013:05:57:09.907",
    "  28203   42776   -99616   -99616 JAN1325A 2025:013:07:07:46.860",
    "  28203   42775   -99616   -99616 JAN1325A 2025:013:07:44:32.626",
    "  28203   42774   -99616   -99616 JAN1325A 2025:013:09:01:57.271",
    "  28203   42772   -99616   -99616 JAN1325A 2025:013:10:28:47.000",
    "  28203   42771   -99616   -99616 JAN1325A 2025:013:10:52:58.000",
    "  28203   42770   -99616   -99616 JAN1325A 2025:013:12:13:46.779",
    "  28203   28538   -99616    75624 JAN1325A 2025:013:13:10:11.281",
    "  28203   29875   -99616    75624 JAN1325A 2025:013:22:22:50.269",
    "  28203   29836   -99616    92560 JAN1325A 2025:014:03:09:59.659",
    "  28203   29623   -99616    91576 JAN1325A 2025:014:07:27:04.516",
    "  28203   30730   -99616    75624 JAN1325A 2025:014:13:35:28.530",
    "  28203   30729   -99616    92560 JAN1325A 2025:014:20:15:34.620",
    "  28203   30373   -99616    75624 JAN1325A 2025:015:01:13:39.560",
    "  28565   28565    75624    75624 JAN1525A 2025:015:06:34:03.769",  # Replan
    "  29963   29963    75624    75624 JAN1525A 2025:015:15:17:21.528",
    "  42779   42779   -99616   -99616 JAN1525A 2025:015:18:31:28.826",
    "  42778   42778   -99616   -99616 JAN1525A 2025:015:19:52:46.485",
]

# SCS-107 filter + "Command" filter removes all CMT_EVT commands.
# There are a number of "Command" event related to the HRC shutoff due to SCS-107
# during an HRC-S observation.
cmd_evt_params_2024366 = []

# SCS-107 filter leaves in two source="CMD_EVT" commands. These get checked in the test.
cmd_evt_params_2025012 = [
    {"event": "Command", "event_date": "2025:013:15:14:38"},
    {
        "event": "Load_not_run",
        "event_date": "2025:014:02:55:00",
        "event_type": "LOAD_NOT_RUN",
        "load": "JAN1425A",
    },
]

cases = [
    # Includes a manual "Obsid" update event
    {
        "start": "2024:366",
        "stop": "2025:004",
        "exp_lines": exp_2024366,
        "len_as_run": 36,  # One extra observation due to the manual obsid update
        "len_planned": 35,
        "default_stop": 20,
        "event_filter": (
            kc.filter_scs107_events,
            kc.filter_cmd_events_by_event("Command"),  # Remove all HRC turnoff commands
        ),
        "cmd_evt_params": cmd_evt_params_2024366,
    },
    # Includes an "Observing not run" JAN1325A event
    {
        "start": "2025:012",
        "stop": "2025:016",
        "exp_lines": exp_2025012,
        "len_as_run": 29,
        "len_planned": 29,
        "default_stop": 60,
        "event_filter": kc.filter_scs107_events,
        "cmd_evt_params": cmd_evt_params_2025012,
    },
]


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
@pytest.mark.parametrize("case", cases)
def test_filter_scs107_events(
    clear_caches,
    cmds_dir,
    case,
):
    """Filter the command events associated with SCS-107 events.

    Test that get_observations(), get_states(), and get_cmds() are working as expected
    when the event_filter is applied to the planned observations.
    """
    start = case["start"]
    stop = case["stop"]
    exp_lines = case["exp_lines"]
    len_as_run = case["len_as_run"]
    len_planned = case["len_planned"]
    default_lookback = case["default_stop"]
    event_filter = case["event_filter"]
    cmd_evt_params = case["cmd_evt_params"]

    obss_as_run = apt.Table(kc.get_observations(start, stop))

    with kc.set_time_now(stop), kc.conf.set_temp("default_lookback", default_lookback):
        obss_planned = apt.Table(
            kc.get_observations(start, stop, event_filter=event_filter)
        )
        states_planned = kcs.get_states(
            start, stop, event_filter=event_filter, state_keys=["obsid"]
        )
        cmds_planned = kc.get_cmds(start, stop, event_filter=event_filter)

    # get_states() is working as expected
    assert np.all(states_planned["obsid"] == obss_planned["obsid"])

    # get_cmds() is working as expected - no CMD_EVT in the planned observations
    ok = cmds_planned["source"] == "CMD_EVT"
    assert np.all(cmds_planned[ok]["params"] == cmd_evt_params)
    assert len(obss_as_run) == len_as_run
    assert len(obss_planned) == len_planned

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
        "obsid_1",
        "obsid_2",
        "simpos_1",
        "simpos_2",
        "source_1",
        "starcat_date",
    ].pformat()

    assert lines == exp_lines

    keys = [
        # scenario, stop, default_lookback, event_filter
        (None, None, 30, None),
        (None, stop, default_lookback, event_filter),
    ]
    assert list(kcc2.CMDS_RECENT) == keys
    assert list(kco.OBSERVATIONS) == keys
