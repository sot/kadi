import numpy as np

from ... import cmds as commands
from ...cmds import states

import Chandra.cmd_states as cmd_states
from Chandra.Time import DateTime
from Ska.engarchive import fetch


def get_states(start, stop, state_keys):
    start = DateTime(start)
    stop = DateTime(stop)

    cstates = cmd_states.fetch_states(start, stop)
    rcstates = states.reduce_states(cstates, state_keys)
    lenr = len(rcstates)

    cmds = commands.filter(start - 7, stop)
    kstates = states.get_states_for_cmds(cmds, state_keys)
    rkstates = states.reduce_states(kstates, state_keys)[-lenr:]

    return rcstates, rkstates


def compare_states(start, stop, state_keys, compare_state_keys=None):
    rcstates, rkstates = get_states(start, stop, state_keys)

    if compare_state_keys is None:
        compare_state_keys = state_keys

    for colname in compare_state_keys:
        assert np.all(rkstates[colname] == rcstates[colname])

    return rcstates, rkstates


def test_acis():
    """
    Test all ACIS states include vid_board for late-2017
    """
    state_keys = ['clocking', 'power_cmd',  'fep_count', 'si_mode',  'ccd_count', 'vid_board']
    compare_states('2017:280', '2017:360', state_keys, state_keys + ['datestart'])


def test_states_2017():
    """
    Test for 200 days in 2017.  Includes 2017:066, 068, 090 anomalies and
    2017:250-254 SCS107 + 251 CTI.

    Skip 'vid_board' because https://github.com/sot/cmd_states/pull/31 was put
    in place around 2017:276 and so the behavior of this changed then.

    Skip 'pitch' because Chandra.cmd_states only avoids pitch breaks within actual
    maneuver commanding (so the NMAN period before maneuver starts is OK) while kadi
    will insert pitch breaks only in NPNT.
    """
    state_keys = (['obsid', 'clocking', 'power_cmd',  'fep_count',
                   'si_mode',  'ccd_count'] +
                  ['q1', 'q2', 'q3', 'q4', 'pcad_mode'])
    rcstates, rkstates = compare_states('2017:060', '2017:260', state_keys)

    # Check state datestart.  There are 4 known discrepancies of 0.001 sec
    # due to a slight difference in the start time for NSUN maneuvers.  Cmd_states
    # uses the exact floating point time to start while kadi uses the string time
    # rounded to the nearest msec.  The float command time is not stored in kadi.
    bad = np.flatnonzero(rkstates['datestart'] != rcstates['datestart'])
    assert len(bad) == 4
    tk = DateTime(rkstates['datestart'][bad]).secs
    tc = DateTime(rcstates['datestart'][bad]).secs
    assert np.all(np.abs(tk - tc) < 0.0015)


def test_pitch_2017():
    """
    Test pitch for 100 days in 2017.  Includes 2017:066, 068, 090 anomalies.  This is done
    by interpolating states (at 200 second intervals) because the pitch generation differs
    slightly between kadi and Chandra.cmd_states.  (See test_states_2017 note).

    Make sure that pitch matches to within 0.5 deg in all samples, and 0.05 deg during
    NPNT.
    """
    rcstates, rkstates = get_states('2017:060', '2017:160', ['pcad_mode', 'pitch'])

    rcstates['tstop'] = DateTime(rcstates['datestop']).secs
    rkstates['tstop'] = DateTime(rkstates['datestop']).secs

    times = np.arange(rcstates['tstop'][0], rcstates['tstop'][-2], 200.0)
    rci = cmd_states.interpolate_states(rcstates, times)
    rki = cmd_states.interpolate_states(rkstates, times)

    dp = np.abs(rci['pitch'] - rki['pitch'])
    assert np.all(dp < 0.5)
    assert np.all(rci['pcad_mode'] == rki['pcad_mode'])
    ok = rci['pcad_mode'] == 'NPNT'
    assert np.all(dp[ok] < 0.05)


def test_sun_vec_versus_telemetry():
    """
    Test sun vector values `pitch` and `off_nominal_roll` versus flight telem.  Include
    Load maneuver at 2017:349:20:52:37.719 in DEC1117 with large pitch and
    off_nominal_roll change (from around zero to -17 deg).

    State values are within 1.5 degrees of telemetry.
    """

    state_keys = ['pitch', 'off_nom_roll']
    start, stop = '2017:349:10:00:00', '2017:350:10:00:00'
    cmds = commands.filter(start, stop)
    kstates = states.get_states_for_cmds(cmds, state_keys)[-20:-1]
    rk = states.reduce_states(kstates, state_keys)

    tstart = DateTime(rk['datestart']).secs
    tstop = DateTime(rk['datestop']).secs
    tmid = (tstart + tstop) / 2

    # Pitch from telemetry
    dat = fetch.Msid('pitch', tstart[0] - 100, tstop[-1] + 100)
    dat.interpolate(times=tmid)
    delta = np.abs(dat.vals - rk['pitch'])
    assert np.max(rk['pitch']) - np.min(rk['pitch']) > 75  # Big maneuver
    assert np.all(delta < 1.5)

    # Off nominal roll (not roll from ra,dec,roll) from telemetry
    dat = fetch.Msid('roll', tstart[0] - 100, tstop[-1] + 100)
    dat.interpolate(times=tmid)
    delta = np.abs(dat.vals - rk['off_nom_roll'])
    assert np.max(rk['off_nom_roll']) - np.min(rk['off_nom_roll']) > 20  # Large range
    assert np.all(delta < 1.5)
