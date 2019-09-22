import os
import numpy as np

from .. import commands, states
import pytest

import Chandra.cmd_states as cmd_states
from Chandra.Time import DateTime
from Ska.engarchive import fetch
from astropy.io import ascii
from astropy.table import Table

try:
    fetch.get_time_range('dp_pitch')
    HAS_PITCH = True
except Exception:
    HAS_PITCH = False


def assert_all_close_states(rc, rk, keys):
    """
    Compare all ``key`` columns of the commanded states table ``rc`` and
    the kadi states table ``rk``.
    """
    for key in keys:
        rcdtype = rc[key].dtype
        if rcdtype.kind == 'f':
            assert np.allclose(rk[key].astype(rcdtype), rc[key])
        else:
            assert np.all(rk[key] == rc[key])


def get_states_test(start, stop, state_keys, continuity=None):
    """
    Helper for getting states from kadi and cmd_states
    """
    start = DateTime(start)
    stop = DateTime(stop)

    cstates = Table(cmd_states.fetch_states(start, stop))
    trans_keys = [set(val.split(',')) for val in cstates['trans_keys']]
    cstates.remove_column('trans_keys')  # Necessary for older astropy
    cstates['trans_keys'] = trans_keys
    rcstates = states.reduce_states(cstates, state_keys, merge_identical=True)
    lenr = len(rcstates)

    cmds = commands.get_cmds(start - 7, stop)
    kstates = states.get_states(state_keys=state_keys, cmds=cmds,
                                continuity=continuity, reduce=False)
    rkstates = states.reduce_states(kstates, state_keys, merge_identical=True)[-lenr:]

    return rcstates, rkstates


def compare_states(start, stop, state_keys, compare_state_keys=None, continuity=None,
                   compare_dates=True):
    """
    Helper for comparing states from kadi and cmd_states
    """
    rcstates, rkstates = get_states_test(start, stop, state_keys, continuity=continuity)

    if compare_state_keys is None:
        compare_state_keys = state_keys

    assert_all_close_states(rcstates, rkstates, compare_state_keys)

    if compare_dates:
        assert np.all(rcstates['datestart'][1:] == rkstates['datestart'][1:])
        assert np.all(rcstates['datestop'][:-1] == rkstates['datestop'][:-1])

    return rcstates, rkstates


def test_acis():
    """
    Test all ACIS states include vid_board for late-2017
    """
    state_keys = ['clocking', 'power_cmd', 'fep_count', 'si_mode', 'vid_board']
    rc, rk = compare_states('2017:280', '2017:360', state_keys, state_keys)


def test_cmd_line_interface(tmpdir):
    """
    Test command line interface
    """
    filename = os.path.join(str(tmpdir), 'out.txt')
    states.get_chandra_states(['--outfile', filename,
                               '--start', '2017:001:21:00:00',
                               '--stop', '2017:002:11:30:00',
                               '--state-keys', 'obsid,si_mode,pcad_mode'])
    with open(filename, 'r') as fh:
        out = fh.read()
    assert out.splitlines() == [
        '             datestart               datestop  obsid   si_mode  pcad_mode ',
        ' 2017:001:21:00:00.000  2017:001:21:02:06.467  18140  TE_008FC       NPNT ',
        ' 2017:001:21:02:06.467  2017:001:21:05:06.467  18140  TE_008FC       NMAN ',
        ' 2017:001:21:05:06.467  2017:001:21:06:41.467  19973  TE_008FC       NMAN ',
        ' 2017:001:21:06:41.467  2017:001:21:23:02.282  19973  TE_00A58       NMAN ',
        ' 2017:001:21:23:02.282  2017:002:11:23:43.185  19973  TE_00A58       NPNT ',
        ' 2017:002:11:23:43.185  2017:002:11:26:43.185  19973  TE_00A58       NMAN ',
        ' 2017:002:11:26:43.185  2017:002:11:29:29.870  50432  TE_00A58       NMAN ',
        ' 2017:002:11:29:29.870  2017:002:11:30:00.000  50432  TE_00A58       NPNT ']


def test_quick():
    """
    Test for a few days in 2017.  Sanity check for refactoring etc.
    """
    state_keys = (['obsid', 'clocking', 'power_cmd', 'fep_count', 'vid_board',
                   'si_mode', 'ccd_count'] +
                  ['q1', 'q2', 'q3', 'q4', 'pcad_mode', 'dither', 'ra', 'dec', 'roll'] +
                  ['letg', 'hetg'] +
                  ['simpos', 'simfa_pos'])
    continuity = {'letg': 'RETR', 'hetg': 'RETR'}  # Not necessarily set within 7 days
    rc, rk = compare_states('2018:235', '2018:245', state_keys, continuity=continuity)

    # Now test using start/stop pair with start/stop and no supplied cmds or continuity.
    # This also tests the API kwarg order: datestart, datestop, state_keys, ..)
    sts = states.get_states('2018:235', '2018:245', state_keys, reduce=False)
    rk = states.reduce_states(sts, state_keys, merge_identical=True)
    assert len(rc) == len(rk)

    assert_all_close_states(rc, rk, state_keys)

    assert np.all(rc['datestart'][1:] == rk['datestart'][1:])
    assert np.all(rc['datestop'][:-1] == rk['datestop'][:-1])


def test_states_2017():
    """
    Test for 200 days in 2017.  Includes 2017:066, 068, 090 anomalies and
    2017:250-254 SCS107 + 251 CTI.

    Skip 'vid_board' because https://github.com/sot/cmd_states/pull/31 was put
    in place around 2017:276 and so the behavior of this changed then. (Tested later).

    Skip 'ccd_count' because https://github.com/sot/cmd_states/pull/39 changed
    that behavior.

    Skip 'pitch' because Chandra.cmd_states only avoids pitch breaks within actual
    maneuver commanding (so the NMAN period before maneuver starts is OK) while kadi
    will insert pitch breaks only in NPNT.  (Tested later).
    """
    state_keys = (['obsid', 'clocking', 'power_cmd', 'fep_count',
                   'si_mode'] +
                  ['q1', 'q2', 'q3', 'q4', 'pcad_mode', 'dither', 'ra', 'dec', 'roll'] +
                  ['letg', 'hetg'] +
                  ['simpos', 'simfa_pos'])
    rcstates, rkstates = compare_states('2017:060', '2017:260', state_keys, compare_dates=False)

    # Check state datestart.  There are 4 known discrepancies of 0.001 sec
    # due to a slight difference in the start time for NSUN maneuvers.  Cmd_states
    # uses the exact floating point time to start while kadi uses the string time
    # rounded to the nearest msec.  The float command time is not stored in kadi.
    # For this test drop the first state, which has a datestart mismatch because
    # of the difference in startup between Chandra.cmd_states and kadi.states.
    rkstates = rkstates[1:]
    rcstates = rcstates[1:]
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
    rcstates, rkstates = get_states_test('2017:060', '2017:160', ['pcad_mode', 'pitch'])

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


@pytest.mark.skipif('not HAS_PITCH')
def test_sun_vec_versus_telemetry():
    """
    Test sun vector values `pitch` and `off_nominal_roll` versus flight telem.  Include
    Load maneuver at 2017:349:20:52:37.719 in DEC1117 with large pitch and
    off_nominal_roll change (from around zero to -17 deg).

    State values are within 1.5 degrees of telemetry.
    """

    state_keys = ['pitch', 'off_nom_roll']
    start, stop = '2017:349:10:00:00', '2017:350:10:00:00'
    cmds = commands.get_cmds(start, stop)
    rk = states.get_states(state_keys=state_keys, cmds=cmds, merge_identical=True)[-20:-1]

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


def test_dither():
    """Values look reasonable given load commands"""
    cmds = commands.get_cmds('2017:341:21:40:05', '2017:350:00:00:00')
    rk = states.get_states(state_keys=['dither_phase_pitch', 'dither_phase_yaw',
                                       'dither_ampl_pitch', 'dither_ampl_yaw',
                                       'dither_period_pitch', 'dither_period_yaw'],
                           cmds=cmds)

    assert np.all(rk['datestart'] == ['2017:341:21:40:05.265',
                                      '2017:342:08:26:34.023',
                                      '2017:345:02:45:50.318',
                                      '2017:345:09:02:21.704',
                                      '2017:345:17:18:08.893',
                                      '2017:346:04:35:44.546',
                                      '2017:349:21:28:42.426'])
    assert np.all(rk['dither_phase_pitch'] == 0.0)
    assert np.all(rk['dither_phase_yaw'] == 0.0)
    ampls = [20.00149628163879,
             7.9989482580707696,
             20.00149628163879,
             7.9989482580707696,
             20.00149628163879,
             7.9989482580707696,
             20.00149628163879]
    assert np.allclose(rk['dither_ampl_pitch'], ampls, atol=1e-6, rtol=0)
    assert np.allclose(rk['dither_ampl_yaw'], ampls, atol=1e-6, rtol=0)
    assert np.allclose(rk['dither_period_pitch'], [768.5740994184024,
                                                   707.13038356352104,
                                                   768.5740994184024,
                                                   707.13038356352104,
                                                   768.5740994184024,
                                                   707.13038356352104,
                                                   768.5740994184024], atol=1e-6, rtol=0)

    assert np.allclose(rk['dither_period_yaw'], [1086.9567572759399,
                                                 999.99938521341483,
                                                 1086.9567572759399,
                                                 999.99938521341483,
                                                 1086.9567572759399,
                                                 999.99938521341483,
                                                 1086.9567572759399], atol=1e-6, rtol=0)


def test_get_continuity_regress():
    """Regression test against values produced by get_continuity during development.
    Correctness not validated for all values.
    The particular time of 2018:001:12:00:00 happens during a maneuver, so this
    tests a bug fix where maneuver transitions were leaking past the stop time.
    It also tests that all continuity times are before the stop time.
    """
    expected = {'ccd_count': 3,
                'clocking': 1,
                'dec': 32.166641023063612,
                'dither': 'ENAB',
                'fep_count': 3,
                'hetg': 'RETR',
                'letg': 'RETR',
                'obsid': 20392,
                'off_nom_roll': -2.0300858116326026,
                'pcad_mode': 'NMAN',
                'pitch': 134.5392571808533,
                'power_cmd': 'XTZ0000005',
                'q1': -0.32430877626423488,
                'q2': -0.59794754520454407,
                'q3': -0.73138983148061287,
                'q4': 0.048491903554710794,
                'ra': 158.0145560201608,
                'roll': 84.946493470873875,
                'si_mode': 'TE_005C6',
                'simfa_pos': -468,
                'simpos': 75624,
                'targ_q1': 0.304190361,
                'targ_q2': 0.445053899,
                'targ_q3': 0.787398757,
                'targ_q4': 0.298995734,
                'vid_board': 1}

    dates = {'ccd_count': '2018:001:11:58:21.735',
             'clocking': '2018:001:11:59:28.735',
             'dec': '2018:001:11:57:47.798',
             'dither': '2017:364:11:51:48.955',
             'fep_count': '2018:001:11:58:21.735',
             'hetg': '2018:001:02:58:48.143',
             'letg': '2017:364:10:50:43.995',
             'obsid': '2018:001:11:55:05.818',
             'off_nom_roll': '2018:001:11:57:47.798',
             'pcad_mode': '2018:001:11:52:05.818',
             'pitch': '2018:001:11:57:47.798',
             'power_cmd': '2018:001:11:59:28.735',
             'q1': '2018:001:11:57:47.798',
             'q2': '2018:001:11:57:47.798',
             'q3': '2018:001:11:57:47.798',
             'q4': '2018:001:11:57:47.798',
             'ra': '2018:001:11:57:47.798',
             'roll': '2018:001:11:57:47.798',
             'si_mode': '2018:001:11:59:24.735',
             'simfa_pos': '2017:364:11:39:00.159',
             'simpos': '2018:001:02:55:13.804',
             'targ_q1': '2018:001:11:52:10.175',
             'targ_q2': '2018:001:11:52:10.175',
             'targ_q3': '2018:001:11:52:10.175',
             'targ_q4': '2018:001:11:52:10.175',
             'vid_board': '2018:001:11:58:21.735'}

    continuity = states.get_continuity('2018:001:12:00:00')

    for key, val in expected.items():
        if isinstance(val, (int, str)):
            assert continuity[key] == val
        else:
            assert np.isclose(continuity[key], val, rtol=0, atol=1e-7)
        assert continuity['__dates__'][key] == dates[key]
        assert continuity['__dates__'][key] < '2018:001:12:00:00.000'
        # Transitions with no spacecraft command (instead from injected maneuver state breaks)
        manvr_keys = ('pitch', 'off_nom_roll', 'ra', 'dec', 'roll', 'q1', 'q2', 'q3', 'q4')
        if key not in manvr_keys:
            cmds = commands.get_cmds(date=continuity['__dates__'][key])
            assert len(cmds) > 0


def test_get_continuity_vs_states():
    """
    Functional test: continuity for a certain date should be the same as the last states
    when fed in cmds up through that date.
    """
    date0 = '2017:014'
    # Get last state up through `date0`.  Hardwire the lookback here to 21 days.
    cmds = commands.get_cmds('2016:360', date0)
    sts = states.get_states(cmds=cmds)
    sts0 = sts[-1]

    continuity = states.get_continuity(date0)
    del continuity['__dates__']

    for key, val in continuity.items():
        if isinstance(val, (int, str)):
            assert sts0[key] == val
        else:
            assert np.isclose(sts0[key], val, rtol=0, atol=1e-7)


def test_get_continuity_keys():
    """Test that output has only the desired state keys. Also test that one can
    provide a string instead of list of state keys"""
    continuity = states.get_continuity('2017:014', 'clocking')
    assert set(continuity) == set(['clocking', '__dates__'])


def test_get_continuity_fail():
    with pytest.raises(ValueError) as err:
        states.get_continuity('2017:014', 'letg', lookbacks=[3])
    assert 'did not find transitions' in str(err)


def test_reduce_states_merge_identical():
    datestart = DateTime(np.arange(0, 5)).date
    datestop = DateTime(np.arange(1, 6)).date

    # Table with something that changes every time
    vals = np.arange(5)
    dat = Table([datestart, datestop, vals], names=['datestart', 'datestop', 'vals'])
    dat['val1'] = 1
    dr = states.reduce_states(dat, ['vals', 'val1'], merge_identical=True)
    assert np.all(dr[dat.colnames] == dat)

    # Table with nothing that changes
    vals = np.ones(5)
    dat = Table([datestart, datestop, vals], names=['datestart', 'datestop', 'vals'])
    dat['val1'] = 1
    dr = states.reduce_states(dat, ['vals', 'val1'], merge_identical=True)
    assert len(dr) == 1
    assert dr['datestart'][0] == dat['datestart'][0]
    assert dr['datestop'][0] == dat['datestop'][-1]

    # Table with edge changes
    vals = [1, 0, 0, 0, 1]
    dat = Table([datestart, datestop, vals], names=['datestart', 'datestop', 'vals'])
    dr = states.reduce_states(dat, ['vals'], merge_identical=True)
    assert len(dr) == 3
    assert np.all(dr['datestart'] == dat['datestart'][[0, 1, 4]])
    assert np.all(dr['datestop'] == dat['datestop'][[0, 3, 4]])
    assert np.all(dr['vals'] == [1, 0, 1])

    # Table with multiple changes
    val1 = [1, 0, 1, 1, 1]
    val2 = [1, 1, 1, 1, 0]
    dat = Table([datestart, datestop, val1, val2], names=['datestart', 'datestop', 'val1', 'val2'])
    dr = states.reduce_states(dat, ['val1', 'val2'], merge_identical=True)
    assert len(dr) == 4
    assert np.all(dr['datestart'] == dat['datestart'][[0, 1, 2, 4]])
    assert np.all(dr['datestop'] == dat['datestop'][[0, 1, 3, 4]])
    assert np.all(dr['val1'] == [1, 0, 1, 1])
    assert np.all(dr['val2'] == [1, 1, 1, 0])
    assert str(dr['trans_keys'][0]) == 'val1,val2'
    assert dr['trans_keys'][0] == set(['val1', 'val2'])
    assert dr['trans_keys'][1] == set(['val1'])
    assert dr['trans_keys'][2] == set(['val1'])
    assert dr['trans_keys'][3] == set(['val2'])


def test_reduce_states_cmd_states():
    """
    Test that simple get_states() call with defaults gives the same results
    as calling cmd_states.fetch_states().
    """
    cs = cmd_states.fetch_states('2018:235', '2018:245', allow_identical=True)
    cs = Table(cs)

    state_keys = (set(cmd_states.STATE0) -
                  set(['datestart', 'datestop', 'trans_keys', 'tstart', 'tstop']))

    # Default setting is reduce states with merge_identical=False, which is the same
    # as cmd_states.
    ksr = states.get_states('2018:235', '2018:245', state_keys)

    assert len(ksr) == len(cs)

    assert_all_close_states(cs, ksr, set(state_keys) - set(['trans_keys']))

    assert np.all(ksr['datestart'][1:] == cs['datestart'][1:])
    assert np.all(ksr['datestop'][:-1] == cs['datestop'][:-1])

    # Transition keys after first should match.  cmd_states is a comma-delimited string.
    for k_trans_keys, c_trans_keys in zip(ksr['trans_keys'][1:], cs['trans_keys'][1:]):
        assert k_trans_keys == set(c_trans_keys.split(','))


###########################################################################
# Comparing to backstop history files or regression outputs
###########################################################################

def compare_backstop_history(history, state_key, compare_val=True):
    hist = ascii.read(history, guess=False, format='no_header',
                      converters={'col1': [ascii.convert_numpy(np.str)]})
    start = DateTime(hist['col1'][0], format='greta') - 1 / 86400.
    stop = DateTime(hist['col1'][-1], format='greta') + 1 / 86400.
    sts = states.get_states(start=start, stop=stop, state_keys=state_key)
    sts = sts[1:]  # Drop the first state (which is continuity at start time)
    assert len(sts) == len(hist)
    assert np.all(DateTime(sts['datestart']).greta == hist['col1'])
    if compare_val:
        assert np.all(sts[state_key] == hist['col3'])


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
    compare_backstop_history(history, 'format')


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
    compare_backstop_history(history, 'subformat')


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
    compare_backstop_history(history, 'sun_pos_mon')


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
    cmds = commands.get_cmds('2017:087:03:04:02', '2017:088:00:00:00')
    sts = states.get_states(state_keys=['sun_pos_mon'], cmds=cmds)

    assert len(sts) == len(spm)
    assert np.all(sts['datestart'] == spm['datestart'])
    assert np.all(sts['sun_pos_mon'] == spm['sun_pos_mon'])


def test_backstop_ephem_update():
    history = """
2017107.051708675 | EPHEMERIS
2017109.204526240 | EPHEMERIS
2017112.121320933 | EPHEMERIS
2017115.034040060 | EPHEMERIS
2017117.190746809 | EPHEMERIS
"""
    compare_backstop_history(history, 'ephem_update', compare_val=False)


def test_backstop_radmon_no_scs107():
    """
    Test radmon states for nominal loads.  The current non-load commands from
    Chandra.cmd_states does not have the RADMON disable associated with SCS107
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
    compare_backstop_history(history, 'radmon')


@pytest.mark.xfail()
def test_backstop_radmon_with_scs107():
    """
    Test radmon states for nominal loads.  The current non-load commands from
    Chandra.cmd_states does not have the RADMON disable associated with SCS107
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
    compare_backstop_history(history, 'radmon')


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
    compare_backstop_history(history, 'scs98')


def test_backstop_scs84():
    """
    SCS 84 has not changed from DISA since 2006 due to a change in operations.
    """
    sts = states.get_states(start='2017:001', stop='2017:300', state_keys=['scs84'])
    assert len(sts) == 1
    assert sts[0]['scs84'] == 'DISA'
    assert sts[0]['datestart'] == '2017:001:12:00:00.000'
    assert sts[0]['datestop'] == '2017:300:12:00:00.000'


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
    compare_backstop_history(history, 'simpos')


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
    compare_backstop_history(history, 'simfa_pos')


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
    compare_backstop_history(history, 'grating')


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
    compare_backstop_history(history, 'eclipse_timer')


def compare_regress_output(regress, state_key):
    regr = ascii.read(regress, format='fixed_width_two_line', fill_values=None)
    start = regr['datestart'][0]
    stop = regr['datestop'][-1]
    sts = states.get_states(start, stop, state_keys=[state_key])
    for colname in ('datestart', 'datestop', state_key):
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
    compare_regress_output(regress, 'eclipse')

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
    compare_regress_output(regress, 'eclipse')


def test_continuity_just_after_command():
    """
    Fix issue where continuity required having at least one other
    command after the relevant continuity command.
    """
    cont = states.get_continuity('2018:001:12:00:00', 'targ_q1')
    assert cont['__dates__']['targ_q1'] == '2018:001:11:52:10.175'

    # 1 msec later
    cont = states.get_continuity('2018:001:11:52:10.176', 'targ_q1')
    assert cont['__dates__']['targ_q1'] == '2018:001:11:52:10.175'


def test_continuity_far_future():
    """
    Test that using a continuity date in the future (well past commamds)
    works.
    """
    # Now + 150 days.  Don't know the answer but just make sure this
    # runs to completion.  The lookbacks of 7 and 30 days should fail
    # but 180 days will get something.
    cont = states.get_continuity(DateTime() + 150, 'obsid')
    assert 'obsid' in cont
    assert 'obsid' in cont['__dates__']


def test_acis_power_cmds():
    start = "2017:359:13:37:50"
    stop = "2017:360:00:46:00"
    state_keys = ["power_cmd", "ccd_count", "fep_count", "vid_board"]
    cmds = commands.get_cmds(start, stop)
    continuity = states.get_continuity(start, state_keys)
    test_states = states.get_states(cmds=cmds, continuity=continuity,
                                    state_keys=state_keys, reduce=False)
    vid_dn = np.where(test_states["power_cmd"] == "WSVIDALLDN")[0]
    assert (test_states["ccd_count"][vid_dn] == 0).all()
    assert (test_states["fep_count"][vid_dn] == test_states["fep_count"][vid_dn[0] - 1]).all()
    pow_zero = np.where(test_states["power_cmd"] == "WSPOW00000")[0]
    assert (test_states["ccd_count"][pow_zero] == 0).all()
    assert (test_states["fep_count"][pow_zero] == 0).all()


def test_continuity_with_transitions_SPM():
    """
    Test that continuity returns a dict that has the __transitions__ key set
    to a list of transitions that are needed for correct continuity.  Part of
    fix for #125.

    Start time is ne minute before auto-re-enable of SPM during eclipse handling
    This is in the middle of the transitions generated by SPMEnableTransition.
    The test here is seeing the ENAB transition that is one minute after the
    continuity start time.  This will be used by get_states() to get things
    right using this continuity dict.
    """
    start = '2017:087:08:20:35.838'
    stop = '2017:087:10:20:35.838'
    cont = states.get_continuity(start, state_keys=['sun_pos_mon'])
    assert cont == {'__dates__': {'sun_pos_mon': '2017:087:07:44:55.838'},
                    '__transitions__': [{'date': '2017:087:08:21:35.838', 'sun_pos_mon': 'ENAB'}],
                    'sun_pos_mon': 'DISA'}

    exp = ['      datestart              datestop       sun_pos_mon  trans_keys',
           '--------------------- --------------------- ----------- -----------',
           '2017:087:08:20:35.838 2017:087:08:21:35.838        DISA            ',
           '2017:087:08:21:35.838 2017:087:08:30:50.891        ENAB sun_pos_mon',
           '2017:087:08:30:50.891 2017:087:10:20:35.838        DISA sun_pos_mon']
    sts = states.get_states(start, stop, state_keys=['sun_pos_mon'])
    assert sts.pformat(max_lines=-1, max_width=-1) == exp


def test_continuity_with_no_transitions_SPM():
    """Test that continuity returns a dict that does NOT have the __transitions__
    key set if not needed.  Part of fix for #125.

    """
    cont = states.get_continuity('2017:001', state_keys=['sun_pos_mon'])
    assert cont == {'sun_pos_mon': 'DISA',
                    '__dates__': {'sun_pos_mon': '2017:001:04:23:55.764'}}


def test_get_pitch_from_mid_maneuver():
    """Regression test for the fix for #125.  Mostly the same as the test above, but for
    the Maneuver transition class.

    """
    start = '2019:039:14:16:25.002'  # Mid-maneuver
    stop = '2019:039:16:00:00.000'
    exp = ['      datestart              datestop           pitch     pcad_mode',
           '--------------------- --------------------- ------------- ---------',
           '2019:039:14:16:25.002 2019:039:14:16:54.364 63.2696565838      NMAN',
           '2019:039:14:16:54.364 2019:039:14:22:01.825 83.0388345752      NMAN',
           '2019:039:14:22:01.825 2019:039:14:27:09.285 106.057258631      NMAN',
           '2019:039:14:27:09.285 2019:039:14:32:16.745 129.079427541      NMAN',
           '2019:039:14:32:16.745 2019:039:14:37:24.205 148.857427163      NMAN',
           '2019:039:14:37:24.205 2019:039:14:42:31.665 159.541502291      NMAN',
           '2019:039:14:42:31.665 2019:039:16:00:00.000 161.950922135      NPNT']

    sts = states.get_states(start, stop, state_keys=['pitch', 'pcad_mode'])
    del sts['trans_keys']

    assert sts.pformat(max_width=-1, max_lines=-1) == exp
