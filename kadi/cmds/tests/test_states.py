import numpy as np

from ... import cmds as commands
from ...cmds import states

import Chandra.cmd_states as cmd_states
from Chandra.Time import DateTime


def compare_states(start, stop, state_keys, compare_state_keys=None):
    start = DateTime(start)
    stop = DateTime(stop)

    if compare_state_keys is None:
        compare_state_keys = state_keys

    cstates = cmd_states.fetch_states(start, stop)
    rcstates = states.reduce_states(cstates, state_keys)
    lenr = len(rcstates)

    cmds = commands.filter(start - 7, stop)
    kstates = states.get_states_for_cmds(cmds, state_keys)
    rkstates = states.reduce_states(kstates, state_keys)[-lenr:]

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
    """
    state_keys = (['obsid', 'clocking', 'power_cmd',  'fep_count',
                   'si_mode',  'ccd_count'] +
                  ['q1', 'q2', 'q3', 'q4', 'pcad_mode', 'pitch'])
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
