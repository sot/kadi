import numpy as np

from ... import cmds as commands
from ...cmds import states

import Chandra.cmd_states as cmd_states


def test_states_acis_simple():
    state_keys = ['obsid', 'clocking', 'power_cmd',  'vid_board', 'fep_count',
                  'si_mode',  'ccd_count']
    cmds = commands.filter('2014:028', '2014:030:11:00:00')
    kstates = states.get_states_for_cmds(cmds, state_keys)

    cstates = cmd_states.fetch_states('2014:030:02:50:00', '2014:030:11:00:00')
    rstates = cmd_states.reduce_states(cstates, state_keys, allow_identical=False)

    lenr = len(rstates)
    for colname in ['datestart'] + state_keys:
        assert np.all(kstates[-lenr:][colname] == rstates[colname])


def test_states_manvr():
    state_keys = ['q1', 'q2', 'q3', 'q4', 'pcad_mode', 'obsid']
    cmds = commands.filter('2014:028', '2014:030:11:00:00')
    kstates = states.get_states_for_cmds(cmds, state_keys)
    rkstates = states.reduce_states(kstates, state_keys)

    cstates = cmd_states.fetch_states('2014:030:07:15:00', '2014:030:11:00:00')
    rcstates = states.reduce_states(cstates, state_keys)

    lenr = len(rcstates)
    for colname in ['datestart'] + state_keys:
        assert np.all(rkstates[-lenr:][colname] == rcstates[colname])
