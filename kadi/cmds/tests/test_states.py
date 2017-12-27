import numpy as np

from ... import cmds as commands
import Chandra.cmd_states as cmd_states


def test_states_acis_simple():
    colnames = ['obsid', 'clocking', 'power_cmd',  'vid_board', 'fep_count',
                'si_mode',  'ccd_count']
    cmds = commands.filter('2014:028', '2014:030:11:00:00')
    kstates = commands.states.get_states_for_cmds(cmds, ['obsid', 'acis'])

    cstates = cmd_states.fetch_states('2014:030:02:50:00', '2014:030:11:00:00')
    rstates = cmd_states.reduce_states(cstates, colnames, allow_identical=False)

    lenr = len(rstates)
    for colname in ['datestart'] + colnames:
        assert np.all(kstates[-lenr:][colname] == rstates[colname])
