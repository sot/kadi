from kadi import commands
from kadi.commands import states
import numpy as np

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
    assert (test_states["fep_count"][vid_dn] == test_states["fep_count"][vid_dn[0]-1]).all()
    pow_zero = np.where(test_states["power_cmd"] == "WSPOW00000")[0]
    assert (test_states["ccd_count"][pow_zero] == 0).all()
    assert (test_states["fep_count"][pow_zero] == 0).all()
