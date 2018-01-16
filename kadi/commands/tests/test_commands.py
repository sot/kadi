import numpy as np
from astropy.table import Table

# Import cmds module directly (not kadi.cmds package, which is from ... import cmds)
from .. import commands


def test_find():
    cs = commands._find('2012:029', '2012:030')
    assert isinstance(cs, Table)
    assert len(cs) == 147
    assert np.all(cs['timeline_id'][:10] == 426098447)
    assert np.all(cs['timeline_id'][-10:] == 426098448)
    assert cs['date'][0] == '2012:029:13:00:00.000'
    assert cs['date'][-1] == '2012:030:11:00:01.285'
    assert cs['tlmsid'][-1] == 'CTXBON'

    cs = commands._find('2012:029', '2012:030', type='simtrans')
    assert len(cs) == 2
    assert np.all(cs['date'] == ['2012:030:02:00:00.000', '2012:030:08:27:02.000'])

    cs = commands._find('2012:015', '2012:030', type='acispkt', tlmsid='wsvidalldn')
    assert len(cs) == 3
    assert np.all(cs['date'] == ['2012:018:01:16:15.798', '2012:020:16:51:17.713',
                                 '2012:026:05:28:09.000'])

    cs = commands._find('2011:001', '2014:001', msid='aflcrset')
    assert len(cs) == 2494


def test_get_cmds():
    cs = commands.get_cmds('2012:029', '2012:030')
    assert isinstance(cs, commands.CommandTable)
    assert len(cs) == 147
    assert np.all(cs['timeline_id'][:10] == 426098447)
    assert np.all(cs['timeline_id'][-10:] == 426098448)
    assert cs['date'][0] == '2012:029:13:00:00.000'
    assert cs['date'][-1] == '2012:030:11:00:01.285'
    assert cs['tlmsid'][-1] == 'CTXBON'

    cs = commands.get_cmds('2012:029', '2012:030', type='simtrans')
    assert len(cs) == 2
    assert np.all(cs['date'] == ['2012:030:02:00:00.000', '2012:030:08:27:02.000'])
    assert np.all(cs['pos'] == [75624, 73176])  # from params

    cmd = cs[1]

    assert repr(cmd).startswith('<Cmd 2012:030:08:27:02.000 SIMTRANS')
    assert repr(cmd).endswith('scs=133 step=161 timeline_id=426098449 pos=73176>')
    assert str(cmd).startswith('2012:030:08:27:02.000 SIMTRANS')
    assert str(cmd).endswith('scs=133 step=161 timeline_id=426098449 pos=73176')

    assert cmd['pos'] == 73176
    assert cmd['step'] == 161
