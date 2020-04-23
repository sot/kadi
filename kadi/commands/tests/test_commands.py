from pathlib import Path

import numpy as np
from astropy.table import Table

# Use data file from parse_cm.test for get_cmds_from_backstop test.
# This package is a dependency
import parse_cm.tests

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
    assert repr(cmd).endswith('scs=133 step=161 timeline_id=426098449 vcdu=15639968 pos=73176>')
    assert str(cmd).startswith('2012:030:08:27:02.000 SIMTRANS')
    assert str(cmd).endswith('scs=133 step=161 timeline_id=426098449 vcdu=15639968 pos=73176')

    assert cmd['pos'] == 73176
    assert cmd['step'] == 161


def test_get_cmds_zero_length_result():
    cmds = commands.get_cmds(date='2017:001')
    assert len(cmds) == 0
    assert cmds.colnames == ['idx', 'date', 'type', 'tlmsid', 'scs',
                             'step', 'timeline_id', 'vcdu', 'params']


def test_get_cmds_inclusive_stop():
    """get_cmds returns start <= date < stop for inclusive_stop=False (default)
    or start <= date <= stop for inclusive_stop=True.
    """
    # Query over a range that includes two commands at exactly start and stop.
    start, stop = '2020:001:15:50:00.000', '2020:001:15:50:00.257'
    cmds = commands.get_cmds(start, stop)
    assert np.all(cmds['date'] == [start])

    cmds = commands.get_cmds(start, stop, inclusive_stop=True)
    assert np.all(cmds['date'] == [start, stop])


def test_get_cmds_from_backstop_and_add_cmds():
    bs_file = Path(parse_cm.tests.__file__).parent / 'data' / 'CR182_0803.backstop'
    bs_cmds = commands.get_cmds_from_backstop(bs_file)

    cmds = commands.get_cmds(start='2018:182:00:00:00',
                             stop='2018:182:08:00:00')

    assert len(bs_cmds) == 674
    assert len(cmds) == 56

    assert bs_cmds.colnames == cmds.colnames
    for bs_col, col in zip(bs_cmds.itercols(), cmds.itercols()):
        assert bs_col.dtype == col.dtype

    new_cmds = cmds.add_cmds(bs_cmds)
    assert len(new_cmds) == len(cmds) + len(bs_cmds)

    # No MP_STARCAT command parameters by default
    ok = bs_cmds['type'] == 'MP_STARCAT'
    assert np.count_nonzero(ok) == 15
    assert np.all(bs_cmds['params'][ok] == {})

    # Accept MP_STARCAT commands
    bs_cmds = commands.get_cmds_from_backstop(bs_file, remove_starcat=False)
    ok = bs_cmds['type'] == 'MP_STARCAT'
    assert np.count_nonzero(ok) == 15
    assert np.all(bs_cmds['params'][ok] != {})
