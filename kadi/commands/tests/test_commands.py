from pathlib import Path
import os

import numpy as np
from astropy.table import Table
import pytest

# Use data file from parse_cm.test for get_cmds_from_backstop test.
# This package is a dependency
import parse_cm.tests
from Chandra.Time import secs2date
from cxotime import CxoTime

from kadi import commands
from kadi.commands import core, commands_v1, commands_v2
from kadi.scripts import update_cmds_v1


HAS_MPDIR = Path(os.environ['SKA'], 'data', 'mpcrit1', 'mplogs', '2020').exists()


@pytest.fixture(scope="module", params=["1", "2"])
def version(request):
    return request.param


@pytest.fixture
def version_env(monkeypatch, version):
    if version is None:
        monkeypatch.delenv('KADI_COMMANDS_VERSION', raising=False)
    else:
        monkeypatch.setenv('KADI_COMMANDS_VERSION', version)
    return version


@pytest.mark.parametrize('version', [1, 2])
def test_find(version):
    if version == 1:
        idx_cmds = commands_v1.IDX_CMDS
        pars_dict = commands_v1.PARS_DICT
    else:
        idx_cmds = commands_v2.IDX_CMDS
        pars_dict = commands_v2.PARS_DICT

    cs = core._find('2012:029:12:00:00', '2012:030:12:00:00',
                    idx_cmds=idx_cmds, pars_dict=pars_dict)
    assert isinstance(cs, Table)
    assert len(cs) == 147
    if version == 1:
        assert np.all(cs['timeline_id'][:10] == 426098447)
        assert np.all(cs['timeline_id'][-10:] == 426098448)
    else:
        assert np.all(cs['source'][:10] == 'JAN2612A')
        assert np.all(cs['source'][-10:] == 'JAN3012C')
    assert cs['date'][0] == '2012:029:13:00:00.000'
    assert cs['date'][-1] == '2012:030:11:00:01.285'
    assert cs['tlmsid'][-1] == 'CTXBON'

    cs = core._find('2012:029:12:00:00', '2012:030:12:00:00', type='simtrans',
                    idx_cmds=idx_cmds, pars_dict=pars_dict)
    assert len(cs) == 2
    assert np.all(cs['date'] == ['2012:030:02:00:00.000', '2012:030:08:27:02.000'])

    cs = core._find('2012:015:12:00:00', '2012:030:12:00:00',
                    idx_cmds=idx_cmds, pars_dict=pars_dict,
                    type='acispkt', tlmsid='wsvidalldn',)
    assert len(cs) == 3
    assert np.all(cs['date'] == ['2012:018:01:16:15.798', '2012:020:16:51:17.713',
                                 '2012:026:05:28:09.000'])

    cs = core._find('2011:001:12:00:00', '2014:001:12:00:00', msid='aflcrset',
                    idx_cmds=idx_cmds, pars_dict=pars_dict)
    assert len(cs) == 2494


def test_get_cmds(version_env):
    cs = commands.get_cmds('2012:029:12:00:00', '2012:030:12:00:00')
    assert isinstance(cs, commands.CommandTable)
    assert len(cs) == 147
    if version_env == '2':
        assert np.all(cs['source'][:10] == 'JAN2612A')
        assert np.all(cs['source'][-10:] == 'JAN3012C')
    else:
        assert np.all(cs['timeline_id'][:10] == 426098447)
        assert np.all(cs['timeline_id'][-10:] == 426098448)
    assert cs['date'][0] == '2012:029:13:00:00.000'
    assert cs['date'][-1] == '2012:030:11:00:01.285'
    assert cs['tlmsid'][-1] == 'CTXBON'

    cs = commands.get_cmds('2012:029:12:00:00', '2012:030:12:00:00', type='simtrans')
    assert len(cs) == 2
    assert np.all(cs['date'] == ['2012:030:02:00:00.000', '2012:030:08:27:02.000'])
    assert np.all(cs['pos'] == [75624, 73176])  # from params

    cmd = cs[1]

    assert repr(cmd).startswith('<Cmd 2012:030:08:27:02.000 SIMTRANS')
    assert str(cmd).startswith('2012:030:08:27:02.000 SIMTRANS')
    if version_env == '2':
        assert repr(cmd).endswith('scs=133 step=161 source=JAN3012C vcdu=15639968 pos=73176>')
        assert str(cmd).endswith('scs=133 step=161 source=JAN3012C vcdu=15639968 pos=73176')
    else:
        assert repr(cmd).endswith('scs=133 step=161 timeline_id=426098449 vcdu=15639968 pos=73176>')
        assert str(cmd).endswith('scs=133 step=161 timeline_id=426098449 vcdu=15639968 pos=73176')

    assert cmd['pos'] == 73176
    assert cmd['step'] == 161


def test_get_cmds_zero_length_result(version_env):
    cmds = commands.get_cmds(date='2017:001:12:00:00')
    assert len(cmds) == 0
    source_name = 'source' if version_env == '2' else 'timeline_id'
    assert cmds.colnames == ['idx', 'date', 'type', 'tlmsid', 'scs',
                             'step', 'time', source_name, 'vcdu', 'params']


def test_get_cmds_inclusive_stop(version_env):
    """get_cmds returns start <= date < stop for inclusive_stop=False (default)
    or start <= date <= stop for inclusive_stop=True.
    """
    # Query over a range that includes two commands at exactly start and stop.
    start, stop = '2020:001:15:50:00.000', '2020:001:15:50:00.257'
    cmds = commands.get_cmds(start, stop)
    assert np.all(cmds['date'] == [start])

    cmds = commands.get_cmds(start, stop, inclusive_stop=True)
    assert np.all(cmds['date'] == [start, stop])


def test_cmds_as_list_of_dict(version_env):
    cmds = commands.get_cmds('2020:140', '2020:141')
    cmds_list = cmds.as_list_of_dict()
    assert isinstance(cmds_list, list)
    assert isinstance(cmds_list[0], dict)
    cmds_rt = commands.CommandTable(cmds)
    assert set(cmds_rt.colnames) == set(cmds.colnames)
    for name in cmds.colnames:
        assert np.all(cmds_rt[name] == cmds[name])


def test_cmds_as_list_of_dict_ska_parsecm():
    """Test the ska_parsecm=True compatibility mode for list_of_dict"""
    cmds = commands.get_cmds('2020:140', '2020:141')
    cmds_list = cmds.as_list_of_dict(ska_parsecm=True)
    assert isinstance(cmds_list, list)
    assert isinstance(cmds_list[0], dict)
    assert cmds_list[0] == {
        'cmd': 'COMMAND_HW',  # Cmd parameter exists and matches type
        'date': '2020:140:00:00:00.000',
        'idx': 21387,
        'params': {'HEX': '7C063C0', 'MSID': 'CIU1024T'},  # Keys are upper case
        'scs': 129,
        'step': 496,
        'time': 706233669.184,
        'timeline_id': 426104285,
        'tlmsid': 'CIMODESL',
        'type': 'COMMAND_HW',
        'vcdu': 12516929}
    for cmd in cmds_list:
        assert cmd.get('cmd') == cmd.get('type')
        assert all(param.upper() == param for param in cmd['params'])


def test_get_cmds_from_backstop_and_add_cmds():
    bs_file = Path(parse_cm.tests.__file__).parent / 'data' / 'CR182_0803.backstop'
    bs_cmds = commands.get_cmds_from_backstop(bs_file, remove_starcat=True)

    cmds = commands.get_cmds(start='2018:182:00:00:00', stop='2018:182:08:00:00')

    assert len(bs_cmds) == 674
    assert len(cmds) == 56

    assert bs_cmds.colnames == cmds.colnames
    for bs_col, col in zip(bs_cmds.itercols(), cmds.itercols()):
        assert bs_col.dtype == col.dtype

    assert np.all(secs2date(cmds['time']) == cmds['date'])
    assert np.all(secs2date(bs_cmds['time']) == bs_cmds['date'])

    new_cmds = cmds.add_cmds(bs_cmds)
    assert len(new_cmds) == len(cmds) + len(bs_cmds)

    # No MP_STARCAT command parameters by default
    ok = bs_cmds['type'] == 'MP_STARCAT'
    assert np.count_nonzero(ok) == 15
    assert np.all(bs_cmds['params'][ok] == {})

    # Accept MP_STARCAT commands (also check read_backstop command)
    bs_cmds = commands.read_backstop(bs_file)
    ok = bs_cmds['type'] == 'MP_STARCAT'
    assert np.count_nonzero(ok) == 15
    assert np.all(bs_cmds['params'][ok] != {})


@pytest.mark.skipif('not HAS_MPDIR')
def test_commands_v1_create_archive_regress(tmpdir):
    """Create cmds archive from scratch and test that it matches flight"""
    kadi_orig = os.environ.get('KADI')
    start = CxoTime('2020:159:00:00:00')
    stop = start + 30
    cmds_flight = commands_v1.get_cmds(start + 3, stop - 3)
    cmds_flight.fetch_params()

    try:
        os.environ['KADI'] = str(tmpdir)
        update_cmds_v1.main(
            (f'--start={start.date}',
             f'--stop={stop.date}',
             f'--data-root={tmpdir}'))
        # Force reload of LazyVal
        del commands_v1.IDX_CMDS._val
        del commands_v1.PARS_DICT._val
        del commands_v1.REV_PARS_DICT._val

        # Make sure we are seeing the temporary cmds archive
        cmds_empty = commands_v1.get_cmds(start - 60, start - 50)
        assert len(cmds_empty) == 0

        cmds_local = commands_v1.get_cmds(start + 3, stop - 3)
        cmds_local.fetch_params()
        assert len(cmds_flight) == len(cmds_local)
        for attr in ('tlmsid', 'date', 'params'):
            assert np.all(cmds_flight[attr] == cmds_local[attr])

    finally:
        if kadi_orig is None:
            del os.environ['KADI']
        else:
            os.environ['KADI'] = kadi_orig

        # Force reload
        del commands_v1.IDX_CMDS._val
        del commands_v1.PARS_DICT._val
        del commands_v1.REV_PARS_DICT._val
