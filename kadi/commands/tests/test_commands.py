from pathlib import Path
import os

import numpy as np
from astropy.table import Table
import astropy.units as u
import pytest

# Use data file from parse_cm.test for get_cmds_from_backstop test.
# This package is a dependency
import parse_cm.tests
from testr.test_helper import has_internet
from Chandra.Time import secs2date
from cxotime import CxoTime

from kadi import commands
from kadi.commands import (core, commands_v1, commands_v2, conf, get_starcats,
                           get_observations)
from kadi.scripts import update_cmds_v1, update_cmds_v2
from kadi.commands.command_sets import get_cmds_from_event

HAS_MPDIR = Path(os.environ['SKA'], 'data', 'mpcrit1', 'mplogs', '2020').exists()
HAS_INTERNET = has_internet()
VERSIONS = ['1', '2'] if HAS_INTERNET else ['1']


@pytest.fixture(scope="module", params=VERSIONS)
def version(request):
    return request.param


@pytest.fixture
def version_env(monkeypatch, version):
    if version is None:
        monkeypatch.delenv('KADI_COMMANDS_VERSION', raising=False)
    else:
        monkeypatch.setenv('KADI_COMMANDS_VERSION', version)
    return version


@pytest.fixture(scope="module", autouse=True)
def cmds_dir(tmp_path_factory):
    with commands_v2.conf.set_temp('cache_loads_in_astropy_cache', True):
        with commands_v2.conf.set_temp('clean_loads_dir', False):
            cmds_dir = tmp_path_factory.mktemp('cmds_dir')
            with commands_v2.conf.set_temp('commands_dir', str(cmds_dir)):
                yield


def test_find(version):
    if version == '1':
        idx_cmds = commands_v1.IDX_CMDS
        pars_dict = commands_v1.PARS_DICT
    else:
        idx_cmds = commands_v2.IDX_CMDS
        pars_dict = commands_v2.PARS_DICT

    cs = core._find('2012:029:12:00:00', '2012:030:12:00:00',
                    idx_cmds=idx_cmds, pars_dict=pars_dict)
    assert isinstance(cs, Table)
    assert len(cs) == 147 if version == '1' else 151  # OBS commands in v2 only
    if version == '1':
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
    assert len(cs) == 147 if version_env == '1' else 151  # OBS commands in v2 only
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
def test_commands_create_archive_regress(tmpdir, version_env):
    """Create cmds archive from scratch and test that it matches flight

    This tests over an eventful month that includes IU reset/NSM, SCS-107
    (radiation), fast replan, loads approved but not uplinked, etc.
    """
    update_cmds = update_cmds_v2 if version_env == "2" else update_cmds_v1
    commands = commands_v2 if version_env == "2" else commands_v1

    kadi_orig = os.environ.get('KADI')
    start = CxoTime('2021:290')
    stop = start + 30
    cmds_flight = commands.get_cmds(start + 3, stop - 3)
    cmds_flight.fetch_params()

    with conf.set_temp('commands_dir', str(tmpdir)):
        try:
            os.environ['KADI'] = str(tmpdir)
            update_cmds.main(
                ('--lookback=30' if version_env == "2" else f'--start={start.date}',
                 f'--stop={stop.date}',
                 f'--data-root={tmpdir}'))
            # Force reload of LazyVal
            del commands.IDX_CMDS._val
            del commands.PARS_DICT._val
            del commands.REV_PARS_DICT._val

            # Make sure we are seeing the temporary cmds archive
            cmds_empty = commands.get_cmds(start - 60, start - 50)
            assert len(cmds_empty) == 0

            cmds_local = commands.get_cmds(start + 3, stop - 3)
            cmds_local.fetch_params()
            assert len(cmds_flight) == len(cmds_local)

            # 'starcat_idx' param in OBS cmd does not match since the pickle files
            # are different, so remove it.
            for cmds in (cmds_local, cmds_flight):
                for cmd in cmds:
                    if cmd['tlmsid'] == 'OBS' and 'starcat_idx' in cmd['params']:
                        del cmd['params']['starcat_idx']

            for attr in ('tlmsid', 'date', 'params'):
                assert np.all(cmds_flight[attr] == cmds_local[attr])

        finally:
            if kadi_orig is None:
                del os.environ['KADI']
            else:
                os.environ['KADI'] = kadi_orig

            # Force reload
            del commands.IDX_CMDS._val
            del commands.PARS_DICT._val
            del commands.REV_PARS_DICT._val


def stop_date_fixture_factory(stop_date):
    @pytest.fixture()
    def stop_date_fixture(monkeypatch):
        commands_v2.clear_caches()
        monkeypatch.setenv('KADI_COMMANDS_DEFAULT_STOP', stop_date)
        cmds_dir = Path(conf.commands_dir) / stop_date
        with commands_v2.conf.set_temp('commands_dir', str(cmds_dir)):
            yield
    return stop_date_fixture


stop_date_2021_10_24 = stop_date_fixture_factory('2021-10-24')
stop_date_2020_12_03 = stop_date_fixture_factory('2020-12-03')


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_cmds_v2_arch_only(stop_date_2020_12_03):
    cmds = commands_v2.get_cmds(start='2020-01-01', stop='2020-01-02')
    cmds = cmds[cmds['tlmsid'] != 'OBS']
    assert len(cmds) == 153
    assert np.all(cmds['idx'] != -1)
    # Also do a zero-length query
    cmds = commands_v2.get_cmds(start='2020-01-01', stop='2020-01-01')
    assert len(cmds) == 0


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_cmds_v2_arch_recent(stop_date_2020_12_03):
    cmds = commands_v2.get_cmds(start='2020-09-01', stop='2020-12-01')
    cmds = cmds[cmds['tlmsid'] != 'OBS']

    # Since recent matches arch in the past, even though the results are a mix
    # of arch and recent, they commands actually come from the arch because of
    # how the matching block is used (commands come from arch up through the end
    # of the matching block).
    assert np.all(cmds['idx'] != -1)
    assert len(cmds) == 17640

    loads = commands_v2.get_loads()
    assert loads.pformat_all() == [
        '  name         cmd_start              cmd_stop       observing_stop vehicle_stop          rltt          scheduled_stop_time ',  # noqa
        '-------- --------------------- --------------------- -------------- ------------ --------------------- ---------------------',  # noqa
        'NOV0920A 2020:314:12:13:00.000 2020:321:00:48:01.673             --           -- 2020:314:12:16:00.000 2020:321:00:48:01.673',  # noqa
        'NOV1620A 2020:321:00:45:01.673 2020:327:19:26:00.000             --           -- 2020:321:00:48:01.673 2020:327:19:26:00.000',  # noqa
        'NOV2320A 2020:327:19:23:00.000 2020:334:20:44:27.758             --           -- 2020:327:19:26:00.000 2020:334:20:44:27.758',  # noqa
        'NOV3020A 2020:334:20:41:27.758 2020:342:06:04:34.287             --           -- 2020:334:20:44:27.758 2020:342:06:04:34.287'  # noqa
    ]


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_cmds_v2_recent_only(stop_date_2020_12_03):
    # This query stop is well beyond the default stop date, so it should get
    # only commands out to the end of the NOV3020A loads (~ Dec 7).
    cmds = commands_v2.get_cmds(start='2020-12-01', stop='2021-01-01')
    cmds = cmds[cmds['tlmsid'] != 'OBS']
    assert len(cmds) == 1523
    assert np.all(cmds['idx'] == -1)
    assert cmds[:5].pformat_like_backstop() == [
        '2020:336:00:08:38.610 | COMMAND_HW       | CNOOP      | NOV3020A | hex=7E00000, msid=CNOOPLR, scs=128',  # noqa
        '2020:336:00:08:39.635 | COMMAND_HW       | CNOOP      | NOV3020A | hex=7E00000, msid=CNOOPLR, scs=128',  # noqa
        '2020:336:00:12:55.214 | ACISPKT          | AA00000000 | NOV3020A | cmds=3, words=3, scs=131',  # noqa
        '2020:336:00:12:55.214 | ORBPOINT         | None       | NOV3020A | event_type=XEF1000, scs=0',  # noqa
        '2020:336:00:12:59.214 | ACISPKT          | AA00000000 | NOV3020A | cmds=3, words=3, scs=131'  # noqa
    ]
    assert cmds[-5:].pformat_like_backstop() == [
        '2020:342:03:15:02.313 | COMMAND_SW       | OFMTSNRM   | NOV3020A | hex=8010A00, msid=OFMTSNRM, scs=130',  # noqa
        '2020:342:03:15:02.313 | COMMAND_SW       | COSCSEND   | NOV3020A | hex=C800000, msid=OBC_END_SCS, scs=130',  # noqa
        '2020:342:06:04:34.287 | ACISPKT          | AA00000000 | NOV3020A | cmds=3, words=3, scs=133',  # noqa
        '2020:342:06:04:34.287 | COMMAND_SW       | COSCSEND   | NOV3020A | hex=C800000, msid=OBC_END_SCS, scs=133',  # noqa
        '2020:342:06:04:34.287 | LOAD_EVENT       | None       | NOV3020A | event_type=SCHEDULED_STOP_TIME, scs=0'  # noqa
    ]

    # Same for no stop date
    cmds = commands_v2.get_cmds(start='2020-12-01', stop=None)
    cmds = cmds[cmds['tlmsid'] != 'OBS']
    assert len(cmds) == 1523
    assert np.all(cmds['idx'] == -1)

    # Sanity check on the loads
    loads = commands_v2.get_loads()
    assert np.all(loads['name'] == ['NOV0920A', 'NOV1620A', 'NOV2320A', 'NOV3020A'])

    # zero-length query
    cmds = commands_v2.get_cmds(start='2020-12-01', stop='2020-12-01')
    assert len(cmds) == 0


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_get_cmds_nsm_2021(stop_date_2021_10_24):
    """NSM at ~2021:296:10:41. This tests non-load commands from cmd_events.
    """
    cmds = commands_v2.get_cmds('2021:296:10:35:00')  # , '2021:298:01:58:00')
    cmds = cmds[cmds['tlmsid'] != 'OBS']
    exp = ['2021:296:10:35:00.000 | COMMAND_HW       | CIMODESL   | OCT1821A | '
           'hex=7C067C0, msid=CIU1024X, scs=128',
           '2021:296:10:35:00.257 | COMMAND_HW       | CTXAOF     | OCT1821A | '
           'hex=780000C, msid=CTXAOF, scs=128',
           '2021:296:10:35:00.514 | COMMAND_HW       | CPAAOF     | OCT1821A | '
           'hex=780001E, msid=CPAAOF, scs=128',
           '2021:296:10:35:00.771 | COMMAND_HW       | CTXBOF     | OCT1821A | '
           'hex=780004C, msid=CTXBOF, scs=128',
           '2021:296:10:35:01.028 | COMMAND_HW       | CPABON     | OCT1821A | '
           'hex=7800056, msid=CPABON, scs=128',
           '2021:296:10:35:01.285 | COMMAND_HW       | CTXBON     | OCT1821A | '
           'hex=7800044, msid=CTXBON, scs=128',
           '2021:296:10:41:57.000 | COMMAND_SW       | AONSMSAF   | CMD_EVT  | '
           'event=NSM, event_date=2021:296:10:41:57, scs=0',
           '2021:296:10:41:57.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | '
           'event=NSM, event_date=2021:296:10:41:57, scs=0',
           '2021:296:10:41:58.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | '
           'event=NSM, event_date=2021:296:10:41:57, msid=AFLCRSET, scs=0',
           '2021:296:10:41:58.025 | SIMTRANS         | None       | CMD_EVT  | '
           'event=NSM, event_date=2021:296:10:41:57, pos=-99616, scs=0',
           '2021:296:10:42:20.000 | MP_OBSID         | COAOSQID   | CMD_EVT  | '
           'event=Obsid, event_date=2021:296:10:42:20, id=0, scs=0',
           '2021:296:10:43:03.685 | ACISPKT          | AA00000000 | CMD_EVT  | '
           'event=NSM, event_date=2021:296:10:41:57, scs=0',
           '2021:296:10:43:04.710 | ACISPKT          | AA00000000 | CMD_EVT  | '
           'event=NSM, event_date=2021:296:10:41:57, scs=0',
           '2021:296:10:43:14.960 | ACISPKT          | WSPOW0002A | CMD_EVT  | '
           'event=NSM, event_date=2021:296:10:41:57, scs=0',
           '2021:296:10:43:14.960 | COMMAND_SW       | AODSDITH   | CMD_EVT  | '
           'event=NSM, event_date=2021:296:10:41:57, scs=0',
           '2021:297:01:41:01.000 | COMMAND_SW       | AONMMODE   | CMD_EVT  | '
           'event=Maneuver, event_date=2021:297:01:41:01, msid=AONMMODE, scs=0',
           '2021:297:01:41:01.256 | COMMAND_SW       | AONM2NPE   | CMD_EVT  | '
           'event=Maneuver, event_date=2021:297:01:41:01, msid=AONM2NPE, scs=0',
           '2021:297:01:41:05.356 | MP_TARGQUAT      | AOUPTARQ   | CMD_EVT  | '
           'event=Maneuver, event_date=2021:297:01:41:01, q1=7.05469070e-01, '
           'q2=3.29883070e-01, q3=5.34409010e-01, q4=3.28477660e-01, scs=0',
           '2021:297:01:41:11.250 | COMMAND_SW       | AOMANUVR   | CMD_EVT  | '
           'event=Maneuver, event_date=2021:297:01:41:01, msid=AOMANUVR, scs=0',
           '2021:297:02:12:42.886 | ORBPOINT         | None       | OCT1821A | '
           'event_type=EQF003M, scs=0',
           '2021:297:03:40:42.886 | ORBPOINT         | None       | OCT1821A | '
           'event_type=EQF005M, scs=0',
           '2021:297:03:40:42.886 | ORBPOINT         | None       | OCT1821A | '
           'event_type=EQF015M, scs=0',
           '2021:297:04:43:26.016 | ORBPOINT         | None       | OCT1821A | '
           'event_type=EALT1, scs=0',
           '2021:297:04:43:27.301 | ORBPOINT         | None       | OCT1821A | '
           'event_type=XALT1, scs=0',
           '2021:297:12:42:42.886 | ORBPOINT         | None       | OCT1821A | '
           'event_type=EQF013M, scs=0',
           '2021:297:13:59:39.602 | ORBPOINT         | None       | OCT1821A | '
           'event_type=EEF1000, scs=0']
    assert cmds.pformat_like_backstop(max_params_width=200) == exp


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_cmds_scenario(stop_date_2020_12_03):
    """Test custom scenario with a couple of ACIS commands"""
    # First make the cmd_events.csv file for the scenario
    scenario = 'test_acis'
    cmds_dir = Path(commands_v2.conf.commands_dir) / scenario
    cmds_dir.mkdir(exist_ok=True)
    # Note variation in format of date, since this comes from humans.
    cmd_evts_text = """\
Date,Event,Params,Author,Comment
2020-12-01T00:08:30,Command,ACISPKT | TLMSID=WSPOW00000",Tom Aldcroft,
2020-12-01 00:08:39,Command,"ACISPKT | TLMSID=WSVIDALLDN",Tom Aldcroft,
"""
    (cmds_dir / 'cmd_events.csv').write_text(cmd_evts_text)

    # Now get commands in a time range that includes the new command events
    cmds = commands_v2.get_cmds('2020-12-01 00:08:00', '2020-12-01 00:09:00',
                                scenario=scenario)
    cmds = cmds[cmds['tlmsid'] != 'OBS']
    exp = [
        '2020:336:00:08:30.000 | ACISPKT          | WSPOW00000 | CMD_EVT  | event=Command, event_date=2020:336:00:08:30, scs=0',  # noqa
        '2020:336:00:08:38.610 | COMMAND_HW       | CNOOP      | NOV3020A | hex=7E00000, msid=CNOOPLR, scs=128',  # noqa
        '2020:336:00:08:39.000 | ACISPKT          | WSVIDALLDN | CMD_EVT  | event=Command, event_date=2020:336:00:08:39, scs=0',  # noqa
        '2020:336:00:08:39.635 | COMMAND_HW       | CNOOP      | NOV3020A | hex=7E00000, msid=CNOOPLR, scs=128'  # noqa
    ]
    assert cmds.pformat_like_backstop() == exp


def test_command_set_bsh():
    cmds = get_cmds_from_event('2000:001', 'Bright star hold', '')
    exp = [
        '2000:001:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2000:001:00:00:00, msid=AFLCRSET, scs=0',
        '2000:001:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2000:001:00:00:00, pos=-99616, scs=0',
        '2000:001:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:01:17.960 | COMMAND_SW       | AODSDITH   | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2000:001:00:00:00, scs=0']

    assert cmds.pformat_like_backstop() == exp


def test_command_set_safe_mode():
    cmds = get_cmds_from_event('2000:001', 'Safe mode', '')
    exp = [
        '2000:001:00:00:00.000 | COMMAND_SW       | ACPCSFSU   | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:00:00.000 | COMMAND_SW       | CSELFMT5   | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:00:00.000 | COMMAND_SW       | AONSMSAF   | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, msid=AFLCRSET, scs=0',
        '2000:001:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, pos=-99616, scs=0',
        '2000:001:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, scs=0',
        '2000:001:00:01:17.960 | COMMAND_SW       | AODSDITH   | CMD_EVT  | '
        'event=Safe_mode, event_date=2000:001:00:00:00, scs=0']

    assert cmds.pformat_like_backstop() == exp


@pytest.mark.skipif(not HAS_INTERNET, reason="No internet connection")
def test_bright_star_hold_event(cmds_dir, stop_date_2020_12_03):
    """Make a scenario with a bright star hold event.

    Confirm that this inserts expected commands and interrupts all load commands.
    """
    bsh_dir = Path(conf.commands_dir) / 'bsh'
    bsh_dir.mkdir(parents=True, exist_ok=True)
    cmd_events_file = bsh_dir / 'cmd_events.csv'
    cmd_events_file.write_text("""\
State,Date,Event,Params,Author,Comment
Definitive,2020:337:00:00:00,Bright star hold,,Tom Aldcroft,
""")
    cmds = commands_v2.get_cmds(start='2020:336:21:48:00', stop='2020:338', scenario='bsh')
    exp = [
        '2020:336:21:48:03.312 | LOAD_EVENT       | OBS        | NOV3020A | '
        'manvr_start=2020:336:21:09:24.361, prev_att=(-0.242373434, -0.348723922, '
        '0.42827',
        '2020:336:21:48:06.387 | COMMAND_SW       | CODISASX   | NOV3020A | '
        'hex=8456200, msid=CODISASX, codisas1=98 , scs=128',
        '2020:336:21:48:07.412 | COMMAND_SW       | AOFUNCEN   | NOV3020A | '
        'hex=803031E, msid=AOFUNCEN, aopcadse=30 , scs=128',
        '2020:336:21:54:23.061 | COMMAND_SW       | AOFUNCEN   | NOV3020A | '
        'hex=8030320, msid=AOFUNCEN, aopcadse=32 , scs=128',
        '2020:336:21:55:23.061 | COMMAND_SW       | AOFUNCEN   | NOV3020A | '
        'hex=8030315, msid=AOFUNCEN, aopcadse=21 , scs=128',
        # BSH interrupt at 2020:337
        '2020:337:00:00:00.000 | COMMAND_SW       | OORMPDS    | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0',
        '2020:337:00:00:01.025 | COMMAND_HW       | AFIDP      | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2020:337:00:00:00, msid=AFLCRSET, scs=0',
        '2020:337:00:00:01.025 | SIMTRANS         | None       | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2020:337:00:00:00, pos=-99616, scs=0',
        '2020:337:00:01:06.685 | ACISPKT          | AA00000000 | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0',
        '2020:337:00:01:07.710 | ACISPKT          | AA00000000 | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0',
        '2020:337:00:01:17.960 | ACISPKT          | WSPOW00000 | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0',
        '2020:337:00:01:17.960 | COMMAND_SW       | AODSDITH   | CMD_EVT  | '
        'event=Bright_star_hold, event_date=2020:337:00:00:00, scs=0',
        # Only ORBPOINT from here on
        '2020:337:02:07:03.790 | ORBPOINT         | None       | NOV3020A | '
        'event_type=EAPOGEE, scs=0',
        '2020:337:21:15:45.455 | ORBPOINT         | None       | NOV3020A | '
        'event_type=EALT1, scs=0',
        '2020:337:21:15:46.227 | ORBPOINT         | None       | NOV3020A | '
        'event_type=XALT1, scs=0']
    assert cmds.pformat_like_backstop() == exp


def test_get_observations_by_obsid_single():
    obss = get_observations(8008)
    assert len(obss) == 1
    assert obss == [
        {'obsid': 8008,
         'simpos': 92904,
         'obs_stop': '2007:002:18:04:28.965',
         'manvr_start': '2007:002:04:31:48.216',
         'targ_att': (0.149614271, 0.490896707, 0.831470649, 0.21282047),
         'npnt_enab': True,
         'obs_start': '2007:002:04:46:58.056',
         'prev_att': (0.319214732, 0.535685207, 0.766039803, 0.155969017),
         'starcat_idx': 144398,
         'source': 'DEC2506C'}
    ]


def test_get_observations_by_obsid_multi():
    obss = get_observations(47912)  # Following ACA high background NSM 2019:248
    assert obss == [
        {'obsid': 47912,
         'simpos': -99616,
         'obs_stop': '2019:248:16:51:18.000',
         'manvr_start': '2019:248:14:52:35.407',
         'targ_att': (-0.564950617, 0.252299958, -0.165669121, 0.767938327),
         'npnt_enab': True,
         'obs_start': '2019:248:15:27:35.289',
         'prev_att': (-0.218410783, 0.748632452, -0.580771797, 0.233560059),
         'starcat_idx': 165938,
         'source': 'SEP0219B'},
        {'obsid': 47912,
         'simpos': -99616,
         'obs_stop': '2019:249:01:59:00.000',
         'manvr_start': '2019:248:16:51:18.000',
         'targ_att': (-0.3594375808951632,
                      0.6553454859043244,
                      -0.4661410647781301,
                      0.47332803366853643),
         'npnt_enab': False,
         'obs_start': '2019:248:17:18:17.732',
         'prev_att': (-0.564950617, 0.252299958, -0.165669121, 0.767938327),
         'source': 'CMD_EVT'},
        {'obsid': 47912,
         'simpos': -99616,
         'obs_stop': '2019:249:23:30:00.000',
         'manvr_start': '2019:249:01:59:10.250',
         'targ_att': (-0.54577727, 0.27602874, -0.17407247, 0.77177334),
         'npnt_enab': True,
         'obs_start': '2019:249:02:25:31.907',
         'prev_att': (-0.3594375808951632,
                      0.6553454859043244,
                      -0.4661410647781301,
                      0.47332803366853643),
         'starcat_idx': 165938,
         'source': 'CMD_EVT'}
    ]


def test_get_observations_by_start_date():
    # Test observations from a 6 months ago onward
    obss = get_observations(start=CxoTime.now() - 180 * u.day)
    assert len(obss) > 500
    # Latest obs should also be no less than 14 days old
    assert obss[-1]['obs_start'] > (CxoTime.now() - 14 * u.day).date


def test_get_observations_by_start_stop_date_with_scenario():
    # Test observations in a range and use the scenario keyword
    obss = get_observations(start='2022:001', stop='2022:002', scenario='flight')
    assert len(obss) == 6
    assert obss[0]['obsid'] == 45814
    assert obss[0]['obs_start'] == '2022:001:05:48:44.808'
    assert obss[-1]['obsid'] == 23800
    assert obss[-1]['obs_start'] == '2022:001:17:33:53.255'


def test_get_observations_no_match():
    with pytest.raises(ValueError, match='No matching observations for obsid=8008'):
        get_observations(obsid=8008, start='2022:001', stop='2022:002', scenario='flight')

    obss = get_observations(start='2022:001', stop='2022:001', scenario='flight')
    assert len(obss) == 0


years = np.arange(2003, CxoTime.now().ymdhms.year + 1)


@pytest.mark.parametrize('year', years)
def test_get_starcats_each_year(year):
    starcats = get_starcats(start=f'{year}:001', stop=f'{year}:004')
    assert len(starcats) > 2
    for starcat in starcats:
        # Make sure fids and stars are all ID'd
        ok = starcat['type'] != 'MON'
        assert np.all(starcat['id'][ok] != -999)


def test_get_starcats_with_cmds():
    cmds = commands_v2.get_cmds(start='2022:001', stop='2022:002')
    starcats0 = get_starcats(start='2022:001', stop='2022:002')
    starcats1 = get_starcats(cmds=cmds)
    assert len(starcats0) == len(starcats1)
    for starcat0, starcat1 in zip(starcats0, starcats1):
        eq = starcat0.values_equal(starcat1)
        for col in eq.itercols():
            assert np.all(col)


def test_get_starcats_obsid():
    from mica.starcheck import get_starcat
    sc_kadi = get_starcats(obsid=26330)[0]
    sc_mica = get_starcat(26330)
    assert len(sc_kadi) == len(sc_mica)
    assert sc_kadi.colnames == ['slot', 'idx', 'id', 'type', 'sz', 'mag',
                                'maxmag', 'yang', 'zang', 'dim', 'res', 'halfw']
    for name in sc_kadi.colnames:
        if name == 'mag':
            continue  # kadi mag is latest from agasc, could change
        elif name == 'maxmag':
            assert np.allclose(sc_kadi[name], sc_mica[name], atol=0.001, rtol=0)
        elif name in ('yang', 'zang'):
            assert np.all(np.abs(sc_kadi[name] - sc_mica[name]) < 1)
        else:
            assert np.all(sc_kadi[name] == sc_mica[name])
