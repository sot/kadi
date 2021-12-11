# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path


def _version_str(version):
    """Legacy version string"""
    return '' if version in (None, 1) else str(version)


def SKA_DATA():
    return Path(os.environ.get('SKA', '/proj/sot/ska'), 'data/kadi')


def DATA_DIR():
    return Path(os.environ.get('KADI', SKA_DATA())).absolute()


def EVENTS_DB_PATH():
    return DATA_DIR() / 'events3.db3'


def IDX_CMDS_PATH(version=None):
    version = _version_str(version)
    return DATA_DIR() / f'cmds{version}.h5'


def PARS_DICT_PATH(version=None):
    version = _version_str(version)
    return DATA_DIR() / f'cmds{version}.pkl'


def CMDS_DIR(scenario=None):
    from kadi.commands import conf

    # Special case for "flight" scenario. This is hardwired to $SKA/data/kadi
    # and is intended for use in load review tools run on HEAD.
    if scenario == 'flight':
        cmds_dir = SKA_DATA()
    else:
        cmds_dir = Path(conf.commands_dir).expanduser()

    return cmds_dir.absolute()


def LOADS_ARCHIVE_DIR():
    out = CMDS_DIR() / 'loads'
    return out


def LOADS_BACKSTOP_PATH(load_name):
    out = LOADS_ARCHIVE_DIR() / f'{load_name}.pkl.gz'
    return out


def SCENARIO_DIR(scenario=None):
    scenario_dir = CMDS_DIR()
    if scenario is None:
        scenario = os.environ.get('KADI_SCENARIO')
    if scenario:
        scenario_dir = scenario_dir / scenario
    return scenario_dir


def LOADS_TABLE_PATH(scenario=None):
    return SCENARIO_DIR(scenario) / 'loads.csv'


def CMD_EVENTS_PATH(scenario=None):
    return SCENARIO_DIR(scenario) / 'cmd_events.csv'
