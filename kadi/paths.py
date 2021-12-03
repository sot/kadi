# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from pathlib import Path


def _version_str(version):
    """Legacy version string"""
    return '' if version is None else f'_v{version}'


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


def CMDS_DIR(cmds_dir=None):
    if cmds_dir is None:
        if 'KADI_CMDS_DIR' in os.environ:
            cmds_dir = Path(os.environ['KADI_CMDS_DIR'])
        else:
            cmds_dir = Path('.')  # DATA_DIR() ?
    else:
        cmds_dir = Path(cmds_dir)

    return cmds_dir.absolute()


def LOADS_ARCHIVE_DIR(cmds_dir=None, load_name=None):
    out = CMDS_DIR(cmds_dir) / 'loads'
    if load_name is not None:
        year = load_name[5:7]
        if year == '99':
            year = '1999'
        else:
            year = f'20{year}'
        out = out / year
    return out


def LOADS_BACKSTOP_PATH(cmds_dir=None, load_name=None):
    out = LOADS_ARCHIVE_DIR(cmds_dir, load_name) / f'{load_name}.pkl.gz'
    return out


def SCENARIO_DIR(cmds_dir=None, scenario=None):
    scenario_dir = CMDS_DIR(cmds_dir)
    if scenario is None:
        scenario = os.environ.get('KADI_SCENARIO')
    if scenario:
        scenario_dir = scenario_dir / scenario
    return scenario_dir


def LOADS_TABLE_PATH(cmds_dir=None, scenario=None):
    return SCENARIO_DIR(cmds_dir, scenario) / 'loads.csv'


def CMD_EVENTS_PATH(cmds_dir=None, scenario=None):
    return SCENARIO_DIR(cmds_dir, scenario) / 'cmd_events.csv'
