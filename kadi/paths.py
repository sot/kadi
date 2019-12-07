# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os


def SKA_DATA():
    return os.path.join(os.environ.get('SKA', '/proj/sot/ska'), 'data/kadi')


def DATA_DIR():
    return os.path.abspath(os.environ.get('KADI', SKA_DATA()))


def EVENTS_DB_PATH():
    return os.path.join(DATA_DIR(), 'events3.db3')


def IDX_CMDS_PATH():
    return os.path.join(DATA_DIR(), 'cmds.h5')


def PARS_DICT_PATH():
    return os.path.join(DATA_DIR(), 'cmds.pkl')
