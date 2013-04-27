import os

SKA_DATA = os.path.join(os.environ.get('SKA', '/proj/sot/ska'), 'data/kadi')
DATA_DIR = os.path.abspath(os.environ.get('KADI', SKA_DATA))

EVENTS_DB_PATH = os.path.join(DATA_DIR, 'events.db3')
IDX_CMDS_PATH = os.path.join(DATA_DIR, 'cmds.h5')
PARS_DICT_PATH = os.path.join(DATA_DIR, 'cmds.pkl')
