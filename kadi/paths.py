import os

SKA_DATA = os.path.join(os.environ.get('SKA', '/proj/sot/ska'), 'data/kadi')
DATA_DIR = os.path.abspath(os.environ.get('KADI', SKA_DATA))
