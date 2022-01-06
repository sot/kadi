# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Compare kadi grating_moves events to those derived by MTA circa Jan. 8, 2014.
http://asc.harvard.edu/mta_days/mta_otg/OTG_filtered.html

(as needed)
>>> import sys
>>> sys.path.insert(0, '..')

>>> run -i compare_grating_moves
>>> print kadi_mta_bad
>>> print mta_kadi_bad
"""
import numpy as np
import Ska.Numpy
from astropy.table import Table
from astropy.io import ascii
from kadi import events
from Chandra.Time import DateTime

kadi_moves = events.grating_moves.filter(start='2000:160:12:00:00', stop='2014:008:12:00:00',
                                         grating__contains='ETG').table
mta_moves = Table.read('mta_grating_moves.dat', format='ascii',
                       converters={'START_TIME': [ascii.convert_numpy(str)],
                                   'STOP_TIME': [ascii.convert_numpy(str)]})
mta_moves.sort('START_TIME')

kadi_starts = kadi_moves['tstart']
mta_starts = DateTime(mta_moves['START_TIME'], format='greta').secs

# Kadi to nearest MTA

indexes = np.arange(len(mta_starts))
i_nearest = Ska.Numpy.interpolate(indexes, mta_starts, kadi_starts,
                                  sorted=True, method='nearest')
mta_nearest = mta_moves[i_nearest]
mta_nearest_starts = mta_starts[i_nearest]
dt = kadi_moves['tstart'] - mta_nearest_starts
kadi_mta = Table([i_nearest, kadi_moves['start'], mta_nearest['START_TIME'], dt],
                 names=['i_nearest', 'kadi_date', 'mta_date', 'dt'])
bad = np.abs(kadi_mta['dt']) > 4
kadi_mta_bad = kadi_mta[bad]

# MTA to nearest Kadi

indexes = np.arange(len(kadi_starts))
i_nearest = Ska.Numpy.interpolate(indexes, kadi_starts, mta_starts,
                                  sorted=True, method='nearest')
kadi_nearest = kadi_moves[i_nearest]
dt = kadi_nearest['tstart'] - mta_starts
mta_kadi = Table([i_nearest, kadi_nearest['start'], mta_moves['START_TIME'], dt],
                 names=['i_nearest', 'kadi_date', 'mta_date', 'dt'])
bad = np.abs(mta_kadi['dt']) > 4
mta_kadi_bad = mta_kadi[bad]
