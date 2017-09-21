# Licensed under a 3-clause BSD style license - see LICENSE.rst
from itertools import izip, cycle

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base

from tlmevent import TlmEvent, Manvr, Base

engine = create_engine('sqlite:///test.db3', echo=True)
session = sessionmaker(bind=engine)()
Base.metadata.create_all(engine)

size = 1000
msid_vals = ['pcadmd', 'aofatt', 'aoacaseq', 'cobsrqid']
msids = (msid_vals[i] for i in np.random.randint(4, size=size))
vals = cycle(['nman', 1, 2.3])
priorvals = cycle([3.2, 'npnt', 1])

# Make fake telemetry events
for msid, time, priorval, val in izip(msids, xrange(size),
                                      priorvals, vals):
    tlmevent = TlmEvent(msid, float(time), float(time), priorval, val)
    session.add(tlmevent)
session.commit()

# Make fake maneuvers
for i in range(size // 10):
    manvr = Manvr(session, i * 10.0, i * 10.0 + 10)
    session.add(manvr)
session.commit()

events = session.query(TlmEvent).all()
manvrs = session.query(Manvr).all()
