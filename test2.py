import Ska.engarchive.fetch as fetch
from Chandra.Time import DateTime

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tlmevent import TlmEvent, Manvr, Base

engine = create_engine('sqlite:///test2.db3')
session = sessionmaker(bind=engine)()
Base.metadata.create_all(engine)

# In [6]: dat = fetch.Msid('aofattmd', '2011:001', '2011:365')
# 
# In [7]: set(dat.vals)
# Out[7]: set(['MNVR', 'NULL', 'STDY', 'DTHR'])
# In [10]: dat.tdb.Tsc['STATE_CODE']
# Out[10]:
# rec.array(['MNVR', 'STDY', 'NULL', 'DTHR'],
#       dtype='|S4')

tstart = DateTime('2011:001').secs
tstop = DateTime('2011:005').secs
msids = ['aopcadmd', 'aofattmd', 'aoacaseq', 'aofattup']

dats = fetch.Msidset(msids, tstart, tstop)

for msid in msids:
    dat = dats[msid]
    i_changes = np.flatnonzero(dat.vals[1:] != dat.vals[:-1])
    for i in i_changes:
        tlmevent = TlmEvent(msid, dat.times[i], dat.times[i + 1],
                            dat.vals[i], dat.vals[i + 1])
        session.add(tlmevent)
    session.commit()

query = session.query(TlmEvent)
events = query.filter(TlmEvent.msid == 'aofattmd') \
              .filter(TlmEvent.time >= tstart) \
              .filter(TlmEvent.time <= tstop) \
              .all()

for i, event in enumerate(events):
    if event.val == 'MNVR':
        try:
            event0 = events[i - 1]
            event1 = events[i + 1]
        except IndexError as err:
            # Previous or Post transition is not available, so skip for now
            print err
            pass
        else:
            manvr = Manvr(session, event0.time, event1.time0)
            print 'Adding {}'.format(manvr)
            session.add(manvr)
session.commit()

manvrs = session.query(Manvr).all()
