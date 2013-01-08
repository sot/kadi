import os
import Ska.engarchive.fetch as fetch
from Chandra.Time import DateTime
from collections import Counter

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import subqueryload
from tlmevent import TlmEvent, Manvr, Base

if os.path.exists('test2.db3'):
    os.unlink('test2.db3')
engine = create_engine('sqlite:///test2.db3')
# engine = create_engine('sqlite://')
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
tstop = DateTime('2011:365').secs
msids = ['aopcadmd', 'aofattmd', 'aoacaseq']  # , 'aopsacpr']

print 'fetching data'
dats = fetch.Msidset(msids, tstart, tstop)

print 'Creating tlmevents'
for msid in msids:
    print 'msid=', msid
    dat = dats[msid]
    i_changes = np.flatnonzero(dat.vals[1:] != dat.vals[:-1])
    for i in i_changes:
        tlmevent = TlmEvent(msid, dat.times[i], dat.times[i + 1],
                            dat.vals[i], dat.vals[i + 1])
        session.add(tlmevent)
    session.commit()

print 'Getting tlmevents'
query = session.query(TlmEvent)
events = query.filter(TlmEvent.time > tstart - 1) \
                .filter(TlmEvent.time < tstop + 1).all()
aof_events = [x for x in events if x.msid == 'aofattmd']

trans_events = [x for x in aof_events if x.val0 == 'MNVR' or x.val == 'MNVR']
for i, event in enumerate(trans_events):
    if event.val == 'MNVR':
        try:
            event0 = trans_events[i - 1]  # start of previous dwell
            event1 = event                # start of maneuver
            event2 = trans_events[i + 1]  # end of maneuver
            event3 = trans_events[i + 2]  # end of next dwell
        except IndexError as err:
            # Previous or Post transition is not available, so skip for now
            pass
        else:
            manvr_events = [event0, event1]
            manvr_events.extend([x for x in events
                                 if x.time > event1.time
                                 and x.time < event3.time])
            manvr_events.append(event3)
            manvr = Manvr(manvr_events, event1.time, event2.time0)
            print 'Adding {}'.format(manvr)
            session.add(manvr)
print 'committing maneuvers'
session.commit()

print 'session.query(Manvr).all()[1:-1]'
manvrs = session.query(Manvr).options(subqueryload(Manvr.tlmevents)) \
                             .all()[1:-1]
#seqs = [tuple('{}_{}_{}'.format(x.msid, x.val0, x.val)
#              for x in manvr.tlmevents)
#        for manvr in manvrs]

seqs = []
for manvr in manvrs:
    seq = tuple('{}_{}_{}'.format(x.msid, x.val0, x.val)
                      for x in manvr.tlmevents)
    seqs.append(seq)
    if seq == ('aofattmd_MNVR_STDY', 'aofattmd_STDY_MNVR',
               'aofattmd_STDY_MNVR'):
        print manvr
        for tlmevent in manvr.tlmevents:
            print tlmevent
        break

vals = Counter(seqs)
