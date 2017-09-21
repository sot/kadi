# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import Ska.engarchive.fetch as fetch
from Chandra.Time import DateTime
from collections import Counter

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import subqueryload
from tlmevent import TlmEvent, Manvr, Base

filename = 'test3.db3'
if os.path.exists(filename):
    os.unlink(filename)
engine = create_engine('sqlite:///' + filename)
session = sessionmaker(bind=engine)()
Base.metadata.create_all(engine)


def add_year(year):
    year0 = str(year)
    year1 = str(year + 1)
    print 'Working on year', year0
    tstart = DateTime(year0 + ':001').secs
    tstop = DateTime(year1 + ':001').secs
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


def add_maneuvers(events):
    print 'Getting tlmevents'
    aof_events = [x for x in events if x.msid == 'aofattmd']

    trans_events = [x for x in aof_events if x.val0 == 'MNVR' or
                                             x.val == 'MNVR']
    times = np.array([x.time for x in events])

    for i, event in enumerate(trans_events):
        if event.val == 'MNVR':
            try:
                event0 = trans_events[i - 1]  # start of previous dwell
                event1 = event                # start of maneuver
                event2 = trans_events[i + 1]  # end of maneuver
                event3 = trans_events[i + 2]  # end of next dwell
            except IndexError:
                # Previous or Post transition is not available, so skip for now
                pass
            else:
                manvr_events = [event0, event1]
                i0 = np.searchsorted(times, event1.time, side='left')
                i1 = np.searchsorted(times, event3.time, side='right')
                manvr_events.extend([x for x in events[i0:i1 + 1]
                                     if x.time > event1.time
                                     and x.time < event3.time])
                manvr_events.append(event3)
                manvr = Manvr(manvr_events, event1.time, event2.time0)
                print i, manvr
                session.add(manvr)
                if i % 2800 == -1:
                    print 'committing maneuvers'
                    session.commit()

    print 'committing maneuvers'
    session.commit()


def get_unique():
    print 'session.query(Manvr).all()[1:-1]'
    manvrs = session.query(Manvr).options(subqueryload(Manvr.tlmevents)) \
                                 .all()[1:-1]
    #seqs = [tuple('{}_{}_{}'.format(x.msid, x.val0, x.val)
    #              for x in manvr.tlmevents)
    #        for manvr in manvrs]

    seqs = []
    for manvr in manvrs:
        print manvr
        seqs.append(tuple('{}_{}_{}'.format(x.msid, x.val0, x.val)
                          for x in manvr.tlmevents))

    vals = Counter(seqs)
    return vals


for year in range(2000, 2013):
    add_year(year)

events = session.query(TlmEvent).all()
add_maneuvers()
