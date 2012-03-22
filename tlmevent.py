from itertools import izip, cycle

import numpy as np
from sqlalchemy import Table
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy import PickleType
from sqlalchemy.orm import sessionmaker, relationship

Base = declarative_base()

manvr_event = Table('manvr_event', Base.metadata,
                     Column('manvr_id', Integer,
                            ForeignKey('manvr.id'), index=True,),
                     Column('event_id', Integer, ForeignKey('event.id'))
                     )


class Manvr(Base):
    __tablename__ = 'manvr'

    id = Column(Integer, primary_key=True)
    tstart = Column(Float)
    tstop = Column(Float)
    events = relationship('Event', secondary=manvr_event, backref='manvr')

    def __init__(self, tstart, tstop):
        self.tstart = tstart
        self.tstop = tstop
        events = session.query(Event).filter(Event.time >= tstart) \
                                     .filter(Event.time <= tstop)
        for event in events:
            self.events.append(event)

    def __repr__(self):
        return "<Manvr('{}','{}', '{}')>".format(
            self.tstart, self.tstop, self.events)


class Event(Base):
    __tablename__ = 'event'

    id = Column(Integer, primary_key=True)
    msid = Column(String)
    time = Column(Float, index=True)
    priorval = Column(PickleType)
    val = Column(PickleType)

    def __init__(self, msid, time, priorval, val):
        self.msid = msid
        self.time = time
        self.priorval = priorval
        self.val = val

    def __repr__(self):
        return "<Event({}, {}, {}, {})>".format(
            repr(self.msid), repr(self.time),
            repr(self.priorval), repr(self.val))


engine = create_engine('sqlite:///test.db3', echo=False)
session = sessionmaker(bind=engine)()
Base.metadata.create_all(engine)

size = 100
msid_vals = ['pcadmd', 'aofatt', 'aoacaseq', 'cobsrqid']
msids = (msid_vals[i] for i in np.random.randint(4, size=size))
for msid, time, priorval, val in izip(msids, xrange(size),
                       cycle(['nman', 1, 2.3]),
                       cycle([3.2, 'npnt', 1])):
    event = Event(msid, float(time), priorval, val)
    session.add(event)
    if time % 10 == 0:
        session.commit()
session.commit()

for i in range(size // 10):
    manvr = Manvr(i * 10.0, i * 10.0 + 10)
    session.add(manvr)

session.commit()


# timeit manvrs = session.query(Manvr).options(subqueryload(Manvr.events)).all()
