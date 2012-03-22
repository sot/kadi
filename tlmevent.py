from sqlalchemy import Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy import PickleType
from sqlalchemy.orm import relationship

Base = declarative_base()


class Manvr(Base):
    __tablename__ = 'manvr'
    assoc_table = Table(__tablename__ + '_tlmevent',
                        Base.metadata,
                        Column('manvr_id', Integer,
                               ForeignKey('manvr.id'), index=True),
                        Column('tlmevent_id', Integer,
                               ForeignKey('tlmevent.id'))
                        )
    tlmevents = relationship('TlmEvent', secondary=assoc_table,
                             backref=__tablename__)

    id = Column(Integer, primary_key=True)
    tstart = Column(Float)
    tstop = Column(Float)

    def __init__(self, session, tstart, tstop):
        self.tstart = tstart
        self.tstop = tstop
        tlmevents = session.query(TlmEvent).filter(TlmEvent.time >= tstart) \
                                     .filter(TlmEvent.time <= tstop)
        for tlmevent in tlmevents:
            self.tlmevents.append(tlmevent)

    def __repr__(self):
        return "<Manvr('{}','{}')>".format(
            self.tstart, self.tstop)


class TlmEvent(Base):
    __tablename__ = 'tlmevent'

    id = Column(Integer, primary_key=True)
    msid = Column(String)
    time0 = Column(Float)
    time = Column(Float, index=True)
    val0 = Column(PickleType)
    val = Column(PickleType)

    def __init__(self, msid, time0, time, val0, val):
        self.msid = msid
        self.time0 = time0
        self.time = time
        self.val0 = val0
        self.val = val

    def __repr__(self):
        return "<TlmEvent({}, {}, {}, {})>".format(
            repr(self.msid), repr(self.time),
            repr(self.val0), repr(self.val))
