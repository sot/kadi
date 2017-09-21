# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
    dwell0start = Column(Float)
    manstart = Column(Float)
    manstop = Column(Float)
    dwell1stop = Column(Float)

    def __init__(self, tlmevents, manstart, manstop):
        self.manstart = manstart
        self.manstop = manstop
        self.dwell0start = tlmevents[0].time
        self.dwell1stop = tlmevents[-1].time0

        if 0:
            self.dwell0_start
            self.dwell0_nman_npnt
            self.dwell0_npnt_nman
            self.manvr_start
            self.manvr_stop
            self.dwell1_nman_npnt
            self.dwell1_init_acq0
            self.dwell1_acq_guid0
            self.dwell1_guid_kalm0
            self.dwell1_init_acq1
            self.dwell1_acq_guid1
            self.dwell1_guid_kalm1
            self.dwell1_init_acq2
            self.dwell1_acq_guid2
            self.dwell1_guid_kalm2
            self.dwell1_npnt_nman
            self.dwell1_stop

        self.tlmevents = tlmevents
        # for tlmevent in tlmevents:
        #    self.tlmevents.append(tlmevent)

    def __repr__(self):
        return "<Manvr('{}','{}')>".format(
            self.manstart, self.manstop)


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
