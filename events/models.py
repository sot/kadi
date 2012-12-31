from itertools import izip

from django.db import models

from .manvr_templates import get_manvr_templates
from .json_field import JSONField

# Fool pyflakes into thinking these are defined
fetch = None
interpolate = None
DateTime = None
np = None

ZERO_DT = -1e-4


def get_si(simpos):
    """
    Get SI corresponding to the given SIM position.
    """
    if ((simpos >= 82109) and (simpos <= 104839)):
        si = 'ACIS-I'
    elif ((simpos >= 70736) and (simpos <= 82108)):
        si = 'ACIS-S'
    elif ((simpos >= -86147) and (simpos <= -20000)):
        si = ' HRC-I'
    elif ((simpos >= -104362) and (simpos <= -86148)):
        si = ' HRC-S'
    else:
        si = '  NONE'
    return si


def get_msid_changes(msids, sortmsids={}):
    """
    For the list of fetch MSID objects, return a sorted structured array
    of each time any MSID value changes.
    """
    changes = []
    for msid in msids:
        i_changes = np.flatnonzero(msid.vals[1:] != msid.vals[:-1])
        for i in i_changes:
            change = (msid.msid,
                      sortmsids.get(msid.msid, 10),
                      msid.vals[i], msid.vals[i + 1],
                      DateTime(msid.times[i]).date, DateTime(msid.times[i + 1]).date,
                      0.0,
                      msid.times[i], msid.times[i + 1],
                      )
            changes.append(change)
    changes = np.rec.fromrecords(changes, names=('msid', 'sortmsid', 'val0', 'val',
                                                 'date0', 'date', 'dt', 'time0', 'time'))
    changes.sort(order=['time', 'sortmsid'])
    return changes


def import_ska(func):
    """
    Decorator to lazy import useful Ska functions.  Web app should not
    need to do this.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from Chandra.Time import DateTime
        from Ska.Numpy import interpolate
        import Ska.engarchive.fetch_eng as fetch
        import numpy as np
        globals()['interpolate'] = interpolate
        globals()['DateTime'] = DateTime
        globals()['fetch'] = fetch
        globals()['np'] = np
        return func(*args, **kwargs)
    return wrapper


class EventUpdate(models.Model):
    """
    Last telemetry which was searched for an update.
    """
    name = models.CharField(max_length=30, primary_key=True)  # model name
    date = models.CharField(max_length=21)


class Event(models.Model):
    datestart = models.CharField(max_length=21, primary_key=True)
    datestop = models.CharField(max_length=21)
    tstart = models.FloatField(db_index=True)
    tstop = models.FloatField()
    lookback = 21  # days of lookback into telemetry

    class Meta:
        abstract = True

    @classmethod
    def from_dict(cls, event_dict, logger=None):
        """
        Set Event model from a dict `event_dict` which might have extra stuff not in
        Model.  If `logger` is supplied then log output at debug level.
        """
        event_model = cls()
        for key, val in event_dict.items():
            if hasattr(event_model, key):
                if logger is not None:
                    logger.debug('Setting {} model with {}={}'
                                 .format(event_model.name, key, val))
                setattr(event_model, key, val)
        return event_model

    def __unicode__(self):
        return ('datestart={}'.format(self.datestart))


class TlmEvent(Event):
    event_msid = None  # must be overridden by derived class
    event_val = None
    rel_msids = None

    class Meta:
        abstract = True

    @classmethod
    def add_extras(cls, event, event_msid, rel_msids):
        pass

    @classmethod
    def get_msids_states(cls, start, stop):
        """
        Get event and related MSIDs and compute the states corresponding
        to the event.
        """
        tstart = DateTime(start).secs
        tstop = DateTime(stop).secs

        # Get the related MSIDs
        rel_msids = fetch.Msidset(cls.rel_msids, tstart, tstop) if cls.rel_msids else {}

        # Get the MSID that defines the event and find state intervals on that MSID
        event_msid = fetch.Msid(cls.event_msid, tstart, tstop)
        states = event_msid.state_intervals()

        # Require that the event states be flanked by a non-event state
        # to ensure that the full event was seen in telemetry.
        if states[0]['val'] == cls.event_val:
            states = states[1:]
        if states[-1]['val'] == cls.event_val:
            states = states[:-1]

        # Select event states that are entirely contained within interval
        ok = ((states['val'] == cls.event_val) &
              (states['tstart'] >= tstart) & (states['tstop'] <= tstop))
        states = states[ok]

        return states, event_msid, rel_msids

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get events from telemetry defined by a simple rule that the value of
        `event_msid` == `event_val`.  Related MSIDs in the list `rel_msid`
        are fetched and put into the event at the time `rel_dt` seconds
        before the event start and after the event end.
        """
        states, event_msid, rel_msids = cls.get_msids_states(start, stop)

        # Assemble a list of dicts corresponding to events in this tlm interval
        events = []
        for state in states:
            tstart = state['tstart']
            tstop = state['tstop']
            event = dict(tstart=tstart,
                         tstop=tstop,
                         datestart=DateTime(tstart).date,
                         datestop=DateTime(tstop).date)

            for rel_msid in rel_msids.values():
                vals = interpolate(rel_msid.vals, rel_msid.times,
                                   [tstart - cls.rel_dt, tstop + cls.rel_dt],
                                   method='nearest')
                event['start_{}'.format(rel_msid.msid)] = vals[0]
                event['stop_{}'.format(rel_msid.msid)] = vals[1]

            # Custom processing defined by subclasses to add more attrs to event
            cls.add_extras(event, event_msid, rel_msids)

            events.append(event)

        return events

    @property
    def msidset(self):
        """
        fetch.MSIDset of self.fetch_event_msids.  By default filter_bad is True.
        """
        if not hasattr(self, '_msidset'):
            self._msidset = self.fetch_event()
        return self._msidset

    @import_ska
    def fetch_event(self, pad=None, extra_msids=None, filter_bad=True):
        """
        Fetch an MSIDset of self.fetch_msids.
        """
        if pad is None:
            pad = self.fetch_event_pad
        msids = self.fetch_event_msids[:]
        if extra_msids is not None:
            msids.extend(extra_msids)
        msidset = fetch.MSIDset(msids, self.tstart - pad, self.tstop + pad,
                                filter_bad=filter_bad)
        return msidset


class TscMove(TlmEvent):
    name = 'tsc_move'
    event_msid = '3tscmove'
    event_val = 'T'
    rel_msids = ['3tscpos']
    rel_dt = 66  # just over 2 major frame rate samples

    start_3tscpos = models.IntegerField()
    stop_3tscpos = models.IntegerField()

    @classmethod
    def add_extras(cls, event, event_msid, rel_msids):
        event['start_det'] = get_si(event['start_3tscpos'])
        event['stop_det'] = get_si(event['stop_3tscpos'])

    def __unicode__(self):
        return ('datestart={} start_3tscpos={} stop_3tscpos={}'
                .format(self.datestart, self.start_3tscpos, self.stop_3tscpos))


class FaMove(TlmEvent):
    name = 'fa_move'
    event_msid = '3famove'
    event_val = 'T'
    rel_msids = ['3fapos']
    rel_dt = 16.4  # 1/2 major frame (FA moves can be within 2 minutes)

    start_3fapos = models.IntegerField()
    stop_3fapos = models.IntegerField()

    def __unicode__(self):
        return ('datestart={} start_3fapos={} stop_3fapos={}'
                .format(self.datestart, self.start_3fapos, self.stop_3fapos))


class Dump(TlmEvent):
    name = 'dump'
    event_msid = 'aounload'
    event_val = 'GRND'


class Eclipse(TlmEvent):
    name = 'eclipse'
    event_msid = 'aoeclips'
    event_val = 'ECL '


class Manvr(TlmEvent):
    name = 'manvr'
    event_msid = 'aofattmd'  # MNVR STDY NULL DTHR
    event_val = 'MNVR'
    rel_msids = ['aopcadmd', 'aoacaseq', 'aopsacpr']
    fetch_event_msids = ['one_shot', 'aofattmd', 'aopcadmd', 'aoacaseq',
                         'aopsacpr', 'aounload',
                         'aoattqt1', 'aoattqt2', 'aoattqt3', 'aoattqt4',
                         'aogyrct1', 'aogyrct2', 'aogyrct3', 'aogyrct4',
                         'aoatupq1', 'aoatupq2', 'aoatupq3',
                         'aotarqt1', 'aotarqt2', 'aotarqt3',
                         'aoatter1', 'aoatter2', 'aoatter3',
                         'aogbias1', 'aogbias2', 'aogbias3']
    fetch_event_pad = 600

    prev_manvr_stop = models.CharField(max_length=21, null=True)
    prev_npnt_start = models.CharField(max_length=21, null=True)
    nman_start = models.CharField(max_length=21, null=True)
    manvr_start = models.CharField(max_length=21, null=True)
    manvr_stop = models.CharField(max_length=21, null=True)
    npnt_start = models.CharField(max_length=21, null=True)
    acq_start = models.CharField(max_length=21, null=True)
    guide_start = models.CharField(max_length=21, null=True)
    kalman_start = models.CharField(max_length=21, null=True)
    aca_proc_act_start = models.CharField(max_length=21, null=True)
    npnt_stop = models.CharField(max_length=21, null=True)
    next_nman_start = models.CharField(max_length=21, null=True)
    next_manvr_start = models.CharField(max_length=21, null=True)
    n_dwell = models.IntegerField()
    n_acq = models.IntegerField()
    n_guide = models.IntegerField()
    n_kalman = models.IntegerField()
    anomalous = models.BooleanField()
    template = models.CharField(max_length=16)
    dwell_start = models.CharField(max_length=21, null=True)
    dwell_stop = models.CharField(max_length=21, null=True)
    dwell_rel_tstart = models.FloatField(null=True)
    dwell_dur = models.FloatField(null=True)
    dwell2_start = models.CharField(max_length=21, null=True)
    dwell2_stop = models.CharField(max_length=21, null=True)
    dwell2_rel_tstart = models.FloatField(null=True)
    dwell2_dur = models.FloatField(null=True)
    tlm = JSONField()

    def __unicode__(self):
        dwell_dur = ('None' if self.dwell_dur is None
                     else '{:.1f} ks'.format(self.dwell_dur / 1000.))
        return ('datestart={} dwell_dur={} template={}'
                .format(self.datestart[:-4], dwell_dur, self.template))

    @classmethod
    def get_dwells(cls, changes):
        dwells = []
        state = None
        t0 = 0
        ok = changes['dt'] >= ZERO_DT
        dwell = {}
        for change in changes[ok]:
            # Not in a dwell and ACA sequence is KALMAN => start dwell.
            if (state is None
                    and change['msid'] == 'aoacaseq'
                    and change['val'] == 'KALM'):
                t0 = change['time']
                dwell['dt'] = change['dt']
                dwell['tstart'] = change['time']
                dwell['datestart'] = change['date']
                state = 'dwell'

            # Another KALMAN within 400 secs of previous KALMAN in dwell.
            # This is another acquisition sequence and moves the dwell start back.
            elif (state == 'dwell'
                  and change['msid'] == 'aoacaseq'
                  and change['val'] == 'KALM'
                  and change['time'] - t0 < 400):
                t0 = change['time']
                dwell['dt'] = change['dt']
                dwell['tstart'] = change['time']
                dwell['datestart'] = change['date']

            # End of dwell because of NPNT => NMAN transition OR another acquisition
            elif (state == 'dwell'
                  and ((change['msid'] == 'aopcadmd' and change['val'] == 'NMAN') or
                       (change['msid'] == 'aoacaseq' and change['time'] - t0 > 400))):
                dwell['tstop'] = change['time0']
                dwell['datestop'] = change['date0']
                dwells.append(dwell)
                dwell = {}
                state = None

        return dwells

    @classmethod
    def get_manvr_attrs(cls, changes):
        """
        Get attributes of the maneuver event and possible dwells based on
        the MSID `changes`.
        """
        def match(msid, val, idx=None, filter=None):
            """
            Find a match for the given `msid` and `val`.  The `filter` can
            be either 'before' or 'after' to select changes before or after
            the maneuver end.  The `idx` value then selects form the matching
            changes.  If the desired match is not available then None is returned.
            """
            ok = (changes['msid'] == msid)
            if val.startswith('!'):
                ok &= (changes['val'] != val[1:])
            else:
                ok &= (changes['val'] == val)
            if filter == 'before':
                ok &= (changes['dt'] < ZERO_DT)
            elif filter == 'after':
                ok &= (changes['dt'] >= ZERO_DT)
            try:
                if idx is None:
                    return changes[ok]['date']
                else:
                    return changes[ok][idx]['date']
            except IndexError:
                return None

        # Get the dwells from the sequence of changes
        dwells = cls.get_dwells(changes)

        # Check for any telemetry values that are off-nominal
        nom_vals = {'aopcadmd': ('NPNT', 'NMAN'),
                    'aoacaseq': ('GUID', 'KALM', 'AQXN'),
                    'aofattmd': ('MNVR', 'STDY'),
                    'aopsacpr': ('INIT', 'INAC', 'ACT '),
                    'aounload': ('MON ', 'GRND')}
        anomalous = False
        for change in changes[changes['dt'] >= ZERO_DT]:
            if change['val'] not in nom_vals[change['msid']]:
                anomalous = True
                break

        # Templates of previously seen maneuver sequences. These cover sequences seen at
        # least twice as of ~Mar 2012.
        manvr_templates = get_manvr_templates()
        seqs = ['{}_{}_{}'.format(c['msid'], c['val0'], c['val']) for c in changes
                if (c['msid'] in ('aopcadmd', 'aofattmd', 'aoacaseq') and
                    c['dt'] >= ZERO_DT)]
        for name, manvr_template in manvr_templates:
            if seqs == manvr_template[2:]:  # skip first two which are STDY-MNVR and MNVR-STDY
                template = name
                break
        else:
            template = 'unknown'

        manvr_attrs = dict(
            prev_manvr_stop=match('aofattmd', '!MNVR', -1, 'before'),  # Last STDY before this manvr
            prev_npnt_start=match('aopcadmd', 'NPNT', -1, 'before'),  # Last NPNT before this manvr

            nman_start=match('aopcadmd', 'NMAN', -1, 'before'),  # NMAN that precedes this manvr
            manvr_start=match('aofattmd', 'MNVR', -1, 'before'),  # start of this manvr
            manvr_stop=match('aofattmd', '!MNVR', 0, 'after'),

            npnt_start=match('aopcadmd', 'NPNT', 0, 'after'),
            acq_start=match('aoacaseq', 'AQXN', 0, 'after'),
            guide_start=match('aoacaseq', 'GUID', 0, 'after'),
            kalman_start=match('aoacaseq', 'KALM', 0, 'after'),
            aca_proc_act_start=match('aopsacpr', 'ACT ', 0, 'after'),
            npnt_stop=match('aopcadmd', '!NPNT', -1, 'after'),

            next_nman_start=match('aopcadmd', 'NMAN', -1, 'after'),
            next_manvr_start=match('aofattmd', 'MNVR', -1, 'after'),
            n_dwell=len(dwells),
            n_acq=len(match('aoacaseq', 'AQXN', None, 'after')),
            n_guide=len(match('aoacaseq', 'GUID', None, 'after')),
            n_kalman=len(match('aoacaseq', 'KALM', None, 'after')),
            anomalous=anomalous,
            template=template,
            )

        for i, dwell in enumerate(dwells):
            prefix = 'dwell_' if i == 0 else 'dwell{}_'.format(i + 1)
            manvr_attrs[prefix + 'start'] = dwell['datestart']
            manvr_attrs[prefix + 'stop'] = dwell['datestop']
            manvr_attrs[prefix + 'rel_tstart'] = dwell['dt']  # dwell_dt is ambiguous
            manvr_attrs[prefix + 'dur'] = dwell['tstop'] - dwell['tstart']

        return manvr_attrs

    @classmethod
    def get_tlm(cls, event):
        tlm = {'start': {'aoatterr': [0.0, 0, 0, 0]},
               'stop': {'aoatterr': [1.0, 0, 0, 1]},
               }
        return tlm

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get maneuver events from telemetry.
        """
        states, event_msid, rel_msids = cls.get_msids_states(start, stop)
        changes = get_msid_changes([event_msid] + rel_msids.values(),
                                   sortmsids={'aofattmd': 1, 'aopcadmd': 2,
                                              'aoacaseq': 3, 'aopsacpr': 4})

        events = []
        for manvr_prev, manvr, manvr_next in izip(states, states[1:], states[2:]):
            tstart = manvr['tstart']
            tstop = manvr['tstop']
            i0 = np.searchsorted(changes['time'], manvr_prev['tstop'])
            i1 = np.searchsorted(changes['time'], manvr_next['tstart'])
            sequence = changes[i0:i1 + 1]
            sequence['dt'] = (sequence['time'] + sequence['time0']) / 2.0 - manvr['tstop']
            ok = ((sequence['dt'] >= ZERO_DT) | (sequence['msid'] == 'aofattmd') |
                  (sequence['msid'] == 'aopcadmd'))
            sequence = sequence[ok]
            manvr_attrs = cls.get_manvr_attrs(sequence)

            event = dict(tstart=tstart,
                         tstop=tstop,
                         datestart=DateTime(tstart).date,
                         datestop=DateTime(tstop).date,
                         foreign={'ManvrSeq': sequence},
                         )
            event.update(manvr_attrs)
            event['tlm'] = cls.get_tlm(event)

            events.append(event)

        return events

    def plot(self, figsize=(8, 10), fig=None):
        from .plot import plot_manvr
        plot_manvr(self, figsize, fig)


class ManvrSeq(models.Model):
    manvr = models.ForeignKey(Manvr)
    msid = models.CharField(max_length=8)
    val0 = models.CharField(max_length=4)
    val = models.CharField(max_length=4)
    date0 = models.CharField(max_length=21)
    dt = models.FloatField()
    time0 = models.FloatField()
    time = models.FloatField()
