from itertools import count, izip

from django.db import models

import pyyaks.logger
from .manvr_templates import get_manvr_templates
from .json_field import JSONField

# Fool pyflakes into thinking these are defined
fetch = None
interpolate = None
DateTime = None
np = None

ZERO_DT = -1e-4

logger = pyyaks.logger.get_logger(name='events', level=pyyaks.logger.INFO,
                                  format="%(asctime)s %(message)s")


def _get_si(simpos):
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


def _get_start_stop_vals(event, event_msidset, msids, rel_dt):
    """
    Get the values of related telemetry MSIDs `msids` to the `event` as
    `start_<MSID>` and `stop_<MSID>`.  The value is taken from telemetry
    within `event_msidset` at event start - `rel_dt` and stop + `rel_dt`.
    Returns a dict of <MSID>: <VAL> pairs.
    """
    out = {}
    rel_msids = [event_msidset[msid] for msid in msids]
    for rel_msid in rel_msids:
        vals = interpolate(rel_msid.vals, rel_msid.times,
                           [event['tstart'] - rel_dt, event['tstop'] + rel_dt],
                           method='nearest')
        out['start_{}'.format(rel_msid.msid)] = vals[0]
        out['stop_{}'.format(rel_msid.msid)] = vals[1]

    return out


def _get_msid_changes(msids, sortmsids={}):
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
    changes = np.rec.fromrecords(changes, names=('msid', 'sortmsid', 'prev_val', 'val',
                                                 'prev_date', 'date', 'dt', 'prev_time', 'time'))
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


@import_ska
def fuzz_states(states, t_fuzz):
    """
    For a set of `states` (from fetch.MSID.state_intervals()), merge any that are
    within `t_fuzz` seconds of each other.
    """
    done = False
    while not done:
        for i, state0, state1 in izip(count(), states, states[1:]):
            if state1['tstart'] - state0['tstop'] < t_fuzz and state0['val'] == state1['val']:
                # Merge state1 into state0 and delete state1
                state0['tstop'] = state1['tstop']
                state0['datestop'] = state1['datestop']
                state0['duration'] = state0['tstop'] - state0['tstart']
                states = np.concatenate([states[:i + 1], states[i + 2:]])
                break
        else:
            done = True

    return states


class Update(models.Model):
    """
    Last telemetry which was searched for an update.
    """
    name = models.CharField(max_length=30, primary_key=True)  # model name
    date = models.CharField(max_length=21)

    def __unicode__(self):
        return ('name={} date={}'.format(self.name, self.date))


class BaseModel(models.Model):
    class Meta:
        abstract = True

    @classmethod
    def from_dict(cls, model_dict, logger=None):
        """
        Set model from a dict `model_dict` which might have extra stuff not in
        Model.  If `logger` is supplied then log output at debug level.
        """
        model = cls()
        for key, val in model_dict.items():
            if hasattr(model, key):
                if logger is not None:
                    logger.debug('Setting {} model with {}={}'
                                 .format(model.model_name, key, val))
                setattr(model, key, val)
        return model

    @property
    def model_name(self):
        if not hasattr(self, '_model_name'):
            cc_name = self.__class__.__name__
            chars = []
            for c0, c1 in izip(cc_name[:-1], cc_name[1:]):
                # Lower case followed by Upper case then insert "_"
                chars.append(c0.lower())
                if c0.lower() == c0 and c1.lower() != c1:
                    chars.append('_')
            chars.append(c1.lower())
            self._model_name = ''.join(chars)
        return self._model_name

    @classmethod
    @import_ska
    def find(cls, start=None, stop=None, subset=None, **kwargs):
        objs = cls.objects.all()
        if start is not None:
            kwargs['start__gte'] = DateTime(start).date
        if stop is not None:
            field_names = [x.name for x in cls._meta.fields]
            attr = ('stop__lte' if 'stop' in field_names else 'start__lte')
            kwargs[attr] = DateTime(stop).date
        if kwargs:
            objs = objs.filter(**kwargs)
        if subset:
            if not isinstance(subset, slice):
                raise ValueError('subset parameter must be a slice() object')
            objs = objs[subset]
        names = [f.name for f in cls._meta.fields]
        rows = objs.values_list()
        dat = np.rec.fromrecords(rows, names=names)
        return dat.view(np.ndarray)


class BaseEvent(BaseModel):
    """
    Base class for any event that gets updated in update_events.main().  Note
    that BaseModel is the base class for models like ManvrSeq that get
    generated as part of another event class.
    """
    class Meta:
        abstract = True

    lookback = 21  # days of lookback

    def __unicode__(self):
        return ('start={}'.format(self.start))


class Event(BaseEvent):
    start = models.CharField(max_length=21, primary_key=True)
    stop = models.CharField(max_length=21)
    tstart = models.FloatField(db_index=True)
    tstop = models.FloatField()
    dur = models.FloatField()

    class Meta:
        abstract = True

    def __unicode__(self):
        return ('start={} dur={:.0f}'.format(self.start, self.dur))


class TlmEvent(Event):
    event_msids = None  # must be overridden by derived class
    event_val = None

    class Meta:
        abstract = True

    def plot(self, figsize=None, fig=None):
        """
        Wrapper interface to plotting routines in plot module.  This is factored out
        of this module (models) to reduce loading of other modules (matplotlib) req'd
        for plotting.
        """
        from . import plot
        try:
            plot_func = getattr(plot, self.model_name)
        except AttributeError:
            plot_func = plot.tlm_event
        plot_func(self, figsize, fig)

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Get extra stuff for the event based on telemetry available in event_msidset.
        This is a hook within get_events() that should be overridden in individual
        classes.
        """
        return {}

    @classmethod
    @import_ska
    def get_msids_states(cls, start, stop):
        """
        Get event and related MSIDs and compute the states corresponding
        to the event.
        """
        tstart = DateTime(start).secs
        tstop = DateTime(stop).secs
        event_time_fuzz = cls.event_time_fuzz if hasattr(cls, 'event_time_fuzz') else None

        # Get the event telemetry MSID objects
        event_msidset = fetch.Msidset(cls.event_msids, tstart, tstop)

        try:
            # Telemetry values for event_msids[0] define the states
            states = event_msidset[cls.event_msids[0]].state_intervals()
        except ValueError:
            if event_time_fuzz is None:
                logger.warn('Warning: No telemetry available for {}'
                            .format(cls.__name__))
            return [], event_msidset

        # Events like Safe Sun Mode just appear in telemetry without any surrounding
        # non-event states.  In this case use a 'event_time_fuzz' class attribute t
        if event_time_fuzz:
            if (tstart + event_time_fuzz > states[0]['tstart']
                    or tstop - event_time_fuzz < states[-1]['tstop']):
                # Tstart or tstop is within event_time_fuzz of the start and stop of states so
                # bail out and don't return any states.
                logger.warn('Warning: dropping {} states because of insufficent event time pad'
                            .format(cls.__name__))
                return [], event_msidset
        else:
            # Require that the event states be flanked by a non-event state
            # to ensure that the full event was seen in telemetry.
            if states[0]['val'] == cls.event_val:
                states = states[1:]
            if states[-1]['val'] == cls.event_val:
                states = states[:-1]

        # Select event states that have the right value and are contained within interval
        ok = ((states['val'] == cls.event_val) &
              (states['tstart'] >= tstart) & (states['tstop'] <= tstop))
        states = states[ok]

        if event_time_fuzz:
            states = fuzz_states(states, event_time_fuzz)

        return states, event_msidset

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get events from telemetry defined by a simple rule that the value of
        `event_msids[0]` == `event_val`.
        """
        states, event_msidset = cls.get_msids_states(start, stop)

        # Assemble a list of dicts corresponding to events in this tlm interval
        events = []
        for state in states:
            tstart = state['tstart']
            tstop = state['tstop']
            event = dict(tstart=tstart,
                         tstop=tstop,
                         dur=tstop - tstart,
                         start=DateTime(tstart).date,
                         stop=DateTime(tstop).date)

            # Custom processing defined by subclasses to add more attrs to event
            event.update(cls.get_extras(event, event_msidset))

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
    event_msids = ['3tscmove', '3tscpos']
    event_val = 'T'

    start_3tscpos = models.IntegerField()
    stop_3tscpos = models.IntegerField()
    start_det = models.CharField(max_length=6)
    stop_det = models.CharField(max_length=6)

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define start/stop_3tscpos and start/stop_det.
        """
        out = _get_start_stop_vals(event, event_msidset, msids=['3tscpos'], rel_dt=66.0)
        out['start_det'] = _get_si(out['start_3tscpos'])
        out['stop_det'] = _get_si(out['stop_3tscpos'])
        return out

    def __unicode__(self):
        return ('start={} dur={:.0f} start_3tscpos={} stop_3tscpos={}'
                .format(self.start, self.dur, self.start_3tscpos, self.stop_3tscpos))


class FaMove(TlmEvent):
    event_msids = ['3famove', '3fapos']
    event_val = 'T'

    start_3fapos = models.IntegerField()
    stop_3fapos = models.IntegerField()

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define start/stop_3fapos.
        """
        out = _get_start_stop_vals(event, event_msidset, msids=['3fapos'], rel_dt=16.4)
        return out

    def __unicode__(self):
        return ('start={} dur={:.0f} start_3fapos={} stop_3fapos={}'
                .format(self.start, self.dur, self.start_3fapos, self.stop_3fapos))


class Dump(TlmEvent):
    event_msids = ['aounload']
    event_val = 'GRND'


class Eclipse(TlmEvent):
    event_msids = ['aoeclips']
    event_val = 'ECL '
    fetch_event_msids = ['aoeclips', 'eb1k5', 'eb2k5', 'eb3k5']


class Manvr(TlmEvent):
    event_msids = ['aofattmd', 'aopcadmd', 'aoacaseq', 'aopsacpr']
    event_val = 'MNVR'

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
    tlm = JSONField()

    def __unicode__(self):
        return ('start={} dur={:.0f} n_dwell={} template={}'
                .format(self.start, self.dur, self.n_dwell, self.template))

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
                dwell['rel_tstart'] = change['dt']
                dwell['tstart'] = change['time']
                dwell['start'] = change['date']
                state = 'dwell'

            # Another KALMAN within 400 secs of previous KALMAN in dwell.
            # This is another acquisition sequence and moves the dwell start back.
            elif (state == 'dwell'
                  and change['msid'] == 'aoacaseq'
                  and change['val'] == 'KALM'
                  and change['time'] - t0 < 400):
                t0 = change['time']
                dwell['rel_tstart'] = change['dt']
                dwell['tstart'] = change['time']
                dwell['start'] = change['date']

            # End of dwell because of NPNT => NMAN transition OR another acquisition
            elif (state == 'dwell'
                  and ((change['msid'] == 'aopcadmd' and change['val'] == 'NMAN') or
                       (change['msid'] == 'aoacaseq' and change['time'] - t0 > 400))):
                dwell['tstop'] = change['prev_time']
                dwell['stop'] = change['prev_date']
                dwell['dur'] = dwell['tstop'] - dwell['tstart']
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
        seqs = ['{}_{}_{}'.format(c['msid'], c['prev_val'], c['val']) for c in changes
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
            n_acq=len(match('aoacaseq', 'AQXN', None, 'after')),
            n_guide=len(match('aoacaseq', 'GUID', None, 'after')),
            n_kalman=len(match('aoacaseq', 'KALM', None, 'after')),
            anomalous=anomalous,
            template=template,
            )

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
        states, event_msidset = cls.get_msids_states(start, stop)
        changes = _get_msid_changes(event_msidset.values(),
                                    sortmsids={'aofattmd': 1, 'aopcadmd': 2,
                                               'aoacaseq': 3, 'aopsacpr': 4})

        events = []
        for manvr_prev, manvr, manvr_next in izip(states, states[1:], states[2:]):
            tstart = manvr['tstart']
            tstop = manvr['tstop']
            i0 = np.searchsorted(changes['time'], manvr_prev['tstop'])
            i1 = np.searchsorted(changes['time'], manvr_next['tstart'])
            sequence = changes[i0:i1 + 1]
            sequence['dt'] = (sequence['time'] + sequence['prev_time']) / 2.0 - manvr['tstop']
            ok = ((sequence['dt'] >= ZERO_DT) | (sequence['msid'] == 'aofattmd') |
                  (sequence['msid'] == 'aopcadmd'))
            sequence = sequence[ok]
            manvr_attrs = cls.get_manvr_attrs(sequence)
            dwells = cls.get_dwells(sequence)

            event = dict(tstart=tstart,
                         tstop=tstop,
                         dur=tstop - tstart,
                         start=DateTime(tstart).date,
                         stop=DateTime(tstop).date,
                         n_dwell=len(dwells),
                         foreign={'ManvrSeq': sequence,
                                  'Dwell': dwells},
                         )
            event.update(manvr_attrs)
            event['tlm'] = cls.get_tlm(event)

            events.append(event)

        return events


class Dwell(Event):
    """
    Represent an NPM dwell.  This is not derived from TlmEvent because it is created as a
    by-product of Manvr processing.
    """
    rel_tstart = models.FloatField()
    manvr = models.ForeignKey(Manvr)

    # To do: add ra dec roll quaternion

    def __unicode__(self):
        # TODO add ra, dec, roll
        return ('start={} dur={:.0f}'
                .format(self.start, self.dur))


class ManvrSeq(BaseModel):
    manvr = models.ForeignKey(Manvr)
    msid = models.CharField(max_length=8)
    prev_val = models.CharField(max_length=4)
    val = models.CharField(max_length=4)
    date = models.CharField(max_length=21)
    dt = models.FloatField()
    time = models.FloatField()
    prev_date = models.CharField(max_length=21)
    prev_time = models.FloatField()

    def __unicode__(self):
        return ('{}: {} => {} at {}'
                .format(self.msid.upper(), self.prev_val, self.val, self.date))


class SafeSun(TlmEvent):
    notes = models.TextField()

    event_msids = ['61psts02']
    event_val = 'SSM'
    event_time_fuzz = 86400  # One full day of fuzz / pad


class MajorEvent(BaseEvent):
    key = models.CharField(max_length=24, primary_key=True)
    start = models.CharField(max_length=8, db_index=True)
    date = models.CharField(max_length=11)
    tstart = models.FloatField(db_index=True)
    descr = models.TextField()
    note = models.TextField()
    source = models.CharField(max_length=3)

    def __unicode__(self):
        descr = self.descr
        if len(descr) > 30:
            descr = descr[:27] + '...'
        note = self.note
        if note:
            descr += ' (' + (note if len(note) < 30 else note[:27] + '...') + ')'
        return ('{} ({}) {}: {}'.format(self.start, self.date[5:], self.source, descr))

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get Major Events from FDB and FOT tables on the OCCweb
        """
        from . import scrape
        import hashlib

        tstart = DateTime(start).secs
        tstop = DateTime(stop).secs

        events = scrape.get_fot_major_events() + scrape.get_fdb_major_events()

        # Select events within time range and sort by tstart key
        events = sorted((x for x in events if tstart <= x['tstart'] <= tstop),
                        key=lambda x: x['tstart'])

        # Manually generate a unique key for event since date is not unique
        for event in events:
            key = ''.join(event[x] for x in ('start', 'descr', 'note', 'source'))
            event['key'] = hashlib.sha1(key).hexdigest()[:24]

        return events


class IFotEvent(BaseEvent):
    """
    Event from iFOT database
    """
    ifot_id = models.IntegerField(primary_key=True)
    start = models.CharField(max_length=21)
    stop = models.CharField(max_length=21)

    class Meta:
        abstract = True

    ifot_columns = ['id', 'tstart', 'tstop']
    ifot_props = []
    ifot_types = {}  # Override automatic type inference for properties or columns

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get Major Events from FDB and FOT tables on the OCCweb
        """
        from . import occweb

        datestart = DateTime(start).date
        datestop = DateTime(stop).date

        # def get_ifot(event_type, start=None, stop=None, props=[], columns=[], timeout=TIMEOUT):
        ifot_evts = occweb.get_ifot(cls.ifot_type_desc, start=datestart, stop=datestop,
                                    props=cls.ifot_props, columns=cls.ifot_columns,
                                    types=cls.ifot_types)

        events = []
        for ifot_evt in ifot_evts:
            event = {key.lower(): ifot_evt[cls.ifot_type_desc + '.' + key].tolist()
                     for key in cls.ifot_props}
            # Prefer start or stop from props, but if not there use column tstart/tstop
            for st in ('start', 'stop'):
                if st not in event or not event[st].strip():
                    event[st] = ifot_evt['t' + st]
            events.append(event)

        return events

    def __unicode__(self):
        return ('{}: {} {}'
                .format(self.ifot_id, self.start[:17]))


class CAP(IFotEvent):
    """
    CAP from iFOT database
    """
    num = models.CharField(max_length=15)
    title = models.TextField()
    descr = models.TextField()
    notes = models.TextField()
    link = models.CharField(max_length=250)

    ifot_type_desc = 'CAP'
    ifot_props = ['NUM', 'START', 'STOP', 'TITLE', 'LINK', 'DESC']

    def __unicode__(self):
        return ('{}: {} {}'
                .format(self.num, self.start[:17], self.title))


class DsnComm(IFotEvent):
    """
    Scheduled DSN comm period
    """
    bot = models.CharField(max_length=4)
    eot = models.CharField(max_length=4)
    activity = models.CharField(max_length=30)
    config = models.CharField(max_length=10)
    data_rate = models.CharField(max_length=9)
    site = models.CharField(max_length=12)
    soe = models.CharField(max_length=4)
    station = models.CharField(max_length=6)

    ifot_type_desc = 'DSN_COMM'
    ifot_props = ['bot', 'eot', 'activity', 'config', 'data_rate', 'site', 'soe', 'station']
    ifot_types = {'DSN_COMM.bot': 'str', 'DSN_COMM.eot': 'str'}

    def __unicode__(self):
        return ('{}: {} {}-{} {}'
                .format(self.station, self.start[:17], self.bot, self.eot, self.activity))


class Orbit(BaseEvent):
    """
    Full orbit, with dates corresponding to start (ORBIT ASCENDING NODE CROSSING), stop,
    apogee, perigee, radzone start and radzone stop.  Radzone is defined as the time
    covering perigee when radmon is disabled by command.  This corresponds to the planned
    values and may differ from actual in the case of events that run SCS107 and
    prematurely disable RADMON.

    """
    start = models.CharField(max_length=21)
    stop = models.CharField(max_length=21)
    tstart = models.FloatField(db_index=True)
    tstop = models.FloatField()
    dur = models.FloatField()
    orbit_num = models.IntegerField(primary_key=True)
    perigee = models.CharField(max_length=21)
    apogee = models.CharField(max_length=21)
    t_perigee = models.FloatField()
    start_radzone = models.CharField(max_length=21)
    stop_radzone = models.CharField(max_length=21)
    dt_start_radzone = models.FloatField()
    dt_stop_radzone = models.FloatField()

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get Orbit Events from timeline reports.
        """
        from . import orbit_funcs

        datestart = DateTime(start).date
        datestop = DateTime(stop).date
        years = sorted(set(x[:4] for x in (datestart, datestop)))
        file_dates = []
        for year in years:
            file_dates.extend(orbit_funcs.get_tlr_files(year))
        tlr_files = [x['name'] for x in file_dates if datestart <= x['date'] <= datestop]

        # Get all orbit points from the tlr files as a list of tuples
        orbit_points = orbit_funcs.get_orbit_points(tlr_files)

        # Process the points, doing various cleanup and return as a np.array
        orbit_points = orbit_funcs.process_orbit_points(orbit_points)

        # Get the orbits from the orbit points
        orbits = orbit_funcs.get_orbits(orbit_points)

        events = []
        for orbit in orbits:
            ok = orbit_points['orbit_num'] == orbit['orbit_num']
            event = {key: orbit[key] for key in orbit.dtype.names}
            event['foreign'] = {'OrbitPoint': orbit_points[ok]}
            events.append(event)

        return events

    def __unicode__(self):
        return ('{} {} dur={:.1f} radzone={:.1f} to {:.1f} ksec'
                .format(self.orbit_num, self.start[:17], self.dur / 1000,
                        self.dt_start_radzone / 1000, self.dt_stop_radzone / 1000))


class OrbitPoint(BaseModel):
    orbit = models.ForeignKey(Orbit)
    date = models.CharField(max_length=21)
    name = models.CharField(max_length=9)
    orbit_num = models.IntegerField()
    descr = models.CharField(max_length=50)

    def __unicode__(self):
        return ('{} (orbit {}) {}: {}'
                .format(self.date[:17], self.orbit_num, self.name, self.descr[:30]))
