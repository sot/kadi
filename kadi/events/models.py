import os

from itertools import count, izip

if 'DJANGO_SETTINGS_MODULE' not in os.environ:
    os.environ['DJANGO_SETTINGS_MODULE'] = 'kadi.settings'

from django.db import models

import pyyaks.logger
from .manvr_templates import get_manvr_templates

# Fool pyflakes into thinking these are defined
fetch = None
interpolate = None
DateTime = None
np = None
Quat = None

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


def _get_start_stop_vals(tstart, tstop, msidset, msids):
    """
    Get the values of related telemetry MSIDs ``msids`` to the event as ``start_<MSID>``
    and ``stop_<MSID>``.  The value is taken from telemetry within `msidset` at ``tstart``
    and ``tstop``.  Returns a dict of <MSID>: <VAL> pairs.
    """
    out = {}
    rel_msids = [msidset[msid] for msid in msids]
    for rel_msid in rel_msids:
        vals = interpolate(rel_msid.vals, rel_msid.times,
                           [tstart, tstop],
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


def get_event_models(baseclass=None):
    """
    Get all Event models that represent actual events (and are not base
    or meta classes).

    :returns: dict of {model_name:ModelClass, ...}
    """
    import inspect

    models = {}
    for name, var in globals().items():
        if inspect.isclass(var) and issubclass(var, baseclass or BaseEvent):
            # Make an instance of event class to discover if it is an abstact base class.
            event = var()
            if not event._meta.abstract:
                models[event.model_name] = var

    return models


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
        from Quaternion import Quat
        import Ska.engarchive.fetch_eng as fetch
        import numpy as np
        globals()['interpolate'] = interpolate
        globals()['DateTime'] = DateTime
        globals()['fetch'] = fetch
        globals()['np'] = np
        globals()['Quat'] = Quat
        return func(*args, **kwargs)
    return wrapper


@import_ska
def _state_intervals(vals, times):
    transitions = np.hstack([[True], vals[:-1] != vals[1:], [True]])
    t0 = times[0] - (times[1] - times[0]) / 2
    t1 = times[-1] + (times[-1] - times[-2]) / 2
    midtimes = np.hstack([[t0], (times[:-1] + times[1:]) / 2, [t1]])

    state_vals = vals[transitions[1:]]
    state_times = midtimes[transitions]

    intervals = {'datestart': DateTime(state_times[:-1]).date,
                 'datestop': DateTime(state_times[1:]).date,
                 'tstart': state_times[:-1],
                 'tstop': state_times[1:],
                 'duration': state_times[1:] - state_times[:-1],
                 'val': state_vals}

    import Ska.Numpy
    return Ska.Numpy.structured_array(intervals)


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


class Pad(object):
    def __init__(self, start=None, stop=None):
        self.start = start or 0.0
        self.stop = stop or 0.0

    def __repr__(self):
        return '<{} start={} stop={} at 0x{:x}>'.format(
            self.__class__.__name__, self.start, self.stop, id(self))


class IntervalPad(object):
    """
    Data descriptor that sets and gets an interval pad.  This pad has
    two values that are applied to the start and stop times for an interval,
    respectively.
    """

    def __get__(self, instance, owner):
        if not hasattr(instance, '_pad'):
            instance._pad = Pad(0, 0)
        return instance._pad

    def __set__(self, instance, val):
        if val is None:
            instance._pad = Pad(0, 0)
        elif isinstance(val, Pad):
            instance._pad = Pad(val.start, val.stop)
        else:
            try:
                len_val = len(val)
            except TypeError:
                instance._pad = Pad(float(val), float(val))
            else:
                if len_val == 2:
                    instance._pad = Pad(val[0], val[1])
                else:
                    raise ValueError('interval_pad must be a float scalar or 2-element list')


class Update(models.Model):
    """
    Last telemetry which was searched for an update.
    """
    name = models.CharField(max_length=30, primary_key=True)  # model name
    date = models.CharField(max_length=21)

    def __unicode__(self):
        return ('name={} date={}'.format(self.name, self.date))


class MyManager(models.Manager):
    """
    Custom query manager that allows for overriding the default __repr__.
    The purpose is to make a more user friendly output for event queries.

    http://stackoverflow.com/questions/2163151/custom-queryset-and-manager-without-breaking-dry
    https://docs.djangoproject.com/en/1.5/topics/db/managers/#custom-managers
    """
    def get_query_set(self):
        return self.model.QuerySet(self.model)


class BaseModel(models.Model):
    objects = MyManager()  # Custom manager to use custom QuerySet below

    class QuerySet(models.query.QuerySet):
        """
        More user-friendly output from event queries.
        """
        def __repr__(self):
            data = list(self[:models.query.REPR_OUTPUT_SIZE + 1])
            if len(data) > models.query.REPR_OUTPUT_SIZE:
                data[-1] = "...(remaining elements truncated)..."
            return '\n'.join(repr(x) for x in data)

        @property
        def table(self):
            def un_unicode(vals):
                return tuple(val.encode('ascii') if isinstance(val, unicode) else val
                             for val in vals)

            import numpy as np
            from astropy.table import Table

            names = [f.name for f in self.model._meta.fields]
            rows = [un_unicode(vals) for vals in self.values_list()]
            cols = (zip(*rows) if len(rows) > 0 else None)
            dat = Table(cols, names=names)

            drop_names = [name for name in dat.dtype.names if dat[name].dtype.kind == 'O']
            drop_names.extend([f.name for f in self.model._meta.fields
                               if getattr(f, '_kadi_hidden', False)])
            if drop_names:
                dat.remove_columns(drop_names)

            return dat

    class Meta:
        abstract = True
        ordering = ['start']

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
    def get_date_intervals(cls, start, stop, pad=None):
        # OPTIMIZE ME!

        # Initially get events within padded date range.  Filter on only
        # the "start" field because this is always indexed, and filtering
        # on two fields is much slower in SQL.
        if pad is None:
            pad = Pad()
        elif not isinstance(pad, Pad):
            raise TypeError('pad arg must be a Pad object')

        datestart = (DateTime(start) - cls.lookback).date
        datestop = (DateTime(stop) + cls.lookback).date
        events = cls.objects.filter(start__gte=datestart, start__lte=datestop)

        datestart = DateTime(start).date
        datestop = DateTime(stop).date

        intervals = []
        for event in events:
            event_datestart = DateTime(event.tstart - pad.start, format='secs').date
            event_datestop = DateTime(event.tstop + pad.stop, format='secs').date

            if event_datestart <= datestop and event_datestop >= datestart:
                intervals.append((max(event_datestart, datestart),
                                  min(event_datestop, datestop)))

        return intervals


class BaseEvent(BaseModel):
    """
    Base class for any event that gets updated in update_events.main().  Note
    that BaseModel is the base class for models like ManvrSeq that get
    generated as part of another event class.
    """
    class Meta:
        abstract = True
        ordering = ['start']

    lookback = 21  # days of lookback
    interval_pad = IntervalPad()  # interval padding before/ after event start/stop

    def get_commands(self):
        """
        Get load commands within start/stop interval for this event.
        Apply padding defined by interval_pad attribute.
        """
        from .. import cmds as commands
        cmds = commands.filter(self.tstart - self.interval_pad.start,
                               self.tstop + self.interval_pad.stop)
        return cmds

    def __unicode__(self):
        return ('start={}'.format(self.start))


class Event(BaseEvent):
    start = models.CharField(max_length=21, primary_key=True,
                             help_text='Start time (YYYY:DDD:HH:MM:SS)')
    stop = models.CharField(max_length=21, help_text='Stop time (YYYY:DDD:HH:MM:SS)')
    tstart = models.FloatField(db_index=True, help_text='Start time (CXC secs)')
    tstop = models.FloatField(help_text='Stop time (CXC secs)')
    dur = models.FloatField(help_text='Duration (secs)')

    class Meta:
        abstract = True
        ordering = ['start']

    def __unicode__(self):
        return ('start={} dur={:.0f}'.format(self.start, self.dur))


class TlmEvent(Event):
    event_msids = None  # must be overridden by derived class
    event_val = None

    class Meta:
        abstract = True
        ordering = ['start']

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

        # When `event_time_fuzz` is specified, e.g. for events like Safe Sun Mode
        # or normal sun mode then ensure that the end of the event is at least
        # event_time_fuzz from the end of the search interval.  If not the event
        # might be split between the current search interval and the next.  Since
        # the next search interval will step forward in time, it is sure that
        # eventually the event will be fully contained.
        if event_time_fuzz:
            if (tstop - event_time_fuzz < states[-1]['tstop']
                    and states[-1]['val'] == cls.event_val):
                # Event tstop is within event_time_fuzz of the stop of states so
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


class Obsid(TlmEvent):
    """
    Observation identifier

    **Event definition**: interval where ``COBSRQID`` is unchanged.

    **Fields**

    ======== ========== ================================
     Field      Type              Description
    ======== ========== ================================
      start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
       stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
     tstart      Float            Start time (CXC secs)
      tstop      Float             Stop time (CXC secs)
        dur      Float                  Duration (secs)
      obsid    Integer        Observation ID (COBSRQID)
    ======== ========== ================================
    """
    event_msids = ['cobsrqid']

    obsid = models.IntegerField(help_text='Observation ID (COBSRQID)')

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get obsid events from telemetry.  A event is defined by a
        contiguous interval of the telemetered obsid.
        """
        # Get the event telemetry MSID objects
        event_msidset = fetch.Msidset(cls.event_msids, start, stop)
        obsid = event_msidset['cobsrqid']
        states = obsid.state_intervals()
        events = []
        # Skip the first and last states as they are likely incomplete
        for state in states[1:-1]:
            event = dict(start=state['datestart'],
                         stop=state['datestop'],
                         tstart=state['tstart'],
                         tstop=state['tstop'],
                         dur=state['tstop'] - state['tstart'],
                         obsid=state['val'])
            events.append(event)
        return events

    def __unicode__(self):
        return ('start={} dur={:.0f} obsid={}'
                .format(self.start, self.dur, self.obsid))


class TscMove(TlmEvent):
    """
    SIM TSC translation

    **Event definition**: interval where ``3TSCMOVE = MOVE``

    In addition to reporting the start and stop TSC position, these positions are also
    converted to the corresponding science instrument detector name, one of ``ACIS-I``,
    ``ACIS-S``, ``HRC-I``, or ``HRC-S``.  The maximum PWM value ``3MRMMXMV`` (sampled at
    the stop time + 66 seconds) is also included.

    **Fields**

    =============== ========== ============================================
         Field         Type                    Description
    =============== ========== ============================================
             start   Char(21)               Start time (YYYY:DDD:HH:MM:SS)
              stop   Char(21)                Stop time (YYYY:DDD:HH:MM:SS)
            tstart      Float                        Start time (CXC secs)
             tstop      Float                         Stop time (CXC secs)
               dur      Float                              Duration (secs)
     start_3tscpos    Integer                   Start TSC position (steps)
      stop_3tscpos    Integer                    Stop TSC position (steps)
         start_det    Char(6)   Start detector (ACIS-I ACIS-S HRC-I HRC-S)
          stop_det    Char(6)    Stop detector (ACIS-I ACIS-S HRC-I HRC-S)
           max_pwm    Integer                   Max PWM during translation
    =============== ========== ============================================
    """
    event_msids = ['3tscmove', '3tscpos', '3mrmmxmv']
    event_val = 'T'

    start_3tscpos = models.IntegerField(help_text='Start TSC position (steps)')
    stop_3tscpos = models.IntegerField(help_text='Stop TSC position (steps)')
    start_det = models.CharField(max_length=6,
                                 help_text='Start detector (ACIS-I ACIS-S HRC-I HRC-S)')
    stop_det = models.CharField(max_length=6,
                                help_text='Stop detector (ACIS-I ACIS-S HRC-I HRC-S)')
    max_pwm = models.IntegerField(help_text='Max PWM during translation')

    interval_pad = IntervalPad()  # interval padding before/ after event start/stop

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define start/stop_3tscpos and start/stop_det.
        """
        out = _get_start_stop_vals(event['tstart'] - 66, event['tstop'] + 66,
                                   event_msidset, msids=['3tscpos'])
        out['start_det'] = _get_si(out['start_3tscpos'])
        out['stop_det'] = _get_si(out['stop_3tscpos'])
        pwm = event_msidset['3mrmmxmv']
        out['max_pwm'] = interpolate(pwm.vals, pwm.times, [event['tstop'] + 66])[0]
        return out

    def __unicode__(self):
        return ('start={} dur={:.0f} start_3tscpos={} stop_3tscpos={}'
                .format(self.start, self.dur, self.start_3tscpos, self.stop_3tscpos))


class Scs107(TlmEvent):
    """
    SCS107 run

    **Event definition**: interval with the following combination of state values::

      3TSCMOVE = MOVE
      AORWBIAS = DISA
      CORADMEN = DISA

    These MSIDs are first sampled onto a common time sequence of 16.4 sec samples
    so the start / stop times are accurate only to that resolution.

    Early in the mission there were two SIM TSC translations during an SCS107 run.
    By the above rules this would generate two SCS107 events, but instead any two
    SCS107 events within 600 seconds are combined into a single event.

    **Fields**

    ======== ========== ================================
     Field      Type              Description
    ======== ========== ================================
      start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
       stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
     tstart      Float            Start time (CXC secs)
      tstop      Float             Stop time (CXC secs)
        dur      Float                  Duration (secs)
      notes       Text               Supplemental notes
    ======== ========== ================================
    """
    notes = models.TextField(help_text='Supplemental notes')

    event_msids = ['3tscmove', 'aorwbias', 'coradmen']

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        msidset = fetch.MSIDset(cls.event_msids, start, stop)
        # Interpolate all MSIDs to a common time and make a common bads array
        msidset.interpolate(16.4, filter_bad=False)
        common_bads = np.zeros(len(msidset.times), dtype=bool)
        for msid in msidset.values():
            common_bads |= msid.bads

        # Apply the common bads array and filter out these bad values
        for msid in msidset.values():
            msid.bads = common_bads
            msid.filter_bad()

        scs107 = ((msidset['3tscmove'].vals == 'T') & (msidset['aorwbias'].vals == 'DISA')
                  & (msidset['coradmen'].vals == 'DISA'))
        states = _state_intervals(scs107, msidset.times)
        if states[0]['val'] is True:
            states = states[1:]
        if states[-1]['val'] is True:
            states = states[:-1]

        # Select event states that are True (SCS107 in progress)
        ok = (states['val']
              & (states['tstart'] >= DateTime(start).secs)
              & (states['tstop'] <= DateTime(stop).secs))
        states = states[ok]

        # Earlier in the mission there were two SIM translations, which generates
        # two states here.  So fuzz them together.
        states = fuzz_states(states, 600)

        # Assemble a list of dicts corresponding to events in this tlm interval
        events = []
        for state in states:
            tstart = state['tstart']
            tstop = state['tstop']
            event = dict(tstart=tstart,
                         tstop=tstop,
                         dur=tstop - tstart,
                         start=DateTime(tstart).date,
                         stop=DateTime(tstop).date,
                         notes='')

            events.append(event)

        return events


class FaMove(TlmEvent):
    """
    SIM FA translation

    **Event definition**: interval where ``3FAMOVE = MOVE``

    **Fields**

    ============== ========== ================================
        Field         Type              Description
    ============== ========== ================================
            start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
             stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
           tstart      Float            Start time (CXC secs)
            tstop      Float             Stop time (CXC secs)
              dur      Float                  Duration (secs)
     start_3fapos    Integer        Start FA position (steps)
      stop_3fapos    Integer         Stop FA position (steps)
    ============== ========== ================================
    """
    event_msids = ['3famove', '3fapos']
    event_val = 'T'

    start_3fapos = models.IntegerField(help_text='Start FA position (steps)')
    stop_3fapos = models.IntegerField(help_text='Stop FA position (steps)')

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define start/stop_3fapos.
        """
        out = _get_start_stop_vals(event['tstart'] - 16.4, event['tstop'] + 16.4,
                                   event_msidset, msids=['3fapos'])
        return out

    def __unicode__(self):
        return ('start={} dur={:.0f} start_3fapos={} stop_3fapos={}'
                .format(self.start, self.dur, self.start_3fapos, self.stop_3fapos))


class Dump(TlmEvent):
    """
    Ground commanded momentum dump

    **Event definition**: interval where ``AOUNLOAD = GRND``

    **Fields**

    ======== ========== ================================
     Field      Type              Description
    ======== ========== ================================
      start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
       stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
     tstart      Float            Start time (CXC secs)
      tstop      Float             Stop time (CXC secs)
        dur      Float                  Duration (secs)
    ======== ========== ================================
    """
    event_msids = ['aounload']
    event_val = 'GRND'


class Eclipse(TlmEvent):
    """
    Eclipse

    **Event definition**: interval where ``AOECLIPS = ECL``

    **Fields**

    ======== ========== ================================
     Field      Type              Description
    ======== ========== ================================
      start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
       stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
     tstart      Float            Start time (CXC secs)
      tstop      Float             Stop time (CXC secs)
        dur      Float                  Duration (secs)
    ======== ========== ================================
    """
    event_msids = ['aoeclips']
    event_val = 'ECL '
    fetch_event_msids = ['aoeclips', 'eb1k5', 'eb2k5', 'eb3k5']


class Manvr(TlmEvent):
    """
    Maneuver

    **Event definition**: interval where ``AOFATTMD = MNVR`` (spacecraft actually maneuvering)

    The maneuver event includes a number of attributes that give a detailed
    characterization of the timing and nature of the maneuver and corresponding
    star acquisitions and normal point model dwells.

    The ``start`` and ``stop`` time attributes for a maneuver event correspond exactly to
    the start and stop of the actual maneuver.  However, the full maneuver event
    contains information covering a larger time span from the end of the previous maneuver
    to the start of the next maneuver::

      Previous maneuver
                             <---- Start of included information
        Previous MANV end
        Previous NPNT start

        ==> Maneuver <==

        Star acquisition
        Transition to KALM
        Kalman dwell
          Optional: more dwells, star acq sequences, NMAN/NPNT

        Transition to NMAN
        Transition to MANV
                             <---- End of included information
      Next maneuver

    **Fields**

    ==================== ========== ============================================================
           Field            Type                            Description
    ==================== ========== ============================================================
                  start   Char(21)                               Start time (YYYY:DDD:HH:MM:SS)
                   stop   Char(21)                                Stop time (YYYY:DDD:HH:MM:SS)
                 tstart      Float                                        Start time (CXC secs)
                  tstop      Float                                         Stop time (CXC secs)
                    dur      Float                                              Duration (secs)
        prev_manvr_stop   Char(21)             Stop time of previous AOFATTMD=MNVR before manvr
        prev_npnt_start   Char(21)            Start time of previous AOPCADMD=NPNT before manvr
             nman_start   Char(21)                        Start time of AOPCADMD=NMAN for manvr
            manvr_start   Char(21)                        Start time of AOFATTMD=MNVR for manvr
             manvr_stop   Char(21)                         Stop time of AOFATTMD=MNVR for manvr
             npnt_start   Char(21)                      Start time of AOPCADMD=NPNT after manvr
              acq_start   Char(21)                      Start time of AOACASEQ=AQXN after manvr
            guide_start   Char(21)                      Start time of AOACASEQ=GUID after manvr
           kalman_start   Char(21)                      Start time of AOACASEQ=KALM after manvr
     aca_proc_act_start   Char(21)                       Start time of AOPSACPR=ACT after manvr
              npnt_stop   Char(21)                       Stop time of AOPCADMD=NPNT after manvr
        next_nman_start   Char(21)                 Start time of next AOPCADMD=NMAN after manvr
       next_manvr_start   Char(21)                 Start time of next AOFATTMD=MNVR after manvr
                n_dwell    Integer    Number of kalman dwells after manvr and before next manvr
                  n_acq    Integer   Number of AQXN intervals after manvr and before next manvr
                n_guide    Integer   Number of GUID intervals after manvr and before next manvr
               n_kalman    Integer   Number of KALM intervals after manvr and before next manvr
              anomalous    Boolean                             Key MSID shows off-nominal value
               template   Char(16)                                    Matched maneuver template
               start_ra      Float                           Start right ascension before manvr
              start_dec      Float                               Start declination before manvr
             start_roll      Float                                Start roll angle before manvr
                stop_ra      Float                             Stop right ascension after manvr
               stop_dec      Float                                 Stop declination after manvr
              stop_roll      Float                                  Stop roll angle after manvr
                  angle      Float                                         Maneuver angle (deg)
    ==================== ========== ============================================================

    ``n_acq``, ``n_guide``, and ``n_kalman``: these provide a count of the number of times
        after the maneuver ends that ``AOACASEQ`` changes value from anything to ``AQXN``,
        ``GUID``, and ``KALM`` respectively.

    ``anomalous``: this is ``True`` if the following MSIDs have values that are
        not in the list of nominal state values:

        ==========  ===========================
           MSID          Nominal state values
        ==========  ===========================
         AOPCADMD       NPNT NMAN
         AOACASEQ       GUID KALM AQXN
         AOFATTMD       MNVR STDY
         AOPSACPR       INIT INAC ACT
         AOUNLOAD       MON  GRND
        ==========  ===========================

    ``template``: this indicates which of the pre-defined maneuver sequence templates were
        matched by this maneuver.  For details see :ref:`maneuver_templates`.
    """
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

    interval_pad = IntervalPad()  # interval padding before/ after event start/stop

    prev_manvr_stop = models.CharField(max_length=21, null=True,
                                       help_text='Stop time of previous AOFATTMD=MNVR before manvr')
    prev_npnt_start = models.CharField(max_length=21, null=True, help_text='Start time of previous '
                                       'AOPCADMD=NPNT before manvr')
    nman_start = models.CharField(max_length=21, null=True,
                                  help_text='Start time of AOPCADMD=NMAN for manvr')
    manvr_start = models.CharField(max_length=21, null=True,
                                   help_text='Start time of AOFATTMD=MNVR for manvr')
    manvr_stop = models.CharField(max_length=21, null=True,
                                  help_text='Stop time of AOFATTMD=MNVR for manvr')
    npnt_start = models.CharField(max_length=21, null=True,
                                  help_text='Start time of AOPCADMD=NPNT after manvr')
    acq_start = models.CharField(max_length=21, null=True,
                                 help_text='Start time of AOACASEQ=AQXN after manvr')
    guide_start = models.CharField(max_length=21, null=True,
                                   help_text='Start time of AOACASEQ=GUID after manvr')
    kalman_start = models.CharField(max_length=21, null=True,
                                    help_text='Start time of AOACASEQ=KALM after manvr')
    aca_proc_act_start = models.CharField(max_length=21, null=True,
                                          help_text='Start time of AOPSACPR=ACT after manvr')
    npnt_stop = models.CharField(max_length=21, null=True,
                                 help_text='Stop time of AOPCADMD=NPNT after manvr')
    next_nman_start = models.CharField(max_length=21, null=True,
                                       help_text='Start time of next AOPCADMD=NMAN after manvr')
    next_manvr_start = models.CharField(max_length=21, null=True,
                                        help_text='Start time of next AOFATTMD=MNVR after manvr')
    n_dwell = models.IntegerField(help_text=
                                  'Number of kalman dwells after manvr and before next manvr')
    n_acq = models.IntegerField(help_text='Number of AQXN intervals after '
                                'manvr and before next manvr')
    n_guide = models.IntegerField(help_text='Number of GUID intervals after '
                                  'manvr and before next manvr')
    n_kalman = models.IntegerField(help_text='Number of KALM intervals after '
                                   'manvr and before next manvr')
    anomalous = models.BooleanField(help_text='Key MSID shows off-nominal value')
    template = models.CharField(max_length=16, help_text='Matched maneuver template')
    start_ra = models.FloatField(help_text='Start right ascension before manvr')
    start_dec = models.FloatField(help_text='Start declination before manvr')
    start_roll = models.FloatField(help_text='Start roll angle before manvr')
    stop_ra = models.FloatField(help_text='Stop right ascension after manvr')
    stop_dec = models.FloatField(help_text='Stop declination after manvr')
    stop_roll = models.FloatField(help_text='Stop roll angle after manvr')
    angle = models.FloatField(help_text='Maneuver angle (deg)')

    class Meta:
        ordering = ['start']

    def __unicode__(self):
        return ('start={} dur={:.0f} n_dwell={} template={}'
                .format(self.start, self.dur, self.n_dwell, self.template))

    @classmethod
    def get_dwells(cls, event, changes):
        """
        Get the Kalman dwells associated with a maneuver event.

        A Kalman dwell is the contiguous interval of AOACASEQ = KALM between::

          Start: AOACASEQ ==> KALM  (transition from any state to KALM)
          Stop:  AOACASEQ ==> not KALM (transition to any state other than KALM)
                        **or**
                 AOPCADMD ==> NMAN

        Short Kalman dwells that are less than 400 seconds long are *ignored* and
        are not recorded in the database.  These are typically associated with monitor
        window commanding or multiple acquisition attempts).
        """
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

        for dwell in dwells:
            for att in ('ra', 'dec', 'roll'):
                dwell[att] = event['stop_' + att]

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
            Kalman_start=match('aoacaseq', 'KALM', 0, 'after'),
            aca_proc_act_start=match('aopsacpr', 'ACT ', 0, 'after'),
            npnt_stop=match('aopcadmd', '!NPNT', -1, 'after'),

            next_nman_start=match('aopcadmd', 'NMAN', -1, 'after'),
            next_manvr_start=match('aofattmd', 'MNVR', -1, 'after'),
            n_acq=len(match('aoacaseq', 'AQXN', None, 'after')),
            n_guide=len(match('aoacaseq', 'GUID', None, 'after')),
            n_Kalman=len(match('aoacaseq', 'KALM', None, 'after')),
            anomalous=anomalous,
            template=template,
            )

        return manvr_attrs

    @classmethod
    @import_ska
    def get_target_attitudes(cls, event, msidset):
        """
        Define start/stop_aotarqt<1..4> and start/stop_ra,dec,roll
        """
        out = {}
        quats = {}
        for label, dt in (('start', -60), ('stop', 60)):
            time = event['tstart'] + dt
            q123 = []
            for i in range(1, 4):
                name = 'aotarqt{}'.format(i)
                msid = msidset[name]
                q123.append(interpolate(msid.vals, msid.times, [time], method='nearest')[0])
            q123 = np.array(q123)
            sum_q123_sq = np.sum(q123 ** 2)
            q4 = np.sqrt(np.abs(1.0 - sum_q123_sq))
            norm = np.sqrt(sum_q123_sq + q4 ** 2)
            quat = Quat(np.concatenate([q123, [q4]]) / norm)
            quats[label] = quat
            out[label + '_aotarqt1'] = float(quat.q[0])
            out[label + '_aotarqt2'] = float(quat.q[1])
            out[label + '_aotarqt3'] = float(quat.q[2])
            out[label + '_aotarqt4'] = float(quat.q[3])
            out[label + '_ra'] = float(quat.ra)
            out[label + '_dec'] = float(quat.dec)
            out[label + '_roll'] = float(quat.roll)

        dq = quats['stop'] / quats['start']  # = (wx * sa2, wy * sa2, wz * sa2, ca2)
        q3 = np.abs(dq.q[3])
        if q3 >= 1.0:  # Floating point error possible
            out['angle'] = 0.0
        else:
            out['angle'] = float(np.arccos(q3) * 2 * 180 / np.pi)

        return out

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get maneuver events from telemetry.
        """
        tarqt_msidset = fetch.Msidset(['aotarqt1', 'aotarqt2', 'aotarqt3'], start, stop)
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

            event = dict(tstart=tstart,
                         tstop=tstop,
                         dur=tstop - tstart,
                         start=DateTime(tstart).date,
                         stop=DateTime(tstop).date,
                         foreign={'ManvrSeq': sequence},
                         )
            event.update(manvr_attrs)
            event.update(cls.get_target_attitudes(event, tarqt_msidset))

            dwells = cls.get_dwells(event, sequence)
            event['foreign']['Dwell'] = dwells
            event['n_dwell'] = len(dwells)

            events.append(event)

        return events


class Dwell(Event):
    """
    Dwell in Kalman mode

    **Event definition**: contiguous interval of AOACASEQ = KALM between::

      Start: AOACASEQ ==> KALM  (transition from any state to KALM)
      Stop:  AOACASEQ ==> not KALM (transition to any state other than KALM)
                    **or**
             AOPCADMD ==> NMAN

    Short Kalman dwells that are less than 400 seconds long are *ignored* and
    are not recorded in the database.  These are typically associated with monitor
    window commanding or multiple acquisition attempts).

    **Fields**

    ============ ============ ========================================
       Field         Type                   Description
    ============ ============ ========================================
          start     Char(21)           Start time (YYYY:DDD:HH:MM:SS)
           stop     Char(21)            Stop time (YYYY:DDD:HH:MM:SS)
         tstart        Float                    Start time (CXC secs)
          tstop        Float                     Stop time (CXC secs)
            dur        Float                          Duration (secs)
     rel_tstart        Float   Start time relative to manvr end (sec)
          manvr   ForeignKey        Maneuver that contains this dwell
             ra        Float                    Right ascension (deg)
            dec        Float                        Declination (deg)
           roll        Float                         Roll angle (deg)
    ============ ============ ========================================
    """
    rel_tstart = models.FloatField(help_text='Start time relative to manvr end (sec)')
    manvr = models.ForeignKey(Manvr, help_text='Maneuver that contains this dwell')
    ra = models.FloatField(help_text='Right ascension (deg)')
    dec = models.FloatField(help_text='Declination (deg)')
    roll = models.FloatField(help_text='Roll angle (deg)')

    # To do: add ra dec roll quaternion

    def __unicode__(self):
        # TODO add ra, dec, roll
        return ('start={} dur={:.0f}'
                .format(self.start, self.dur))


class ManvrSeq(BaseModel):
    """
    Maneuver sequence event

    Each entry in this table corresponds to a state transition for an MSID
    that is relevant to the sequence of events comprising a maneuver event.

    **Fields**

    =========== ============ ===========
       Field        Type     Description
    =========== ============ ===========
         manvr   ForeignKey
          msid      Char(8)
      prev_val      Char(4)
           val      Char(4)
          date     Char(21)
            dt        Float
          time        Float
     prev_date     Char(21)
     prev_time        Float
    =========== ============ ===========
    """

    manvr = models.ForeignKey(Manvr)
    msid = models.CharField(max_length=8)
    prev_val = models.CharField(max_length=4)
    val = models.CharField(max_length=4)
    date = models.CharField(max_length=21)
    dt = models.FloatField()
    time = models.FloatField()
    prev_date = models.CharField(max_length=21)
    prev_time = models.FloatField()

    class Meta:
        ordering = ['manvr', 'date']

    def __unicode__(self):
        return ('{}: {} => {} at {}'
                .format(self.msid.upper(), self.prev_val, self.val, self.date))


class SafeSun(TlmEvent):
    """
    Safe sun event

    **Event definition**: interval when CPE PCAD mode ``61PSTS02 = SSM``

    During a safing event and recovery this MSID can toggle to different values,
    so SafeSun events within 24 hours of each other are merged.

    **Fields**

    ======== ========== ================================
     Field      Type              Description
    ======== ========== ================================
      start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
       stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
     tstart      Float            Start time (CXC secs)
      tstop      Float             Stop time (CXC secs)
        dur      Float                  Duration (secs)
      notes       Text
    ======== ========== ================================
    """
    notes = models.TextField()

    event_msids = ['61psts02']
    event_val = 'SSM'
    event_time_fuzz = 86400  # One full day of fuzz / pad


class NormalSun(TlmEvent):
    """
    Normal sun mode event

    **Event definition**: interval when PCAD mode ``AOPCADMD = NSUN``

    During a safing event and recovery this MSID can toggle to different values,
    so NormalSun events within 4 hours of each other are merged.

    **Fields**

    ======== ========== ================================
     Field      Type              Description
    ======== ========== ================================
      start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
       stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
     tstart      Float            Start time (CXC secs)
      tstop      Float             Stop time (CXC secs)
        dur      Float                  Duration (secs)
    ======== ========== ================================
    """
    event_msids = ['aopcadmd']
    event_val = 'NSUN'
    event_time_fuzz = 86400  # One full day of fuzz


class MajorEvent(BaseEvent):
    """
    Major event

    **Event definition**: events from the two lists maintained by the FOT and
      the FDB (systems engineering).

    Two lists of major event related to Chandra are available on OCCweb:

    - http://occweb.cfa.harvard.edu/occweb/web/fot_web/eng/reports/Chandra_major_events.htm
    - http://occweb.cfa.harvard.edu/occweb/web/fdb_web/Major_Events.html

    These two event lists are scraped from OCCweb and merged into a single list with a
    common structure.  Unlike most kadi event types, the MajorEvent class does not
    represent an interval of time (``start`` and ``stop``) but only has ``start``
    (YYYY:DOY) and ``date`` (YYYY-Mon-DD) attributes to indicate the time.

    **Fields**

    ======== ========== =============================================
     Field      Type                     Description
    ======== ========== =============================================
        key   Char(24)                     Unique key for this event
      start    Char(8)      Event time to the nearest day (YYYY:DOY)
       date   Char(11)   Event time to the nearest day (YYYY-Mon-DD)
     tstart      Float       Event time to the nearest day (CXC sec)
      descr       Text                             Event description
       note       Text          Note (comments or CAP # or FSW PR #)
     source    Char(3)                     Event source (FDB or FOT)
    ======== ========== =============================================
    """
    key = models.CharField(max_length=24, primary_key=True,
                           help_text='Unique key for this event')
    key._kadi_hidden = True
    start = models.CharField(max_length=8, db_index=True,
                             help_text='Event time to the nearest day (YYYY:DOY)')
    date = models.CharField(max_length=11,
                            help_text='Event time to the nearest day (YYYY-Mon-DD)')
    tstart = models.FloatField(db_index=True,
                               help_text='Event time to the nearest day (CXC sec)')
    descr = models.TextField(help_text='Event description')
    note = models.TextField(help_text='Note (comments or CAP # or FSW PR #)')
    source = models.CharField(max_length=3, help_text='Event source (FDB or FOT)')

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
    Base class for events from the iFOT database
    """
    ifot_id = models.IntegerField(primary_key=True)
    start = models.CharField(max_length=21)
    stop = models.CharField(max_length=21)
    tstart = models.FloatField(db_index=True, help_text='Start time (CXC secs)')
    tstop = models.FloatField(help_text='Stop time (CXC secs)')
    dur = models.FloatField(help_text='Duration (secs)')

    class Meta:
        abstract = True
        ordering = ['start']

    ifot_columns = ['id', 'tstart', 'tstop']
    ifot_props = []
    ifot_types = {}  # Override automatic type inference for properties or columns

    @classmethod
    @import_ska
    def get_events(cls, start, stop=None):
        """
        Get events from iFOT web interface
        """
        from .. import occweb

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
                tst = 't{}'.format(st)
                if st not in event or not event[st].strip():
                    event[st] = ifot_evt[tst]
                # The above still might not be OK because sometimes what is in the
                # <prop>.START/STOP values is not a valid date.
                try:
                    DateTime(event[st]).date
                except:
                    # Fail, roll back to the tstart/tstop version
                    logger.info('WARNING: Bad value of ifot_evt[{}.{}] = {}'
                                .format(cls.ifot_type_desc, st, event[st]))
                    event[st] = ifot_evt[tst]

            event['ifot_id'] = ifot_evt['id']
            event['tstart'] = DateTime(event['start']).secs
            event['tstop'] = DateTime(event['stop']).secs
            event['dur'] = event['tstop'] - event['tstart']
            events.append(event)

        return events

    def __unicode__(self):
        return ('{}: {} {}'
                .format(self.ifot_id, self.start[:17]))


class CAP(IFotEvent):
    """
    CAP from iFOT database

    **Event definition**: CAP from iFOT database

    **Fields**

    ========= =========== =======================
      Field       Type          Description
    ========= =========== =======================
     ifot_id     Integer
       start    Char(21)
        stop    Char(21)
      tstart       Float   Start time (CXC secs)
       tstop       Float    Stop time (CXC secs)
         dur       Float         Duration (secs)
         num    Char(15)              CAP number
       title        Text               CAP title
       descr        Text         CAP description
       notes        Text               CAP notes
        link   Char(250)                CAP link
    ========= =========== =======================
    """
    num = models.CharField(max_length=15, help_text='CAP number')
    title = models.TextField(help_text='CAP title')
    descr = models.TextField(help_text='CAP description')
    notes = models.TextField(help_text='CAP notes')
    link = models.CharField(max_length=250, help_text='CAP link')

    ifot_type_desc = 'CAP'
    ifot_props = ['NUM', 'START', 'STOP', 'TITLE', 'LINK', 'DESC']

    def __unicode__(self):
        return ('{}: {} {}'
                .format(self.num, self.start[:17], self.title))


class DsnComm(IFotEvent):
    """
    Scheduled DSN comm period

    **Event definition**: DSN comm pass beginning of support to end of support (not
      beginning / end of track).

    **Fields**

    =========== ========== ========================
       Field       Type          Description
    =========== ========== ========================
       ifot_id    Integer
         start   Char(21)
          stop   Char(21)
        tstart      Float    Start time (CXC secs)
         tstop      Float     Stop time (CXC secs)
           dur      Float          Duration (secs)
           bot    Char(4)       Beginning of track
           eot    Char(4)             End of track
      activity   Char(30)     Activity description
        config   Char(10)            Configuration
     data_rate    Char(9)                Data rate
          site   Char(12)                 DSN site
           soe    Char(4)   DSN Sequence Of Events
       station    Char(6)              DSN station
    =========== ========== ========================
    """
    bot = models.CharField(max_length=4, help_text='Beginning of track')
    eot = models.CharField(max_length=4, help_text='End of track')
    activity = models.CharField(max_length=30, help_text='Activity description')
    config = models.CharField(max_length=10, help_text='Configuration')
    data_rate = models.CharField(max_length=9, help_text='Data rate')
    site = models.CharField(max_length=12, help_text='DSN site')
    soe = models.CharField(max_length=4, help_text='DSN Sequence Of Events')
    station = models.CharField(max_length=6, help_text='DSN station')

    ifot_type_desc = 'DSN_COMM'
    ifot_props = ['bot', 'eot', 'activity', 'config', 'data_rate', 'site', 'soe', 'station']
    ifot_types = {'DSN_COMM.bot': 'str', 'DSN_COMM.eot': 'str'}

    def __unicode__(self):
        return ('{}: {} {}-{} {}'
                .format(self.station, self.start[:17], self.bot, self.eot, self.activity))


class Orbit(BaseEvent):
    """
    Orbit

    **Event definition**: single Chandra orbit starting from ascending node crossing

    Full orbit, with dates corresponding to start (ORBIT ASCENDING NODE CROSSING), stop,
    apogee, perigee, radzone start and radzone stop.  Radzone is defined as the time
    covering perigee when radmon is disabled by command.  This corresponds to the planned
    values and may differ from actual in the case of events that run SCS107 and
    prematurely disable RADMON.

    **Fields**

    ================== ========== ==================================================
          Field           Type                       Description
    ================== ========== ==================================================
                start   Char(21)         Start time (orbit ascending node crossing)
                 stop   Char(21)     Stop time (next orbit ascending node crossing)
               tstart      Float         Start time (orbit ascending node crossing)
                tstop      Float     Stop time (next orbit ascending node crossing)
                  dur      Float                               Orbit duration (sec)
            orbit_num    Integer                                       Orbit number
              perigee   Char(21)                                       Perigee time
               apogee   Char(21)                                        Apogee time
            t_perigee      Float                             Perigee time (CXC sec)
        start_radzone   Char(21)                             Start time of rad zone
         stop_radzone   Char(21)                              Stop time of rad zone
     dt_start_radzone      Float   Start time of rad zone relative to perigee (sec)
      dt_stop_radzone      Float    Stop time of rad zone relative to perigee (sec)
    ================== ========== ==================================================
    """
    start = models.CharField(max_length=21,
                             help_text='Start time (orbit ascending node crossing)')
    stop = models.CharField(max_length=21,
                            help_text='Stop time (next orbit ascending node crossing)')
    tstart = models.FloatField(db_index=True,
                               help_text='Start time (orbit ascending node crossing)')
    tstop = models.FloatField(help_text='Stop time (next orbit ascending node crossing)')
    dur = models.FloatField(help_text='Orbit duration (sec)')
    orbit_num = models.IntegerField(primary_key=True, help_text='Orbit number')
    perigee = models.CharField(max_length=21, help_text='Perigee time')
    apogee = models.CharField(max_length=21, help_text='Apogee time')
    t_perigee = models.FloatField(help_text='Perigee time (CXC sec)')
    start_radzone = models.CharField(max_length=21, help_text='Start time of rad zone')
    stop_radzone = models.CharField(max_length=21, help_text='Stop time of rad zone')
    dt_start_radzone = models.FloatField(help_text='Start time of rad zone relative '
                                         'to perigee (sec)')
    dt_stop_radzone = models.FloatField(help_text='Stop time of rad zone relative '
                                        'to perigee (sec)')

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
            event['foreign'] = {'OrbitPoint': orbit_points[ok],
                                'RadZone': [orbit_funcs.get_radzone_from_orbit(orbit)]}
            events.append(event)

        return events

    def __unicode__(self):
        return ('{} {} dur={:.1f} radzone={:.1f} to {:.1f} ksec'
                .format(self.orbit_num, self.start[:17], self.dur / 1000,
                        self.dt_start_radzone / 1000, self.dt_stop_radzone / 1000))


class OrbitPoint(BaseModel):
    """
    Orbit point

    **Fields**

    =========== ============ ===========
       Field        Type     Description
    =========== ============ ===========
         orbit   ForeignKey
          date     Char(21)
          name      Char(9)
     orbit_num      Integer
         descr     Char(50)
    =========== ============ ===========
    """
    orbit = models.ForeignKey(Orbit)
    date = models.CharField(max_length=21)
    name = models.CharField(max_length=9)
    orbit_num = models.IntegerField()
    descr = models.CharField(max_length=50)

    class Meta:
        ordering = ['date']

    def __unicode__(self):
        return ('{} (orbit {}) {}: {}'
                .format(self.date[:17], self.orbit_num, self.name, self.descr[:30]))


class RadZone(Event):
    """
    Radiation zone

    **Fields**

    =========== ============ ================================
       Field        Type               Description
    =========== ============ ================================
         start     Char(21)   Start time (YYYY:DDD:HH:MM:SS)
          stop     Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
        tstart        Float            Start time (CXC secs)
         tstop        Float             Stop time (CXC secs)
           dur        Float                  Duration (secs)
         orbit   ForeignKey
     orbit_num      Integer
       perigee     Char(21)
    =========== ============ ================================
    """
    orbit = models.ForeignKey(Orbit)
    orbit_num = models.IntegerField()
    perigee = models.CharField(max_length=21)

    def __unicode__(self):
        return ('{} {} {} dur={:.1f} ksec'
                .format(self.orbit_num, self.start[:17], self.stop[:17],
                        self.dur / 1000))
