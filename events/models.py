
from django.db import models

# Fool pyflakes into thinking these are defined
fetch = None
interpolate = None
DateTime = None


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
        globals()['interpolate'] = interpolate
        globals()['DateTime'] = DateTime
        globals()['fetch'] = fetch
        return func(*args, **kwargs)
    return wrapper


class Event(models.Model):
    tstart = models.FloatField()
    tstop = models.FloatField()
    datestart = models.CharField(max_length=21)
    datestop = models.CharField(max_length=21)

    class Meta:
        abstract = True


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
    @import_ska
    def get_events(cls, tstart, tstop):
        """
        Get events from telemetry defined by a simple rule that the value of
        `event_msid` == `event_val`.  Related MSIDs in the list `rel_msid`
        are fetched and put into the event at the time `rel_dt` seconds
        before the event start and after the event end.
        """
        tstart = DateTime(tstart).secs
        tstop = DateTime(tstop).secs

        # Get the related MSIDs
        rel_msids = fetch.Msidset(cls.rel_msids, tstart, tstop) if cls.rel_msids else []

        # Get the MSID that defines the event and find state intervals on that MSID
        event_msid = fetch.Msid(cls.event_msid, tstart, tstop)
        states = event_msid.state_intervals()

        # Require that the event states be flanked by a non-event state
        # to ensure that the full event was seen in telemetry.
        if states[0]['val'] == cls.event_val:
            states = states[1:]
        if states[-1]['val'] == cls.event_val:
            states = states[:-1]

        # import pdb; pdb.set_trace()

        # Select event states that are entirely contained within interval
        ok = ((states['val'] == cls.event_val) &
              (states['tstart'] >= tstart) & (states['tstop'] <= tstop))
        states = states[ok]

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


class FaMove(TlmEvent):
    name = 'fa_move'
    event_msid = '3famove'
    event_val = 'T'
    rel_msids = ['3fapos']
    rel_dt = 16.4  # 1/2 major frame (FA moves can be within 2 minutes)

    start_3fapos = models.IntegerField()
    stop_3fapos = models.IntegerField()
