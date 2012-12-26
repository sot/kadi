
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


class TscMove(Event):
    name = 'tsc_move'
    tlm_msids = ['3tscmove', '3tscpos']

    tscpos_start = models.IntegerField()
    tscpos_stop = models.IntegerField()
    det_start = models.CharField(max_length=6)
    det_stop = models.CharField(max_length=6)

    @import_ska
    @staticmethod
    def get_events(tstart, tstop):
        tstart = DateTime(tstart).secs
        tstop = DateTime(tstop).secs
        msids = fetch.Msidset(TscMove.tlm_msids, tstart, tstop)
        pos = msids['3tscpos']
        states = msids['3tscmove'].state_intervals()

        # Require that the processed moves be flanked by a non-move state
        # to ensure that the full move was seen in telemetry.
        if states[0]['val'] == 'T':
            states = states[1:]
        if states[-1]['val'] == 'T':
            states = states[:-1]

        # Select states with val=T (moving) that are entirely contained within interval
        ok = (states['val'] == 'T') & (states['tstart'] >= tstart) & (states['tstop'] <= tstop)
        states = states[ok]

        for state in states:
            tstart = state['tstart']
            tstop = state['tstop']
            tscposs = interpolate(pos.vals, pos.times, [tstart - 70, tstop + 70],
                                  method='nearest')
            dets = [get_si(tscpos) for tscpos in tscposs]
            ts = {'t{}'.format(i): 10000.0 for i in range(40)}
            tsc_move = TscMove(tstart=tstart,
                               tstop=tstop,
                               datestart=DateTime(tstart).date,
                               datestop=DateTime(tstop).date,
                               tscpos_start=tscposs[0],
                               tscpos_stop=tscposs[1],
                               det_start=dets[0],
                               det_stop=dets[1],
                               )
            print 'Saving {}'.format(tsc_move)
            tsc_move.save()
            tsc_move.tscmovetimes_set.create(**ts)


class TscMoveTimes(models.Model):
    tscmove = models.ForeignKey(TscMove)
    t0 = models.FloatField()
    t1 = models.FloatField()
    t2 = models.FloatField()
    t3 = models.FloatField()
    t4 = models.FloatField()
    t5 = models.FloatField()
    t6 = models.FloatField()
    t7 = models.FloatField()
    t8 = models.FloatField()
    t9 = models.FloatField()
    t10 = models.FloatField()
    t11 = models.FloatField()
    t12 = models.FloatField()
    t10 = models.FloatField()
    t11 = models.FloatField()
    t12 = models.FloatField()
    t13 = models.FloatField()
    t14 = models.FloatField()
    t15 = models.FloatField()
    t16 = models.FloatField()
    t17 = models.FloatField()
    t18 = models.FloatField()
    t19 = models.FloatField()
    t20 = models.FloatField()
    t21 = models.FloatField()
    t22 = models.FloatField()
    t23 = models.FloatField()
    t24 = models.FloatField()
    t25 = models.FloatField()
    t26 = models.FloatField()
    t27 = models.FloatField()
    t28 = models.FloatField()
    t29 = models.FloatField()
    t30 = models.FloatField()
    t31 = models.FloatField()
    t32 = models.FloatField()
    t33 = models.FloatField()
    t34 = models.FloatField()
    t35 = models.FloatField()
    t36 = models.FloatField()
    t37 = models.FloatField()
    t38 = models.FloatField()
    t39 = models.FloatField()


class FaMove(models.Model):
    name = 'fa_move'
    tlm_msids = ['3famov', '3fapos']

    tstart = models.FloatField()
    tstop = models.FloatField()
    start_3fapos = models.IntegerField()
    stop_3fapos = models.IntegerField()
