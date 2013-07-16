import operator
from itertools import izip

import numpy as np

from Chandra.Time import DateTime

from . import models
from .models import IntervalPad

# This gets updated dynamically by code at the end
__all__ = ['get_dates_vals', 'EventQuery']


def un_unicode(vals):
    return tuple(val.encode('ascii') if isinstance(val, unicode) else val
                 for val in vals)


def get_true_intervals(dates, vals):
    """
    For an interval-like quantity ``vals`` at ``dates``, return the
    contiguous intervals when ``vals`` is True.  "Interval-like" means
    that the transitions always occur at a discrete point in time.  E.g.::

      dates = [0,    1,    1,     2,     3    ]
      vals =  [True, True, False, False, False]

    Returns a list of 2-tuple intervals [(date0, date1), ...].
    """
    dates = np.array(dates)
    vals = np.array(vals)

    transitions = np.hstack([[True], vals[:-1] != vals[1:], [True]])

    state_vals = vals[transitions[1:]]
    datestarts = dates[transitions[:-1]]
    datestops = dates[transitions[1:]]

    intervals_iter = izip(state_vals, datestarts, datestops)
    intervals = [(date0, date1)
                 for state_val, date0, date1 in intervals_iter
                 if state_val]

    return intervals


def merge_dates_vals(vals0, dates0, dates1):
    """
    Merge dates1 into (dates0, vals0), taking care to not introduce
    duplicate dates.  In other words return a new vals and dates that
    is evaluated at the union of dates0 and dates1.
    """
    idxs = np.searchsorted(dates0, dates1)
    out_dates = dates0[:]
    out_vals = vals0[:]
    for idx, date1 in izip(idxs[::-1], dates1[::-1]):
        if dates0[idx] != date1:
            out_dates.insert(idx, date1)
            out_vals.insert(idx, vals0[idx])

    return np.array(out_dates), np.array(out_vals)


def get_dates_vals(intervals, start, stop):
    """
    Convert a list of date 2-tuple intervals into a contiguous "curve" of
    dates, vals values.  The dates and vals could be plotted.
    """
    datestart = DateTime(start).date
    datestop = DateTime(stop).date
    dates = []
    vals = []

    # If the first interval does not start exactly at datestart then add
    # a "False" interval from datestart to the beginning of the first interval
    if intervals[0][0] > datestart:
        dates.extend([datestart, intervals[0][0]])
        vals.extend([False, False])

    for interval0, interval1 in izip(intervals[:-1], intervals[1:]):
        dates.extend(list(interval0))
        vals.extend([True, True])

        dates.extend([interval0[1], interval1[0]])
        vals.extend([False, False])

    # Put in the last interval
    dates.extend(list(intervals[-1]))
    vals.extend([True, True])

    # If the last interval does not end exactly at datestop then add
    # a "False" interval from the end of the last interval to  datestop
    if intervals[-1][1] < datestop:
        dates.extend([intervals[-1][1], datestop])
        vals.extend([False, False])

    return dates, vals


def combine_intervals(op, intervals0, intervals1, start, stop):
    dates0, vals0 = get_dates_vals(intervals0, start, stop)
    dates1, vals1 = get_dates_vals(intervals1, start, stop)

    merge_dates0, merge_vals0 = merge_dates_vals(vals0, dates0, dates1)
    merge_dates1, merge_vals1 = merge_dates_vals(vals1, dates1, dates0)
    if np.any(merge_dates0 != merge_dates1):
        raise ValueError('Failure to properly merge intervals')

    intervals = get_true_intervals(merge_dates0, op(merge_vals0, merge_vals1))

    return intervals


class EventQuery(object):
    """
    High-level interface for handling event queries.

    This includes a few key methods:

    - filter() : filter events matching criteria and return Django query set
    - find() : filter events matching criteria and return np array
    - intervals(): return time intervals between event start/stop times

    A key feature is that EventQuery objects can be combined with boolean
    and, or, and not logic to generate composite EventQuery objects.  From
    there the intervals() output can be used to select or remove the intervals
    from Ska.engarchive fetch datasets.
    """

    interval_pad = IntervalPad()  # descriptor defining a Pad for intervals

    def __init__(self, cls=None, left=None, right=None, op=None):
        self.cls = cls
        self.left = left
        self.right = right
        self.op = op
        self.interval_pad = getattr(cls, 'interval_pad', None)

    @property
    def name(self):
        return self.cls.__name__

    def __and__(self, other):
        return EventQuery(left=self, right=other, op=operator.and_)

    def __or__(self, other):
        return EventQuery(left=self, right=other, op=operator.or_)

    def __invert__(self):
        return EventQuery(left=self, op=operator.not_)

    def intervals(self, start, stop):
        if self.op is not None:
            intervals0 = self.left.intervals(start, stop)
            if self.right is None:
                # This assumes any unary operator is ~.  FIX ME!
                intervals1 = [(DateTime(start).date, DateTime(stop).date)]
                return combine_intervals(operator.xor, intervals0, intervals1, start, stop)
            else:
                intervals1 = self.right.intervals(start, stop)
                return combine_intervals(self.op, intervals0, intervals1, start, stop)
        else:
            date_intervals = self.cls.get_date_intervals(start, stop, self.interval_pad)
            return date_intervals

    @property
    def table(self):
        return self.all().table

    def filter(self, start=None, stop=None, subset=None, **kwargs):
        """
        Find events between ``start`` and ``stop`` which match the filter
        attributes in ``**kwargs``.  The matching events are returned as a
        Django query set.  If ``start`` or ``stop`` are not supplied they
        default to the beginning / end of available data.  The optional
        ``subset`` arg must be a Python slice() object and allows slicing
        of the filtered output.

        Example::

          >>> from kadi import events
          >>> events.manvrs.filter('2011:001', '2012:001', n_dwell__exact=1, angle__gte=140)
          >>> events.manvrs.filter('2011:001', '2012:001', subset=slice(None, 5))  # first 5

        :param start: start time (DateTime compatible format)
        :param stop: stop time (DateTime compatible format)
        :param subset: subset of matching events that are output
        :param start: start time (DateTime compatible format)

        :returns: Django query set with matching events
        """
        cls = self.cls
        objs = cls.objects.all()
        if stop is not None:
            kwargs['start__lte'] = DateTime(stop).date
        if start is not None:
            field_names = [x.name for x in cls._meta.fields]
            attr = ('stop__gte' if 'stop' in field_names else 'start__lte')
            kwargs[attr] = DateTime(start).date
        if kwargs:
            objs = objs.filter(**kwargs)
        if subset:
            if not isinstance(subset, slice):
                raise ValueError('subset parameter must be a slice() object')
            objs = objs[subset]

        return objs

    def all(self):
        """
        Return all events as a Django query set object.

        Example::

          >>> from kadi import events
          >>> print events.safe_suns.all()
          <SafeSun: start=1999:229:20:18:22.688 dur=105043>
          <SafeSun: start=1999:250:16:31:46.461 dur=1697905>
          <SafeSun: start=2000:048:08:09:30.216 dur=68689>
          <SafeSun: start=2011:187:12:29:22.579 dur=288496>
          <SafeSun: start=2012:150:03:33:45.816 dur=118577>

          >>> print events.safe_suns.all().table
                  start                  stop             tstart        tstop          dur      notes
          --------------------- --------------------- ------------- ------------- ------------- -----
          1999:229:20:18:22.688 1999:231:01:29:05.885 51308366.8723  51413410.069 105043.196657
          1999:250:16:31:46.461 1999:270:08:10:11.850 53109170.6451 54807076.0338 1697905.38868
          2000:048:08:09:30.216 2000:049:03:14:19.260 67162234.4001 67230923.4436 68689.0435828
          2011:187:12:29:22.579 2011:190:20:37:38.914 426342628.763 426631125.098 288496.334723
          2012:150:03:33:45.816 2012:151:12:30:03.213   454649692.0 454768269.397 118577.396626
        """
        return self.filter()


# Put EventQuery objects for each query-able model class into module globals
event_models = models.get_event_models()
for model_name, model_class in event_models.items():
    query_name = model_name + 's'  # simple pluralization
    globals()[query_name] = EventQuery(cls=model_class)
    __all__.append(query_name)
