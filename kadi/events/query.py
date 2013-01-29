import inspect
import operator
from itertools import izip

import numpy as np

from Chandra.Time import DateTime

from . import models

__all__ = []  # this gets updated dynamically by code at the end


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
    def __init__(self, cls=None, left=None, right=None, op=None):
        self.cls = cls
        self.left = left
        self.right = right
        self.op = op
        self._interval_pad = getattr(cls, 'interval_pad', (0.0, 0.0))

    @property
    def name(self):
        return self.cls.__name__

    def __and__(self, other):
        return EventQuery(left=self, right=other, op=operator.and_)

    def __or__(self, other):
        return EventQuery(left=self, right=other, op=operator.or_)

    def __invert__(self):
        return EventQuery(left=self, op=operator.not_)

    @property
    def interval_pad(self):
        return self._interval_pad

    @interval_pad.setter
    def interval_pad(self, val):
        try:
            len_val = len(val)
        except TypeError:
            self._interval_pad = (float(val), float(val))
        else:
            if len_val == 2:
                self._interval_pad = tuple(val)
            else:
                raise ValueError('interval_pad must be a float scalar or 2-element list')

    def intervals(self, start, stop):
        if self.op is not None:
            intervals0 = self.left.intervals(start, stop)
            if self.right is None:
                return intervals0
            else:
                intervals1 = self.right.intervals(start, stop)
                return combine_intervals(self.op, intervals0, intervals1, start, stop)
        else:
            date_intervals = self.cls.get_date_intervals(start, stop, self.interval_pad)
            return date_intervals

# Put EventQuery objects for each query-able model class into module globals
for name, var in vars(models).items():
    if inspect.isclass(var) and issubclass(var, models.BaseEvent):
        event = var()  # make an instance of the event class
        if not event._meta.abstract:
            query_name = event.model_name + 's'  # simple pluralization
            globals()[query_name] = EventQuery(cls=var)
            __all__.append(query_name)
