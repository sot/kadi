# Licensed under a 3-clause BSD style license - see LICENSE.rst
import operator
import warnings

import numpy as np

from Chandra.Time import DateTime

import django
django.setup()

from . import models
from .models import IntervalPad

# This gets updated dynamically by code at the end
__all__ = ['get_dates_vals', 'EventQuery']


def un_unicode(vals):
    # I think this code is orphaned and should never be called.
    warnings.warn('unexpected call to query.un_unicode', stacklevel=2)
    out = tuple(vals)

    return out


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

    intervals_iter = zip(state_vals, datestarts, datestops)
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
    for idx, date1 in zip(idxs[::-1], dates1[::-1]):
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

    # Handle the trivial case of an empty list of intervals => all False in range
    if not intervals:
        dates = [datestart, datestop]
        vals = [False, False]
        return dates, vals

    # If the first interval does not start exactly at datestart then add
    # a "False" interval from datestart to the beginning of the first interval
    if intervals[0][0] > datestart:
        dates.extend([datestart, intervals[0][0]])
        vals.extend([False, False])

    for interval0, interval1 in zip(intervals[:-1], intervals[1:]):
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
    - intervals(): return time intervals between event start/stop times

    An EventQuery object can be pre-filtered via any of the expressions
    described in the ``filter()`` doc string.  In this way the corresponding
    ``intervals()`` and fetch ``remove_intervals`` / ``select_intervals``
    outputs can be likewise filtered.

    A key feature is that EventQuery objects can be combined with boolean
    and, or, and not logic to generate composite EventQuery objects.  From
    there the intervals() output can be used to select or remove the intervals
    from Ska.engarchive fetch datasets.
    """

    interval_pad = IntervalPad()  # descriptor defining a Pad for intervals

    def __init__(self, cls=None, left=None, right=None, op=None, pad=None, **filter_kwargs):
        self.cls = cls
        self.left = left
        self.right = right
        self.op = op
        self.interval_pad = pad
        self.filter_kwargs = filter_kwargs

    def __repr__(self):
        if self.cls is None:
            op_name = {'and_': 'AND',
                       'or_': 'OR'}.get(self.op.__name__, 'UNKNOWN_OP')
            if self.right is None:
                # This assumes any unary operator is ~.  FIX ME!
                return 'NOT {}'.format(self.left)
            else:
                return '({} {} {})'.format(self.left, op_name, self.right)
        else:
            bits = ['<EventQuery: ', self.cls.__name__]
            if self.interval_pad.start != self.interval_pad.stop:
                bits.append(' pad=({}, {})'.format(self.interval_pad.start, self.interval_pad.stop))
            else:
                if self.interval_pad.start != 0:
                    bits.append(' pad={}'.format(self.interval_pad.start))
            if self.filter_kwargs:
                bits.append(
                    ' ' + ' '.join('{}={!r}'.format(k, v)
                                   for k, v in self.filter_kwargs.items()))
            bits.append('>')
            return ''.join(bits)

    def __call__(self, pad=None, **filter_kwargs):
        """
        Generate new EventQuery event for the same model class but with different pad.
        """
        return EventQuery(cls=self.cls, pad=pad, **filter_kwargs)

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
            date_intervals = self.cls.get_date_intervals(start, stop, self.interval_pad,
                                                         **self.filter_kwargs)
            return date_intervals

    @property
    def table(self):
        return self.all().table

    def filter(self, start=None, stop=None, obsid=None, subset=None, **kwargs):
        """
        Find events between ``start`` and ``stop``, or with the given ``obsid``, which
        match the filter attributes in subsequent keyword arguments.  The matching events
        are returned as a Django query set [1].

        If ``start`` or ``stop`` are not supplied they default to the beginning / end of
        available data.  The optional ``subset`` arg must be a Python slice() object and
        allows slicing of the filtered output.

        This function allows for the powerful field lookups from the underlying
        Django model implementation.  A field lookup is similar to an SQL ``WHERE``
        clause with the form ``<field_name>__<filter_type>=<value>`` (with a double
        underscore between).  For instance ``n_dwell__lte=1`` would be the same as
        ``SELECT ... WHERE n_dwell <= 1``.  Common filter types are:

        - ``exact`` (exact match), ``contains`` (contains string)
        - ``startswith``, ``endswith`` (starts or ends with string)
        - ``gt``, ``gte``, ``lt``, ``lte`` (comparisons)
        - ``isnull`` (field value is missing, e.g. manvrs.aca_proc_act_start)

        For the common case of testing equality (``exact``) there is a shortcut where
        the ``__exact`` can be skipped, so for instance ``n_dwell=1`` selects
        maneuver events with one dwell.  The full list of field lookups is at [2].

        Examples::

          >>> from kadi import events
          >>> events.manvrs.filter('2011:001', '2012:001', n_dwell=1, angle__gte=140)
          >>> events.manvrs.filter('2011:001', '2012:001', subset=slice(None, 5))  # first 5
          >>> events.manvrs.filter(obsid=14305)

        [1]: https://docs.djangoproject.com/en/1.5/topics/db/queries/
        [2]: https://docs.djangoproject.com/en/1.5/ref/models/querysets/#field-lookups

        :param start: start time (DateTime compatible format)
        :param stop: stop time (DateTime compatible format)
        :param obsid: obsid for event
        :param subset: subset of matching events that are output

        :returns: Django query set with matching events
        """
        cls = self.cls
        objs = cls.objects.all()

        # Start from self.filter_kwargs as the default and update with kwargs
        new_kwargs = self.filter_kwargs.copy()
        new_kwargs.update(kwargs)
        kwargs = new_kwargs

        if obsid is not None:
            if start or stop:
                raise ValueError('Cannot set both obsid and start or stop')

            # If obsid is set then define a filter so that the start of the event occurs
            # in the interval of the requested obsid.  First get the interval.
            obsid_events = obsids.filter(obsid__exact=obsid)
            if len(obsid_events) != 1:
                raise ValueError('Error: Found {} events matching obsid={}'
                                 .format(len(obsid_events), obsid))
            start = obsid_events[0].start
            stop = obsid_events[0].stop

            # Normally obsid_attr is 'start' so that this filter requires
            # that (event start >= obsid start) & (event start < obsid stop).
            # Notable exception in Manvr where the obsid is taken at the
            # end of the maneuver to correspond to the contained dwells.
            obsid_attr = cls._get_obsid_start_attr
            kwargs[obsid_attr + '__gte'] = start
            kwargs[obsid_attr + '__lt'] = stop

        if stop is not None:
            kwargs['start__lte'] = DateTime(stop).date

        if start is not None:
            field_names = [x.name for x in cls.get_model_fields()]
            attr = ('stop__gt' if 'stop' in field_names else 'start__gt')
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
          <SafeSun: start=1999:229:20:17:50.616 dur=105091>
          <SafeSun: start=1999:269:20:22:50.616 dur=43165>
          <SafeSun: start=2000:048:08:08:54.216 dur=68798>
          <SafeSun: start=2011:187:12:28:53.816 dur=288624>
          <SafeSun: start=2012:150:03:33:09.816 dur=118720>

          >>> print events.safe_suns.all().table
                  start                  stop            tstart      tstop      dur    notes
          --------------------- --------------------- ----------- ----------- -------- -----
          1999:229:20:17:50.616 1999:231:01:29:21.816  51308334.8  51413426.0 105091.2
          1999:269:20:22:50.616 1999:270:08:22:15.416  54764634.8  54807799.6  43164.8
          2000:048:08:08:54.216 2000:049:03:15:32.216  67162198.4  67230996.4  68798.0
          2011:187:12:28:53.816 2011:190:20:39:17.416 426342600.0 426631223.6 288623.6
          2012:150:03:33:09.816 2012:151:12:31:49.416 454649656.0 454768375.6 118719.6
        """
        return self.filter()


class LttBadEventQuery(EventQuery):
    def __call__(self, pad=None, **filter_kwargs):
        """
        Generate new EventQuery event for the same model class but with different pad.
        """
        if 'msid' in filter_kwargs:
            filter_kwargs['msid__in'] = ['*', filter_kwargs.pop('msid')]

        return EventQuery(cls=self.cls, pad=pad, **filter_kwargs)


# Put EventQuery objects for each query-able model class into module globals
obsids = None  # silence pyflakes
event_models = models.get_event_models()
for model_name, model_class in event_models.items():
    query_name = model_name + 's'  # simple pluralization
    event_query_class = LttBadEventQuery if model_name == 'ltt_bad' else EventQuery
    query_instance = event_query_class(cls=model_class)
    query_instance.__doc__ = model_class.__doc__
    globals()[query_name] = query_instance
    __all__.append(query_name)
