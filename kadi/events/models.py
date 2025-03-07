# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import operator
import sys
from itertools import count
from pathlib import Path

import cheta.fetch_eng as fetch
import numpy as np
from astropy import table
from chandra_time import DateTime
from cheta import utils
from django.db import models
from Quaternion import Quat
from ska_numpy import interpolate

from .manvr_templates import get_manvr_templates

models.query.REPR_OUTPUT_SIZE = 1000  # Increase default number of rows printed

ZERO_DT = -1e-4
MAX_GAP = 328.1  # Max gap (seconds) in telemetry for state intervals
R2A = 206264.8  # Convert from radians to arcsec

logger = logging.getLogger(__name__)


def msidset_interpolate(msidset, dt, time0):
    """
    Interpolate the ``msidset`` in-place to a common time basis, starting at ``time0``
    and stepping by ``dt``.  This assumes an unfiltered MSIDset, and returns
    a filtered MSIDset.
    """
    tstart = (time0 // dt) * dt
    times = np.arange((msidset.tstop - tstart) // dt) * dt + tstart
    msidset.interpolate(times=times, filter_bad=False)
    common_bads = np.zeros(len(msidset.times), dtype=bool)
    for msid in msidset.values():
        common_bads |= msid.bads

    # Apply the common bads array and filter out these bad values
    for msid in msidset.values():
        msid.bads = common_bads
        msid.filter_bad()
    msidset.times = msidset.times[~common_bads]


def _get_si(simpos):
    """
    Get SI corresponding to the given SIM position.
    """
    if (simpos >= 82109) and (simpos <= 104839):
        si = "ACIS-I"
    elif (simpos >= 70736) and (simpos <= 82108):
        si = "ACIS-S"
    elif (simpos >= -86147) and (simpos <= -20000):
        si = " HRC-I"
    elif (simpos >= -104362) and (simpos <= -86148):
        si = " HRC-S"
    else:
        si = "  NONE"
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
        vals = interpolate(
            rel_msid.vals, rel_msid.times, [tstart, tstop], method="nearest"
        )
        out["start_{}".format(rel_msid.msid)] = vals[0]
        out["stop_{}".format(rel_msid.msid)] = vals[1]

    return out


def _get_msid_changes(msids, sortmsids=None):
    """
    For the list of fetch MSID objects, return a sorted structured array
    of each time any MSID value changes.
    """
    if sortmsids is None:
        sortmsids = {}
    changes = []
    for msid in msids:
        i_changes = np.flatnonzero(msid.vals[1:] != msid.vals[:-1])
        for i in i_changes:
            change = (
                msid.msid,
                sortmsids.get(msid.msid, 10),
                msid.vals[i],
                msid.vals[i + 1],
                DateTime(msid.times[i]).date,
                DateTime(msid.times[i + 1]).date,
                0.0,
                msid.times[i],
                msid.times[i + 1],
            )
            changes.append(change)
    changes = np.rec.fromrecords(
        changes,
        names=(
            "msid",
            "sortmsid",
            "prev_val",
            "val",
            "prev_date",
            "date",
            "dt",
            "prev_time",
            "time",
        ),
    )
    changes.sort(order=["time", "sortmsid"])
    return changes


def get_event_models(baseclass=None):
    """
    Get all Event models that represent actual events (and are not base
    or meta classes).

    Returns
    -------
    dict of {model_name:ModelClass, ...}
    """
    import inspect

    models = {}
    for var in globals().values():
        if inspect.isclass(var) and issubclass(var, baseclass or BaseEvent):
            # Make an instance of event class to discover if it is an abstact base class.
            event = var()
            if not event._meta.abstract:
                models[event.model_name] = var

    return models


def fuzz_states(states, t_fuzz):
    """
    For a set of `states` (from fetch.MSID.state_intervals()) or intervals (from
    fetch.MSID.logical_intervals), merge any that are within `t_fuzz` seconds of each
    other.  Logical intervals are just the subset of states with 'val' equal to a
    particular value.

    Parameters
    ----------
    states
        table of states or intervals
    t_fuzz : fuzz time in seconds
        :returns fuzzed_states: table
    """
    done = False
    state_has_val = "val" in states.dtype.names
    while not done:
        for i, state0, state1 in zip(count(), states, states[1:]):
            # Logical intervals all have a 'val' of True by definition, while for state
            # intervals we need to check that adjacent intervals have same value.
            state_equal = state0["val"] == state1["val"] if state_has_val else True
            if state1["tstart"] - state0["tstop"] < t_fuzz and state_equal:
                # Merge state1 into state0 and delete state1
                state0["tstop"] = state1["tstop"]
                state0["datestop"] = state1["datestop"]
                state0["duration"] = state0["tstop"] - state0["tstart"]
                states = table.vstack([states[: i + 1], states[i + 2 :]])
                break
        else:
            done = True

    return states


class Pad(object):
    """
    Time padding at the start and stop of an interval.

    Positive values always make the interval *bigger* in each direction, so
    a pad of 300 seconds makes the interval a total of 10 minutes longer (5 minutes
    on each side).  A pad of -300 seconds makes the interval start 5 minutes
    later and end 5 minutes earlier.
    """

    def __init__(self, start=None, stop=None):
        self.start = start or 0.0
        self.stop = stop or 0.0

    def __repr__(self):
        return "<{} start={} stop={} seconds>".format(
            self.__class__.__name__, self.start, self.stop
        )


class IntervalPad(object):
    """
    Data descriptor that sets and gets an interval pad.  This pad has
    two values that are applied to the start and stop times for an interval,
    respectively.
    """

    def __get__(self, instance, owner):
        if not hasattr(instance, "_pad"):
            instance._pad = Pad(0, 0)
        return instance._pad

    def __set__(self, instance, val):
        if val is None:
            val_start = val_stop = 0
        elif isinstance(val, Pad):
            val_start, val_stop = val.start, val.stop
        else:
            try:
                len_val = len(val)
            except TypeError:
                val_start = val_stop = val
            else:
                if len_val == 0:
                    val_start = val_stop = 0
                elif len_val == 1:
                    val_start = val_stop = val[0]
                elif len_val == 2:
                    val_start, val_stop = val[0], val[1]
                else:
                    raise ValueError(
                        "interval_pad must be a float scalar, or 1 or 2-element list"
                    )

        instance._pad = Pad(float(val_start), float(val_stop))


class Update(models.Model):
    """
    Last telemetry which was searched for an update.
    """

    name = models.CharField(max_length=30, primary_key=True)  # model name
    date = models.CharField(max_length=21)

    def __unicode__(self):
        return "name={} date={}".format(self.name, self.date)


class MyManager(models.Manager):
    """
    Custom query manager that allows for overriding the default __repr__.
    The purpose is to make a more user friendly output for event queries.

    http://stackoverflow.com/questions/2163151/custom-queryset-and-manager-without-breaking-dry
    https://docs.djangoproject.com/en/3.1/topics/db/managers/#custom-managers
    """

    def get_query_set(self):
        return self.model.QuerySet(self.model)


class BaseModel(models.Model):
    """
    Base class for for all models.
    """

    _get_obsid_start_attr = "date"  # Attribute to use for getting event obsid
    update_priority = 0  # Priority order in update processing (higher => earlier)

    class QuerySet(models.query.QuerySet):
        """
        More user-friendly output from event queries.
        """

        def __repr__(self):
            data = list(self[: models.query.REPR_OUTPUT_SIZE + 1])
            if len(data) > models.query.REPR_OUTPUT_SIZE:
                data[-1] = "...(remaining elements truncated)..."
            return "\n".join(repr(x) for x in data)

        @property
        def table(self):
            def un_unicode(vals):
                return tuple(
                    val.encode("ascii") if isinstance(val, str) else val for val in vals
                )

            from astropy.table import Table

            model_fields = self.model.get_model_fields()
            names = [f.name for f in model_fields]
            rows = self.values_list()
            cols = list(zip(*rows)) if len(rows) > 0 else None
            dat = Table(cols, names=names)

            drop_names = [
                name for name in dat.dtype.names if dat[name].dtype.kind == "O"
            ]
            drop_names.extend(
                [f.name for f in model_fields if getattr(f, "_kadi_hidden", False)]
            )
            if drop_names:
                dat.remove_columns(drop_names)

            return dat

        def select_overlapping(self, query_event, allow_partial=True):
            """
            Select events which overlap with the specified ``query_event``.

            By default partial overlap between the self events and the ``query_event``
            intervals is sufficient.  However, if ``allow_partial=False``, then
            complete overlap is required.  As an example this would be selecting
            maneuvers that are *entirely* within the radiation zone.

            Examples::

              >>> from kadi import events
              >>> manvrs = events.manvrs.filter('2001:001:00:00:00', '2001:003:00:00:00')
              >>> non_rad_manvrs = manvrs.select_overlapping(~events.rad_zones)
              >>> rad_manvrs = manvrs.select_overlapping(events.rad_zones)
              >>> fully_radzone_manvrs = manvrs.select_overlapping(events.rad_zones,
              ...                                                  allow_partial=False)

            :param query_event: QueryEvent object (e.g. events.tsc_moves or a composite
                                boolean expression)
            :param allow_partial: return events where there is at least partial overlap
                            (default=True)

            :returns: list of overlapping events
            """
            from chandra_time import DateTime

            from .query import combine_intervals

            # First find the intervals corresponding to overlaps between self events and
            # `query_event`.  Do this over an interval that covers an extra 30 days on
            # either end, assuming that is the longest possible event.  (This padding might
            # not actually be necessary, but it doesn't hurt too much).
            events = list(self)
            start = DateTime(events[0].tstart) - 30
            stop = DateTime(events[-1].tstop) + 30
            datestarts = [event.start for event in events]
            datestops = [event.stop for event in events]
            intervals = list(zip(datestarts, datestops))
            qe_intervals = query_event.intervals(start, stop)
            overlap_intervals = combine_intervals(
                operator.and_, intervals, qe_intervals, start, stop
            )

            overlap_starts, overlap_stops = list(zip(*overlap_intervals))

            datestarts = np.array(datestarts, dtype="S21")
            datestops = np.array(datestops, dtype="S21")
            overlap_starts = np.array(overlap_starts, dtype="S21")
            overlap_stops = np.array(overlap_stops, dtype="S21")

            func = (
                self._get_partial_overlaps if allow_partial else self._get_full_overlaps
            )
            indices = func(overlap_starts, overlap_stops, datestarts, datestops)
            return [events[i] for i in indices]

        @staticmethod
        def _get_full_overlaps(overlap_starts, overlap_stops, datestarts, datestops):
            """
            Find which of the new overlap_intervals are the same as an original interval,
            indicating that there is a full overlap.
            """
            if len(datestops) == 0 or len(overlap_stops) == 0:
                return np.array([])

            # Find the indices of matching datestart/stops for the overlap start/stop times.
            indices = np.searchsorted(datestops, overlap_stops)

            match_datestarts = datestarts[indices]
            match_datestops = datestops[indices]

            ok = (overlap_starts == match_datestarts) & (
                overlap_stops == match_datestops
            )
            return indices[ok]

        @staticmethod
        def _get_partial_overlaps(overlap_starts, overlap_stops, datestarts, datestops):
            """
            Find which of the new overlap_intervals are the same as an original interval,
            indicating that there is a full overlap.
            """
            if len(datestops) == 0 or len(overlap_stops) == 0:
                return np.array([])

            indices = np.searchsorted(datestops, overlap_starts)
            indices = np.unique(indices)

            return indices

    try:
        # Django >= 1.8 (which is req'd for Py3)
        # https://docs.djangoproject.com/en/1.10/topics/db/managers/#creating-a-manager-with-queryset-methods
        objects = QuerySet.as_manager()  # Custom manager to use custom QuerySet below
    except AttributeError:
        objects = MyManager()

    class Meta:
        abstract = True
        ordering = ["start"]

    @classmethod
    def from_dict(cls, model_dict, logger=None) -> "BaseModel":
        """
        Set model from a dict `model_dict` which might have extra stuff not in
        Model.  If `logger` is supplied then log output at debug level.

        Parameters
        ----------
        model_dict : dict
            Dictionary of model attributes
        logger : logging.Logger, optional
            Logger object to log debug output

        Returns
        -------
        model : BaseModel
            Model instance
        """
        model = cls()

        for key, val in model_dict.items():
            if hasattr(model, key):
                if logger is not None:
                    logger.debug(
                        "Setting {} model with {}={}".format(model.model_name, key, val)
                    )
                setattr(model, key, val)

        return model

    @classmethod
    def get_model_fields(cls):
        """
        Return a list of model fields (works from class or instance).
        """
        return cls._meta.fields

    @property
    def model_name(self):
        if not hasattr(self, "_model_name"):
            cc_name = self.__class__.__name__
            chars = []
            for c0, c1 in zip(cc_name[:-1], cc_name[1:]):
                # Lower case followed by Upper case then insert "_"
                chars.append(c0.lower())
                if c0.lower() == c0 and c1.lower() != c1:
                    chars.append("_")
            chars.append(c1.lower())
            self._model_name = "".join(chars)
        return self._model_name

    @classmethod
    def get_date_intervals(cls, start, stop, pad=None, **filter_kwargs):
        # OPTIMIZE ME!

        # Initially get events within padded date range.  Filter on only
        # the "start" field because this is always indexed, and filtering
        # on two fields is much slower in SQL.
        if pad is None:
            pad = Pad()
        elif not isinstance(pad, Pad):
            raise TypeError("pad arg must be a Pad object")

        datestart = (DateTime(start) - cls.lookback).date
        datestop = (DateTime(stop) + cls.lookback).date
        events = cls.objects.filter(
            start__gte=datestart, start__lte=datestop, **filter_kwargs
        )

        datestart = DateTime(start).date
        datestop = DateTime(stop).date

        intervals = []
        for event in events:
            event_datestart = DateTime(event.tstart - pad.start, format="secs").date
            event_datestop = DateTime(event.tstop + pad.stop, format="secs").date

            # Negative padding might make an interval entirely disappear
            if event_datestart >= event_datestop:
                continue

            if event_datestart <= datestop and event_datestop >= datestart:
                interval_datestart = max(event_datestart, datestart)
                interval_datestop = min(event_datestop, datestop)

                # If there is a previous interval and the end of the previous interval
                # is after the start of this new interval, then merge the two intervals
                if intervals and intervals[-1][1] >= interval_datestart:
                    intervals[-1] = (intervals[-1][0], interval_datestop)
                else:
                    # Otherwise just create a new interval
                    intervals.append((interval_datestart, interval_datestop))

        return intervals

    def get_obsid(self):
        """
        Return the obsid associated with the event.

        Typically this is the obsid at the start of the event, but for maneuvers it is the
        obsid at the end of the maneuver.

        Returns
        -------
        obsid
        """
        from . import query

        # Get the start of the event.  If derived from Event or BaseEvent then
        # self will have a start attr.  Otherwise it must be from BaseModel in
        # which case it will have a date attr.
        start = getattr(self, self._get_obsid_start_attr)
        obsids = query.obsids.filter(start, start)
        if len(obsids) != 1:
            raise ValueError(
                "Expected one obsid at {} but got {}".format(start, obsids)
            )
        return obsids[0].obsid

    def __bytes__(self):
        return str(self).encode("utf-8")

    def __str__(self):
        return self.__unicode__()


class BaseEvent(BaseModel):
    """
    Base class for any event that gets updated in update_events.main().  Note
    that BaseModel is the base class for models like ManvrSeq that get
    generated as part of another event class.
    """

    class Meta:
        abstract = True
        ordering = ["start"]

    _get_obsid_start_attr = "start"  # Attribute to use for getting event obsid
    lookback = 21  # days of lookback
    interval_pad = IntervalPad()  # interval padding before/ after event start/stop

    def get_commands(self):
        """
        Get load commands within start/stop interval for this event.
        Apply padding defined by interval_pad attribute.
        """
        from kadi import cmds as commands

        cmds = commands.filter(
            self.tstart - self.interval_pad.start, self.tstop + self.interval_pad.stop
        )
        return cmds

    def __unicode__(self):
        return "start={}".format(self.start)

    def get_next(self, queryset=None):
        """
        Get the next object by primary key order
        """
        if queryset is None:
            queryset = self.__class__.objects.all()
        next = queryset.filter(pk__gt=self.pk)
        try:
            return next[0]
        except IndexError:
            return False

    def get_previous(self, queryset=None):
        """
        Get the previous object by primary key order
        """
        if queryset is None:
            queryset = self.__class__.objects.all()
        prev = queryset.filter(pk__lt=self.pk).order_by("-pk")
        try:
            return prev[0]
        except IndexError:
            return False


class Event(BaseEvent):
    start = models.CharField(
        max_length=21, primary_key=True, help_text="Start time (YYYY:DDD:HH:MM:SS)"
    )
    stop = models.CharField(max_length=21, help_text="Stop time (YYYY:DDD:HH:MM:SS)")
    tstart = models.FloatField(db_index=True, help_text="Start time (CXC secs)")
    tstop = models.FloatField(help_text="Stop time (CXC secs)")
    dur = models.FloatField(help_text="Duration (secs)")
    dur._kadi_format = "{:.1f}"

    class Meta:
        abstract = True
        ordering = ["start"]

    def __unicode__(self):
        return "start={} dur={:.0f}".format(self.start, self.dur)


class TlmEvent(Event):
    obsid = models.IntegerField(help_text="Observation ID (COBSRQID)")

    event_msids = None  # must be overridden by derived class
    event_val = None
    event_filter_bad = True  # Normally remove bad quality data immediately
    aux_msids = None  # Additional MSIDs to fetch for event

    class Meta:
        abstract = True
        ordering = ["start"]

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
    def get_state_times_bools(cls, event_msidset):
        """
        Get the boolean True/False array indicating when ``event_msid`` is in the
        desired state for this event type.  The default is when
        ``event_msid == cls.event_val``, but subclasses may override this method.

        Parameters
        ----------
        event_msid
            fetch.MSID object

        Returns
        -------
        boolean ndarray
        """
        event_msid = event_msidset[cls.event_msids[0]]
        vals = event_msid.vals
        if vals.dtype.kind == "U":
            # Strip leading/trailing whitespace from string values. This is needed
            # because the CXC includes trailing whitespace in the telemetry values while
            # MAUDE (more sensibly) does not.
            vals = np.char.strip(vals)
        bools = (
            np.isin(vals, cls.event_val)
            if isinstance(cls.event_val, list)
            else vals == cls.event_val
        )
        return event_msid.times, bools

    @classmethod
    def get_msids_states(cls, start, stop):
        """
        Get event and related MSIDs and compute the states corresponding
        to the event.
        """
        tstart = DateTime(start).secs
        tstop = DateTime(stop).secs
        event_time_fuzz = (
            cls.event_time_fuzz if hasattr(cls, "event_time_fuzz") else None
        )

        # Get the event telemetry MSID objects
        event_msidset = fetch.MSIDset(
            cls.event_msids, tstart, tstop, filter_bad=cls.event_filter_bad
        )

        try:
            # Telemetry values for event_msids[0] define the states.  Don't allow a logical
            # interval that spans a telemetry gap of more than 10 major frames.
            times, bools = cls.get_state_times_bools(event_msidset)
            states = utils.logical_intervals(
                times, bools, max_gap=MAX_GAP, complete_intervals=True
            )
        except (IndexError, ValueError):
            if event_time_fuzz is None:
                logger.warning(
                    "Warning: No telemetry available for {}".format(cls.__name__)
                )
            return [], event_msidset

        if len(states) > 0:
            # When `event_time_fuzz` is specified, e.g. for events like Safe Sun Mode
            # or normal sun mode then ensure that the end of the event is at least
            # event_time_fuzz from the end of the search interval.  If not the event
            # might be split between the current search interval and the next.  Since
            # the next search interval will step forward in time, it is sure that
            # eventually the event will be fully contained.
            if event_time_fuzz:
                while tstop - event_time_fuzz < states[-1]["tstop"]:
                    # Event tstop is within event_time_fuzz of the stop of states so
                    # bail out and don't return any states.
                    logger.warning(
                        "Warning: dropping state because of "
                        "insufficent event time pad:\n{}\n".format(states[-1:])
                    )
                    states = states[:-1]
                    if len(states) == 0:
                        return [], event_msidset

            # Select event states that are contained within start/stop interval
            ok = (states["tstart"] >= tstart) & (states["tstop"] <= tstop)
            states = states[ok]

            if event_time_fuzz:
                states = fuzz_states(states, event_time_fuzz)

        return states, event_msidset

    @classmethod
    def from_dict(cls, model_dict, logger=None) -> "BaseModel":
        """
        Set model from a dict `model_dict` which might have extra stuff not in
        Model.  If `logger` is supplied then log output at debug level.

        Parameters
        ----------
        model_dict : dict
            Dictionary of model attributes
        logger : logging.Logger, optional
            Logger object to log debug output

        Returns
        -------
        model : BaseModel
            Model instance
        """
        # Get the obsid at the appropriate time for the event (typically "start"
        # but "stop" in the case of Manvr). But don't do this for Obsid.
        if cls is not Obsid:
            tstart = DateTime(model_dict[cls._get_obsid_start_attr]).secs
            obsrq = fetch.Msid("cobsrqid", tstart, tstart + 200)
            if len(obsrq.vals) == 0:
                logger.warning(
                    "WARNING: unable to get COBSRQID near "
                    f"{model_dict[cls._get_obsid_start_attr]}, "
                    "using obsid=-999"
                )
                model_dict["obsid"] = -999
            else:
                # MAUDE telemetry can include corrupted values near IU reset,
                # so set obsid to 0 in that case. Obsid=0 will be the next valid value.
                if (obsid := obsrq.vals[0]) > 65535:
                    logger.info(f"Setting obsid=0 to replace corrupted obsid={obsid}")
                    obsid = 0
                model_dict["obsid"] = obsid

        return super().from_dict(model_dict, logger)

    @classmethod
    def get_events(cls, start, stop=None):
        """
        Get events from telemetry defined by a simple rule that the value of
        `event_msids[0]` == `event_val`.
        """
        states, event_msidset = cls.get_msids_states(start, stop)

        # Assemble a list of dicts corresponding to events in this tlm interval
        events = []
        for state in states:
            tstart = state["tstart"]
            tstop = state["tstop"]
            event = dict(
                tstart=tstart,
                tstop=tstop,
                dur=tstop - tstart,
                start=DateTime(tstart).date,
                stop=DateTime(tstop).date,
            )

            # Reject events that are shorter than the minimum duration
            if hasattr(cls, "event_min_dur") and event["dur"] < cls.event_min_dur:
                continue

            # Custom processing defined by subclasses to add more attrs to event
            event.update(cls.get_extras(event, event_msidset))

            events.append(event)

        return events

    @property
    def msidset(self):
        """
        fetch.MSIDset of self.fetch_event_msids.  By default filter_bad is True.
        """
        if not hasattr(self, "_msidset"):
            self._msidset = self.fetch_event()
        return self._msidset

    def fetch_event(self, pad=None, extra_msids=None, filter_bad=True):
        """
        Fetch an MSIDset of self.fetch_msids.
        """
        if pad is None:
            pad = self.fetch_event_pad
        msids = self.fetch_event_msids[:]
        if extra_msids is not None:
            msids.extend(extra_msids)
        msidset = fetch.MSIDset(
            msids, self.tstart - pad, self.tstop + pad, filter_bad=filter_bad
        )
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

    event_msids = ["cobsrqid"]

    update_priority = 1000  # Process Obsid first

    @classmethod
    def get_events(cls, start, stop=None):
        """
        Get obsid events from telemetry.  A event is defined by a
        contiguous interval of the telemetered obsid.
        """
        events = []
        # Get the event telemetry MSID objects
        event_msidset = fetch.Msidset(cls.event_msids, start, stop)
        obsid: fetch.Msid = event_msidset["cobsrqid"]

        # Bad telemetry following an IU reset and/or Safe Mode
        bad = (obsid.vals < 0) | (obsid.vals > 65535)
        if np.count_nonzero(bad) > 0:
            logger.info(
                f"Setting bad COBSRQID values {obsid.vals[bad]} at "
                f"{DateTime(obsid.times[bad]).date} to 0"
            )
        obsid.vals[bad] = 0

        if len(obsid) < 2:
            # Not enough telemetry for state_intervals, return no events
            return events

        states = obsid.state_intervals()
        # Skip the first and last states as they are likely incomplete
        for state in states[1:-1]:
            event = dict(
                start=state["datestart"],
                stop=state["datestop"],
                tstart=state["tstart"],
                tstop=state["tstop"],
                dur=state["tstop"] - state["tstart"],
                obsid=state["val"],
            )
            events.append(event)
        return events

    def __unicode__(self):
        return "start={} dur={:.0f} obsid={}".format(self.start, self.dur, self.obsid)


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
             obsid    Integer                    Observation ID (COBSRQID)
     start_3tscpos    Integer                   Start TSC position (steps)
      stop_3tscpos    Integer                    Stop TSC position (steps)
         start_det    Char(6)   Start detector (ACIS-I ACIS-S HRC-I HRC-S)
          stop_det    Char(6)    Stop detector (ACIS-I ACIS-S HRC-I HRC-S)
           max_pwm    Integer                   Max PWM during translation
    =============== ========== ============================================
    """

    event_msids = ["3tscmove", "3tscpos", "3mrmmxmv"]
    event_val = ["T", "MOVE"]

    start_3tscpos = models.IntegerField(help_text="Start TSC position (steps)")
    stop_3tscpos = models.IntegerField(help_text="Stop TSC position (steps)")
    start_det = models.CharField(
        max_length=6, help_text="Start detector (ACIS-I ACIS-S HRC-I HRC-S)"
    )
    stop_det = models.CharField(
        max_length=6, help_text="Stop detector (ACIS-I ACIS-S HRC-I HRC-S)"
    )
    max_pwm = models.IntegerField(help_text="Max PWM during translation")

    interval_pad = IntervalPad()  # interval padding before/ after event start/stop

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define start/stop_3tscpos and start/stop_det.
        """
        out = _get_start_stop_vals(
            event["tstart"] - 66, event["tstop"] + 66, event_msidset, msids=["3tscpos"]
        )
        out["start_det"] = _get_si(out["start_3tscpos"])
        out["stop_det"] = _get_si(out["stop_3tscpos"])
        pwm = event_msidset["3mrmmxmv"]
        out["max_pwm"] = interpolate(pwm.vals, pwm.times, [event["tstop"] + 66])[0]
        return out

    def __unicode__(self):
        return "start={} dur={:.0f} start_3tscpos={} stop_3tscpos={}".format(
            self.start, self.dur, self.start_3tscpos, self.stop_3tscpos
        )


class DarkCalReplica(TlmEvent):
    """
    ACA dark current calibration replica

    **Event definition**: interval where ``CIUMACAC = ON``

    CIUMACAC is the IU MODE ACA CALIBRATION INDICATOR.

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

    event_msids = ["ciumacac"]
    event_val = "ON"
    event_min_dur = 300


class DarkCal(TlmEvent):
    """
    ACA dark current calibration event

    **Event definition**: interval where ``CIUMACAC = ON``

    CIUMACAC is the IU MODE ACA CALIBRATION INDICATOR.  Individual intervals
    within one day are joined together to a single calibration event.

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

    event_msids = ["ciumacac"]
    event_val = "ON"
    event_time_fuzz = 86400  # One full day of fuzz / pad
    event_min_dur = 8000


class Scs107(TlmEvent):
    """
    SCS107 run

    **Event definition**: reaction wheel bias disabled between 500 to 1000 sec::

      AORWBIAS = DISA

    This is commanded by SCS-107 with a roughly constant time delay from RW Bias
    disable to the subsequent re-enable. Over the mission the delay has varied from
    around 900 seconds (early) to ~550 secs (circa 2025), as SCS-107 has been modified.
    See `notebooks/scs107-via-RW-bias-disable.ipynb` for the supporting analysis.

    For this event, intervals of RW Bias being disabled for 500 to 1000 seconds are
    selected as a proxy for SCS-107 runs.

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
      notes       Text               Supplemental notes
    ======== ========== ================================
    """

    notes = models.TextField(help_text="Supplemental notes")

    event_msids = ["aorwbias"]
    event_val = "DISA"

    @classmethod
    def get_events(cls, start, stop=None):
        events = super().get_events(start, stop)
        # See docstring for why we select only 500 to 1000 second intervals
        events = [event for event in events if 500 < event["dur"] < 1000]
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
            obsid    Integer        Observation ID (COBSRQID)
     start_3fapos    Integer        Start FA position (steps)
      stop_3fapos    Integer         Stop FA position (steps)
    ============== ========== ================================
    """

    event_msids = ["3famove", "3fapos"]
    event_val = ["T", "MOVE"]

    start_3fapos = models.IntegerField(help_text="Start FA position (steps)")
    stop_3fapos = models.IntegerField(help_text="Stop FA position (steps)")

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define start/stop_3fapos.
        """
        out = _get_start_stop_vals(
            event["tstart"] - 16.4,
            event["tstop"] + 16.4,
            event_msidset,
            msids=["3fapos"],
        )
        return out

    def __unicode__(self):
        return "start={} dur={:.0f} start_3fapos={} stop_3fapos={}".format(
            self.start, self.dur, self.start_3fapos, self.stop_3fapos
        )


class GratingMove(TlmEvent):
    """
    Grating movement (HETG or LETG)

    **Event definition**: interval with 4MP28AV > 2.0 V  (MCE A + 28 VOLT MONITOR)

    This event detects grating motion via the MCE-A 28 volt monitor.  Due to
    changes in the on-board software over the years, this appears to be the
    most reliable method.

    Short movements of less than 4 seconds are classified with grating=BUMP.
    In a handful of cases in 2000, there are intervals with 4MP28AV > 2.0
    where no grating motion is seen.  These have grating=UNKN (unknown).

    **Fields**

    ================ ========== =========================================
         Field          Type                   Description
    ================ ========== =========================================
              start   Char(21)            Start time (YYYY:DDD:HH:MM:SS)
               stop   Char(21)             Stop time (YYYY:DDD:HH:MM:SS)
             tstart      Float                     Start time (CXC secs)
              tstop      Float                      Stop time (CXC secs)
                dur      Float                           Duration (secs)
              obsid    Integer                 Observation ID (COBSRQID)
     start_4lposaro      Float             Start LETG position (degrees)
      stop_4lposaro      Float              Stop LETG position (degrees)
     start_4hposaro      Float             Start HETG position (degrees)
      stop_4hposaro      Float              Stop HETG position (degrees)
            grating    Char(4)   Grating in motion (UNKN LETG HETG BUMP)
          direction    Char(4)        Grating direction (UNKN INSR RETR)
    ================ ========== =========================================
    """

    start_4lposaro = models.FloatField(help_text="Start LETG position (degrees)")
    stop_4lposaro = models.FloatField(help_text="Stop LETG position (degrees)")
    start_4hposaro = models.FloatField(help_text="Start HETG position (degrees)")
    stop_4hposaro = models.FloatField(help_text="Stop HETG position (degrees)")
    grating = models.CharField(
        max_length=4, help_text="Grating in motion (UNKN LETG HETG BUMP)"
    )
    direction = models.CharField(
        max_length=4, help_text="Grating direction (UNKN INSR RETR)"
    )

    event_msids = ["4mp28av", "4lposaro", "4hposaro"]
    event_time_fuzz = 10

    @classmethod
    def get_state_times_bools(cls, event_msidset):
        event_msid = event_msidset["4mp28av"]
        moving = event_msid.vals > 2.0
        return event_msid.times, moving

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define start/stop grating positions for HETG and LETG
        """
        out = _get_start_stop_vals(
            event["tstart"],
            event["tstop"],
            event_msidset,
            msids=["4lposaro", "4hposaro"],
        )

        if event["dur"] < 4:
            grating = "BUMP"
        else:
            grating = "UNKN"  # Should never stay as 'UNKN'
            if abs(out["start_4lposaro"] - out["stop_4lposaro"]) > 5:
                grating = "LETG"
            if abs(out["start_4hposaro"] - out["stop_4hposaro"]) > 5:
                # If BOTH this is not good (maybe two moves fuzzed together)
                grating = "BOTH" if grating == "LETG" else "HETG"

        if grating == "LETG":
            direction = (
                "INSR" if out["start_4lposaro"] > out["stop_4lposaro"] else "RETR"
            )
        elif grating == "HETG":
            direction = (
                "INSR" if out["start_4hposaro"] > out["stop_4hposaro"] else "RETR"
            )
        else:
            direction = "UNKN"

        out["direction"] = direction
        out["grating"] = grating

        return out

    def __unicode__(self):
        return "start={} dur={:.0f} grating={} direction={}".format(
            self.start, self.dur, self.grating, self.direction
        )


class Dump(TlmEvent):
    """
    Momentum unload either ground commanded or autonomous

    **Event definition**: interval where ``AOUNLOAD = GRND`` or ``AOUNLOAD = AUTO``

    **Fields**

    ======== ========== ==================================
     Field      Type               Description
    ======== ========== ==================================
      start   Char(21)     Start time (YYYY:DDD:HH:MM:SS)
       stop   Char(21)      Stop time (YYYY:DDD:HH:MM:SS)
     tstart      Float              Start time (CXC secs)
      tstop      Float               Stop time (CXC secs)
        dur      Float                    Duration (secs)
      obsid    Integer          Observation ID (COBSRQID)
       type    Char(4)   Momentum unload type (GRND AUTO)
    ======== ========== ==================================
    """

    event_msids = ["aounload"]
    event_val = "GRND"
    event_min_dur = 4.0

    type = models.CharField(max_length=4, help_text="Momentum unload type (GRND AUTO)")

    @classmethod
    def get_state_times_bools(cls, event_msidset):
        event_msid = event_msidset["aounload"]
        unload = np.isin(event_msid.vals, ["GRND", "AUTO"])
        return event_msid.times, unload

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define unload type
        """
        tlm = event_msidset["aounload"]
        t_mid = (event["tstart"] + event["tstop"]) / 2.0
        idx = np.searchsorted(tlm.times, t_mid)
        out = {"type": tlm.vals[idx]}
        return out

    def __str__(self):
        return "start={} dur={:.0f} type={}".format(self.start, self.dur, self.type)


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
      obsid    Integer        Observation ID (COBSRQID)
    ======== ========== ================================
    """

    event_msids = ["aoeclips"]
    event_val = "ECL"
    fetch_event_msids = ["aoeclips", "eb1k5", "eb2k5", "eb3k5"]
    # There are many short periods of ECL that are not real eclipses, typically
    # following a real eclipse, most commonly < 10 s. Require a minimum duration.
    event_min_dur = 100


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
                  obsid    Integer                                    Observation ID (COBSRQID)
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
               one_shot      Float                            One shot attitude update (arcsec)
          one_shot_roll      Float                       One shot attitude update roll (arcsec)
         one_shot_pitch      Float                      One shot attitude update pitch (arcsec)
           one_shot_yaw      Float                        One shot attitude update yaw (arcsec)
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

    ``one_shot``: one shot attitude update following maneuver.  This is -99.0 for maneuvers
        with no corresponding transition to NPM.

    ``one_shot_roll``, ``one_shot_pitch``, and ``one_shot_yaw`` are the values of AOATTER1, 2, and 3
        from samples after the guide transition.
    """

    _get_obsid_start_attr = "stop"  # Attribute to use for getting event obsid
    event_msids = ["aofattmd", "aopcadmd", "aoacaseq", "aopsacpr"]
    event_val = "MNVR"
    # Aux MSIDs to fetch for maneuver events (required for one-shot)
    aux_msids = ["aotarqt1", "aotarqt2", "aotarqt3", "aoatter1", "aoatter2", "aoatter3"]

    fetch_event_msids = [
        "one_shot",
        "aofattmd",
        "aopcadmd",
        "aoacaseq",
        "aopsacpr",
        "aounload",
        "aoattqt1",
        "aoattqt2",
        "aoattqt3",
        "aoattqt4",
        "aogyrct1",
        "aogyrct2",
        "aogyrct3",
        "aogyrct4",
        "aoatupq1",
        "aoatupq2",
        "aoatupq3",
        "aotarqt1",
        "aotarqt2",
        "aotarqt3",
        "aoatter1",
        "aoatter2",
        "aoatter3",
        "aogbias1",
        "aogbias2",
        "aogbias3",
    ]
    fetch_event_pad = 600

    interval_pad = IntervalPad()  # interval padding before/ after event start/stop

    prev_manvr_stop = models.CharField(
        max_length=21,
        null=True,
        help_text="Stop time of previous AOFATTMD=MNVR before manvr",
    )
    prev_npnt_start = models.CharField(
        max_length=21,
        null=True,
        help_text="Start time of previous AOPCADMD=NPNT before manvr",
    )
    nman_start = models.CharField(
        max_length=21, null=True, help_text="Start time of AOPCADMD=NMAN for manvr"
    )
    manvr_start = models.CharField(
        max_length=21, null=True, help_text="Start time of AOFATTMD=MNVR for manvr"
    )
    manvr_stop = models.CharField(
        max_length=21, null=True, help_text="Stop time of AOFATTMD=MNVR for manvr"
    )
    npnt_start = models.CharField(
        max_length=21, null=True, help_text="Start time of AOPCADMD=NPNT after manvr"
    )
    acq_start = models.CharField(
        max_length=21, null=True, help_text="Start time of AOACASEQ=AQXN after manvr"
    )
    guide_start = models.CharField(
        max_length=21, null=True, help_text="Start time of AOACASEQ=GUID after manvr"
    )
    kalman_start = models.CharField(
        max_length=21, null=True, help_text="Start time of AOACASEQ=KALM after manvr"
    )
    aca_proc_act_start = models.CharField(
        max_length=21, null=True, help_text="Start time of AOPSACPR=ACT after manvr"
    )
    npnt_stop = models.CharField(
        max_length=21, null=True, help_text="Stop time of AOPCADMD=NPNT after manvr"
    )
    next_nman_start = models.CharField(
        max_length=21,
        null=True,
        help_text="Start time of next AOPCADMD=NMAN after manvr",
    )
    next_manvr_start = models.CharField(
        max_length=21,
        null=True,
        help_text="Start time of next AOFATTMD=MNVR after manvr",
    )
    n_dwell = models.IntegerField(
        help_text="Number of kalman dwells after manvr and before next manvr"
    )
    n_acq = models.IntegerField(
        help_text="Number of AQXN intervals after manvr and before next manvr"
    )
    n_guide = models.IntegerField(
        help_text="Number of GUID intervals after manvr and before next manvr"
    )
    n_kalman = models.IntegerField(
        help_text="Number of KALM intervals after manvr and before next manvr"
    )
    anomalous = models.BooleanField(help_text="Key MSID shows off-nominal value")
    template = models.CharField(max_length=16, help_text="Matched maneuver template")
    start_ra = models.FloatField(help_text="Start right ascension before manvr")
    start_dec = models.FloatField(help_text="Start declination before manvr")
    start_roll = models.FloatField(help_text="Start roll angle before manvr")
    stop_ra = models.FloatField(help_text="Stop right ascension after manvr")
    stop_dec = models.FloatField(help_text="Stop declination after manvr")
    stop_roll = models.FloatField(help_text="Stop roll angle after manvr")
    angle = models.FloatField(help_text="Maneuver angle (deg)")
    one_shot = models.FloatField(help_text="One shot attitude update (arcsec)")
    one_shot_roll = models.FloatField(
        help_text="One shot attitude update roll (arcsec)"
    )
    one_shot_pitch = models.FloatField(
        help_text="One shot attitude update pitch (arcsec)"
    )
    one_shot_yaw = models.FloatField(help_text="One shot attitude update yaw (arcsec)")

    one_shot._kadi_format = "{:.1f}"
    one_shot_roll._kadi_format = "{:.1f}"
    one_shot_pitch._kadi_format = "{:.1f}"
    one_shot_yaw._kadi_format = "{:.1f}"
    angle._kadi_format = "{:.2f}"
    start_ra._kadi_format = "{:.5f}"
    start_dec._kadi_format = "{:.5f}"
    start_roll._kadi_format = "{:.5f}"
    stop_ra._kadi_format = "{:.5f}"
    stop_dec._kadi_format = "{:.5f}"
    stop_roll._kadi_format = "{:.5f}"

    class Meta:
        ordering = ["start"]

    def __unicode__(self):
        return "start={} dur={:.0f} n_dwell={} template={}".format(
            self.start, self.dur, self.n_dwell, self.template
        )

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
        ok = changes["dt"] >= ZERO_DT
        dwell = {}
        for change in changes[ok]:
            # Not in a dwell and ACA sequence is KALMAN => start dwell.
            if (
                state is None
                and change["msid"] == "aoacaseq"
                and change["val"] == "KALM"
            ):
                t0 = change["time"]
                dwell["rel_tstart"] = change["dt"]
                dwell["tstart"] = change["time"]
                dwell["start"] = change["date"]
                state = "dwell"

            # Another KALMAN within 400 secs of previous KALMAN in dwell.
            # This is another acquisition sequence and moves the dwell start back.
            elif (
                state == "dwell"
                and change["msid"] == "aoacaseq"
                and change["val"] == "KALM"
                and change["time"] - t0 < 400
            ):
                t0 = change["time"]
                dwell["rel_tstart"] = change["dt"]
                dwell["tstart"] = change["time"]
                dwell["start"] = change["date"]

            # End of dwell because of NPNT => NMAN transition OR another acquisition
            elif state == "dwell" and (
                (change["msid"] == "aopcadmd" and change["val"] == "NMAN")
                or (change["msid"] == "aoacaseq" and change["time"] - t0 > 400)
            ):
                dwell["tstop"] = change["prev_time"]
                dwell["stop"] = change["prev_date"]
                dwell["dur"] = dwell["tstop"] - dwell["tstart"]
                dwells.append(dwell)
                dwell = {}
                state = None

        for dwell in dwells:
            for att in ("ra", "dec", "roll"):
                dwell[att] = event["stop_" + att]

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
            ok = changes["msid"] == msid
            if val.startswith("!"):
                ok &= changes["val"] != val[1:]
            else:
                ok &= changes["val"] == val
            if filter == "before":
                ok &= changes["dt"] < ZERO_DT
            elif filter == "after":
                ok &= changes["dt"] >= ZERO_DT
            try:
                if idx is None:
                    return changes[ok]["date"]
                else:
                    return changes[ok][idx]["date"]
            except IndexError:
                return None

        # Check for any telemetry values that are off-nominal
        nom_vals = {
            "aopcadmd": ("NPNT", "NMAN"),
            "aoacaseq": ("GUID", "KALM", "AQXN"),
            "aofattmd": ("MNVR", "STDY"),
            "aopsacpr": ("INIT", "INAC", "ACT"),
            "aounload": ("MON", "GRND"),
        }
        anomalous = False
        for change in changes[changes["dt"] >= ZERO_DT]:
            if change["val"].strip() not in nom_vals[change["msid"]]:
                anomalous = True
                break

        # Templates of previously seen maneuver sequences. These cover sequences seen at
        # least twice as of ~Mar 2012.
        manvr_templates = get_manvr_templates()
        seqs = [
            "{}_{}_{}".format(c["msid"], c["prev_val"], c["val"])
            for c in changes
            if (
                c["msid"] in ("aopcadmd", "aofattmd", "aoacaseq") and c["dt"] >= ZERO_DT
            )
        ]
        for name, manvr_template in manvr_templates:
            if (
                seqs == manvr_template[2:]
            ):  # skip first two which are STDY-MNVR and MNVR-STDY
                template = name
                break
        else:
            template = "unknown"

        manvr_attrs = dict(
            prev_manvr_stop=match(
                "aofattmd", "!MNVR", -1, "before"
            ),  # Last STDY before this manvr
            prev_npnt_start=match(
                "aopcadmd", "NPNT", -1, "before"
            ),  # Last NPNT before this manvr
            nman_start=match(
                "aopcadmd", "NMAN", -1, "before"
            ),  # NMAN that precedes this manvr
            manvr_start=match("aofattmd", "MNVR", -1, "before"),  # start of this manvr
            manvr_stop=match("aofattmd", "!MNVR", 0, "after"),
            npnt_start=match("aopcadmd", "NPNT", 0, "after"),
            acq_start=match("aoacaseq", "AQXN", 0, "after"),
            guide_start=match("aoacaseq", "GUID", 0, "after"),
            kalman_start=match("aoacaseq", "KALM", 0, "after"),
            aca_proc_act_start=match("aopsacpr", "ACT ", 0, "after"),
            npnt_stop=match("aopcadmd", "!NPNT", -1, "after"),
            next_nman_start=match("aopcadmd", "NMAN", -1, "after"),
            next_manvr_start=match("aofattmd", "MNVR", -1, "after"),
            n_acq=len(match("aoacaseq", "AQXN", None, "after")),
            n_guide=len(match("aoacaseq", "GUID", None, "after")),
            n_kalman=len(match("aoacaseq", "KALM", None, "after")),
            anomalous=anomalous,
            template=template,
        )

        return manvr_attrs

    @classmethod
    def get_target_attitudes(cls, event, msidset):
        """
        Define start/stop_aotarqt<1..4> and start/stop_ra,dec,roll
        """
        out = {}
        quats = {}
        for label, dt in (("start", -60), ("stop", 60)):
            time = event["tstart"] + dt
            q123 = []
            for i in range(1, 4):
                name = "aotarqt{}".format(i)
                msid = msidset[name]
                q123.append(
                    interpolate(msid.vals, msid.times, [time], method="nearest")[0]
                )
            q123 = np.array(q123)
            sum_q123_sq = np.sum(q123**2)
            q4 = np.sqrt(np.abs(1.0 - sum_q123_sq))
            norm = np.sqrt(sum_q123_sq + q4**2)
            quat = Quat(np.concatenate([q123, [q4]]) / norm)
            quats[label] = quat
            out[label + "_aotarqt1"] = float(quat.q[0])
            out[label + "_aotarqt2"] = float(quat.q[1])
            out[label + "_aotarqt3"] = float(quat.q[2])
            out[label + "_aotarqt4"] = float(quat.q[3])
            out[label + "_ra"] = float(quat.ra)
            out[label + "_dec"] = float(quat.dec)
            out[label + "_roll"] = float(quat.roll)

        dq = quats["stop"] / quats["start"]  # = (wx * sa2, wy * sa2, wz * sa2, ca2)
        q3 = np.abs(dq.q[3])
        if q3 >= 1.0:  # Floating point error possible
            out["angle"] = 0.0
        else:
            out["angle"] = float(np.arccos(q3) * 2 * 180 / np.pi)

        return out

    @classmethod
    def get_one_shot(cls, guide_start, aux_msidset):
        """
        Define one_shot attribute (one-shot attitude update following maneuver)
        """
        # No acq => guide transition so no one-shot defined.  Use -99.0 as a convenience for
        # the one shot length, -999.0 for the axis values.
        if guide_start is None:
            return {
                "one_shot": -99.0,
                "one_shot_roll": -999.0,
                "one_shot_pitch": -999.0,
                "one_shot_yaw": -999.0,
            }
        # Save the first post maneuver / post ACQ attitude error sample for each AOATTER axis
        pm_att_err = {}
        for ax in [1, 2, 3]:
            msid = aux_msidset["aoatter{}".format(ax)]
            ok = (msid.times >= DateTime(guide_start).secs + 1.1) & (
                msid.times < DateTime(guide_start).secs + 30
            )
            if not np.any(ok):
                raise ValueError(
                    "No AOATTER{} times for guide transition at {} for one shot".format(
                        ax, guide_start
                    )
                )
            pm_att_err[ax] = msid.vals[ok][0] * R2A
        return {
            "one_shot": np.sqrt(pm_att_err[2] ** 2 + pm_att_err[3] ** 2),
            "one_shot_roll": pm_att_err[1],
            "one_shot_pitch": pm_att_err[2],
            "one_shot_yaw": pm_att_err[3],
        }

    @classmethod
    def get_events(cls, start, stop=None):
        """
        Get maneuver events from telemetry.
        """
        events = []
        # Auxiliary information
        aux_msidset = fetch.Msidset(cls.aux_msids, start, stop)

        # Need at least 2 samples to get states. Having no samples typically
        # happens when building the event tables and telemetry queries are just
        # before telemetry starts.
        if any(len(aux_msid) < 2 for aux_msid in aux_msidset.values()):
            return events

        states, event_msidset = cls.get_msids_states(start, stop)
        changes = _get_msid_changes(
            list(event_msidset.values()),
            sortmsids={"aofattmd": 1, "aopcadmd": 2, "aoacaseq": 3, "aopsacpr": 4},
        )

        for manvr_prev, manvr, manvr_next in zip(states, states[1:], states[2:]):
            tstart = manvr["tstart"]
            tstop = manvr["tstop"]

            # Make sure the aux_msidset (used for stop/stop target attitudes and one shot)
            # is complete through the end of maneuver + one hour.  Finish event processing
            # if that is not the case.
            min_aux_tstop = min(aux_msid.times[-1] for aux_msid in aux_msidset.values())
            if min_aux_tstop < tstop + 3600:
                logger.info(
                    "Breaking out of maneuver processing at manvr start={} because "
                    "min_aux_stop={} < manvr stop + 1hr={}".format(
                        DateTime(tstart).date,
                        DateTime(min_aux_tstop).date,
                        DateTime(tstop + 3600).date,
                    )
                )
                break

            i0 = np.searchsorted(changes["time"], manvr_prev["tstop"])
            i1 = np.searchsorted(changes["time"], manvr_next["tstart"])
            sequence = changes[i0 : i1 + 1]
            sequence["dt"] = (sequence["time"] + sequence["prev_time"]) / 2.0 - manvr[
                "tstop"
            ]
            ok = (
                (sequence["dt"] >= ZERO_DT)
                | (sequence["msid"] == "aofattmd")
                | (sequence["msid"] == "aopcadmd")
            )
            sequence = sequence[ok]
            manvr_attrs = cls.get_manvr_attrs(sequence)

            event = dict(
                tstart=tstart,
                tstop=tstop,
                dur=tstop - tstart,
                start=DateTime(tstart).date,
                stop=DateTime(tstop).date,
                foreign={"ManvrSeq": sequence},
            )
            event.update(manvr_attrs)
            event.update(cls.get_target_attitudes(event, aux_msidset))
            one_shot = cls.get_one_shot(manvr_attrs["guide_start"], aux_msidset)
            event.update(one_shot)
            dwells = cls.get_dwells(event, sequence)
            event["foreign"]["Dwell"] = dwells
            event["n_dwell"] = len(dwells)

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

    rel_tstart = models.FloatField(help_text="Start time relative to manvr end (sec)")
    manvr = models.ForeignKey(
        Manvr, on_delete=models.CASCADE, help_text="Maneuver that contains this dwell"
    )
    ra = models.FloatField(help_text="Right ascension (deg)")
    dec = models.FloatField(help_text="Declination (deg)")
    roll = models.FloatField(help_text="Roll angle (deg)")
    rel_tstart._kadi_format = "{:.2f}"
    ra._kadi_format = "{:.5f}"
    dec._kadi_format = "{:.5f}"
    roll._kadi_format = "{:.5f}"

    # To do: add ra dec roll quaternion

    def __unicode__(self):
        # TODO add ra, dec, roll
        return "start={} dur={:.0f}".format(self.start, self.dur)


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

    manvr = models.ForeignKey(Manvr, on_delete=models.CASCADE)
    msid = models.CharField(max_length=8)
    prev_val = models.CharField(max_length=4)
    val = models.CharField(max_length=4)
    date = models.CharField(max_length=21)
    dt = models.FloatField()
    time = models.FloatField()
    prev_date = models.CharField(max_length=21)
    prev_time = models.FloatField()

    class Meta:
        ordering = ["manvr", "date"]

    def __unicode__(self):
        return "{}: {} => {} at {}".format(
            self.msid.upper(), self.prev_val, self.val, self.date
        )


class SafeSun(TlmEvent):
    """
    Safe sun event

    **Event definition**: interval from safe mode entry to recovery to OBC control.

    Specifically, it is considered part of the safe mode condition if any of the
    following are True::

       CONLOFP != 'NRML'  # OFP state
       CTUFMTSL = 'FMT5'  # CTU telemetry format
       C1SQAX != 'ENAB'   # Sequencer A enable/disable

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
      notes       Text
    ======== ========== ================================
    """

    notes = models.TextField()
    event_msids = ["conlofp", "ctufmtsl", "c1sqax"]
    event_filter_bad = False
    event_min_dur = 3600

    fetch_event_pad = 86400 / 2
    fetch_event_msids = ["conlofp", "ctufmtsl", "c1sqax", "aopcadmd", "61psts02"]

    @classmethod
    def get_state_times_bools(cls, msidset):
        # Second safemode has bad telemetry for the entire event so the standard
        # detection algorithm fails.  Just hardwire safemode #1 and fix up the
        # telemetry within this range so that the standard detection works.
        safemode2_tstart = DateTime("1999:269:20:22:45").secs
        safemode2_tstop = DateTime("1999:270:08:22:30").secs
        if msidset.tstart < safemode2_tstop and msidset.tstop > safemode2_tstart:
            for msid in msidset.values():
                ok = (msid.times > safemode2_tstart) & (msid.times < safemode2_tstop)
                msid.bads[ok] = False
                force_val = {"conlofp": "NRML", "ctufmtsl": "FMT5", "c1sqax": "ENAB"}[
                    msid.msid
                ]
                msid.vals[ok] = force_val

        # Interpolate all MSIDs to a common time.  Sync to the start of CTUFMTSL which is
        # sampled at 32.8 seconds
        msidset_interpolate(msidset, 32.8, msidset["ctufmtsl"].times[0])

        # Define intervals when in safe mode ~(A & B & C) == (~A | ~B | ~C)
        safe_mode = (
            (msidset["conlofp"].vals != "NRML")
            | (msidset["ctufmtsl"].vals == "FMT5")
            | (msidset["c1sqax"].vals != "ENAB")
        )

        # Telemetry indicates a safemode around 1999:221 which isn't real
        bogus_tstart = DateTime("1999:225:12:00:00").secs
        if msidset.tstart < bogus_tstart:
            ok = msidset.times < bogus_tstart
            safe_mode[ok] = False

        return msidset.times, safe_mode


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
      obsid    Integer        Observation ID (COBSRQID)
    ======== ========== ================================
    """

    event_msids = ["aopcadmd"]
    event_val = "NSUN"


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

    key = models.CharField(
        max_length=24, primary_key=True, help_text="Unique key for this event"
    )
    key._kadi_hidden = True
    start = models.CharField(
        max_length=8,
        db_index=True,
        help_text="Event time to the nearest day (YYYY:DOY)",
    )
    date = models.CharField(
        max_length=11, help_text="Event time to the nearest day (YYYY-Mon-DD)"
    )
    tstart = models.FloatField(
        db_index=True, help_text="Event time to the nearest day (CXC sec)"
    )
    descr = models.TextField(help_text="Event description")
    note = models.TextField(help_text="Note (comments or CAP # or FSW PR #)")
    source = models.CharField(max_length=3, help_text="Event source (FDB or FOT)")

    # Allow for hand-edits of the source HTML tables back about two years. Make
    # sure the lookback for getting new events is a bit further back than delete
    # lookback.
    lookback = 365 * 2 + 10
    lookback_delete = 365 * 2

    class Meta:
        ordering = ["key"]

    def __unicode__(self):
        descr = self.descr
        if len(descr) > 30:
            descr = descr[:27] + "..."
        note = self.note
        if note:
            descr += " (" + (note if len(note) < 30 else note[:27] + "...") + ")"
        return "{} ({}) {}: {}".format(self.start, self.date[5:], self.source, descr)

    @classmethod
    def get_events(cls, start, stop=None):
        """
        Get Major Events from FDB and FOT tables on the OCCweb
        """
        import hashlib

        from . import scrape

        tstart = DateTime(start).secs
        tstop = DateTime(stop).secs

        events = scrape.get_fot_major_events() + scrape.get_fdb_major_events()

        # Select events within time range and sort by tstart key
        events = sorted(
            (x for x in events if tstart <= x["tstart"] <= tstop),
            key=lambda x: x["tstart"],
        )

        # Manually generate a unique key for event since date is not unique
        for event in events:
            key = "".join(event[x] for x in ("start", "descr", "note", "source"))
            key = key.encode(
                "ascii", "replace"
            )  # Should already be clean ASCII but make sure
            event["key"] = event["start"] + ":" + hashlib.sha1(key).hexdigest()[:6]

        return events


class IFotEvent(BaseEvent):
    """
    Base class for events from the iFOT database
    """

    ifot_id = models.IntegerField(primary_key=True, help_text="iFOT identifier")
    start = models.CharField(
        max_length=21, db_index=True, help_text="Start time (date)"
    )
    stop = models.CharField(max_length=21, help_text="Stop time (date)")
    tstart = models.FloatField(db_index=True, help_text="Start time (CXC secs)")
    tstop = models.FloatField(help_text="Stop time (CXC secs)")
    dur = models.FloatField(help_text="Duration (secs)")
    dur._kadi_format = "{:.1f}"

    class Meta:
        abstract = True
        ordering = ["start"]

    ifot_columns = ["id", "tstart", "tstop"]
    ifot_props = []
    ifot_types = {}  # Override automatic type inference for properties or columns

    @classmethod
    def get_events(cls, start, stop=None):
        """
        Get events from iFOT web interface
        """
        from kadi import occweb

        datestart = DateTime(start).date
        datestop = DateTime(stop).date

        # def get_ifot(event_type, start=None, stop=None, props=[], columns=[], timeout=TIMEOUT):
        ifot_evts = occweb.get_ifot(
            cls.ifot_type_desc,
            start=datestart,
            stop=datestop,
            props=cls.ifot_props,
            columns=cls.ifot_columns,
            types=cls.ifot_types,
        )

        events = []
        for ifot_evt in ifot_evts:
            event = {
                key.lower(): ifot_evt[cls.ifot_type_desc + "." + key].tolist()
                for key in cls.ifot_props
            }
            # Prefer start or stop from props, but if not there use column tstart/tstop
            for st in ("start", "stop"):
                tst = "t{}".format(st)
                if st not in event or not event[st].strip():
                    event[st] = ifot_evt[tst]
                # The above still might not be OK because sometimes what is in the
                # <prop>.START/STOP values is not a valid date.
                try:
                    DateTime(event[st]).date  # noqa: B018
                except Exception:
                    # Fail, roll back to the tstart/tstop version
                    logger.info(
                        'WARNING: Bad value of ifot_evt[{}.{}] = "{}" at {}'.format(
                            cls.ifot_type_desc, st, event[st], ifot_evt["tstart"]
                        )
                    )
                    event[st] = ifot_evt[tst]

            event["ifot_id"] = ifot_evt["id"]
            event["tstart"] = DateTime(event["start"]).secs
            event["tstop"] = DateTime(event["stop"]).secs
            event["dur"] = event["tstop"] - event["tstart"]

            # Custom processing defined by subclasses to add more attrs to event
            if hasattr(cls, "get_extras"):
                event.update(cls.get_extras(event, None))

            events.append(event)

        return events

    def __unicode__(self):
        return "{}: {}".format(self.ifot_id, self.start[:17])


class CAP(IFotEvent):
    """
    CAP from iFOT database

    **Event definition**: CAP from iFOT database

    **Fields**

    ========= =========== =======================
      Field       Type          Description
    ========= =========== =======================
     ifot_id     Integer         iFOT identifier
       start    Char(21)       Start time (date)
        stop    Char(21)        Stop time (date)
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

    num = models.CharField(max_length=15, help_text="CAP number")
    title = models.TextField(help_text="CAP title")
    descr = models.TextField(help_text="CAP description")
    notes = models.TextField(help_text="CAP notes")
    link = models.CharField(max_length=250, help_text="CAP link")

    ifot_type_desc = "CAP"
    ifot_props = ["NUM", "START", "STOP", "TITLE", "LINK", "DESC"]

    def __unicode__(self):
        return "{}: {} {}".format(self.num, self.start[:17], self.title)


class LoadSegment(IFotEvent):
    """
    Load segment

    **Event definition**: Load segment from iFOT database

    **Fields**

    =========== ========== =======================
       Field       Type          Description
    =========== ========== =======================
       ifot_id    Integer         iFOT identifier
         start   Char(21)       Start time (date)
          stop   Char(21)        Stop time (date)
        tstart      Float   Start time (CXC secs)
         tstop      Float    Stop time (CXC secs)
           dur      Float         Duration (secs)
          name   Char(12)       Load segment name
           scs    Integer                SCS slot
     load_name   Char(10)               Load name
       comment       Text                 Comment
    =========== ========== =======================
    """

    name = models.CharField(max_length=12, help_text="Load segment name")
    scs = models.IntegerField(help_text="SCS slot")
    load_name = models.CharField(max_length=10, help_text="Load name")
    comment = models.TextField(help_text="Comment")

    ifot_type_desc = "LOADSEG"
    ifot_props = ["NAME", "SCS", "LOAD_NAME", "COMMENT"]

    lookback = 40  # days of lookback
    lookback_delete = 20  # Remove load segments in database prior to 20 days ago
    # to account for potential load changes.
    lookforward = 28  # Accept load segments planned up to 28 days in advance

    def __unicode__(self):
        return "{}: {} {} scs={}".format(
            self.name, self.start[:17], self.load_name, self.scs
        )


class PassPlan(IFotEvent):
    """
    Pass plan

    **Event definition**: Pass plan from iFOT

    **Fields**

    ==================== ========== =======================
           Field            Type          Description
    ==================== ========== =======================
                ifot_id    Integer         iFOT identifier
                  start   Char(21)       Start time (date)
                   stop   Char(21)        Stop time (date)
                 tstart      Float   Start time (CXC secs)
                  tstop      Float    Stop time (CXC secs)
                    dur      Float         Duration (secs)
                     oc   Char(30)                 OC crew
                     cc   Char(30)                 CC crew
                    got   Char(30)                GOT crew
                station    Char(6)             DSN station
           est_datetime   Char(20)              Date local
     sched_support_time   Char(13)            Support time
               activity   Char(20)                Activity
                    bot    Char(4)      Beginning of track
                    eot    Char(4)            End of track
              data_rate   Char(10)               Data rate
                 config    Char(8)           Configuration
                    lga    Char(1)                     LGA
                  power    Char(6)                   Power
                rxa_rsl   Char(10)                Rx-A RSL
                rxb_rsl   Char(10)                Rx-B RSL
                err_log   Char(10)               Error log
              cmd_count   Char(15)           Command count
    ==================== ========== =======================
    """

    oc = models.CharField(max_length=30, help_text="OC crew")
    cc = models.CharField(max_length=30, help_text="CC crew")
    got = models.CharField(max_length=30, help_text="GOT crew")
    station = models.CharField(max_length=6, help_text="DSN station")
    est_datetime = models.CharField(max_length=20, help_text="Date local")
    sched_support_time = models.CharField(max_length=13, help_text="Support time")
    activity = models.CharField(max_length=20, help_text="Activity")
    bot = models.CharField(max_length=4, help_text="Beginning of track")
    eot = models.CharField(max_length=4, help_text="End of track")
    data_rate = models.CharField(max_length=10, help_text="Data rate")
    config = models.CharField(max_length=8, help_text="Configuration")
    lga = models.CharField(max_length=1, help_text="LGA")
    power = models.CharField(max_length=6, help_text="Power")
    rxa_rsl = models.CharField(max_length=10, help_text="Rx-A RSL")
    rxb_rsl = models.CharField(max_length=10, help_text="Rx-B RSL")
    err_log = models.CharField(max_length=10, help_text="Error log")
    cmd_count = models.CharField(max_length=15, help_text="Command count")

    ifot_type_desc = "PASSPLAN"
    ifot_props = (
        "oc cc got station EST_datetime sched_support_time activity bot eot "
        "data_rate config lga power rxa_rsl rxb_rsl "
        "err_log cmd_count"
    ).split()

    lookback = 21  # days of lookback
    lookback_delete = 7  # Remove all comms in database prior to 7 days ago to account
    # for potential schedule changes.
    lookforward = 28  # Accept comms scheduled up to 28 days in advance

    update_priority = 500  # Must be before DsnComm

    def __unicode__(self):
        return "{} {} {} OC:{} CC:{}".format(
            self.station, self.start[:17], self.est_datetime, self.oc, self.cc
        )


class DsnComm(IFotEvent):
    """
    DSN comm period

    **Event definition**: DSN comm pass beginning of support to end of support (not
      beginning / end of track).

    **Fields**

    =========== ========== ========================
       Field       Type          Description
    =========== ========== ========================
       ifot_id    Integer          iFOT identifier
         start   Char(21)        Start time (date)
          stop   Char(21)         Stop time (date)
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
            oc   Char(30)                  OC crew
            cc   Char(30)                  CC crew
     pass_plan   OneToOne                Pass plan
    =========== ========== ========================
    """

    bot = models.CharField(max_length=4, help_text="Beginning of track")
    eot = models.CharField(max_length=4, help_text="End of track")
    activity = models.CharField(max_length=30, help_text="Activity description")
    config = models.CharField(max_length=10, help_text="Configuration")
    data_rate = models.CharField(max_length=9, help_text="Data rate")
    site = models.CharField(max_length=12, help_text="DSN site")
    soe = models.CharField(max_length=4, help_text="DSN Sequence Of Events")
    station = models.CharField(max_length=6, help_text="DSN station")
    oc = models.CharField(max_length=30, help_text="OC crew")
    cc = models.CharField(max_length=30, help_text="CC crew")
    pass_plan = models.OneToOneField(
        PassPlan,
        help_text="Pass plan",
        null=True,
        related_name="dsn_comm",
        on_delete=models.CASCADE,
    )

    ifot_type_desc = "DSN_COMM"
    ifot_props = [
        "bot",
        "eot",
        "activity",
        "config",
        "data_rate",
        "site",
        "soe",
        "station",
    ]
    ifot_types = {"DSN_COMM.bot": str, "DSN_COMM.eot": str}

    lookback = 21  # days of lookback
    lookback_delete = 7  # Remove all comms in database prior to 7 days ago to account
    # for potential schedule changes.
    lookforward = 28  # Accept comms scheduled up to 28 days in advance

    @classmethod
    def get_extras(cls, event, event_msidset):
        """
        Define OC, CC and pass_plan if available
        """
        out = {}
        pass_plans = PassPlan.objects.filter(start=event["start"])
        if len(pass_plans) > 0:
            # Multiple pass plans possible (e.g. two stations), just take first
            pass_plan = pass_plans[0]
            out["oc"] = pass_plan.oc
            out["cc"] = pass_plan.cc
            out["pass_plan"] = pass_plan
        if len(pass_plans) > 1:
            logger.warning(
                "Multiple pass plans found at {}: {}".format(event["start"], pass_plans)
            )

        return out

    def __unicode__(self):
        return "{}: {} {}-{} {}".format(
            self.station, self.start[:17], self.bot, self.eot, self.activity
        )


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

    start = models.CharField(
        max_length=21, help_text="Start time (orbit ascending node crossing)"
    )
    stop = models.CharField(
        max_length=21, help_text="Stop time (next orbit ascending node crossing)"
    )
    tstart = models.FloatField(
        db_index=True, help_text="Start time (orbit ascending node crossing)"
    )
    tstop = models.FloatField(
        help_text="Stop time (next orbit ascending node crossing)"
    )
    dur = models.FloatField(help_text="Orbit duration (sec)")
    orbit_num = models.IntegerField(primary_key=True, help_text="Orbit number")
    perigee = models.CharField(max_length=21, help_text="Perigee time")
    apogee = models.CharField(max_length=21, help_text="Apogee time")
    t_perigee = models.FloatField(help_text="Perigee time (CXC sec)")
    start_radzone = models.CharField(max_length=21, help_text="Start time of rad zone")
    stop_radzone = models.CharField(max_length=21, help_text="Stop time of rad zone")
    dt_start_radzone = models.FloatField(
        help_text="Start time of rad zone relative to perigee (sec)"
    )
    dt_stop_radzone = models.FloatField(
        help_text="Stop time of rad zone relative to perigee (sec)"
    )
    dur._kadi_format = "{:.1f}"
    t_perigee._kadi_format = "{:.1f}"
    dt_start_radzone._kadi_format = "{:.1f}"
    dt_stop_radzone._kadi_format = "{:.1f}"

    lookforward = 28  # Accept orbits planned up to 28 days in advance

    @classmethod
    def get_events(cls, start, stop=None):
        """
        Get Orbit Events from timeline reports.
        """
        from . import orbit_funcs

        datestart = DateTime(start).date
        datestop = DateTime(stop).date
        years = sorted({x[:4] for x in (datestart, datestop)})
        file_dates = []
        for year in years:
            file_dates.extend(orbit_funcs.get_tlr_files(year))
        tlr_files = [
            x["name"] for x in file_dates if datestart <= x["date"] <= datestop
        ]

        # Get all orbit points from the tlr files as a list of tuples
        orbit_points = orbit_funcs.get_orbit_points(tlr_files)

        # Process the points, doing various cleanup and return as a np.array
        orbit_points = orbit_funcs.process_orbit_points(orbit_points)

        # Get the orbits from the orbit points
        orbits = orbit_funcs.get_orbits(orbit_points)

        events = []
        for orbit in orbits:
            ok = orbit_points["orbit_num"] == orbit["orbit_num"]
            event = {key: orbit[key] for key in orbit.dtype.names}
            event["foreign"] = {
                "OrbitPoint": orbit_points[ok],
                "RadZone": [orbit_funcs.get_radzone_from_orbit(orbit)],
            }
            events.append(event)

        return events

    def __unicode__(self):
        return "{} {} dur={:.1f} radzone={:.1f} to {:.1f} ksec".format(
            self.orbit_num,
            self.start[:17],
            self.dur / 1000,
            self.dt_start_radzone / 1000,
            self.dt_stop_radzone / 1000,
        )


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

    orbit = models.ForeignKey(Orbit, on_delete=models.CASCADE)
    date = models.CharField(max_length=21)
    name = models.CharField(max_length=9)
    orbit_num = models.IntegerField()
    descr = models.CharField(max_length=50)

    class Meta:
        ordering = ["date"]

    def __unicode__(self):
        return "{} (orbit {}) {}: {}".format(
            self.date[:17], self.orbit_num, self.name, self.descr[:30]
        )


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

    orbit = models.ForeignKey(Orbit, on_delete=models.CASCADE)
    orbit_num = models.IntegerField()
    perigee = models.CharField(max_length=21)

    def __unicode__(self):
        return "{} {} {} dur={:.1f} ksec".format(
            self.orbit_num, self.start[:17], self.stop[:17], self.dur / 1000
        )


class AsciiTableEvent(BaseEvent):
    """
    Base class for events defined by a simple quasi-static text table file.
    Subclasses need to define the file name (lives in DATA_DIR()/<filename>)
    and the start and stop column names.
    """

    class Meta:
        abstract = True
        ordering = ["start"]

    @classmethod
    def get_extras(cls, event, interval):
        """
        Get extra stuff for the event based on fields in the interval.  This is a hook
        within get_events() that should be overridden in individual classes.
        """
        return {}

    @classmethod
    def get_events(cls, start, stop=None):
        """
        Get events from telemetry defined by a simple rule that the value of
        `event_msids[0]` == `event_val`.
        """
        from kadi.paths import DATA_DIR

        start = DateTime(start).date
        stop = DateTime(stop).date

        filename = Path(cls.intervals_file)
        if not filename.absolute():
            filename = Path(DATA_DIR(), filename)
        intervals = table.Table.read(str(filename), **cls.table_read_kwargs)

        # Custom in-place processing of raw intervals
        cls.process_intervals(intervals)

        ok = (DateTime(intervals[cls.start_column]).date > start) & (
            DateTime(intervals[cls.stop_column]).date < stop
        )

        # Assemble a list of dicts corresponding to events in this tlm interval
        events = []
        for interval in intervals[ok]:
            tstart = DateTime(interval[cls.start_column]).secs
            tstop = DateTime(interval[cls.stop_column]).secs
            event = dict(
                tstart=tstart,
                tstop=tstop,
                dur=tstop - tstart,
                start=DateTime(tstart).date,
                stop=DateTime(tstop).date,
            )

            # Reject events that are shorter than the minimum duration
            if hasattr(cls, "event_min_dur") and event["dur"] < cls.event_min_dur:
                continue

            # Custom processing defined by subclasses to add more attrs to event
            event.update(cls.get_extras(event, interval))

            events.append(event)

        return events


class LttBad(AsciiTableEvent):
    """
    LTT bad intervals

    **Fields**

    ======== ========== ================================
     Field      Type              Description
    ======== ========== ================================
        key   Char(38)        Unique key for this event
      start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
       stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
     tstart      Float            Start time (CXC secs)
      tstop      Float             Stop time (CXC secs)
        dur      Float                  Duration (secs)
       msid   Char(20)                             MSID
       flag    Char(2)                             Flag
    ======== ========== ================================
    """

    key = models.CharField(
        max_length=38, primary_key=True, help_text="Unique key for this event"
    )
    start = models.CharField(max_length=21, help_text="Start time (YYYY:DDD:HH:MM:SS)")
    stop = models.CharField(max_length=21, help_text="Stop time (YYYY:DDD:HH:MM:SS)")
    tstart = models.FloatField(db_index=True, help_text="Start time (CXC secs)")
    tstop = models.FloatField(help_text="Stop time (CXC secs)")
    dur = models.FloatField(help_text="Duration (secs)")
    msid = models.CharField(max_length=20, help_text="MSID")
    flag = models.CharField(max_length=2, help_text="Flag")

    key._kadi_hidden = True
    dur._kadi_format = "{:.1f}"

    intervals_file = Path(sys.prefix) / "share" / "kadi" / "ltt_bads.dat"
    # Table.read keyword args
    table_read_kwargs = dict(
        format="ascii", data_start=2, delimiter="|", guess=False, fill_values=()
    )
    start_column = "start"
    stop_column = "stop"

    @classmethod
    def process_intervals(cls, intervals):
        intervals["start"] = DateTime(intervals["tstart"]).date
        intervals["stop"] = (DateTime(intervals["tstart"]) + 1).date
        intervals.sort("start")

    @classmethod
    def get_extras(cls, event, interval):
        out = {}
        for key in ("msid", "flag"):
            out[key] = interval[key].tolist()
        out["key"] = event["start"][:17] + out["msid"]
        return out

    def __unicode__(self):
        return "start={} msid={} flag={}".format(self.start, self.msid, self.flag)
