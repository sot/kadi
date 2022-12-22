# Licensed under a 3-clause BSD style license - see LICENSE.rst

import functools
import logging
from dataclasses import dataclass
from typing import List, Optional

import cheta.fetch_eng as fetch
import numpy as np
import plotly.graph_objects as pgo
from astropy.table import Table
from cxotime import CxoTime, CxoTimeLike

__all__ = [
    "add_figure_regions",
    "compress_time_series",
    "convert_state_code_to_raw_val",
    "get_telem_values",
    "fill_gaps_with_nan",
    "NoTelemetryError",
    "get_ofp_states",
    "get_time_series_chunks",
    "TimeSeriesChunk",
    "TimeSeriesPoint",
]

logger = logging.getLogger(__name__)


class NoTelemetryError(Exception):
    """No telemetry available for the specified interval"""


@dataclass
class TimeSeriesPoint:
    time: float
    val: float

    @property
    def date(self):
        return CxoTime(self.time).date

    def __repr__(self):
        date = CxoTime(self.time).date
        out = f"<{self.__class__.__name__} date={date} val={self.val}>"
        return out


@dataclass
class TimeSeriesChunk:
    first: TimeSeriesPoint = None
    min: TimeSeriesPoint = None
    max: TimeSeriesPoint = None
    last: TimeSeriesPoint = None

    def __repr__(self):
        out = "\n".join(
            [
                f"<{self.__class__.__name__}",
                f"first={self.first}",
                f"min={self.min}",
                f"max={self.max}",
                f"last={self.last}>",
            ]
        )
        return out


def get_telem_values(msids: list, stop, days: float = 14) -> Table:
    """
    Fetch last ``days`` of available ``msids`` telemetry values before
    time ``tstart``.

    :param msids: fetch msids list
    :param stop: stop time for telemetry (CxoTime-like)
    :param days: length of telemetry request before ``tstart``
    :returns: Table of requested telemetry values from fetch
    """
    stop = CxoTime(stop)
    start = stop - days
    logger.info(f"Fetching telemetry for {msids} between {start.date} and {stop.date}")

    with fetch.data_source("cxc", "maude allow_subset=False"):
        msidset = fetch.MSIDset(msids, start.date, stop.date)

    # Use the first MSID as the primary one to set the time base
    msid0 = msidset[msids[0]]

    if len(msids) == 1:
        # Only one MSID so just filter any bad values
        msid0.filter_bad()
        times = msid0.times
    else:
        # Multiple MSIDs so interpolate all to the same time base The assumption
        # here is that all MSIDs have the same basic time base, e.g. AOCMDQT1-3.
        msidset.interpolate(times=msid0.times, bad_union=True)
        times = msidset.times

    # Finished when we found at least 10 good records (5 mins)
    if len(times) < 10:
        raise NoTelemetryError(
            f"Found no telemetry for {msids!r} within {days} days of {stop}"
        )

    names = ["time"] + msids
    out = Table([times] + [msidset[x].vals for x in msids], names=names)
    return out


@functools.lru_cache(maxsize=1)
def get_ofp_states(stop, days):
    """Get the Onboard Flight Program states for ``stop`` and ``days`` lookback

    This is normally "NRML" but in safe mode it is "SAFE" or other values. State codes:
    ['NNRM' 'STDB' 'STBS' 'NRML' 'NSTB' 'SUOF' 'SYON' 'DPLY' 'SYSF' 'STUP' 'SAFE']
    """
    import astropy.table as tbl
    from cheta.utils import logical_intervals

    msid = "conlofp"
    tlm = get_telem_values([msid], stop, days)
    states_list = []
    for state_code in np.unique(tlm[msid]):
        states = logical_intervals(
            tlm["time"], tlm[msid] == state_code, max_gap=2.1, complete_intervals=False
        )
        states["val"] = state_code
        states_list.append(states)
    states = tbl.vstack(states_list)
    states.sort("datestart")

    return states


def add_figure_regions(
    fig: pgo.Figure,
    figure_start: CxoTimeLike,
    figure_stop: CxoTimeLike,
    region_starts: List[CxoTimeLike],
    region_stops: List[CxoTimeLike],
    color: str = "black",
    opacity: float = 0.2,
    line_width: float = 3,
):
    """Add regions to a figure with a date-based x-axis

    ``figure_start`` and ``figure_stop`` are the start and stop times for the figure.
    ``region_starts`` and ``region_stops`` are lists of start/stop times for regions.
    """
    # Add "background" grey rectangles for figure time regions to vs-time plot
    # Plot time-axis limits in datetime64 format
    dt0 = CxoTime(figure_start).datetime64
    dt1 = CxoTime(figure_stop).datetime64

    for region_start, region_stop in zip(region_starts, region_stops):
        region_dt0 = CxoTime(region_start).datetime64
        region_dt1 = CxoTime(region_stop).datetime64
        if (region_dt1 >= dt0) & (region_dt0 <= dt1):
            # Note oddity/bug in plotly: need to cast np.datetime64 values to [ms]
            # otherwise the rectangle is not drawn. CxoTime.datetime64 is [ns].
            kwargs = dict(
                x0=max(region_dt0, dt0).astype("datetime64[ms]"),
                x1=min(region_dt1, dt1).astype("datetime64[ms]"),
                line_width=line_width,
                line_color=color,
                fillcolor=color,
                opacity=opacity,
            )
            fig.add_vrect(**kwargs)


def convert_state_code_to_raw_val(state_vals, state_codes):
    raw_vals = np.zeros(len(state_vals), dtype=int)
    for raw_val, state_code in state_codes:
        ok = state_vals == state_code
        raw_vals[ok] = raw_val
    return raw_vals


def get_time_series_chunks(
    times: np.ndarray,
    vals: np.ndarray,
    max_delta_val: float = 0,
    max_delta_time: Optional[float] = None,
) -> List[TimeSeriesChunk]:
    # Turning max_delta_time from None to a big number simplifies the later code.
    if max_delta_time is None:
        max_delta_time = np.finfo(np.float64).max

    chunks = []
    chunk = None

    idx = 0
    while idx < len(times):
        time_ = times[idx]
        val = vals[idx]

        if np.isnan(val):
            # If current value is NaN then end the current chunk and create a
            # new one that is all NaNs spanning a gap. This is coupled to the behavior
            # of fill_gap_with_nan() where a gap is exactly two NaNs (one at the
            # beginning and one at the end of the gap).

            # First end the current chunk
            chunks.append(chunk)

            # Make a new NaN chunk
            first = TimeSeriesPoint(time=time_, val=val)
            last = TimeSeriesPoint(time=times[idx + 1], val=vals[idx + 1])
            chunk = TimeSeriesChunk(first=first, min=first, max=first, last=last)
            chunks.append(chunk)

            # Start a new empty chunk past the two NaNs
            idx += 2
            chunk = None
            continue

        if chunk is None:
            point = TimeSeriesPoint(time=time_, val=val)
            chunk = TimeSeriesChunk(first=point, min=point, max=point, last=point)

        new_min = min(chunk.min.val, val)
        new_max = max(chunk.max.val, val)
        delta_val = new_max - new_min
        delta_time = time_ - chunk.first.time

        if delta_val > max_delta_val or delta_time > max_delta_time:
            # This chunk is complete so add it to the list and start a new one
            chunks.append(chunk)

            if (
                abs(val - chunk.last.val) > max_delta_val
                or time_ - chunk.last.time > max_delta_time
            ):
                # If the value has changed by more than the threshold then start
                # a blank new chunk with the next `idx` point (via the chunk is None
                # bit above)
                chunk = None
            else:
                # Otherwise start a new chunk from the last point in previous chunk
                point = chunk.last
                chunk = TimeSeriesChunk(first=point, min=point, max=point, last=point)

        else:
            # Still within the bounds of the chunk so update the min, max, last
            if new_min < chunk.min.val:
                chunk.min = TimeSeriesPoint(time=time_, val=new_min)
            if new_max > chunk.max.val:
                chunk.max = TimeSeriesPoint(time=time_, val=new_max)
            chunk.last = TimeSeriesPoint(time=time_, val=val)
            idx += 1

    # Add the last chunk if it exists
    if chunk is not None:
        chunks.append(chunk)

    return chunks


def compress_time_series(
    times: np.ndarray,
    vals: np.ndarray,
    max_delta_val: float = 0,
    max_delta_time: Optional[float] = None,
    max_gap: Optional[float] = None,
):
    times = np.array(times, dtype=float)
    vals = np.array(vals, dtype=float)

    if max_gap is not None:
        times, vals = fill_gaps_with_nan(times, vals, max_gap=max_gap)

    chunks = get_time_series_chunks(times, vals, max_delta_val, max_delta_time)
    out_times = []
    out_vals = []
    if len(chunks) == 0:
        return out_times, out_vals

    out_times.append(chunks[0].first.time)
    out_vals.append(chunks[0].first.val)

    for chunk in chunks:
        chunk_times = [
            getattr(chunk, attr).time for attr in ["first", "min", "max", "last"]
        ]
        chunk_vals = [
            getattr(chunk, attr).val for attr in ["first", "min", "max", "last"]
        ]
        for idx in np.argsort(chunk_times):
            if (time_ := chunk_times[idx]) != out_times[-1]:
                out_times.append(time_)
                out_vals.append(chunk_vals[idx])

    return out_times, out_vals


def fill_gaps_with_nan(times: list, vals: list, max_gap: float, dt: float = 0.001):
    """Fill gaps in ``vals`` with NaNs where gaps in ``times`` is more than ``max_gap``.

    Do something like::

      times = [1, 2, 20, 21, 22]  => [1, 2, 2.001, 19.999, 20, 21, 22]
      vals = [1, 2, 3, 4, 5] => [1.0, 2.0, NaN, NaN, 3.0, 4.0, 5.0]

    :param times: times
    :param vals: values
    :param max_gap: maximum gap in seconds
    :param dt: time delta to use for filling gaps
    :returns: times, vals with gaps filled with NaNs
    """
    times = np.asarray(times, dtype=float)
    vals = np.asarray(vals, dtype=float)
    idxs = np.where(np.diff(times) > max_gap)[0]
    for idx in idxs[::-1]:
        gap_times = [times[idx] + dt, times[idx + 1] - dt]
        times = np.insert(times, idx + 1, gap_times)
        vals = np.insert(vals, idx + 1, [np.nan, np.nan])
    return times, vals
