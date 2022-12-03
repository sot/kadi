# Licensed under a 3-clause BSD style license - see LICENSE.rst

import functools
import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import astropy.table as tbl
import cheta.fetch_eng as fetch
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as pgo
from astropy.table import Table
from cheta.derived.comps import ComputedMsid
from cheta.derived.pcad import DP_PITCH, DP_ROLL, arccos_clip
from cheta.utils import logical_intervals, state_intervals
from cxotime import CxoTime

# TODO: move this definition to cxotime
# TODO: use npt.NDArray with numpy 1.21
CxoTimeLike = Union[str, float, int, np.ndarray, npt.ArrayLike, None]

__all__ = [
    "add_figure_regions",
    "compress_time_series",
    "convert_state_code_to_raw_val",
    "get_telem_values",
    "NoTelemetryError",
    "CxoTimeLike",
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


@dataclass
class TimeSeriesChunk:
    first: TimeSeriesPoint = None
    min: TimeSeriesPoint = None
    max: TimeSeriesPoint = None
    last: TimeSeriesPoint = None


def get_telem_values(msids: list, stop, days: float = 14) -> Table:
    """
    Fetch last ``days`` of available ``msids`` telemetry values before
    time ``tstart``.

    :param msids: fetch msids list
    :param stop: stop time for telemetry (CxoTime-like)
    :param days: length of telemetry request before ``tstart``
    :param name_map: dict mapping msid to recarray col name
    :returns: Table of requested telemetry values from fetch
    """
    stop = CxoTime(stop)
    start = stop - days
    logger.info(f"Fetching telemetry between {start} and {stop}")

    with fetch.data_source("cxc", "maude allow_subset=False"):
        msidset = fetch.MSIDset(msids, start, stop)

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
    msid = "conlofp"
    tlm = get_telem_values([msid], stop, days)
    states = state_intervals(tlm["time"], tlm[msid])
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
    chunks = []
    chunk = None

    idx = 0
    while idx < len(times):
        time_ = times[idx]
        val = vals[idx]

        if chunk is None:
            point = TimeSeriesPoint(time=time_, val=val)
            chunk = TimeSeriesChunk(first=point, min=point, max=point, last=point)

        new_min = min(chunk.min.val, val)
        new_max = max(chunk.max.val, val)
        delta_val = new_max - new_min
        delta_time = time_ - chunk.first.time

        if delta_val > max_delta_val or (
            max_delta_time is not None and delta_time > max_delta_time
        ):
            # This chunk is complete so add it to the list and start a new one
            chunks.append(chunk)

            if abs(val - chunk.last.val) > max_delta_val:
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
):
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


########################################
# This should all go into cheta/comps.py
########################################


@functools.lru_cache(maxsize=10)
def get_roll_pitch_tlm(start, stop):
    msids = ["6sares1", "6sares2", "6sunsa1", "6sunsa2", "6sunsa3"]
    dat = fetch.MSIDset(msids, start, stop)

    # Filter out of range values. This happens just at the beginning of safe mode.
    for msid in msids:
        if "6sares" in msid:
            x0, x1 = 0, 180
        else:
            x0, x1 = -1, 1
        dat[msid].bads |= (dat[msid].vals < x0) | (dat[msid].vals > x1)

    dat.interpolate(times=dat[msids[0]].times, bad_union=True)
    return dat


def calc_css_pitch_safe(start, stop):
    # Get the raw telemetry value in user-requested unit system
    dat = get_roll_pitch_tlm(start, stop)

    sa_ang_avg = (1.0 * dat["6sares1"].vals + 1.0 * dat["6sares2"].vals) / 2
    sinang = np.sin(np.radians(sa_ang_avg))
    cosang = np.cos(np.radians(sa_ang_avg))
    # Rotate CSS sun vector from SA to ACA frame
    css_aca = np.array(
        [
            sinang * dat["6sunsa1"].vals - cosang * dat["6sunsa3"].vals,
            dat["6sunsa2"].vals * 1.0,
            cosang * dat["6sunsa1"].vals + sinang * dat["6sunsa3"].vals,
        ]
    )
    # Normalize sun vec (again) and compute pitch
    magnitude = np.sqrt((css_aca * css_aca).sum(axis=0))
    magnitude[magnitude == 0.0] = 1.0
    sun_vec_norm = css_aca / magnitude
    vals = np.degrees(arccos_clip(sun_vec_norm[0]))

    return dat.times, vals


def calc_css_roll_safe(start, stop):
    """Off-Nominal Roll Angle from CSS Data in ACA Frame [Deg]

    Defined as the rotation about the ACA X-axis required to align the sun
    vector with the ACA X/Z plane.

    Calculated by rotating the CSS sun vector from the SA-1 frame to ACA frame
    based on the solar array angles 6SARES1 and 6SARES2.

    """
    # Get the raw telemetry value in user-requested unit system
    dat = get_roll_pitch_tlm(start, stop)

    sa_ang_avg = (dat["6sares1"].vals + dat["6sares2"].vals) / 2
    sinang = np.sin(np.radians(sa_ang_avg))
    cosang = np.cos(np.radians(sa_ang_avg))
    # Rotate CSS sun vector from SA to ACA frame
    css_aca = np.array(
        [
            sinang * dat["6sunsa1"].vals - cosang * dat["6sunsa3"].vals,
            dat["6sunsa2"].vals,
            cosang * dat["6sunsa1"].vals + sinang * dat["6sunsa3"].vals,
        ]
    )
    # Normalize sun vec (again) and compute pitch
    magnitude = np.sqrt((css_aca * css_aca).sum(axis=0))
    # Don't exactly know why zero values can happen, but it does, so just drop those.
    ok = magnitude > 0
    magnitude = magnitude[ok]
    times = dat.times[ok]

    sun_vec_norm = css_aca / magnitude
    vals = np.degrees(np.arctan2(-sun_vec_norm[1, :], -sun_vec_norm[2, :]))

    return times, vals


def calc_pitch_roll_obc(start, stop, pitch_roll):
    """Use the code in the PCAD derived parameter classes to get the pitch and off
    nominal roll from OBC quaternion data."""
    dp = DP_PITCH() if pitch_roll == "pitch" else DP_ROLL()
    tlm = dp.fetch(start, stop)
    vals = dp.calc(tlm)
    return tlm.times, vals


# Class name is arbitrary, but by convention start with `Comp_`
class Comp_Pitch_Roll_OBC_Safe(ComputedMsid):
    """
    Computed MSID to return pitch or off-nominal roll angle which is valid in NPNT,
    NMAN, NSUN, and Safe Mode.

    Logic is:
    - Get logical intervals:
      - CONLOFP == "NRML":
        - AOPCADMD in ["NPNT", "NMAN"] => compute pitch/roll from AOATTQT1/2/3/4
        - AOPCADMD == "NSUN" => get pitch/roll from PITCH/ROLL_CSS derived params.
          These are also in MAUDE.
      - CONLOFP == "SAFE":
        - Compute pitch/roll from PITCH/ROLL_CSS_SAFE via calc_pitch/roll_css_safe()
      - Intervals for other CONLOFP values are ignored.

    """

    msid_match = r"(roll|pitch)_obc_safe"

    # `msid_match` is a class attribute that defines a regular expresion to
    # match for this computed MSID.  This must be defined and it must be
    # unambiguous (not matching an existing MSID or other computed MSID).
    #
    # The two groups in parentheses specify the arguments <MSID> and <offset>.
    # These are passed to `get_msid_attrs` as msid_args[0] and msid_args[1].
    # The \w symbol means to match a-z, A-Z, 0-9 and underscore (_).
    # The \d symbol means to match digits 0-9.

    def get_msid_attrs(self, tstart, tstop, msid, msid_args):
        """
        Get attributes for computed MSID: ``vals``, ``bads``, ``times``,
        ``unit``, ``raw_vals``, and ``offset``.  The first four must always
        be provided.

        :param tstart: start time (CXC secs)
        :param tstop: stop time (CXC secs)
        :param msid: full MSID name e.g. tephin_plus_5
        :param msid_args: tuple of regex match groups (msid_name,)
        :returns: dict of MSID attributes
        """
        start = CxoTime(tstart)
        stop = CxoTime(tstop)

        # Whether we are computing "pitch" or "roll", parsed from MSID name
        pitch_roll: str = msid_args[0]

        ofp_states = get_ofp_states(stop, days=(stop - start).jd)

        tlms = []
        for ofp_state in ofp_states:
            if ofp_state["val"] == "NRML":
                dat = fetch.Msid("aopcadmd", ofp_state["tstart"], ofp_state["tstop"])

                # Get states of either NPNT / NMAN or NSUN
                vals = np.isin(dat.vals, ["NPNT", "NMAN"])
                states_npnt_nman = logical_intervals(
                    dat.times, vals, complete_intervals=False
                )
                states_npnt_nman["val"] = np.repeat("NPNT_NMAN", len(states_npnt_nman))
                # Require at least 10 minutes => 2 samples of the ephem data. This is
                # needed for the built-in derived parameter calculation to work.
                ok = states_npnt_nman["duration"] > 10 * 60 + 1
                states_npnt_nman = states_npnt_nman[ok]

                states_nsun = logical_intervals(dat.times, dat.vals == "NSUN")
                states_nsun["val"] = np.repeat("NSUN", len(states_nsun))
                states = tbl.vstack([states_npnt_nman, states_nsun])
                states.sort("tstart")

                for state in states:
                    if state["val"] == "NPNT_NMAN":
                        tlm = calc_pitch_roll_obc(
                            state["tstart"], state["tstop"], pitch_roll
                        )
                        tlms.append(tlm)
                    elif state["val"] == "NSUN":
                        tlm = fetch.Msid(
                            f"{pitch_roll}_css", state["tstart"], state["tstop"]
                        )
                        tlms.append((tlm.times, tlm.vals))

            elif ofp_state["val"] == "SAFE":
                calc_func = globals()[f"calc_css_{msid_args[0]}_safe"]
                tlm = calc_func(ofp_state["datestart"], ofp_state["datestop"])
                tlms.append(tlm)

        times = np.concatenate([tlm[0] for tlm in tlms])
        vals = np.concatenate([tlm[1] for tlm in tlms])

        # Return a dict with at least `vals`, `times`, `bads`, and `unit`.
        # Additional attributes are allowed and will be set on the
        # final MSID object.
        out = {
            "vals": vals,
            "bads": np.zeros(len(vals), dtype=bool),
            "times": times,
            "unit": "DEG",
        }
        return out
