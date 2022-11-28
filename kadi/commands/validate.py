#!/usr/bin/env python

"""
"""
import functools
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cheta.fetch_eng as fetch
import jinja2
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as pgo
import requests
import Ska.Matplotlib
import Ska.Numpy
import Ska.Shell
import Ska.tdb
from astropy.table import Table
from cheta.utils import logical_intervals, state_intervals
from cxotime import CxoTime

import kadi
import kadi.commands
from kadi.commands.states import interpolate_states

# TODO: move this definition to cxotime
# TODO: use npt.NDArray with numpy 1.21
CxoTimeLike = Union[str, float, int, np.ndarray, npt.ArrayLike]


__all__ = [
    "Validate",
    "ValidatePitch",
    "ValidateSimpos",
    "ValidateObsid",
    "ValidateDither",
    "ValidatePcadMode",
    "NoTelemetryError",
    "get_time_series_chunks",
    "compress_time_series",
    "add_figure_regions",
    "get_manual_exclude_intervals",
]

logger = logging.getLogger(__name__)

# URL to download exclude Times google sheet
# See https://stackoverflow.com/questions/33713084 (2nd answer)
EXCLUDE_INTERVALS_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/{doc_id}/export?"
    "format=csv"
    "&id={doc_id}"
    "&gid={gid}"
)


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


class NoTelemetryError(Exception):
    """No telemetry available for the specified interval"""


@dataclass
class PlotAttrs:
    title: str
    ylabel: str


class Validate(ABC):
    subclasses = []

    # Abstract attributes (no elegant solution as of Python 3.11)
    name: str = None
    stop: CxoTime = None
    days: float = None
    state_keys: tuple = None
    plot_attrs: PlotAttrs = None
    validation_limits: tuple = None
    msids: tuple = None
    quantile_fmt = None
    max_delta_val = 0
    max_delta_time = None
    max_gap = 300  # seconds
    min_violation_duration = 32.81  # seconds

    def __init__(self, stop=None, days=14):
        self.stop = CxoTime(stop)
        self.days = days

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            cls.subclasses.append(cls)

    @property
    def tlm(self):
        if not hasattr(self, "_tlm"):
            self._tlm = get_telem_values(
                msids=self.msids, stop=self.stop, days=self.days
            )
            self.update_tlm()
        return self._tlm

    def update_tlm(self):
        """Update the telemetry values with any subclass-specific processing"""

    @property
    def times(self):
        return self.tlm["time"]

    @property
    def msid(self):
        """Validate classes have first MSID as primary telemetry. Override as needed."""
        return self.msids[0]

    @property
    def states(self):
        if not hasattr(self, "_states"):
            self._states = get_states(
                start=self.tlm["time"][0],
                stop=self.tlm["time"][-1],
                state_keys=self.state_keys,
            )
        return self._states

    @property
    def exclude_intervals(self):
        """Intervals that are excluded from state validation.

        This includes manually excluded times from the Command Events sheet
        (e.g. within a few minutes of an IU-reset), or auto-generated
        state-specific intervals like not validating pitch when in NMM.
        """
        if not hasattr(self, "_exclude_intervals"):
            self._exclude_intervals = self.get_exclude_intervals()

        return self._exclude_intervals

    def get_exclude_intervals(self):
        exclude_intervals = get_manual_exclude_intervals()
        exclude_intervals["source"] = "manual"

        # Filter exclude times for this state key
        keep_idxs = []
        for idx, row in enumerate(exclude_intervals):
            states = row["states"].split()
            if row["states"] == "" or self.name in states:
                keep_idxs.append(idx)
        exclude_intervals = exclude_intervals[keep_idxs]
        return exclude_intervals

    @property
    def tlm_vals(self):
        """Get the reference telemetry value for this Validation subclass

        That means the quantity derived from telemetry that gets compared to
        the states that come from kadi commands. This might be a single MSID
        (e.g. for pitch) or it might be something more complicated (e.g. for
        pointing which comes from 3 quaternion components).
        """
        raise NotImplementedError

    @property
    def states_at_times(self):
        """Get the states that correspond to the telemetry times"""
        if not hasattr(self, "_states_at_times"):
            self._states_at_times = interpolate_states(self.states, self.times)
        return self._states_at_times

    @property
    def state_vals(self):
        if not hasattr(self, "_state_vals"):
            self._state_vals = self.states_at_times[self.name].copy()
        return self._state_vals

    @property
    def violations(self):
        if not hasattr(self, "_violations"):
            self._violations = self.get_violations()
        return self._violations

    def get_violations_mask(self):
        """Get the violations mask for this validation class

        This is the default implementation for most validation classes which just checks
        that the telemetry value is within ``max_delta_val`` of the state value.
        """
        bad = np.abs(self.tlm_vals - self.state_vals) > self.max_delta_val
        return bad

    def get_violations(self):
        """Get the violations mask for this validation class

        This is the main method for each validation class. It returns a Table
        with the columns ``start`` and ``stop`` which are date strings.
        """
        violations_mask = self.get_violations_mask()
        mask_ok = get_overlap_mask(self.times, self.exclude_intervals)
        violations_mask[mask_ok] = False

        intervals = logical_intervals(self.times, violations_mask, max_gap=self.max_gap)
        ok = intervals["duration"] > self.min_violation_duration
        intervals = intervals[ok]

        out = Table()
        out["start"] = intervals["datestart"]
        out["stop"] = intervals["datestop"]

        return out

    def get_plot_figure(self) -> pgo.Figure:
        state_vals = self.state_vals
        tlm_vals = self.tlm_vals
        times = self.tlm["time"]

        fig = pgo.Figure()

        for color, vals in [
            ("#1f77b4", tlm_vals),  # muted blue
            ("#ff7f0e", state_vals),  # safety orange
        ]:
            tm, y = compress_time_series(
                times, vals, self.max_delta_val, self.max_delta_time
            )
            trace = pgo.Scatter(
                x=CxoTime(tm).datetime64,
                y=y,
                mode="lines+markers",
                line={"color": color, "width": 3},
                opacity=0.75,
                showlegend=False,
                marker={"opacity": 0.9, "size": 8},
            )
            fig.add_trace(trace)

        fig.update_layout(
            {
                "title": (f"{self.name}"),
                "yaxis": {
                    "title": self.plot_attrs.ylabel,
                },
                "xaxis": {
                    "title": f"Date",
                },
            }
        )

        add_figure_regions(
            fig,
            figure_start=times[0],
            figure_stop=times[-1],
            region_starts=self.exclude_intervals["start"],
            region_stops=self.exclude_intervals["stop"],
            color="black",
            opacity=0.2,
        )

        add_figure_regions(
            fig,
            figure_start=times[0],
            figure_stop=times[-1],
            region_starts=self.violations["start"],
            region_stops=self.violations["stop"],
            color="red",
            opacity=0.4,
        )

        return fig

    def get_plot_html(self, show=False) -> str:
        fig = self.get_plot_figure()
        if show:
            fig.show()

        html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            default_width=800,
            default_height=500,
        )
        return html


class ValidateSingleMsid(Validate):
    @property
    def tlm_vals(self):
        if not hasattr(self, "_tlm_vals"):
            self._tlm_vals = self.tlm[self.msid].copy()
        return self._tlm_vals


class ValidateStateCode(Validate):
    """Base class for validation of state with state codes like PCAD_MODE"""

    @property
    def state_codes(self) -> Table:
        tsc = Ska.tdb.msids.find(self.msid)[0].Tsc
        state_codes = Table(
            [tsc.data["LOW_RAW_COUNT"], tsc.data["STATE_CODE"]],
            names=["raw_val", "state_code"],
        )
        state_codes.sort("raw_val")
        return state_codes

    @property
    def tlm_vals(self):
        if not hasattr(self, "_tlm_vals"):
            self._tlm_vals = convert_state_code_to_raw_val(
                self.tlm, self.msid, self.state_codes
            )
        return self._tlm_vals

    @property
    def state_vals(self):
        if not hasattr(self, "_state_vals"):
            states_interp = interpolate_states(self.states, self.tlm["time"])
            self._state_vals = convert_state_code_to_raw_val(
                states_interp, self.name, self.state_codes
            )
        return self._state_vals

    def get_plot_figure(self) -> pgo.Figure:
        fig = super().get_plot_figure()
        yaxis = dict(
            tickmode="array",
            tickvals=list(self.state_codes["raw_val"]),
            ticktext=list(self.state_codes["state_code"]),
        )
        fig.update_layout(yaxis=yaxis)
        return fig

    def make_plot(self):
        fig, ax = super().make_plot()
        ax.set_yticks(self.state_codes["raw_val"], self.state_codes["state_code"])
        return fig, ax


def convert_state_code_to_raw_val(dat, name, state_codes):
    vals = np.zeros(len(dat))
    for raw_val, state_code in state_codes:
        ok = dat[name] == state_code
        vals[ok] = raw_val
    return vals


class ValidatePitch(ValidateSingleMsid):
    name = "pitch"
    msids = ["aosares1", "conlofp"]  # Also "6sares1"
    state_keys = ["pitch", "pcad_mode"]
    plot_attrs = PlotAttrs(title="Pitch", ylabel="Pitch (degrees)")
    validation_limits = ((1, 7.0), (99, 7.0), (5, 0.5), (95, 0.5))
    quantile_fmt = "%.3f"
    max_delta_val = 1.0  # deg
    max_delta_vals = {
        "NPNT": 1.0,  # deg
        "NMAN": 20.0,  # deg
        "NSUN": 2.0,  # deg
    }

    def get_violations_mask(self):
        """Get the violations mask for the pitch validation class

        This is the main method for each validation class. It returns a Table
        with the columns ``start`` and ``stop`` which are date strings.
        """
        bad = np.zeros(len(self.times), dtype=bool)

        for pcad_mode, max_delta_val in self.max_delta_vals.items():
            mask = self.states_at_times["pcad_mode"] == pcad_mode
            bad |= (np.abs(self.tlm_vals - self.state_vals) > max_delta_val) & mask

        return bad

    def update_tlm(self):
        """Update self.tlm for safe mode"""
        import Ska.Numpy

        # Online flight processing state
        # <MsidView msid="CONLOFP" technical_name="OFP STATE">
        # ['NNRM' 'STDB' 'STBS' 'NRML' 'NSTB' 'SUOF' 'SYON' 'DPLY' 'SYSF' 'STUP' 'SAFE']
        online_states = state_intervals(self.times, self.tlm["conlofp"])
        self.tlm["6sares1"] = np.ma.zeros(len(self.times))
        self.tlm["6sares1"].mask = np.ones(len(self.times), dtype=bool)

        for state in online_states:
            if state["val"] == "NRML":
                continue
            elif state["val"] == "SAFE":
                logger.info(
                    f"{self.name}: found Safe Mode at "
                    f"{state['datestart']} to {state['datestop']}"
                )
                # Safe mode, so stub in the CPE value 6SARES1 for pitch
                with fetch.data_source("cxc", "maude allow_subset=False"):
                    dat = fetch.Msid("6sares1", state["tstart"], state["tstop"])
                vals = Ska.Numpy.interpolate(
                    dat.vals,
                    dat.times,
                    self.times,
                    method="nearest",
                )
                ok = (self.times >= state["tstart"]) & (self.times < state["tstop"])
                self.tlm["6sares1"].mask[ok] = False
                self.tlm["6sares1"][ok] = vals[ok]
            else:
                # Something else, typically STUP (which I don't understand) so
                # just exclude the interval from state validation.
                exclude = {
                    "start": state["datestart"],
                    "stop": state["datestop"],
                    "states": "pitch",
                    "comment": f"CONLOFP={state['val']}",
                }
                logger.info(f"{self.name}: excluding {exclude}")
                self.exclude_intervals.add_row(exclude)

    @property
    def tlm_vals(self):
        if not hasattr(self, "_tlm_vals"):
            tlm_vals = super().tlm_vals
            # Was there a safe mode in the validation interval?
            if "6sares1" in self.tlm.colnames:
                # This value is masked (True) when not in safe mode
                not_safe_mode = self.tlm["6sares1"].mask
                tlm_vals = np.where(
                    not_safe_mode, self.tlm["aosares1"], self.tlm["6sares1"]
                )
                self._tlm_vals = tlm_vals

        return self._tlm_vals


class ValidateSimpos(ValidateSingleMsid):
    name = "simpos"
    msids = ["3tscpos"]
    state_keys = "simpos"
    plot_attrs = PlotAttrs(title="TSCPOS (SIM-Z)", ylabel="SIM-Z (steps)")
    validation_limits = ((1, 2.0), (99, 2.0))
    quantile_fmt = "%d"
    max_delta_val = 10
    # Skip over SIM moves and transient out-of-state values
    min_violation_duration = 328  # seconds


class ValidateObsid(ValidateSingleMsid):
    name = "obsid"
    msids = ["cobsrqid"]
    state_keys = "obsid"
    plot_attrs = PlotAttrs(title="OBSID", ylabel="OBSID")
    quantile_fmt = "%d"


class ValidateDither(ValidateStateCode):
    name = "dither"
    msids = ["aodithen"]
    state_keys = "dither"
    plot_attrs = PlotAttrs(title="DITHER", ylabel="Dither")


class ValidatePcadMode(ValidateStateCode):
    name = "pcad_mode"
    msids = ["aopcadmd"]
    state_keys = "pcad_mode"
    plot_attrs = PlotAttrs(title="PCAD mode", ylabel="PCAD mode")


def get_telem_values(msids: list, stop, *, days: float = 14) -> Table:
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


def get_overlap_mask(times: np.ndarray, intervals: Table):
    """Return a bool mask of ``times`` that are within any of the ``intervals``.

    ``times`` parameter must be an array of CXC seconds (float) times.
    ``intervals`` must be a Table with columns ``start`` and ``stop``.
    """
    mask = np.zeros(len(times), dtype="bool")
    for interval in intervals:
        exclude = (times >= CxoTime(interval["start"]).secs) & (
            times < CxoTime(interval["stop"]).secs
        )
        mask[exclude] = True
    return mask


@functools.lru_cache(maxsize=1)
def get_manual_exclude_intervals(state_key: str = None) -> Table:
    url = EXCLUDE_INTERVALS_SHEET_URL.format(
        doc_id=kadi.commands.conf.cmd_events_flight_id,
        gid=kadi.commands.conf.cmd_events_exclude_intervals_gid,
    )
    logger.info(f"Getting exclude times from {url}")
    req = requests.get(url, timeout=30)
    if req.status_code != 200:
        raise ValueError(f"Failed to get exclude times sheet: {req.status_code}")

    exclude_intervals = Table.read(req.text, format="csv", fill_values=[])
    return exclude_intervals


def get_states(start: CxoTimeLike, stop: CxoTimeLike, state_keys: list) -> Table:
    """Get states exactly covering date range.

    This is a thin wrapper around kadi.commands.states.get_states() that reduces the
    output time span slightly to ensure telemetry interpolation works.

    :param start: start date (CxoTime-like)
    :param stop: stop date (CxoTime-like)
    :param state_keys: list of state keys to get
    :returns: Table of states
    """
    from kadi.commands.states import get_states as get_states_kadi

    start = CxoTime(start)
    stop = CxoTime(stop)

    logger.info("Using kadi.commands.states to get cmd_states")
    logger.info(
        f"Getting commanded states {state_keys!r} between {start.date} - {stop.date}"
    )

    states = get_states_kadi(start, stop, state_keys=state_keys)
    states["tstart"] = CxoTime(states["datestart"]).secs
    states["tstop"] = CxoTime(states["datestop"]).secs

    # Set start and end state date/times to match telemetry span.  Extend the
    # state durations by a small amount because of a precision issue converting
    # to date and back to secs.  (The reference tstop could be just over the
    # 0.001 precision of date and thus cause an out-of-bounds error when
    # interpolating state values).
    states[0]["tstart"] = CxoTime(start).secs - 0.01
    states[0]["datestart"] = CxoTime(states[0]["tstart"]).date
    states[-1]["tstop"] = CxoTime(stop).secs + 0.01
    states[-1]["datestop"] = CxoTime(states[-1]["tstop"]).date

    return states


def get_index_page_html(stop: CxoTimeLike, days: float):
    """Make a simple HTML page with all the validation plots and information.

    :param stop: stop time for validation interval (CxoTime-like, default=now)
    :param days: length of validation interval (days)
    :returns: HTML string
    """
    validators = []
    violations = []
    for cls in Validate.subclasses:
        logger.info(f"Validating {cls.name}")
        instance = cls(stop=stop, days=days)
        validator = {}
        validator["plot_html"] = instance.get_plot_html()
        validator["title"] = instance.plot_attrs.title
        validators.append(validator)

        for violation in instance.violations:
            violations.append(
                {
                    "name": instance.name,
                    "start": violation["start"],
                    "stop": violation["stop"],
                }
            )

    context = {
        "validators": validators,
        "violations": violations,
    }
    index_template_file = Path(__file__).parent / "templates" / "index_validate.html"
    index_template = index_template_file.read_text()
    template = jinja2.Template(index_template)
    html = template.render(context)

    return html
