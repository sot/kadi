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
from cxotime import CxoTime

import kadi
import kadi.commands
from kadi.commands.states import interpolate_states

# TODO: move to cxotime
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
    "add_exclude_regions",
    "get_exclude_times",
]

logger = logging.getLogger(__name__)

# URL to download exclude Times google sheet
# See https://stackoverflow.com/questions/33713084 (2nd answer)
EXCLUDE_TIMES_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/{doc_id}/export?"
    "format=csv"
    "&id={doc_id}"
    "&gid={gid}"
)


@dataclass
class TimeSeriesPoint:
    time: float
    val: float


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
            chunk = None
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


def add_exclude_regions(
    fig: pgo.Figure,
    time0,
    time1,
    exclude_starts,
    exclude_stops,
    color="black",
    opacity=0.2,
    line_width=3,
):
    # Add "background" grey rectangles for excluded time regions to vs-time plot
    # Plot time-axis limits in datetime64 format
    dt0 = CxoTime(time0).datetime64
    dt1 = CxoTime(time1).datetime64

    for exclude_start, exclude_stop in zip(exclude_starts, exclude_stops):
        exclude_dt0 = CxoTime(exclude_start).datetime64
        exclude_dt1 = CxoTime(exclude_stop).datetime64
        if (exclude_dt1 >= dt0) & (exclude_dt0 <= dt1):
            # Note oddity/bug in plotly: need to cast np.datetime64 values to [ms]
            # otherwise the rectangle is not drawn. CxoTime.datetime64 is [ns].
            kwargs = dict(
                x0=max(exclude_dt0, dt0).astype("datetime64[ms]"),
                x1=min(exclude_dt1, dt1).astype("datetime64[ms]"),
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
    dt: float = None
    state_keys: tuple = None
    plot_attrs: PlotAttrs = None
    validation_limits: tuple = None
    msids: tuple = None
    quantile_fmt = None
    max_delta_val = 0
    max_delta_time = None

    def __init__(self, stop=None, days=14, dt=32.8):
        self.stop = CxoTime(stop)
        self.days = days
        self.dt = dt

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name is not None:
            cls.subclasses.append(cls)

    @property
    def tlm(self):
        if not hasattr(self, "_tlm"):
            self._tlm = get_telem_values(
                msids=self.msids, stop=self.stop, days=self.days, dt=self.dt
            )
        return self._tlm

    @property
    def msid(self):
        """Most Validate classes have a single telemetry MSID"""
        if len(self.msids) == 1:
            return self.msids[0]
        else:
            raise ValueError(f"multiple MSIDs {self.msids}")

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
    def exclude_times(self):
        if not hasattr(self, "_exclude_times"):
            exclude_times = get_exclude_times()

            # Filter exclude times for this state key
            keep_idxs = []
            for idx, row in enumerate(exclude_times):
                states = row["states"].split()
                if row["states"] == "" or self.name in states:
                    keep_idxs.append(idx)
            exclude_times = exclude_times[keep_idxs]
            self._exclude_times = exclude_times

        return self._exclude_times

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
    def state_vals(self):
        if not hasattr(self, "_state_vals"):
            states_interp = interpolate_states(self.states, self.tlm["time"])
            self._state_vals = states_interp[self.name].copy()
        return self._state_vals

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
                marker={"opacity": 0.75, "size": 4},
            )
            fig.add_trace(trace)

        fig.update_layout(
            {
                "title": (f"{self.name}"),
                "yaxis": {
                    "title": self.plot_attrs.ylabel,
                },  # "autorange": False, "range": [0, 35]},
                "xaxis": {
                    "title": f"Date",
                },
                # "range": [
                #     (CxoTime.now() - 5 * 365 * u.day).datetime,
                #     CxoTime.now().datetime,
                # ],
                # "autorange": False,
            }
        )

        add_exclude_regions(
            fig,
            times[0],
            times[-1],
            self.exclude_times["start"],
            self.exclude_times["stop"],
            color="black",
            opacity=0.2,
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
    msids = ["pitch"]  # ["aosares1"]  # , "ctufmtsl", "6sares1"]
    state_keys = "pitch"
    plot_attrs = PlotAttrs(title="Pitch", ylabel="Pitch (degrees)")
    validation_limits = ((1, 7.0), (99, 7.0), (5, 0.5), (95, 0.5))
    quantile_fmt = "%.3f"

    # @property
    # def tlm_vals(self):
    #     stop = self.stop
    #     start = stop - self.days
    #     logger.info(f"Fetching telemetry between {start} and {stop}")

    #     # ctufmtsl = fetch.Msid("ctufmtsl", start, stop)

    #     if not hasattr(self, "_tlm_vals"):
    #         # tlm_vals = np.where(
    #         #     self.tlm["6sares1"],
    #         #     self.tlm["aosares1"],
    #         #     self.tlm["ctufmtsl"] == "FMT5",
    #         # )
    #         self._tlm_vals = tlm_vals

    #     return self._tlm_vals


class ValidateSimpos(ValidateSingleMsid):
    name = "simpos"
    msids = ["3tscpos"]
    state_keys = "simpos"
    plot_attrs = PlotAttrs(title="TSCPOS (SIM-Z)", ylabel="SIM-Z (steps)")
    validation_limits = ((1, 2.0), (99, 2.0))
    quantile_fmt = "%d"


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


def get_telem_values(msids: list, stop, *, days: float = 14, dt: float = 32.8) -> Table:
    """
    Fetch last ``days`` of available ``msids`` telemetry values before
    time ``tstart``.

    :param msids: fetch msids list
    :param stop: stop time for telemetry (CxoTime-like)
    :param days: length of telemetry request before ``tstart``
    :param dt: sample time (secs)
    :param name_map: dict mapping msid to recarray col name
    :returns: Table of requested telemetry values from fetch
    """
    stop = CxoTime(stop)
    start = stop - days
    logger.info(f"Fetching telemetry between {start} and {stop}")

    # TODO: also use MAUDE telemetry to get recent data. But this needs some
    # care to ensure the data are valid.
    # with fetch.data_source("cxc", "maude"):
    msidset = fetch.MSIDset(msids, start, stop)

    tstart = max(x.times[0] for x in msidset.values())
    tstop = min(x.times[-1] for x in msidset.values())
    msidset.interpolate(dt, tstart, tstop)

    # Finished when we found at least 10 good records (5 mins)
    if len(msidset.times) < 10:
        raise NoTelemetryError(f"Found no telemetry within {days} days of {stop}")

    names = ["time"] + msids
    out = Table([msidset.times] + [msidset[x].vals for x in msids], names=names)
    return out


def get_exclude_mask(tlm):
    mask = np.zeros(len(tlm), dtype="bool")
    exclude_times = get_exclude_times()
    for interval in exclude_times:
        exclude = (tlm["time"] >= CxoTime(interval["start"]).secs) & (
            tlm["time"] < CxoTime(interval["stop"]).secs
        )
        mask[exclude] = True
    return mask


@functools.lru_cache(maxsize=1)
def get_exclude_times(state_key: str = None) -> Table:
    url = EXCLUDE_TIMES_SHEET_URL.format(
        doc_id=kadi.commands.conf.cmd_events_flight_id,
        gid=kadi.commands.conf.cmd_events_exclude_times_gid,
    )
    logger.info(f"Getting exclude times from {url}")
    req = requests.get(url, timeout=30)
    if req.status_code != 200:
        raise ValueError(f"Failed to get exclude times sheet: {req.status_code}")

    exclude_times = Table.read(req.text, format="csv", fill_values=[])
    return exclude_times


@functools.lru_cache(maxsize=128)
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
    for cls in Validate.subclasses:
        logger.info(f"Validating {cls.name}")
        instance = cls(stop=stop, days=days)
        validator = {}
        validator["plot_html"] = instance.get_plot_html()
        validator["name"] = instance.name
        validators.append(validator)

    context = {
        "validators": validators,
    }
    index_template_file = Path(__file__).parent / "templates" / "index_validate.html"
    index_template = index_template_file.read_text()
    template = jinja2.Template(index_template)
    html = template.render(context)

    return html
