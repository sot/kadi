#!/usr/bin/env python

"""
"""
import base64
import functools
import io
import logging
import os
import time
from abc import ABC
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import List, Optional

import jinja2
import matplotlib
import numpy as np
import plotly.graph_objects as pgo
import requests
from astropy.table import Table, vstack

import kadi
import kadi.commands

matplotlib.use("Agg")
import cheta.fetch_eng as fetch
import matplotlib.pyplot as plt
import matplotlib.style
import Ska.Matplotlib
import Ska.Numpy
import Ska.Shell
import Ska.tdb
from cheta.utils import logical_intervals
from cxotime import CxoTime
from Quaternion import Quat, normalize
from Ska.Matplotlib import cxctime2plotdate as cxc2pd
from Ska.Matplotlib import plot_cxctime

from kadi.commands.states import interpolate_states

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
]

matplotlib.style.use("bmh")

kadi.commands.conf.commands_version = "2"
# kadi.logger.setLevel("DEBUG")
logger = logging.getLogger(__name__)

# URL to download Bad Times google sheet
# See https://stackoverflow.com/questions/33713084 (2nd answer)
BAD_TIMES_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/{doc_id}/export?"
    "format=csv"
    "&id={doc_id}"
    "&gid={gid}"
)

plot_cxctime = functools.partial(plot_cxctime, interactive=False)


plt.rcParams["axes.formatter.limits"] = (-4, 4)
plt.rcParams["font.size"] = 9
TASK = "validate_states"
VERSION = 5
TASK_DATA = os.path.join(os.environ["SKA"], "data", TASK)
URL = "http://cxc.harvard.edu/mta/ASPECT/" + TASK

TITLE = {
    "dp_pitch": "Pitch",
    "obsid": "OBSID",
    "tscpos": "TSCPOS (SIM-Z)",
    "pcad_mode": "PCAD MODE",
    "dither": "DITHER",
    "letg": "LETG",
    "hetg": "HETG",
    "pointing": "Commanded ATT Radial Offset",
    "roll": "Commanded ATT Roll Offset",
}

LABELS = {
    "dp_pitch": "Pitch (degrees)",
    "obsid": "OBSID",
    "tscpos": "SIM-Z (steps)",
    "pcad_mode": "PCAD MODE",
    "dither": "Dither",
    "letg": "LETG",
    "hetg": "HETG",
    "pointing": "Radial Offset (arcsec)",
    "roll": "Roll Offset (arcsec)",
}


FMTS = {
    "dp_pitch": "%.3f",
    "obsid": "%d",
    "dither": "%d",
    "hetg": "%d",
    "letg": "%d",
    "pcad_mode": "%d",
    "tscpos": "%d",
    "pointing": "%.2f",
    "roll": "%.2f",
}

MODE_SOURCE = {"pcad_mode": "aopcadmd", "dither": "aodithen"}

MODE_MSIDS = {
    "pcad_mode": ["NMAN", "NPNT", "NSUN", "NULL", "PWRF", "RMAN", "STBY"],
    "dither": ["ENAB", "DISA"],
    "hetg": ["INSE", "RETR"],
    "letg": ["INSE", "RETR"],
}

# validation limits
# 'msid' : (( quantile, absolute max value ))
# Note that the quantile needs to be in the set (1, 5, 16, 50, 84, 95, 99)
VALIDATION_LIMITS = {
    "DP_PITCH": (
        (1, 7.0),
        (99, 7.0),
        (5, 0.5),
        (95, 0.5),
    ),
    "POINTING": (
        (1, 0.05),
        (99, 0.05),
    ),
    "ROLL": (
        (1, 0.05),
        (99, 0.05),
    ),
    "TSCPOS": (
        (1, 2.0),
        (99, 2.0),
    ),
}

# number of tolerated differences for string / discrete msids
# 'msid' : n differences before violation recorded
# this is scaled by the number of toggles or expected
# changes in the msid
VALIDATION_SCALE_COUNT = {"OBSID": 2, "HETG": 2, "LETG": 2, "PCAD_MODE": 2, "DITHER": 3}


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


class NoTelemetryError(Exception):
    """No telemetry available for the specified interval"""


@dataclass
class PlotAttrs:
    title: str
    ylabel: str


class Validate(ABC):
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

    def make_plot(self):
        # Interpolate states onto the tlm.date grid
        state_vals = self.state_vals
        tlm_vals = self.tlm_vals

        fig, ax = plt.subplots(figsize=(7, 3.5))
        tlm = self.tlm
        plot_cxctime(tlm["time"], tlm_vals, fig=fig, ax=ax, fmt="-", color="C0")
        plot_cxctime(tlm["time"], state_vals, fig=fig, ax=ax, fmt="-", color="C1")

        bad_times = get_bad_times()
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

        # Add "background" grey rectangles for excluded time regions to vs-time plot
        for bad in bad_times:
            bad_start = CxoTime(bad["start"]).plot_date
            bad_stop = CxoTime(bad["stop"]).plot_date
            if not ((bad_stop >= xlims[0]) & (bad_start <= xlims[1])):
                continue
            rect = matplotlib.patches.Rectangle(
                (bad_start, ylims[0]),
                bad_stop - bad_start,
                ylims[1] - ylims[0],
                alpha=0.2,
                facecolor="black",
                edgecolor="none",
            )
            ax.add_patch(rect)

        ax.margins(0.05)
        ax.set_title(self.plot_attrs.title)
        ax.set_ylabel(self.plot_attrs.ylabel)
        return fig, ax


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


def OLD_config_logging(outdir, verbose):
    """Set up file and console logger.
    See http://docs.python.org/library/logging.html#logging-to-multiple-destinations
    """
    # Disable auto-configuration of root logger by adding a null handler.
    # This prevents other modules (e.g. Chandra.cmd_states) from generating
    # a streamhandler by just calling logging.info(..).
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

    rootlogger = logging.getLogger()
    rootlogger.addHandler(NullHandler())

    loglevel = {0: logging.CRITICAL, 1: logging.INFO, 2: logging.DEBUG}.get(
        verbose, logging.INFO
    )

    logger = logging.getLogger(TASK)
    logger.setLevel(loglevel)

    formatter = logging.Formatter("%(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    filehandler = logging.FileHandler(
        filename=os.path.join(outdir, "run.dat"), mode="w"
    )
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)


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


def get_bad_mask(tlm):
    mask = np.zeros(len(tlm), dtype="bool")
    bad_times = get_bad_times()
    for interval in bad_times:
        bad = (tlm["time"] >= CxoTime(interval["start"]).secs) & (
            tlm["time"] < CxoTime(interval["stop"]).secs
        )
        mask[bad] = True
    return mask


@functools.lru_cache(maxsize=1)
def get_bad_times() -> Table:
    url = BAD_TIMES_SHEET_URL.format(
        doc_id=kadi.commands.conf.cmd_events_flight_id,
        gid=kadi.commands.conf.cmd_events_bad_times_gid,
    )
    logger.info(f"Getting bad times from {url}")
    req = requests.get(url, timeout=30)
    if req.status_code != 200:
        raise ValueError(f"Failed to get bad times sheet: {req.status_code}")

    bad_times = Table.read(req.text, format="csv")
    return bad_times


def get_states(start, stop, state_keys) -> Table:
    """Get states exactly covering date range

    :param start: start date
    :param stop: stop date
    :param state_keys: list of state keys to get
    :returns: Table of states
    """
    from kadi.commands.states import get_states as get_states_kadi

    logger.info("Using kadi.commands.states to get cmd_states")
    logger.info(f"Getting commanded states between {start} - {stop}")

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


def validate_cmd_states(days=4, run_start_time=None, scenario=None):
    # Store info relevant to processing for use in outputs
    proc = dict(
        run_user=os.environ["USER"],
        run_time=time.ctime(),
        errors=[],
    )

    scale = 1.0

    tnow = CxoTime(run_start_time).secs
    tstart = tnow

    # Get temperature telemetry for 3 weeks prior to min(tstart, NOW)
    tlm = get_telem_values(
        tstart,
        [
            "3tscpos",
            "dp_pitch",
            "aoacaseq",
            "aodithen",
            "cacalsta",
            "cobsrqid",
            "aofunlst",
            "aopcadmd",
            "4ootgsel",
            "4ootgmtn",
            "aocmdqt1",
            "aocmdqt2",
            "aocmdqt3",
            # "1de28avo",
            # "1deicacu",
            # "1dp28avo",
            # "1dpicacu",
            # "1dp28bvo",
            # "1dpicbcu",
        ],
        days=days,
        name_map={"3tscpos": "tscpos", "cobsrqid": "obsid"},
    )

    states = get_states(tlm[0].date, tlm[-1].date)

    # Get bad time intervals
    bad_time_mask = get_bad_mask(tlm)

    # Interpolate states onto the tlm.date grid
    state_vals = interpolate_states(states, tlm["time"])

    # "Forgive" dither intervals with dark current replicas
    # This will also exclude dither disables that are in cmd states for standard dark cals
    dark_mask = np.zeros(len(tlm), dtype="bool")
    dark_times = []
    # Find dither "disable" states from tlm
    dith_disa_states = logical_intervals(tlm["time"], tlm["aodithen"] == "DISA")
    for state in dith_disa_states:
        # Index back into telemetry for each of these constant dither disable states
        idx0 = np.searchsorted(tlm["time"], state["tstart"], side="left")
        idx1 = np.searchsorted(tlm["time"], state["tstop"], side="right")
        # If any samples have aca calibration flag, mark interval for exclusion.
        if np.any(tlm["cacalsta"][idx0:idx1] != "OFF "):
            dark_mask[idx0:idx1] = True
            dark_times.append({"start": state["datestart"], "stop": state["datestop"]})

    # Calculate the 4th term of the commanded quaternions
    cmd_q4 = np.sqrt(
        np.abs(1.0 - tlm["aocmdqt1"] ** 2 - tlm["aocmdqt2"] ** 2 - tlm["aocmdqt3"] ** 2)
    )
    raw_tlm_q = np.vstack(
        [tlm["aocmdqt1"], tlm["aocmdqt2"], tlm["aocmdqt3"], cmd_q4]
    ).transpose()

    # Calculate angle/roll differences in state cmd vs tlm cmd quaternions
    raw_state_q = np.vstack(
        [state_vals[n] for n in ["q1", "q2", "q3", "q4"]]
    ).transpose()
    tlm_q = normalize(raw_tlm_q)
    # only use values that aren't NaNs
    good = ~np.isnan(np.sum(tlm_q, axis=-1))
    # and are in NPNT
    npnt = tlm["aopcadmd"] == "NPNT"
    # and are in KALM after the first 2 sample of the transition
    not_kalm = tlm["aoacaseq"] != "KALM"
    kalm = ~(not_kalm | np.hstack([[False, False], not_kalm[:-2]]))
    # and aren't during momentum unloads or in the first 2 samples after unloads
    unload = tlm["aofunlst"] != "NONE"
    no_unload = ~(unload | np.hstack([[False, False], unload[:-2]]))
    ok = good & npnt & kalm & no_unload & ~bad_time_mask
    state_q = normalize(raw_state_q)
    dot_q = np.sum(tlm_q[ok] * state_q[ok], axis=-1)
    dot_q[dot_q > 1] = 1
    angle_diff = np.degrees(2 * np.arccos(dot_q))
    angle_diff = np.min([angle_diff, 360 - angle_diff], axis=0)
    roll_diff = Quat(q=tlm_q[ok]).roll - Quat(q=state_q[ok]).roll
    roll_diff = np.min([roll_diff, 360 - roll_diff], axis=0)

    for msid in MODE_SOURCE:
        tlm_col = np.zeros(len(tlm))
        state_col = np.zeros(len(tlm))
        for mode, idx in zip(MODE_MSIDS[msid], count()):
            tlm_col[tlm[MODE_SOURCE[msid]] == mode] = idx
            state_col[state_vals[msid] == mode] = idx
        tlm = Ska.Numpy.add_column(tlm, msid, tlm_col)
        state_vals = Ska.Numpy.add_column(state_vals, "{}_pred".format(msid), state_col)

    for msid in ["letg", "hetg"]:
        txt = np.repeat("RETR", len(tlm))
        # use a combination of the select telemetry and the insertion telem to
        # approximate the state_vals values
        txt[(tlm["4ootgsel"] == msid.upper()) & (tlm["4ootgmtn"] == "INSE")] = "INSE"
        tlm_col = np.zeros(len(tlm))
        state_col = np.zeros(len(tlm))
        for mode, idx in zip(MODE_MSIDS[msid], count()):
            tlm_col[txt == mode] = idx
            state_col[state_vals[msid] == mode] = idx
        tlm = Ska.Numpy.add_column(tlm, msid, tlm_col)
        state_vals = Ska.Numpy.add_column(state_vals, "{}_pred".format(msid), state_col)

    diff_only = {
        "pointing": {"diff": angle_diff * 3600, "time": tlm["time"][ok]},
        "roll": {"diff": roll_diff * 3600, "time": tlm["time"][ok]},
    }

    pred = {
        "dp_pitch": state_vals.pitch,
        "obsid": state_vals.obsid,
        "dither": state_vals["dither_pred"],
        "pcad_mode": state_vals["pcad_mode_pred"],
        "letg": state_vals["letg_pred"],
        "hetg": state_vals["hetg_pred"],
        "tscpos": state_vals.simpos,
        "pointing": 1,
        "roll": 1,
    }

    plots_validation = []
    valid_viols = []
    logger.info("Making validation plots and quantile table")
    quantiles = (1, 5, 16, 50, 84, 95, 99)
    # store lines of quantile table in a string and write out later
    quant_table = ""
    quant_head = ",".join(["MSID"] + ["quant%d" % x for x in quantiles])
    quant_table += quant_head + "\n"
    for fig_id, msid in enumerate(sorted(pred)):
        plot = dict(msid=msid.upper())
        fig, ax = plt.subplots(figsize=(7, 3.5))

        if msid not in diff_only:
            if msid in MODE_MSIDS:
                state_msid = np.zeros(len(tlm))
                for mode, idx in zip(MODE_MSIDS[msid], count()):
                    state_msid[state_vals[msid] == mode] = idx
                ticklocs, fig, ax = plot_cxctime(
                    tlm["time"], tlm[msid], fig=fig, ax=ax, fmt="-r"
                )
                ticklocs, _, _ = plot_cxctime(
                    tlm["time"], state_msid, fig=fig, ax=ax, fmt="-b"
                )
                ax.set_yticks(range(len(MODE_MSIDS[msid])), MODE_MSIDS[msid])
            else:
                ticklocs, fig, ax = plot_cxctime(
                    tlm["time"], tlm[msid] / scale, fig=fig, ax=ax, fmt="-r"
                )
                ticklocs, fig, ax = plot_cxctime(
                    tlm["time"], pred[msid] / scale, fig=fig, ax=ax, fmt="-b"
                )
        else:
            ticklocs, fig, ax = plot_cxctime(
                diff_only[msid]["time"],
                diff_only[msid]["diff"] / scale,
                fig=fig,
                ax=ax,
                fmt="-k",
            )
        plot["diff_only"] = msid in diff_only
        ax.set_title(TITLE[msid])
        ax.set_ylabel(LABELS[msid])
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

        bad_times = get_bad_times()

        # Add the time intervals of dark current calibrations that have been
        # excluded from the diffs to the "bad_times" for validation so they also
        # can be marked with grey rectangles in the plot.  This is only really
        # visible with interactive/zoomed plot.
        if msid in ["dither", "pcad_mode"] and len(dark_times) > 0:
            bad_times = vstack([bad_times, Table(dark_times)])

        # Add "background" grey rectangles for excluded time regions to vs-time plot
        for bad in bad_times:
            bad_start = cxc2pd([CxoTime(bad["start"]).secs])[0]
            bad_stop = cxc2pd([CxoTime(bad["stop"]).secs])[0]
            if not ((bad_stop >= xlims[0]) & (bad_start <= xlims[1])):
                continue
            rect = matplotlib.patches.Rectangle(
                (bad_start, ylims[0]),
                bad_stop - bad_start,
                ylims[1] - ylims[0],
                alpha=0.2,
                facecolor="black",
                edgecolor="none",
            )
            ax.add_patch(rect)

        ax.margins(0.05)
        fig.tight_layout()

        out = io.BytesIO()
        fig.savefig(out, format="JPEG")
        plt.close(fig)
        plot["lines"] = base64.b64encode(out.getvalue()).decode("ascii")

        if msid not in diff_only:
            ok = ~bad_time_mask
            if msid in ["dither", "pcad_mode"]:
                # For these two validations also ignore intervals during a dark
                # current calibration
                ok &= ~dark_mask
            diff = tlm[msid][ok] - pred[msid][ok]
        else:
            diff = diff_only[msid]["diff"]

        # Sort the diffs in-place because we're just using them in aggregate
        diff = np.sort(diff)

        # if there are only a few residuals, don't bother with histograms
        if msid.upper() in VALIDATION_SCALE_COUNT:
            plot["samples"] = len(diff)
            plot["diff_count"] = np.count_nonzero(diff)
            plot["n_changes"] = 1 + np.count_nonzero(pred[msid][1:] - pred[msid][0:-1])
            if plot["diff_count"] < (
                plot["n_changes"] * VALIDATION_SCALE_COUNT[msid.upper()]
            ):
                plots_validation.append(plot)
                continue
            # if the msid exceeds the diff count, add a validation violation
            else:
                viol = {
                    "msid": "{}_diff_count".format(msid),
                    "value": plot["diff_count"],
                    "limit": plot["n_changes"] * VALIDATION_SCALE_COUNT[msid.upper()],
                    "quant": None,
                }
                valid_viols.append(viol)
                logger.info(
                    "WARNING: %s %d discrete diffs exceed limit of %d"
                    % (
                        msid,
                        plot["diff_count"],
                        plot["n_changes"] * VALIDATION_SCALE_COUNT[msid.upper()],
                    )
                )

        # Make quantiles
        if msid != "obsid":
            quant_line = "%s" % msid
            for quant in quantiles:
                quant_val = diff[(len(diff) * quant) // 100]
                plot["quant%02d" % quant] = FMTS[msid] % quant_val
                quant_line += "," + FMTS[msid] % quant_val
            quant_table += quant_line + "\n"

        for histscale in ("lin", "log"):
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(diff / scale, bins=50, log=(histscale == "log"))
            ax.set_title(msid.upper() + " residuals: telem - cmd states", fontsize=11)
            ax.set_xlabel(LABELS[msid])
            fig.subplots_adjust(bottom=0.18)
            fig.tight_layout()

            out = io.BytesIO()
            fig.savefig(out, format="JPEG")
            plt.close(fig)
            plot["hist" + histscale] = base64.b64encode(out.getvalue()).decode("ascii")

        plots_validation.append(plot)

    valid_viols.extend(make_validation_viols(plots_validation))
    html = get_index_html(proc, plots_validation, valid_viols)
    return html


def get_index_html(proc, plots_validation, valid_viols=None):
    """
    Make output text in HTML format in outdir.
    """
    context = {
        "valid_viols": valid_viols,
        "proc": proc,
        "plots_validation": plots_validation,
    }
    index_template_file = Path(__file__).parent / "templates" / "index.html"
    index_template = index_template_file.read_text()
    # index_template = re.sub(r' %}\n', ' %}', index_template)
    template = jinja2.Template(index_template)
    out = template.render(context)
    return out


def make_validation_viols(plots_validation):
    """
    Find limit violations where MSID quantile values are outside the
    allowed range.
    """

    logger.info("Checking for validation violations")

    viols = []

    for plot in plots_validation:
        # 'plot' is actually a structure with plot info and stats about the
        #  plotted data for a particular MSID.  'msid' can be a real MSID
        #  (1PDEAAT) or pseudo like 'POWER'
        msid = plot["msid"]

        # Make sure validation limits exist for this MSID
        if msid not in VALIDATION_LIMITS:
            continue

        # Cycle through defined quantiles (e.g. 99 for 99%) and corresponding
        # limit values for this MSID.
        for quantile, limit in VALIDATION_LIMITS[msid]:
            # Get the quantile statistic as calculated when making plots
            msid_quantile_value = float(plot["quant%02d" % quantile])

            # Check for a violation and take appropriate action
            if abs(msid_quantile_value) > limit:
                viol = {
                    "msid": msid,
                    "value": msid_quantile_value,
                    "limit": limit,
                    "quant": quantile,
                }
                viols.append(viol)
                logger.info(
                    "WARNING: %s %d%% quantile value of %s exceeds limit of %.2f"
                    % (msid, quantile, msid_quantile_value, limit)
                )

    return viols


def get_options():
    from optparse import OptionParser

    parser = OptionParser()
    parser.set_defaults()
    parser.add_option(
        "--days", type="float", default=4.0, help="Days of validation data (days)"
    )
    parser.add_option(
        "--run-start-time", help="Mock tool run time for regression testing"
    )
    parser.add_option(
        "--verbose",
        type="int",
        default=2,
        help="Verbosity (0=quiet, 1=normal, 2=debug)",
    )

    opt, args = parser.parse_args()
    return opt, args


if __name__ == "__main__":
    opt, args = get_options()
    out = validate_cmd_states(days=opt.days, run_start_time=opt.run_start_time)
    Path("index.html").write_text(out)
