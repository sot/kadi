# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Validate kadi command states against telemetry.

This is a set of classes that validate the kadi command states against
Chandra telemetry.  The classes are designed to be run either standalone
or from the command-line application ``kadi_validate_states`` (defined in
``kadi.scripts.validate_states``).
"""
import functools
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import astropy.units as u
import jinja2
import numpy as np
import plotly.graph_objects as pgo
import requests
import Ska.Matplotlib
import Ska.Numpy
import Ska.Shell
import Ska.tdb
from astropy.table import Table
from cheta.utils import logical_intervals
from cxotime import CxoTime

import kadi
import kadi.commands
from kadi.commands.states import interpolate_states
from kadi.commands.utils import (
    CxoTimeLike,
    NoTelemetryError,
    add_figure_regions,
    compress_time_series,
    convert_state_code_to_raw_val,
    get_ofp_states,
    get_telem_values,
)

__all__ = [
    "Validate",
    "ValidatePitch",
    "ValidateRoll",
    "ValidateSimpos",
    "ValidateObsid",
    "ValidateDither",
    "ValidatePcadMode",
    "NoTelemetryError",
    "get_command_sheet_exclude_intervals",
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
class PlotAttrs:
    title: str
    ylabel: str
    range: Optional[list] = None


class Validate(ABC):
    subclasses = []

    # Abstract attributes (no elegant solution as of Python 3.11)
    name: str = None
    stop: CxoTime = None
    days: float = None
    state_keys: tuple = None
    plot_attrs: PlotAttrs = None
    msids: tuple = None
    max_delta_val = 0
    max_delta_time = None
    max_gap = 300  # seconds
    min_violation_duration = 32.81  # seconds

    def __init__(self, stop=None, days: float = 14, no_exclude: bool = False):
        """Base class for validation

        :param stop: stop time for validation
        :param days: number of days for validation
        :param no_exclude: if True then do not exclude any data (for testing)
        """
        self.stop = CxoTime(stop)
        self.days = days
        self.start: CxoTime = self.stop - days * u.day
        self.no_exclude = no_exclude

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
            self.add_exclude_intervals()
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
            self._exclude_intervals = Table(
                names=["start", "stop", "states", "comment"], dtype=[str, str, str, str]
            )
        return self._exclude_intervals

    def add_exclude_intervals(self):
        """Base method to exclude intervals, starting with intervals defined in the
        Chandra Command Events Google Sheet.

        This method gets called at the end of self.tlm.

        Subclasses can override this method to add additional intervals to exclude,
        making sure to call super().add_exclude_intervals() first.
        """
        exclude_intervals = get_command_sheet_exclude_intervals()

        for row in exclude_intervals:
            states = row["states"].split()
            if row["states"] == "" or self.name in states:
                self.add_exclude_interval(
                    start=row["start"], stop=row["stop"], comment=row["comment"]
                )

    def add_exclude_interval(
        self,
        start: CxoTimeLike,
        stop: CxoTimeLike,
        comment: str,
        pad_start: Optional[u.Quantity] = None,
        pad_stop: Optional[u.Quantity] = None,
    ):
        """Add an interval to the exclude_intervals table

        The ``stop`` time is padded by ``pad`` (which is a Quantity with time units).
        """
        # For testing, we can skip all exclude intervals to generate violations
        if self.no_exclude:
            return

        start = CxoTime(start)
        if pad_start is not None:
            start = start - pad_start
        stop = CxoTime(stop)
        if pad_stop is not None:
            stop = stop + pad_stop

        # Ensure interval is contained within validation interval.
        if start >= self.stop or stop <= self.start:
            return
        start = max(start, self.start)
        stop = min(stop, self.stop)

        exclude = {
            "start": start.date,
            "stop": stop.date,
            "states": self.name,
            "comment": comment,
        }
        logger.info(f"{self.name}: excluding interval {start} - {stop}: {comment}")
        self.exclude_intervals.add_row(exclude)

    def exclude_ofp_intervals(self, states_expected: List[str]):
        """Exclude intervals where OFP (on-board flight program) is not in the expected state."""
        ofp_states = get_ofp_states(self.stop, self.days)
        for state in ofp_states:
            if state["val"] not in states_expected:
                self.add_exclude_interval(
                    start=state["datestart"],
                    stop=state["datestop"],
                    pad_stop=1 * u.hour,
                    comment=f"CONLOFP={state['val']}",
                )

    def exclude_srdc_intervals(self):
        """Check for SRDC's and exclude them from validation."""
        start = self.start - 20 * u.min  # Catch SRDC start just before interval
        cmds = kadi.commands.get_cmds(start, self.stop, tlmsid="COACTSX")
        ok = np.isin(cmds["coacts1"], [142, 143])
        for cmd in cmds[ok]:
            date = CxoTime(cmd["date"])
            self.add_exclude_interval(
                start=date,
                stop=date + 13 * u.min,
                comment="SRDC",
            )

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
    def violations(self) -> Table:
        if not hasattr(self, "_violations"):
            self._violations = self.get_violations()
        return self._violations

    def get_violations_mask(self) -> np.ndarray:
        """Get the violations mask for this validation class

        This is the default implementation for most validation classes which just checks
        that the telemetry value is within ``max_delta_val`` of the state value.
        """
        bad = np.abs(self.tlm_vals - self.state_vals) > self.max_delta_val
        return bad

    def get_violations(self) -> Table:
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

        for name, color, vals in [
            ("Telem", "#1f77b4", tlm_vals),  # muted blue
            ("State", "#ff7f0e", state_vals),  # safety orange
        ]:
            tm, y = compress_time_series(
                times, vals, self.max_delta_val, self.max_delta_time
            )
            trace = pgo.Scatter(
                name=name,
                x=CxoTime(tm).datetime64,
                y=y,
                mode="lines+markers",
                line={"color": color, "width": 3},
                opacity=0.75,
                showlegend=False,
                marker={"opacity": 0.9, "size": 8},
                hovertemplate="%{x|%Y:%j:%H:%M:%S} %{y}",
            )
            fig.add_trace(trace)

        # fig.update_layout(
        #     {
        #         # "title": (f"{self.name}"),
        #         "yaxis": {
        #             "title": self.plot_attrs.ylabel,
        #         },
        #         "xaxis": {
        #             "title": f"Date",
        #         },
        #     }
        # )
        fig.update_xaxes(title="Date")
        fig.update_yaxes(title=self.plot_attrs.ylabel)
        if self.plot_attrs.range is not None:
            fig.update_yaxes(range=self.plot_attrs.range)
            fig.update_xaxes()

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
                self.tlm[self.msid], self.state_codes
            )
        return self._tlm_vals

    @property
    def state_vals(self):
        if not hasattr(self, "_state_vals"):
            states_interp = interpolate_states(self.states, self.tlm["time"])
            self._state_vals = convert_state_code_to_raw_val(
                states_interp[self.name], self.state_codes
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


class ValidatePitchRollBase(ValidateSingleMsid):
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

    def add_exclude_intervals(self):
        """Exclude any intervals where online flight processing is not normal or safe
        mode"""
        super().add_exclude_intervals()
        self.exclude_ofp_intervals(states_expected=["NRML", "SAFE"])


class ValidatePitch(ValidatePitchRollBase):
    name = "pitch"
    msids = ["pitch_obc_safe"]
    state_keys = ["pitch", "pcad_mode"]
    plot_attrs = PlotAttrs(title="Pitch", ylabel="Pitch (degrees)", range=[40, 180])
    max_delta_time = 3600  # sec
    max_delta_val = 1.0  # deg
    max_delta_vals = {
        "NPNT": 1.0,  # deg
        "NMAN": 20.0,  # deg
        "NSUN": 2.0,  # deg
    }


class ValidateRoll(ValidatePitchRollBase):
    name = "off_nom_roll"
    msids = ["roll_obc_safe"]
    state_keys = ["off_nom_roll", "pcad_mode"]
    plot_attrs = PlotAttrs(title="roll", ylabel="roll (degrees)", range=[-30, 30])
    max_delta_time = 3600  # sec
    max_delta_val = 0.5  # deg
    max_delta_vals = {
        "NPNT": 4,  # deg
        "NMAN": 10.0,  # deg
        "NSUN": 4.0,  # deg
    }


class ValidateSimpos(ValidateSingleMsid):
    name = "simpos"
    msids = ["3tscpos"]
    state_keys = "simpos"
    plot_attrs = PlotAttrs(title="TSCPOS (SIM-Z)", ylabel="SIM-Z (steps)")
    max_delta_val = 10
    # Skip over SIM moves and transient out-of-state values
    min_violation_duration = 328  # seconds


class ValidateObsid(ValidateSingleMsid):
    name = "obsid"
    msids = ["cobsrqid"]
    state_keys = "obsid"
    plot_attrs = PlotAttrs(title="OBSID", ylabel="OBSID")


class ValidateGrating(ValidateStateCode):
    msids = ["4ootgsel", "4ootgmtn"]
    min_violation_duration = 328  # seconds

    def add_exclude_intervals(self):
        super().add_exclude_intervals()
        self.exclude_ofp_intervals(["NRML"])

    @property
    def state_codes(self) -> Table:
        if not hasattr(self, "_state_codes"):
            rows = [
                [0, "INSE"],
                [1, "INSE_MOVE"],
                [2, "RETR_MOVE"],
                [3, "RETR"],
            ]
            self._state_codes = Table(rows=rows, names=["raw_val", "state_code"])
        return self._state_codes

    @property
    def tlm_vals(self):
        if not hasattr(self, "_tlm_vals"):
            vals = np.repeat("RETR", len(self.tlm))
            # use a combination of the select telemetry and the insertion telem to
            # approximate the appropriate telemetry values
            # fmt: off
            ok = ((self.tlm["4ootgsel"] == self.name.upper())
                  & (self.tlm["4ootgmtn"] == "INSE"))
            # fmt: on
            vals[ok] = "INSE"
            self._tlm_vals = convert_state_code_to_raw_val(vals, self.state_codes)
        return self._tlm_vals


class ValidateLETG(ValidateGrating):
    name = "letg"
    state_keys = "letg"
    plot_attrs = PlotAttrs(title="LETG", ylabel="LETG")


class ValidateHETG(ValidateGrating):
    name = "hetg"
    state_keys = "hetg"
    plot_attrs = PlotAttrs(title="HETG", ylabel="HETG")


class ValidateDither(ValidateStateCode):
    name = "dither"
    msids = ["aodithen"]
    state_keys = "dither"
    plot_attrs = PlotAttrs(title="DITHER", ylabel="Dither")

    def add_exclude_intervals(self):
        super().add_exclude_intervals()
        self.exclude_ofp_intervals(["NRML"])
        self.exclude_srdc_intervals()


class ValidatePcadMode(ValidateStateCode):
    name = "pcad_mode"
    msids = ["aopcadmd"]
    state_keys = "pcad_mode"
    plot_attrs = PlotAttrs(title="PCAD mode", ylabel="PCAD mode")

    def add_exclude_intervals(self):
        super().add_exclude_intervals()
        self.exclude_ofp_intervals(["NRML"])
        self.exclude_srdc_intervals()


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
def get_command_sheet_exclude_intervals(state_key: str = None) -> Table:
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


def get_index_page_html(
    stop: CxoTimeLike, days: float, states: List[str], no_exclude: bool = False
):
    """Make a simple HTML page with all the validation plots and information.

    :param stop: stop time for validation interval (CxoTime-like, default=now)
    :param days: length of validation interval (days)
    :returns: HTML string
    """
    validators = []
    violations = []
    for cls in Validate.subclasses:
        if states and cls.name not in states:
            continue
        logger.info(f"Validating {cls.name}")
        instance: Validate = cls(stop=stop, days=days, no_exclude=no_exclude)
        title = f"{instance.plot_attrs.title} (state name = {instance.name!r})"
        validator = {}
        validator["plot_html"] = instance.get_plot_html()
        validator["title"] = title
        validator["violations"] = instance.violations
        validator["exclude_intervals"] = instance.exclude_intervals
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
