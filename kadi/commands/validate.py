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
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import astropy.units as u
import jinja2
import numpy as np
import plotly.graph_objects as pgo
import requests
import ska_tdb
from astropy.table import Table
from cheta.utils import (
    NoTelemetryError,
    get_ofp_states,
    get_telem_table,
    logical_intervals,
)
from cxotime import CxoTime, CxoTimeLike

import kadi
import kadi.commands
from kadi.commands.states import interpolate_states, reduce_states
from kadi.commands.utils import (
    add_figure_regions,
    compress_time_series,
    convert_state_code_to_raw_val,
)

__all__ = [
    "PlotAttrs",
    "Validate",
    "ValidatePitch",
    "ValidateRoll",
    "ValidateSimpos",
    "ValidateObsid",
    "ValidateDither",
    "ValidatePcadMode",
    "ValidateLETG",
    "ValidateHETG",
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
    """
    Plot attributes for a Validate subclass.

    Parameters
    ----------
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    range : list, optional
        Y-axis range.
    max_delta_time : float, optional
        Maximum time delta before a new data point is plotted.
    max_delta_val : float, default 0
        Maximum value delta before a new data point is plotted.
    max_gap_time : float, default 300
        Maximum gap in time before a plot gap is inserted.
    """

    title: str
    ylabel: str
    range: Optional[list] = None
    max_delta_time: Optional[float] = None
    max_delta_val: float = 0
    max_gap_time: float = 300


class Validate(ABC):  # noqa: B024
    """Validate kadi command states against telemetry base class.

    Class attributes are as follows:

    state_name : str
        Name of state to validate.
    stop : CxoTime
        Stop time.
    days : float
        Number of days to validate.
    state_keys_extra : list, optional
        Extra state keys needed for validation.
    plot_attrs : PlotAttrs
        Attributes for plot.
    msids : list
        MSIDs to fetch for telemetry.
    max_delta_val : float
        Maximum value delta to signal a violation.
    max_gap : float
        Maximum gap in telemetry before breaking an interval (sec).
    min_violation_duration : float
        Minimum duration of a violation (sec).


    Parameters
    ----------
    stop
        stop time for validation
    days
        number of days for validation
    no_exclude
        if True then do not exclude any data (for testing)
    """

    subclasses = []

    # Abstract attributes (no elegant solution as of Python 3.11)
    state_name: str = None
    stop: CxoTime = None
    days: float = None
    state_keys_extra: Optional[list] = None
    plot_attrs: PlotAttrs = None
    msids: list = None
    max_delta_val = 0
    max_gap = 300
    min_violation_duration = 32.81

    def __init__(self, stop=None, days: float = 14, no_exclude: bool = False):
        """Base class for validation"""
        self.stop = CxoTime(stop)
        self.days = days
        self.start: CxoTime = self.stop - days * u.day
        self.no_exclude = no_exclude

        # Get the exclude intervals from the google sheet along with any auto-generated
        # ones. This creates self.exclude_intervals which is a Table. By virtue of `tlm`
        # and `states` properties that get used, this also creates self.tlm and
        # self.states.
        self.add_exclude_intervals()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.state_name is not None:
            cls.subclasses.append(cls)

    @functools.cached_property
    def tlm(self):
        logger.info(
            f"Fetching telemetry for {self.msids} between {self.start.date} and"
            f" {self.stop.date}"
        )
        _tlm = get_telem_table(self.msids, self.start, self.stop)
        if len(_tlm) == 0:
            raise NoTelemetryError(
                f"No telemetry for {self.msids} between {self.start.date} and"
                f" {self.stop.date}"
            )
        return _tlm

    @property
    def times(self):
        return self.tlm["time"]

    @property
    def msid(self):
        """Validate classes have first MSID as primary telemetry. Override as needed."""
        return self.msids[0]

    @functools.cached_property
    def states(self):
        state_keys = [self.state_name] + (self.state_keys_extra or [])
        _states = get_states(
            start=self.tlm["time"][0],
            stop=self.tlm["time"][-1],
            state_keys=state_keys,
        )
        return _states

    @functools.cached_property
    def exclude_intervals(self):
        """Intervals that are excluded from state validation.

        This includes manually excluded times from the Command Events sheet
        (e.g. within a few minutes of an IU-reset), or auto-generated
        state-specific intervals like not validating pitch when in NMM.
        """
        _exclude_intervals = Table(
            names=["start", "stop", "states", "comment", "source"],
            dtype=[str, str, str, str, str],
        )
        return _exclude_intervals

    def add_exclude_intervals(self):
        """Base method to exclude intervals

        Starts with intervals defined in the Chandra Command Events Google Sheet.

        This method gets called at the end of self.tlm.

        Subclasses can override this method to add additional intervals to exclude,
        making sure to call super().add_exclude_intervals() first.
        """
        exclude_intervals = get_command_sheet_exclude_intervals()

        for row in exclude_intervals:
            states = row["states"].split()
            if row["states"] == "" or self.state_name in states:
                self.add_exclude_interval(
                    start=row["start"],
                    stop=row["stop"],
                    comment=row["comment"],
                    source="Command Events sheet",
                )

    def add_exclude_interval(
        self,
        start: CxoTimeLike,
        stop: CxoTimeLike,
        comment: str,
        pad_start: Optional[u.Quantity] = None,
        pad_stop: Optional[u.Quantity] = None,
        source: str = "Auto-generated",
    ):
        """Add an interval to the exclude_intervals table

        The ``start`` / ``stop`` times are padded by ``pad_start`` and
        ``pad_stop`` (which are a Quantity objects with time units).

        ``manual`` is a boolean that is used to indicate whether the interval was
        manually specified in the command sheet. Non-manual intervals are annotated
        in the table accordingly.
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
            "states": self.state_name,
            "comment": comment,
            "source": source,
        }
        logger.info(
            f"{self.state_name}: excluding interval {start} - {stop}: {comment}"
        )
        self.exclude_intervals.add_row(exclude)
        self.exclude_intervals.sort("start")

    def exclude_ofp_intervals_except(self, states_expected: List[str]):
        """
        Exclude intervals where OFP (on-board flight program) is not in expected state.

        This includes a padding of 30 minutes after SAFE mode and 5 minutes for non-NRML
        states other than SAFE like STUP, SYON, SYSF etc.
        """
        ofp_states = get_ofp_states(self.start, self.stop)
        for state in ofp_states:
            if state["val"] not in states_expected:
                pad_stop = 30 * u.min if state["val"] == "SAFE" else 5 * u.min
                self.add_exclude_interval(
                    start=state["datestart"],
                    stop=state["datestop"],
                    pad_stop=pad_stop,
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

    @functools.cached_property
    def states_at_times(self):
        """Get the states that correspond to the telemetry times"""
        return interpolate_states(self.states, self.times)

    @functools.cached_property
    def state_vals(self):
        return self.states_at_times[self.state_name].copy()

    @functools.cached_property
    def violations(self) -> Table:
        return self.get_violations()

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

        intervals = logical_intervals(
            self.times, violations_mask, max_gap=self.max_gap, complete_intervals=False
        )
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
            logger.info(f"Compressing {name} data for state {self.state_name}")
            tm, y = compress_time_series(
                times,
                vals,
                self.plot_attrs.max_delta_val,
                self.plot_attrs.max_delta_time,
                max_gap=self.plot_attrs.max_gap_time,
            )
            logger.info(f"Creating {name} scatter plot for state {self.state_name}")

            # Note: would be nice to use Scattergl here since performance is good,
            # but it seems to have a bug where connectgaps=False does not work,
            # sometimes.
            # env CHETA_FETCH_DATA_GAP="--start=2022:300 --stop=2022:301" \
            #   python -m kadi.scripts.validate_states \
            #     --days=4 --stop=2022:301:01:00:00 --state hetg
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

        logger.info(f"Creating HTML for state {self.state_name}")
        html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            default_width=800,
            default_height=500,
        )
        return html

    def get_context(self) -> dict:
        """Get the standard context for a jinja2 template.

        Returns
        -------
        dict
        """
        title = f"{self.plot_attrs.title} (state name = {self.state_name!r})"
        context = {}
        context["plot_html"] = self.get_plot_html()
        context["title"] = title
        context["state_name"] = self.state_name
        context["violations"] = self.violations
        context["exclude_intervals"] = self.exclude_intervals
        return context

    def get_html(
        self, context: Optional[dict] = None, template_text: Optional[str] = None
    ) -> str:
        """Get HTML for validator including section header, violations, and plot

        Parameters
        ----------
        context
            optional dict of context for jinja2 template
        template_text
            optional Jinja2 template text

        Returns
        -------
        str
            HTML string
        """
        if context is None:
            context = self.get_context()
        if template_text is None:
            template_file = Path(__file__).parent / "templates" / "state_validate.html"
            template_text = template_file.read_text()
        template = jinja2.Template(template_text)
        html = template.render(context)

        return html


class ValidateSingleMsid(Validate):
    @functools.cached_property
    def tlm_vals(self):
        return self.tlm[self.msid].copy()


class ValidateStateCode(Validate):
    """Base class for validation of state with state codes like PCAD_MODE"""

    @functools.cached_property
    def state_codes(self) -> Table:
        tsc = ska_tdb.msids.find(self.msid)[0].Tsc
        _state_codes = Table(
            [tsc.data["LOW_RAW_COUNT"], tsc.data["STATE_CODE"]],
            names=["raw_val", "state_code"],
        )
        _state_codes.sort("raw_val")
        return _state_codes

    @functools.cached_property
    def tlm_vals(self):
        return convert_state_code_to_raw_val(self.tlm[self.msid], self.state_codes)

    @functools.cached_property
    def state_vals(self):
        states_interp = interpolate_states(self.states, self.tlm["time"])
        _state_vals = convert_state_code_to_raw_val(
            states_interp[self.state_name], self.state_codes
        )
        return _state_vals

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
    # This subclass uses a different max_delta_val for each pcad_mode
    max_delta_vals: dict = None

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
        """Exclude intervals where online flight processing is not normal or safe mode"""
        super().add_exclude_intervals()
        self.exclude_ofp_intervals_except(states_expected=["NRML", "SAFE"])

        # Find any NSUN intervals and exclude the first 45 minutes during which
        # the maneuver to normal sun occurs.
        states_pcad_mode = reduce_states(
            self.states, state_keys=["pcad_mode"], merge_identical=True
        )
        for state in states_pcad_mode:
            if state["pcad_mode"] == "NSUN":
                self.add_exclude_interval(
                    state["datestart"],
                    state["datestart"],
                    pad_stop=45 * u.min,
                    comment="NSUN maneuver",
                )


class ValidatePitch(ValidatePitchRollBase):
    state_name = "pitch"
    msids = ["pitch_comp"]
    state_keys_extra = ["pcad_mode"]
    plot_attrs = PlotAttrs(
        title="Pitch",
        ylabel="Pitch (degrees)",
        range=[40, 180],
        max_delta_time=3600,  # sec
        max_delta_val=1.0,  # deg
    )
    max_delta_vals = {
        "NPNT": 1.0,  # deg
        "NMAN": 20.0,  # deg
        "NSUN": 2.0,  # deg
    }


class ValidateRoll(ValidatePitchRollBase):
    state_name = "off_nom_roll"
    msids = ["roll_comp"]
    state_keys_extra = ["pitch", "pcad_mode"]
    plot_attrs = PlotAttrs(
        title="Off-nominal roll",
        ylabel="off_nom_roll (degrees)",
        range=[-30, 30],
        max_delta_time=3600,  # sec
        max_delta_val=0.5,  # deg
    )
    max_delta_vals = {
        "NPNT": 2,  # deg
        "NMAN": 12.0,  # deg
        "NSUN": 4.0,  # deg
    }


class ValidateDither(ValidateStateCode):
    state_name = "dither"
    msids = ["aodithen"]
    plot_attrs = PlotAttrs(title="Dither enable", ylabel="Dither")

    def add_exclude_intervals(self):
        super().add_exclude_intervals()
        self.exclude_ofp_intervals_except(["NRML"])
        self.exclude_srdc_intervals()


class ValidatePcadMode(ValidateStateCode):
    state_name = "pcad_mode"
    msids = ["aopcadmd"]
    plot_attrs = PlotAttrs(title="PCAD mode", ylabel="PCAD mode")
    min_violation_duration = 180  # seconds

    def add_exclude_intervals(self):
        super().add_exclude_intervals()
        self.exclude_ofp_intervals_except(["NRML"])
        self.exclude_srdc_intervals()


class ValidateSimpos(ValidateSingleMsid):
    state_name = "simpos"
    msids = ["3tscpos"]
    plot_attrs = PlotAttrs(
        title="TSCPOS (SIM-Z)", ylabel="SIM-Z (steps)", max_delta_val=10
    )  # steps
    max_delta_val = 10  # steps
    # Skip over SIM moves and transient out-of-state values
    min_violation_duration = 420  # seconds


class ValidateObsid(ValidateSingleMsid):
    state_name = "obsid"
    msids = ["cobsrqid"]
    plot_attrs = PlotAttrs(title="OBSID", ylabel="OBSID")


class ValidateGrating(ValidateStateCode):
    msids = ["4ootgsel", "4ootgmtn"]
    min_violation_duration = 328  # seconds

    def add_exclude_intervals(self):
        super().add_exclude_intervals()
        self.exclude_ofp_intervals_except(["NRML"])

    @functools.cached_property
    def state_codes(self) -> Table:
        rows = [
            [0, "INSE"],
            [1, "INSE_MOVE"],
            [2, "RETR_MOVE"],
            [3, "RETR"],
        ]
        _state_codes = Table(rows=rows, names=["raw_val", "state_code"])
        return _state_codes

    @functools.cached_property
    def tlm_vals(self):
        vals = np.repeat("RETR", len(self.tlm))
        # use a combination of the select telemetry and the insertion telem to
        # approximate the appropriate telemetry values
        # fmt: off
        ok = ((self.tlm["4ootgsel"] == self.state_name.upper())
              & (self.tlm["4ootgmtn"] == "INSE"))
        # fmt: on
        vals[ok] = "INSE"
        _tlm_vals = convert_state_code_to_raw_val(vals, self.state_codes)
        return _tlm_vals


class ValidateLETG(ValidateGrating):
    state_name = "letg"
    plot_attrs = PlotAttrs(title="LETG", ylabel="LETG")


class ValidateHETG(ValidateGrating):
    state_name = "hetg"
    plot_attrs = PlotAttrs(title="HETG", ylabel="HETG")


class ValidateSunPosMon(ValidateStateCode):
    state_name = "sun_pos_mon"
    msids = ["aopssupm"]
    plot_attrs = PlotAttrs(title="Sun position monitor", ylabel="Sun position monitor")
    min_violation_duration = 400

    def add_exclude_intervals(self):
        super().add_exclude_intervals()
        self.exclude_ofp_intervals_except(["NRML"])

    @functools.cached_property
    def state_vals(self):
        """Convert ENAB (commanded states) to ACT (telemetry).

        The "ENAB" is an artifact of the backstop history sun position monitor states.
        This method is otherwise equivalent to the ValidateStateCode method.
        """
        states_interp = interpolate_states(self.states, self.tlm["time"])
        state_vals = states_interp[self.state_name]
        state_vals[state_vals == "ENAB"] = "ACT"
        state_vals_raw = convert_state_code_to_raw_val(state_vals, self.state_codes)
        return state_vals_raw


class ValidateACISStatePower(ValidateSingleMsid):
    state_name = "dpa_power"
    msids = ["dpa_power"]
    state_keys_extra = ["ccd_count", "clocking", "feps", "fep_count", "simpos"]
    plot_attrs = PlotAttrs(title="DPA Power", ylabel="DPA Power (W)")
    min_violation_duration = 600.0  # seconds
    max_delta_val = 1.0  # W

    def __init__(self, stop=None, days: float = 14, no_exclude: bool = False):
        from joblib import load

        super().__init__(stop=stop, days=days, no_exclude=no_exclude)
        self.model, self.scaler_X, self.scaler_y = load("dpa_power_model.joblib")

    @functools.cached_property
    def states(self):
        _states = get_states(
            start=self.tlm["time"][0],
            stop=self.tlm["time"][-1],
            state_keys=self.state_keys_extra,
        )
        return _states

    @functools.cached_property
    def state_vals(self):
        istates = interpolate_states(self.states, self.times)
        feps = defaultdict(list)
        # create on-off states for FEPs
        for row in istates:
            for i in range(6):
                feps[f"FEP{i}"].append(float(str(i) in row["feps"]))
        fep_keys = [f"FEP{i}" for i in range(6)]
        for fk in fep_keys:
            istates[fk] = feps[fk]
        df = istates.to_pandas()
        keep_cols = self.state_keys_extra + fep_keys
        keep_cols.remove("feps")
        XX = df.drop([col for col in istates.colnames if col not in keep_cols], axis=1)
        XX = self.scaler_X.fit_transform(XX.values)
        yy = self.model.predict(XX)
        # Inverse transform the predictions and actual values
        _state_vals = self.scaler_y.inverse_transform(yy.reshape(-1, 1)).flatten()
        return _state_vals

    def add_exclude_intervals(self):
        super().add_exclude_intervals()
        # The following corresponds to an ACIS watchdog reboot
        self.add_exclude_interval(
            "2022:016:00:05:23", "2022:018:18:43:48", "ACIS Watchdog Reboot"
        )


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
def get_command_sheet_exclude_intervals() -> Table:
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

    Parameters
    ----------
    start
        start date (CxoTime-like)
    stop
        stop date (CxoTime-like)
    state_keys
        list of state keys to get

    Returns
    -------
    Table
        States in the interval
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
