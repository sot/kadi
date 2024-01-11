# Licensed under a 3-clause BSD style license - see LICENSE.rst

import functools
import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest
import ska_sun

from kadi.commands.utils import compress_time_series
from kadi.commands.validate import (
    Validate,
    ValidateRoll,
    get_command_sheet_exclude_intervals,
)

# Regression testing for this 5-day period covering a safe mode with plenty of things
# happening. There are a number of violations in this period and a couple of excluded
# intervals.
REGRESSION_STOP = "2022:297"
REGRESSION_DAYS = 5

try:
    get_command_sheet_exclude_intervals()
    CMD_SHEET_AVAILABLE = True
except ConnectionError:
    CMD_SHEET_AVAILABLE = False


def write_regression_data(stop, days, no_exclude):
    cwd = Path(__file__).parent

    print(f"Getting validation regression data for {stop} {days} {no_exclude}")
    data_all = get_all_validator_data(stop, days, no_exclude)
    name = f"validators_{stop.replace(':', '')}_{days}_{no_exclude}.pkl.gz"
    path = cwd / "data" / name
    print(f"Writing validation regression data to {path}")
    with gzip.open(path, "wb") as fh:
        fh.write(pickle.dumps(data_all))


@functools.lru_cache()
def read_regression_data(stop, days, no_exclude):
    cwd = Path(__file__).parent
    name = f"validators_{stop.replace(':', '')}_{days}_{no_exclude}.pkl.gz"
    path = cwd / "data" / name
    if not path.exists():
        raise FileNotFoundError(
            f"validation regression data {path} not found.\n"
            "Run `python -m kadi.commands.tests.test_validate` to generate it."
        )

    print(f"Reading validation regression data from {path}")
    with gzip.open(path, "rb") as fh:
        data_all = pickle.loads(fh.read())

    return data_all


def get_all_validator_data(stop, days, no_exclude):
    data_all = {
        cls.state_name: get_one_validator_data(cls, stop, days, no_exclude)
        for cls in Validate.subclasses
    }

    return data_all


def get_one_validator_data(cls: type[Validate], stop, days, no_exclude):
    """Get a data structure with regression data for one validator

    The structure of the output is below. The values correspond to the "compressed"
    versions that get plotted and used for the violations table.

        data = {
            "vals": {
                "tlm_time": <list of times>,
                "tlm": <list of tlm values>,
                "state_time": <list of times>,
                "state": <list of state values>,
            },
            "violations": Table of violations
        }

    """
    validator = cls(stop=stop, days=days, no_exclude=no_exclude)

    # These property attributes do all the heavy-lifting of getting the data from
    # cheta and kadi.
    validator_vals = {"tlm": validator.tlm_vals, "state": validator.state_vals}
    times = validator.times

    vals_compressed = {}
    for attr, vals in validator_vals.items():
        # TODO: make this a base method
        tm, y = compress_time_series(
            times,
            vals,
            validator.plot_attrs.max_delta_val,
            validator.plot_attrs.max_delta_time,
            max_gap=validator.plot_attrs.max_gap_time,
        )
        vals_compressed[f"{attr}_time"] = tm
        vals_compressed[attr] = y

    data = {"vals": vals_compressed, "violations": validator.violations}
    return data


def test_validate_subclasses():
    """Test that Validate.subclasses matches regression data"""
    data_all_exp = read_regression_data(
        REGRESSION_STOP, REGRESSION_DAYS, no_exclude=False
    )
    assert set(data_all_exp.keys()) == {cls.state_name for cls in Validate.subclasses}


@pytest.mark.skipif(not CMD_SHEET_AVAILABLE, reason="Command sheet not available")
@pytest.mark.parametrize("cls", Validate.subclasses)
@pytest.mark.parametrize("no_exclude", [False, True])
def test_validate_regression(cls, no_exclude, fast_sun_position_method):
    """Test that validator data matches regression data

    This is likely to be fragile. In the future we may need helper function to output
    the data in a more human-readable format to allow for text diffing.
    """
    data_all_exp = read_regression_data(REGRESSION_STOP, REGRESSION_DAYS, no_exclude)
    # Get expected data (from regression pickle file) and actual data from validator
    data_exp = data_all_exp[cls.state_name]
    data_obs = get_one_validator_data(cls, REGRESSION_STOP, REGRESSION_DAYS, no_exclude)

    assert data_obs["vals"].keys() == data_exp["vals"].keys()

    for key, vals_obs in data_obs["vals"].items():
        vals_obs = np.asarray(vals_obs)
        vals_exp = np.asarray(data_exp["vals"][key])
        assert vals_obs.shape == vals_exp.shape
        assert vals_obs.dtype.kind == vals_exp.dtype.kind
        if vals_obs.dtype.kind == "f":
            assert np.allclose(vals_obs, vals_exp, rtol=0, atol=1e-3, equal_nan=True)
        else:
            assert np.all(vals_obs == vals_exp)

    assert np.all(data_obs["violations"] == data_exp["violations"])


@pytest.mark.skipif(not CMD_SHEET_AVAILABLE, reason="Command sheet not available")
def test_off_nominal_roll_violations():
    """Test off_nominal_roll violations over a time range with tail sun observations"""
    # Default sun position method is "accurate".
    off_nom_roll_val = ValidateRoll(stop="2023:327:00:00:00", days=1)
    assert len(off_nom_roll_val.violations) == 0

    with ska_sun.conf.set_temp("sun_position_method_default", "fast"):
        off_nom_roll2 = ValidateRoll(stop="2023:327:00:00:00", days=1)
    assert len(off_nom_roll2.violations) == 3


if __name__ == "__main__":
    write_regression_data(REGRESSION_STOP, REGRESSION_DAYS, no_exclude=False)
    write_regression_data(REGRESSION_STOP, REGRESSION_DAYS, no_exclude=True)
