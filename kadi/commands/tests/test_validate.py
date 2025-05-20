# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Test the validation of command loads.

This relies on regression data files that are generated or updated by running this test
file with the environment variable KADI_VALIDATE_WRITE_REGRESSION set::

  $ env KADI_VALIDATE_WRITE_REGRESSION=1 pytest -k test_validate -s -v

Regression testing uses a 5-day period covering a safe mode with plenty of things
happening. There are a number of violations in this period and a couple of excluded
intervals.
"""

import functools
import os
from pathlib import Path

import numpy as np
import pytest
import ska_sun
import yaml
from cxotime import CxoTime
from ska_helpers.utils import temp_env_var
from testr.test_helper import has_internet

import kadi.commands as kc
from kadi.commands.utils import compress_time_series
from kadi.commands.validate import (
    Validate,
    ValidateACISStatePower,
    ValidateRoll,
)

REGRESSION_STOP = "2022:297"
REGRESSION_DAYS = 5

HAS_INTERNET = has_internet()


@pytest.fixture(scope="module")
def local_testing(tmp_path_factory):
    """Several context managers to set up the environment for local testing."""
    kc.clear_caches()
    cmds_dir = tmp_path_factory.mktemp("cmds_dir")
    with (
        kc.conf.set_temp("cache_loads_in_astropy_cache", True),
        kc.conf.set_temp("clean_loads_dir", False),
        kc.conf.set_temp("commands_dir", str(cmds_dir)),
    ):
        yield
    kc.clear_caches()


@pytest.fixture()
def regress_stop(local_testing):
    """Set the default stop time for regression testing

    This will force regenerating "recent" commands in case command events or command
    sets have changed.

    This fixture cannot be combined with the module-scoped ``local_testing`` fixture
    because this causes problems for other tests that rely on the default stop time.

    To see what is happening behind the scenes, update the code to enable kadi debug
    logging (kadi.commands.logger.setLevel("DEBUG")) and run the tests with ``-s``.
    """
    with temp_env_var("CXOTIME_NOW", REGRESSION_STOP):
        yield


def get_regression_data_path(cls, stop, days, no_exclude):
    stop = stop.replace(":", "")
    name = cls.state_name
    fn = f"validator_{name}_{stop}_{days}_{no_exclude}.yaml"
    return Path(__file__).parent / "data" / fn


def write_regression_data(cls, stop, days, no_exclude, data):
    print(
        "Getting validation regression data for "
        f"{cls.state_name} {stop} {days} {no_exclude}"
    )
    path = get_regression_data_path(cls, stop, days, no_exclude)
    print(f"Writing validation regression data to {path}")
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh)


@functools.lru_cache()
def read_regression_data(cls, stop, days, no_exclude):
    path = get_regression_data_path(cls, stop, days, no_exclude)
    if not path.exists():
        raise FileNotFoundError(
            f"validation regression data {path} not found.\n"
            "Run `python -m kadi.commands.tests.test_validate` to generate it."
        )

    print(f"Reading validation regression data from {path}")
    with open(path) as fh:
        data = yaml.safe_load(fh)

    return data


def get_one_validator_data(cls: type[Validate], stop, days, no_exclude):
    """Get a data structure with regression data for one validator

    The structure of the output is below. The values correspond to the "compressed"
    versions that get plotted and used for the violations table.

        data = {
            "vals_compressed": {
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
        tms, ys = compress_time_series(
            times,
            vals,
            validator.plot_attrs.max_delta_val,
            validator.plot_attrs.max_delta_time,
            max_gap=validator.plot_attrs.max_gap_time,
        )
        ys = np.array(ys)
        if ys.dtype.kind == "f":
            ys = np.round(ys, 3)
        vals_compressed[f"{attr}_time"] = CxoTime(tms).date.tolist()
        vals_compressed[attr] = ys.tolist()

    data = {
        "violations": {
            "start": validator.violations["start"].tolist(),
            "stop": validator.violations["stop"].tolist(),
        },
        "vals_compressed": vals_compressed,
    }
    return data


@pytest.mark.skipif(not HAS_INTERNET, reason="Command sheet not available")
@pytest.mark.parametrize("cls", Validate.subclasses)
@pytest.mark.parametrize("no_exclude", [False, True])
def test_validate_regression(
    cls,
    no_exclude,
    fast_sun_position_method,
    regress_stop,
    disable_hrc_scs107_commanding,
):
    """Test that validator data matches regression data

    This is likely to be fragile. In the future we may need helper function to output
    the data in a more human-readable format to allow for text diffing.
    """
    data_obs = get_one_validator_data(cls, REGRESSION_STOP, REGRESSION_DAYS, no_exclude)
    if os.environ.get("KADI_VALIDATE_WRITE_REGRESSION"):
        write_regression_data(
            cls, REGRESSION_STOP, REGRESSION_DAYS, no_exclude, data_obs
        )
    data_exp = read_regression_data(cls, REGRESSION_STOP, REGRESSION_DAYS, no_exclude)
    # Get expected data (from regression pickle file) and actual data from validator
    # data_exp = data_all_exp[cls.state_name]

    assert data_obs["vals_compressed"].keys() == data_exp["vals_compressed"].keys()

    for key, vals_obs in data_obs["vals_compressed"].items():
        vals_obs = np.asarray(vals_obs)  # noqa: PLW2901
        vals_exp = np.asarray(data_exp["vals_compressed"][key])
        assert vals_obs.shape == vals_exp.shape
        assert vals_obs.dtype.kind == vals_exp.dtype.kind
        if vals_obs.dtype.kind == "f":
            assert np.allclose(vals_obs, vals_exp, rtol=0, atol=1e-3, equal_nan=True)
        else:
            assert np.all(vals_obs == vals_exp)

    for key in ("start", "stop"):
        assert np.all(data_obs["violations"][key] == data_exp["violations"][key])


@pytest.mark.skipif(not HAS_INTERNET, reason="Command sheet not available")
def test_off_nominal_roll_violations():
    """Test off_nominal_roll violations over a time range with tail sun observations"""
    # Default sun position method is "accurate".
    off_nom_roll_val = ValidateRoll(stop="2023:327:00:00:00", days=1)
    assert len(off_nom_roll_val.violations) == 0

    with ska_sun.conf.set_temp("sun_position_method_default", "fast"):
        off_nom_roll2 = ValidateRoll(stop="2023:327:00:00:00", days=1)
    assert len(off_nom_roll2.violations) == 3


@pytest.mark.skipif(not HAS_INTERNET, reason="Command sheet not available")
def test_acis_power_violations():
    start_viols = [
        "2025:133:22:37:17.616",
    ]
    stop_viols = [
        "2025:133:23:00:48.016",
    ]
    acis_power_val = ValidateACISStatePower(stop="2025:134:00:00:00", days=1.0)
    assert len(acis_power_val.violations) == 1
    assert list(acis_power_val.violations["start"]) == start_viols
    assert list(acis_power_val.violations["stop"]) == stop_viols


if __name__ == "__main__":
    for cls in Validate.subclasses:
        write_regression_data(cls, REGRESSION_STOP, REGRESSION_DAYS, no_exclude=False)
        write_regression_data(cls, REGRESSION_STOP, REGRESSION_DAYS, no_exclude=True)
