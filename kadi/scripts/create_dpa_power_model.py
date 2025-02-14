import argparse
import logging
from collections import defaultdict

import astropy.units as u
import numpy as np
from cheta import fetch_sci as fetch
from cxotime import CxoTime
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from kadi import __version__  # noqa: F401
from kadi.commands.states import get_states, interpolate_states

logger = logging.getLogger("kadi")

exclude_intervals = [
    ("2020:023:17:57:00", "2020:023:23:07:30"),  # ACIS FSW v56 Update
    ("2022:016:00:05:23", "2022:018:18:43:48"),  # ACIS Watchdog Reboot
    ("2023:263:00:37:00", "2023:263:07:40:00"),  # ACIS FSW v60 Update
]


def get_args():
    parser = argparse.ArgumentParser(
        description="Create DPA power model by fitting a MLP regressor telemetry "
        "to commanded states, for purposes of state validation."
    )
    parser.add_argument("stop", type=str, help="Stop date for model.")
    parser.add_argument(
        "days",
        type=float,
        help="Number of days backward from start to use in the model.",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.7,
        help="Fraction of data to use for training.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG | INFO | WARNING, default=INFO)",
    )
    return parser.parse_args()


def main():
    args = get_args()

    stop = CxoTime(args.stop)
    start = stop - args.days * u.day

    # States to obtain for the model
    state_keys = [
        "ccd_count",
        "clocking",
        "feps",
        "ccds",
        "fep_count",
        "si_mode",
    ]

    # Get commanded states
    states = get_states(
        start=start, stop=stop, merge_identical=True, state_keys=state_keys
    )

    # Get dpa_power MSID
    msids = fetch.Msidset(["dpa_power"], start, stop, stat="5min")

    # interpolate the states to the times of the MSIDs
    int_states = interpolate_states(states, msids["dpa_power"].times)
    int_states["dpa_power"] = msids["dpa_power"].vals
    int_states["time"] = msids["dpa_power"].times

    logger.setLevel(args.log_level)

    logger.info("Using data from %s to %s", start.yday, stop.yday)

    # create on-off states for FEPs and CCDs
    ccds = [f"I{i}" for i in range(4)] + [f"S{i}" for i in range(6)]
    ccdsfeps = defaultdict(list)
    for row in int_states:
        for i in range(6):
            ccdsfeps[f"FEP{i}"].append(str(i) in row["feps"])
        for key in ccds:
            ccdsfeps[key].append(key in row["ccds"])
    for key in ccdsfeps:
        int_states[key] = np.array(ccdsfeps[key], dtype="float64")

    # check for continuous clocking mode
    int_states["cc"] = np.char.startswith(int_states["si_mode"], "CC").astype("float64")
    int_states["cc"] *= ~int_states["clocking"].astype("bool")

    # Convert to pandas DataFrame, so we can use sklearn
    df = int_states.to_pandas()

    model_keys = ["cc"] + list(ccdsfeps.keys())

    # Separate into features and target
    X = df.drop([col for col in int_states.colnames if col not in model_keys], axis=1)
    y = df["dpa_power"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the data
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Make the model
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),  # Two layers with 64 and 32 neurons
        solver="adam",
        max_iter=500,
    )

    # Train the model
    model.fit(X_train, y_train.ravel())

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info("Mean Squared Error: %2f", mse)
    logger.info("Mean Absolute Error: %2f", mae)
    dump((model, scaler_X, scaler_y), "dpa_power_model.joblib")
