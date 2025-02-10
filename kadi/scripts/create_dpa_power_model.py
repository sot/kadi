import argparse
import logging
from collections import defaultdict

import astropy.units as u
from cheta import fetch_sci as fetch
from cxotime import CxoTime
from joblib import dump
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kadi import __version__  # noqa: F401
from kadi.commands.states import get_states, interpolate_states

logger = logging.getLogger("kadi")

exclude_intervals = [
    ("2022:016:00:05:23", "2022:018:18:43:48"),
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
    return parser.parse_args()


def main():
    args = get_args()

    stop = CxoTime(args.stop)
    start = stop - args.days * u.day

    # States to obtain for the model
    power_keys = ["ccd_count", "clocking", "fep_count", "simpos"]
    state_keys = power_keys + ["feps", "ccds", "power_cmd"]
    fep_keys = [f"FEP{i}" for i in range(6)]

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

    # create on-off states for FEPs
    feps = defaultdict(list)
    for row in int_states:
        for i in range(6):
            feps[f"FEP{i}"].append(float(str(i) in row["feps"]))
    for fk in fep_keys:
        int_states[fk] = feps[fk]

    df = int_states.to_pandas()

    # Separate into features and target
    X = df.drop(
        [col for col in int_states.colnames if col not in power_keys + fep_keys], axis=1
    )
    y = df["dpa_power"]

    # Split into training and testing sets based on time
    t_split = start + (stop - start) * args.train_test_split
    idx_train = df["time"] < t_split.secs
    idx_test = df["time"] >= t_split.secs
    for interval in exclude_intervals:
        int_start = CxoTime(interval[0]).secs
        int_stop = CxoTime(interval[1]).secs
        idx_train &= ~((df["time"] >= int_start) & (df["time"] <= int_stop))
    X_train = X[idx_train]
    X_test = X[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    # Scale the data
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

    # Make the model
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),  # Two layers with 64 and 32 neurons
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info("Mean Squared Error: %2f", mse)
    dump(model, "dpa_power_model.joblib")
