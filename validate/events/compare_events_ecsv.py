import argparse
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
from cxotime import CxoTime


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Process event query parameters.")
    parser.add_argument(
        "data_root_a",
        type=str,
        help="Data root A for test event ECSV outputs.",
    )
    parser.add_argument(
        "data_root_b",
        type=str,
        help="Data root B for test event ECSV outputs.",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=20.0,
        help="Maximum time difference in seconds.",
    )
    return parser


def compare_events(dirname_a, dirname_b, max_time_diff):
    """
    Compare event data from two directories for a specified time range.

    Parameters
    ----------
    dirname_a : str
        The directory where the event files are saved.
    dirname_b : str
        The directory where the event files are saved.
    max_time_diff : float
        The maximum time difference in seconds.

    """
    events_a = Path(dirname_a) / "events"
    print(events_a.absolute().resolve())
    events_b = Path(dirname_b) / "events"
    ecsv_files_a = sorted(events_a.glob("*.ecsv"))
    ecsv_files_b = sorted(events_b.glob("*.ecsv"))
    print(f"Found {len(ecsv_files_a)} ECSV files in {events_a}")

    if len(ecsv_files_a) != len(ecsv_files_b):
        raise ValueError("Number of ECSV files do not match.")

    rows = []
    for ecsv_file_a, ecsv_file_b in zip(ecsv_files_a, ecsv_files_b):
        if ecsv_file_a.name != ecsv_file_b.name:
            raise ValueError("ECSV file names do not match.")

        status, note, rows_a, rows_b = compare_ecsv_files(
            ecsv_file_a, ecsv_file_b, max_time_diff
        )
        row = {
            "status": status,
            "file": ecsv_file_a.name,
            "rows_a": rows_a,
            "rows_b": rows_b,
            "note": note,
        }
        rows.append(row)

    table = Table(rows)
    table["note"].format = "<s"
    table.pprint_all()


def compare_ecsv_files(ecsv_file_a, ecsv_file_b, time_diff_allowed):
    data_a = Table.read(ecsv_file_a)
    data_b = Table.read(ecsv_file_b)

    not_ok = "Not OK"

    rows_a = len(data_a)
    rows_b = len(data_b)

    if len(data_a) != len(data_b):
        return not_ok, "Number of rows do not match.", rows_a, rows_b

    if data_a.colnames != data_b.colnames:
        return not_ok, "Column names do not match.", rows_a, rows_b

    for name in data_a.colnames:
        if np.all(data_a[name] == data_b[name]):
            continue
        if name in ["start", "stop", "tstart", "tstop"]:
            time_diff = (CxoTime(data_a[name]) - CxoTime(data_b[name])).to_value(u.s)
            time_diff_max = np.max(abs(time_diff))
            # print(f"Max time difference for {ecsv_file_a.name}.{name}: {time_diff_max}")
            if time_diff_max > time_diff_allowed:
                return (
                    not_ok,
                    f"Time difference {time_diff_max} greater than {time_diff_allowed} seconds.",
                    rows_a,
                    rows_b,
                )
        elif name == "dur":
            diff = abs(data_a[name] - data_b[name])
            diff_max = np.max(diff)
            if diff_max > time_diff_allowed:
                return (
                    not_ok,
                    f"Duration difference {diff_max} greater than {time_diff_allowed}.",
                    rows_a,
                    rows_b,
                )
        elif name == "notes":
            continue
        else:
            return not_ok, f"Column {name} does not match.", rows_a, rows_b

    return "OK", "", rows_a, rows_b


def main(args=None):
    opts = get_arg_parser().parse_args(args)
    compare_events(opts.data_root_a, opts.data_root_b, opts.max_time_diff)


if __name__ == "__main__":
    main()
