import argparse
import os
from pathlib import Path

from cxotime import CxoTime


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Process event query parameters.")
    parser.add_argument(
        "--start", type=str, default="2023:043", help="Start time in CxoTime format"
    )
    parser.add_argument(
        "--stop", type=str, default="2023:054", help="Stop time in CxoTime format"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Data root for test events3.db3 and event ECSV outputs. "
            "If this ends in 'flight' then use flight events from SKA."
        ),
    )
    return parser


def write_events(start, stop, dirname):
    """
    Write event data to ECSV files for a specified time range.

    Parameters
    ----------
    start : astropy.time.Time
        The start time for the event query.
    stop : astropy.time.Time
        The stop time for the event query.
    dirname : str
        The directory where the event files will be saved.
    """
    import kadi
    import kadi.paths
    from kadi import events
    from kadi.events.query import EventQuery

    print(f"Using kadi from {kadi.__file__} version {kadi.__version__}")
    (outdir := Path(dirname) / "events").mkdir(exist_ok=True, parents=True)
    print(f"Writing events ECSV files to {outdir}")
    print(f"Events path is {kadi.paths.EVENTS_DB_PATH()}")

    query_funcs = {
        name: getattr(events, name)
        for name in dir(events)
        if isinstance(getattr(events, name), EventQuery)
    }
    for name, query_func in query_funcs.items():
        evts = query_func.filter(start=start, stop=stop).table

        # Round every float column to 6 decimal places
        for col in evts.itercols():
            if col.dtype.kind == "f":
                evts[col.name][:] = evts[col.name].round(6)

        path = outdir / f"{name}.ecsv"
        print(f"{len(evts):3d} rows: {path}")
        evts.write(path, overwrite=True)


def main(args=None):
    opts = get_arg_parser().parse_args(args)
    start = CxoTime(opts.start)
    stop = CxoTime(opts.stop)
    data_root = opts.data_root.rstrip("/")

    # Before we import kadi, set the KADI environment variable to the data root
    if Path(data_root).name != "flight":
        os.environ["KADI"] = data_root

    write_events(start, stop, data_root)


if __name__ == "__main__":
    main()
