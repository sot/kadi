import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import astropy.units as u
import maude
from cxotime import CxoTime

SKA = Path(os.environ["SKA"])


def print_section(text):
    print("\n" + "=" * 79)
    print(text)
    print("=" * 79)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start", type=str, default="2023:043", help="Start time in CxoTime format"
    )
    parser.add_argument(
        "--stop", type=str, default="2023:056", help="Stop time in CxoTime format"
    )
    parser.add_argument(
        "--telem-source",
        required=True,
        choices=["cxc", "maude"],
        help="Telemetry source ('cxc' or 'maude')",
    )
    parser.add_argument(
        "--code",
        required=True,
        choices=["flight", "test"],
        help="Use flight kadi code or local test version ('flight' or 'test')",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry run")

    return parser


def run_update_events(opt):
    import kadi
    from kadi.scripts import update_events

    print(f"Using kadi from {kadi.__file__} version {kadi.__version__}")

    data_path = Path.cwd() / f"{opt.code}-{opt.telem_source}"
    data_path.mkdir(exist_ok=True)
    data_dir = str(data_path)

    events3_flight_path = SKA / "data" / "kadi" / "events3.db3"
    args = ["rsync", "-a", "-v", str(events3_flight_path), f"{data_dir}/events3.db3"]
    print(" ".join(args))
    if not opt.dry_run:
        subprocess.check_call(args)

    if (events_dir_path := data_path / "events").exists():
        print(f"Removing {events_dir_path}")
        if not opt.dry_run:
            shutil.rmtree(events_dir_path)

    # Show debug messages from maude for the first update
    maude.set_logger_level("DEBUG")

    # Process the first day
    start = CxoTime(opt.start)
    stop = start + 1 * u.day

    maude_arg = ["--maude"] if opt.telem_source == "maude" else []

    args = [
        f"--start={start}",
        "--delete-from-start",
        f"--stop={stop}",
        f"--data-root={data_dir}",
    ] + maude_arg
    print_section("First call to update_events (deleting from start)")
    if not opt.dry_run:
        sys.argv = ["update_events"] + args
        update_events.main()

    # No need for maude logging any more
    # maude.set_logger_level("INFO")

    while stop <= CxoTime(opt.stop):
        print_section(f"stop = {stop}")
        t0 = time.time()
        args = [
            f"--stop={stop}",
            f"--data-root={data_dir}",
        ] + maude_arg

        if not opt.dry_run:
            sys.argv = ["update_events"] + args
            update_events.main()

        print(f"Elapsed time: {time.time() - t0:.1f} s")
        stop += 1 * u.day


def main():
    opt = get_arg_parser().parse_args()

    if opt.code == "test":
        repo_path = (Path(__file__).parent / ".." / "..").resolve()
        sys.path.insert(0, str(repo_path))

    run_update_events(opt)


if __name__ == "__main__":
    main()
