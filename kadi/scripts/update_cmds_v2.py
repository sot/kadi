# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse
import os

import astropy.units as u
from cxotime import CxoTime
from ska_helpers.run_info import log_run_info

from kadi import __version__
from kadi.commands.commands_v2 import (
    RLTT_ERA_START,
    clear_caches,
    update_cmds_archive,
)


def get_opt(args=None):
    """
    Get options for command line interface to update.
    """
    parser = argparse.ArgumentParser(
        description="Update HDF5 cmds v2 table",
    )
    parser.add_argument(
        "--data-root",
        default=".",
        help="Data root (default='.')",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        help="Lookback (default=30 days)",
    )
    parser.add_argument(
        "--stop",
        help="Stop date for update (default=Now+21 days)",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=10,
        help="Log level (10=debug, 20=info, 30=warnings)",
    )
    parser.add_argument(
        "--scenario",
        help="Scenario for loads and command events outputs (default=None)",
    )
    parser.add_argument(
        "--kadi-cmds-version",
        type=int,
        help="Kadi cmds version (default=latest)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    parser.add_argument(
        "--truncate-from-rltt-start",
        action="store_true",
        help="Truncate cmds archive from the start of the RLTT era (APR1420B)",
    )

    args = parser.parse_args(args)
    return args


def main(args=None):
    """
    Main function for update_cmds_v2
    """
    opt = get_opt(args)

    if opt.kadi_cmds_version is not None:
        os.environ["KADI_CMDS_VERSION"] = str(opt.kadi_cmds_version)

    log_run_info(log_func=print, opt=opt)

    if opt.truncate_from_rltt_start:
        process_from_rltt_start(opt)
    else:
        update_cmds_archive(
            opt.lookback,
            opt.stop,
            opt.log_level,
            opt.scenario,
            opt.data_root,
            opt.truncate_from_rltt_start,
        )


def process_from_rltt_start(opt):
    # Final processing stop
    stop0 = RLTT_ERA_START + 21 * u.day
    stop1 = CxoTime(opt.stop) if opt.stop else CxoTime.now() + 21 * u.day
    step = 365 * u.day
    lookback0 = 30 * u.day
    stops = CxoTime.linspace(stop0, stop1, step_max=step)
    dt = stops[1] - stops[0]

    for stop in stops:
        first_update = stop == stop0
        lookback = lookback0 if first_update else dt + lookback0

        print()
        print("*" * 80)
        print(
            f"Updating cmds archive to {stop} with "
            f"lookback={lookback.to_value(u.day):.1f} days"
        )
        print("*" * 80)
        print()

        update_cmds_archive(
            lookback=lookback.to_value(u.day),
            stop=stop,
            log_level=opt.log_level,
            scenario=opt.scenario,
            data_root=opt.data_root,
            truncate_from_rltt_start=first_update,
        )
        clear_caches()


if __name__ == "__main__":
    main()
