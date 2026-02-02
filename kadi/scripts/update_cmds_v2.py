# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse
import os

import astropy.units as u
from cxotime import CxoTime
from ska_helpers.run_info import log_run_info

from kadi import __version__
from kadi.commands.commands_v2 import RLTT_ERA_START, logger, update_cmds_archive
from kadi.config import conf


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

    # Developer-only options
    dev_group = parser.add_argument_group("Developer-only options")
    dev_group.add_argument(
        "--no-match-prev-cmds",
        action="store_true",
        help="Do not enforce matching previous command block when updating cmds v2 "
        "(experts only, this can produce an invalid commands table)",
    )
    dev_group.add_argument(
        "--matching-block-size",
        type=int,
        help=f"Matching block size (default={conf.matching_block_size})",
    )
    dev_group.add_argument(
        "--match-from-rltt-start",
        action="store_true",
        help="Match previous commands exactly from the start of the RLTT era "
        "(APR1420B). This implies --no-match-prev-cmds.",
    )

    dev_group.add_argument(
        "--reprocess-from-rltt-start",
        action="store_true",
        help="Reprocess cmds archive from the start of the RLTT era",
    )

    args = parser.parse_args(args)
    return args


def main(args=None):
    """
    Main function for update_cmds_v2
    """
    opt = get_opt(args)
    log_run_info(log_func=print, opt=opt)
    if opt.reprocess_from_rltt_start:
        reprocess_from_rltt_start(opt)
    else:
        process_one_update(opt)


def reprocess_from_rltt_start(opt):
    # Start the V2 updates a week and a day after CMDS_V2_START
    step = 365 * u.day
    date = RLTT_ERA_START + step
    stop = CxoTime.now()
    opt.lookback = step.to_value(u.day) + 30

    while date < stop:
        logger.info("*" * 80)
        logger.info(f"Updating cmds2 to {date}")
        logger.info("*" * 80)
        opt.stop = date.date
        process_one_update(opt)
        date += step

    # Final catchup to `stop`
    opt.stop = date.date
    process_one_update(opt)


def process_one_update(opt):
    # Transfer these developer-only options to conf
    for attr in (
        "no_match_prev_cmds",
        "matching_block_size",
        "match_from_rltt_start",
    ):
        if (value := getattr(opt, attr)) is not None:
            setattr(conf, attr, value)
            print(f"Set conf.{attr} = {value}")

    if opt.kadi_cmds_version is not None:
        os.environ["KADI_CMDS_VERSION"] = str(opt.kadi_cmds_version)

    update_cmds_archive(
        lookback=opt.lookback,
        stop=opt.stop,
        log_level=opt.log_level,
        scenario=opt.scenario,
        data_root=opt.data_root,
    )


if __name__ == "__main__":
    main()
