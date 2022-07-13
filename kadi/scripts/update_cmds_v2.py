# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse

from ska_helpers.run_info import log_run_info

from kadi import __version__
from kadi.commands.commands_v2 import update_cmds_archive


def get_opt(args=None):
    """
    Get options for command line interface to update_
    """
    parser = argparse.ArgumentParser(description="Update HDF5 cmds v2 table")
    parser.add_argument("--lookback", type=int, help="Lookback (default=30 days)")
    parser.add_argument("--stop", help="Stop date for update (default=Now+21 days)")
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
    parser.add_argument("--data-root", default=".", help="Data root (default='.')")
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s {version}".format(version=__version__),
    )

    args = parser.parse_args(args)
    return args


def main(args=None):
    """
    Main function for update_cmds_v2
    """
    opt = get_opt(args)
    log_run_info(log_func=print, opt=opt)

    update_cmds_archive(
        lookback=opt.lookback,
        stop=opt.stop,
        log_level=opt.log_level,
        scenario=opt.scenario,
        data_root=opt.data_root,
    )


if __name__ == "__main__":
    main()
