# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse
import logging
import sys
from pathlib import Path

import maude
from cheta import fetch

import kadi
from kadi.commands import conf, validate

logger = logging.getLogger(__name__)


def get_opt():
    parser = argparse.ArgumentParser(description="Validate kadi command states")
    parser.add_argument("--stop", help="Stop date for update (default=Now)")
    parser.add_argument(
        "--days", type=float, default=14.0, help="Lookback days (default=14 days)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG | INFO | WARNING, default=INFO)",
    )
    parser.add_argument(
        "--out-dir", default=".", help="Output directory for index.html (default='.')"
    )
    parser.add_argument(
        "--state",
        action="append",
        default=[],
        dest="states",
        help="State(s) to validate (default=ALL)",
    )
    parser.add_argument(
        "--no-exclude",
        default=False,
        action="store_true",
        help="Do not apply exclude intervals from validation (for testing)",
    )
    parser.add_argument(
        "--in-work",
        default=False,
        action="store_true",
        help="Include in-work events in validation (for checking new events)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {kadi.__version__}",
    )

    return parser


def main(args=None):
    opt = get_opt().parse_args(args)
    logging.getLogger("kadi").setLevel(opt.log_level)

    maude.conf.cache_msid_queries = True
    fetch.CACHE = True

    if opt.in_work:
        conf.include_in_work_command_events = True

    html = validate.get_index_page_html(opt.stop, opt.days, opt.states, opt.no_exclude)

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "index.html").write_text(html)


if __name__ == "__main__":
    main(sys.argv[1:])
