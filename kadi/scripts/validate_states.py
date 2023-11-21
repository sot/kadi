# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse
import logging
import sys
from pathlib import Path

import jinja2
import maude
from cheta import fetch
from cxotime import CxoTime, CxoTimeLike

import kadi
from kadi.commands import conf
from kadi.commands.validate import Validate

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


def get_index_page_html(
    stop: CxoTimeLike, days: float, states: list[str], no_exclude: bool = False
):
    """Make a simple HTML page with all the validation plots and information.

    Parameters
    ----------
    stop
        stop time for validation interval (CxoTime-like, default=now)
    days
        length of validation interval (days)
    states
        list of states to validate (default=all)
    no_exclude
        if True then do not exclude intervals (default=False)

    Returns
    -------
    str
        HTML string
    """
    validators = []
    violations = []
    if stop is None:
        stop = CxoTime.now()

    for cls in Validate.subclasses:
        if states and cls.state_name not in states:
            continue
        logger.info(f"Validating {cls.state_name}")
        validator: Validate = cls(stop=stop, days=days, no_exclude=no_exclude)
        validator.html = validator.get_html()
        validators.append(validator)

        for violation in validator.violations:
            violations.append(
                {
                    "name": validator.state_name,
                    "start": violation["start"],
                    "stop": violation["stop"],
                }
            )

    context = {
        "validators": validators,
        "violations": violations,
    }
    index_template_file = Path(__file__).parent / "templates" / "index_validate.html"
    index_template = index_template_file.read_text()
    template = jinja2.Template(index_template)
    html = template.render(context)

    return html


def main(args=None):
    opt = get_opt().parse_args(args)

    # Enable logging in relevant packages
    logging.getLogger("kadi").setLevel(opt.log_level)
    fetch.add_logging_handler(level=opt.log_level)
    fetch.data_source.set("cxc", "maude allow_subset=False")
    maude.set_logger_level(opt.log_level)

    maude.conf.cache_msid_queries = True
    fetch.CACHE = True

    if opt.in_work:
        conf.include_in_work_command_events = True

    html = get_index_page_html(opt.stop, opt.days, opt.states, opt.no_exclude)

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "index.html").write_text(html)


if __name__ == "__main__":
    main(sys.argv[1:])
