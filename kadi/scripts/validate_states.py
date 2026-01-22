# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse
import logging
import sys
from pathlib import Path

import jinja2
import maude
from astropy.table import Table
from cheta import fetch
from cxotime import CxoTime, CxoTimeLike

import kadi
import kadi.commands
from kadi.commands.validate import Validate

logger = logging.getLogger("kadi")


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
        "--email",
        action="append",
        dest="emails",
        default=[],
        help='Email address for notification (multiple allowed, use "TEST" for testing)',
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {kadi.__version__}",
    )

    return parser


def run_validators(
    stop: CxoTimeLike, days: float, states: list[str], no_exclude: bool = False
) -> list[Validate]:
    """Run applicable validators and return list of validators.

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
    list
        List of validators
    """
    stop = CxoTime(stop)

    validators = []
    for cls in Validate.subclasses:
        if states and cls.state_name not in states:
            continue
        logger.info(f"Validating {cls.state_name}")
        validator: Validate = cls(stop=stop, days=days, no_exclude=no_exclude)
        validator.html = validator.get_html()
        validators.append(validator)

    return validators


def get_violations(validators: list[Validate]) -> list[dict]:
    """Return list of violations from list of validators.

    Parameters
    ----------
    validators : list
        List of validators

    Returns
    -------
    list
        List of violations dicts, keys are "name", "start", "stop"
    """
    violations = [
        {
            "name": validator.state_name,
            "start": violation["start"],
            "stop": violation["stop"],
        }
        for validator in validators
        for violation in validator.violations
    ]
    return violations


def get_index_page_html(validators: list[Validate], violations: list[dict]) -> str:
    """Make a simple HTML page with all the validation plots and information.

    Parameters
    ----------
    validators : list[Validate]
        List of processed validators
    violations : list[dict]
        List of violations dicts, keys are "name", "start", "stop"

    Returns
    -------
    str
        HTML string
    """
    context = {
        "validators": validators,
        "violations": violations,
    }
    index_template_file = (
        Path(kadi.commands.__file__).parent / "templates" / "index_validate.html"
    )
    index_template = index_template_file.read_text()
    template = jinja2.Template(index_template)
    html = template.render(context)

    return html


def get_violations_text(violations: list[dict]) -> str:
    """Get text for email or logger alert of validation violations.

    Parameters
    ----------
    violations : list[dict]
        List of violations dicts (see get_violations())

    Returns
    -------
    str
        Text for alert
    """
    lines = ["kadi validate_states processing found state violation(s):", ""]
    lines.extend(Table(violations).pformat())
    lines.extend(["", "See:https://cxc.harvard.edu/mta/ASPECT/validate_states/", ""])
    text = "\n".join(lines)
    return text


def send_alert_email(text: str, opt: argparse.Namespace) -> None:
    """Send an email alert for validation violations.

    Parameters
    ----------
    text : str
        Text for email
    opt : argparse.Namespace
        Command-line options
    """
    from acdc.common import send_mail

    subject = "kadi validate_states: state violation(s)"
    if opt.emails:
        send_mail(logger, opt, subject, text, __file__)


def main(args=None):
    opt = get_opt().parse_args(args)

    # Enable logging in relevant packages
    logging.getLogger("kadi").setLevel(opt.log_level)
    fetch.add_logging_handler(level=opt.log_level)
    fetch.data_source.set("cxc", "maude allow_subset=False")
    maude.set_logger_level(opt.log_level)

    maude.conf.cache_msid_queries = True
    fetch.CACHE = True

    kadi.commands.conf.include_in_work_command_events = opt.in_work

    validators = run_validators(opt.stop, opt.days, opt.states, opt.no_exclude)
    violations = get_violations(validators)
    html = get_index_page_html(validators, violations)

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "index.html").write_text(html)

    if violations:
        violations_text = get_violations_text(violations)
        logger.warning(violations_text)
        if opt.emails:
            send_alert_email(violations_text, opt)


if __name__ == "__main__":
    main(sys.argv[1:])
