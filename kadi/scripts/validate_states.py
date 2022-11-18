# Licensed under a 3-clause BSD style license - see LICENSE.rst
import argparse
import sys
from pathlib import Path

import jinja2
import ska_helpers.logging
from ska_helpers.run_info import log_run_info

import kadi
from kadi.commands import validate


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
        "--version",
        action="version",
        version=f"%(prog)s {kadi.__version__}",
    )

    return parser


def main(args=None):
    opt = get_opt().parse_args(args)
    logger = ska_helpers.logging.basic_logger(
        validate.validate_states.__name__, level=opt.log_level
    )
    log_run_info(logger.info, opt)

    validators = []
    for cls in validate.Validate.subclasses:
        logger.info(f"Validating {cls.name}")
        instance = cls(stop=opt.stop, days=opt.days)
        validator = {}
        validator["plot_html"] = instance.get_plot_html()
        validator["name"] = instance.name
        validators.append(validator)

    context = {
        "validators": validators,
    }
    index_template_file = Path(__file__).parent / "templates" / "index_validate.html"
    index_template = index_template_file.read_text()
    template = jinja2.Template(index_template)
    html = template.render(context)

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "index.html").write_text(html)


if __name__ == "__main__":
    main(sys.argv[1:])
