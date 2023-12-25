"""Create a scenario to handle HRC commands not run to to HRC being disabled.

This removes HRC state-impacting commands::

    {"tlmsid": "224PCAON"},
    {"tlmsid": "215PCAON"},
    {"tlmsid": "COACTSX", "coacts1": 134},
    {"tlmsid": "COENASX", "coenas1": 89},
    {"tlmsid": "COENASX", "coenas1": 90},

In practice the hardware commands are not in loads since the HRC return to science.

This script creates a scenario ~/.kadi/<scenario> that can be used to remove these
commands from the flight kadi commands database. It then tests that by setting
``KADI_SCENARIO=<scenario>`` and getting kadi states over the time period of interest.

The simplest way to do import to the flight sheet is to import the CSV file into a
temporary Google Sheet, then copy/paste that table into the flight Chandra Command
Events Google Sheet. Remember to create empty rows for the copy/paste.
"""

import argparse

import kadi.paths
from astropy import table
from cxotime import CxoTime
from kadi.commands import get_cmds
from kadi.commands.core import CommandTable
from kadi.commands.states import get_states
from ska_helpers.utils import temp_env_var


# When was F_HRC_SAFING script run in late 2023
hrc_safing_date = "2023:343:02:00:00"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Print HRC state-impacting commands that were not run due to HRC being disabled."
    )
    parser.add_argument(
        "--start",
        default=hrc_safing_date,
        help=f"Start time for searching for commands (default={hrc_safing_date}))",
    )
    parser.add_argument(
        "--stop",
        help="Stop time for searching for commands (default=NOW)",
    )
    parser.add_argument(
        "--status",
        default="Definitive",
        help="Status of command events (Definitive or In-work)",
    )
    parser.add_argument(
        "--scenario",
        default="hrc_disable",
        help="Scenario name (default=hrc_disable). This creates ~/.kadi/<scenario>/cmd_events.csv.",
    )
    return parser


def main():
    opt = get_parser().parse_args()
    make_cmd_events(opt)
    test_cmd_events(opt)


def test_cmd_events(opt):
    def get_states_local():
        states = get_states(
            start=opt.start,
            stop=opt.stop,
            state_keys=["hrc_15v", "hrc_24v", "hrc_i", "hrc_s"],
            merge_identical=True,
        )
        return states

    print("Current flight states:")
    get_states_local().pprint_all()
    print()
    print(f"States with HRC disable scenario {opt.scenario}:")
    with temp_env_var("KADI_SCENARIO", opt.scenario):
        get_states_local().pprint_all()


def make_cmd_events(opt):
    cmd_kwargs_list = [
        {"tlmsid": "224PCAON"},
        {"tlmsid": "215PCAON"},
        {"tlmsid": "COACTSX", "coacts1": 134},
        {"tlmsid": "COENASX", "coenas1": 89},
        {"tlmsid": "COENASX", "coenas1": 90},
    ]

    rows = []
    start = CxoTime(opt.start)
    stop = CxoTime(opt.stop)

    for cmd_kwargs in cmd_kwargs_list:
        cmds: CommandTable = get_cmds(start=start, stop=stop, **cmd_kwargs)
        print(f"{len(cmds)} cmd(s) found for {cmd_kwargs}")
        params_str = ", ".join([f"{k.upper()}={v}" for k, v in cmd_kwargs.items()])
        for cmd in cmds:
            row = (
                opt.status,
                cmd["date"],
                "Command not run",
                f"{cmd['type']} | {params_str}",
                "Tom Aldcroft",
                "Jean Connelly",
                f"Not run due to F_HRC_SAFING at {hrc_safing_date}",
            )
            rows.append(row)

    names = "State Date Event Params Author Reviewer Comment".split()
    cmd_events = table.Table(rows=rows, names=names)
    cmd_events.sort("Date", reverse=True)
    cmd_events.pprint_all()

    cmd_events_path = kadi.paths.CMD_EVENTS_PATH(opt.scenario)
    cmd_events_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    print(f"Writing {len(cmd_events)} events to {cmd_events_path}")
    cmd_events.write(cmd_events_path, format="ascii.csv", overwrite=True)


if __name__ == "__main__":
    main()
