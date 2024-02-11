import os
import subprocess as sp
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/git/kadi"))
kadi_dir = sp.check_output("git rev-parse --abbrev-ref HEAD".split()).decode().strip()
print(f"Running out of branch {kadi_dir!r}")
os.environ["KADI"] = kadi_dir

from astropy.table import Table

import kadi
import kadi.commands as kc
import kadi.commands.core as kcc
from kadi.scripts import update_cmds_v2


def print_last_five_obss(stop):
    print("*" * 80)
    print(f"Last 5 observations as of {stop} using scenario='flight'")
    print("*" * 80)
    obss = Table(
        kc.get_observations(
            start="2024:034", stop="2024:038:02:00:00", scenario="flight"
        )
    )
    obss[
        "obsid",
        "obs_start",
        "obs_stop",
        "simpos",
        "manvr_start",
        "npnt_enab",
    ][-5:].pprint_all()
    print()


def update_archive_through_stop(stop, kadi_dir):
    os.environ["KADI_COMMANDS_DEFAULT_STOP"] = stop
    print(f"Updating commands as of {stop}")
    kc.clear_caches()
    update_cmds_v2.main(["--stop", stop, "--data-root", kadi_dir, "--log-level", "50"])
    kc.clear_caches()


print(f"Using kadi dir: {kadi_dir}")
print(f"kadi file {kadi.__file__}")

SKA = Path(os.environ["SKA"])

print(f"Making kadi dir: {kadi_dir}")
Path(kadi_dir).mkdir(exist_ok=True, parents=True)
kc.conf.commands_dir = kadi_dir

cmds = [
    "rsync",
    "-av",
    str(SKA / "data" / "kadi" / "cmds-from-2024feb04") + "/",
    kadi_dir + "/",
]
print("Syncing archive to state as of 2024-02-04 (about midnight EST)")
print(f"Running: {' '.join(cmds)}")
sp.run(
    cmds,
    check=True,
)

print(f"kadi version: {kadi.__version__}")
print()
os.environ["KADI_COMMANDS_DEFAULT_STOP"] = stop = "2024:035:12:00:00"
print_last_five_obss(stop)

# Now update the commands archive to processing from next day
for stop in (
    "2024:036:12:00:00",
    "2024:037:04:30:00",
    "2024:037:04:50:00",
    "2024:038:12:00:00",
):
    update_archive_through_stop(stop, kadi_dir)
    print_last_five_obss(stop)
