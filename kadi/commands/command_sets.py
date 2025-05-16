# Licensed under a 3-clause BSD style license - see LICENSE.rst
from pathlib import Path

import astropy.units as u
from cxotime import CxoTime
from parse_cm.backstop import parse_backstop_params
from Quaternion import Quat
from ska_helpers.utils import convert_to_int_float_str

import kadi.commands
from kadi.commands.core import CommandTable

RTS_PATH = Path("FOT/configuration/products/rts")


def cmd_set_rts(*args, date=None):
    from parse_cm.csd import csd_cmd_gen

    rts = "SCS_CATEGORY,OBSERVING\n" + "\n".join(args).upper()
    cmds = csd_cmd_gen(rts, date=date)
    return cmds


def cmd_set_obsid(obs_id, date=None):
    """Return a command set that updates the commanded ObsID.

    Parameters
    ----------
    obsid
        obsid

    Returns
    -------
    list of command defs suitable for generate_cmds()
    """
    return (dict(type="MP_OBSID", tlmsid="COAOSQID", params=dict(ID=obs_id)),)


def cmd_set_maneuver(*args, date=None):
    """Return a command set that initiates a maneuver to the given attitude ``att``.

    Parameters
    ----------
    att
        attitude compatible with Quat() initializer

    Returns
    -------
    list of command defs suitable for generate_cmds()
    """
    att = Quat(args)
    return (
        dict(type="COMMAND_SW", tlmsid="AONMMODE", msid="AONMMODE", dur=0.25625),
        dict(type="COMMAND_SW", tlmsid="AONM2NPE", msid="AONM2NPE", dur=4.1),
        dict(
            type="MP_TARGQUAT",
            tlmsid="AOUPTARQ",
            params=dict(Q1=att.q[0], Q2=att.q[1], Q3=att.q[2], Q4=att.q[3]),
            dur=5.894,
        ),
        dict(type="COMMAND_SW", tlmsid="AOMANUVR", msid="AOMANUVR"),
    )


def cmd_set_aciscti(date=None):
    return (
        dict(type="ACISPKT", tlmsid="WSVIDALLDN", dur=1.025),
        dict(type="ACISPKT", tlmsid="WSPOW0CF3F", dur=1.025),
        dict(type="ACISPKT", tlmsid="WT00216024", dur=67),
        dict(type="ACISPKT", tlmsid="XTZ0000005"),
    )


def cmd_set_end_vehicle(date=None):
    cmds = cmd_set_end_scs(128) + cmd_set_end_scs(129) + cmd_set_end_scs(130)
    return cmds


def cmd_set_end_observing(date=None):
    cmds = cmd_set_end_scs(131) + cmd_set_end_scs(132) + cmd_set_end_scs(133)
    return cmds


def cmd_set_scs107(date=None):
    # SCS-106 (which is called by 107) was patched around 2021-Jun-08
    if date is not None and date > CxoTime("2021-06-08").date:
        pow_cmd = "WSPOW0002A"  # 3-FEPS
    else:
        pow_cmd = "WSPOW00000"  # 0-FEPS

    cmds = cmd_set_end_observing()
    cmds += (
        dict(type="COMMAND_SW", dur=1.025, tlmsid="OORMPDS"),
        dict(type="COMMAND_HW", tlmsid="AFIDP", msid="AFLCRSET"),
        dict(type="SIMTRANS", params=dict(POS=-99616), dur=65.66),
        dict(type="ACISPKT", tlmsid="AA00000000", dur=1.025),
        dict(type="ACISPKT", tlmsid="AA00000000", dur=10.25),
        dict(type="ACISPKT", tlmsid=pow_cmd),
    )
    # Note that the following block to include HRC turn-off commands is wrapped in a
    # config variable to make this conditional in support of regression testing. This is
    # needed for tests of command generation vs. commands in the archive which did not
    # include this HRC SCS-107 commanding (prior to #344).
    if not kadi.commands.conf.disable_hrc_scs107_commanding:
        cmds += (
            dict(type="COMMAND_HW", tlmsid="215PCAOF", dur=1.205),
            dict(type="COMMAND_HW", tlmsid="2IMHVOF", dur=1.025),
            dict(type="COMMAND_HW", tlmsid="2SPHVOF", dur=1.025),
            dict(type="COMMAND_HW", tlmsid="2S2STHV", dur=1.025),
            dict(type="COMMAND_HW", tlmsid="2S1STHV", dur=1.025),
            dict(type="COMMAND_HW", tlmsid="2S2HVOF", dur=1.025),
            dict(type="COMMAND_HW", tlmsid="2S1HVOF", dur=1.025),
        )

    return cmds


def cmd_set_dither(state, date=None):
    if state not in ("ON", "OFF"):
        raise ValueError(f"Invalid dither state {state!r}")
    enab = "EN" if state == "ON" else "DS"
    return (
        dict(
            type="COMMAND_SW",
            tlmsid=f"AO{enab}DITH",
        ),
    )


def cmd_set_bright_star_hold(date=None):
    out = cmd_set_end_vehicle()
    out += cmd_set_scs107(date=date)
    return out


def cmd_set_maneuver_sun_pitch(pitch, date=None):
    """Maneuver to ``pitch`` Sun pitch angle (absolute)."""
    cmd = dict(type="LOAD_EVENT", tlmsid="SUN_PITCH", params={"PITCH": pitch})
    return (cmd,)


def cmd_set_maneuver_sun_rasl(rasl, rate=0.025, *, date=None):
    """Maneuver by ``angle`` degrees in roll about Sun line (relative to current)."""
    cmd = dict(
        type="LOAD_EVENT", tlmsid="SUN_RASL", params={"RASL": rasl, "RATE": rate}
    )
    return (cmd,)


def _cmd_set_nsm_or_safe_mode(*args, tlmsid=None, date=None):
    """Return a command set that sets the NSM or safe mode pitch to ``args[0]``.

    Includes the end of vehicle and SCS-107 commands and dither disable commands.
    """
    nsm_cmd = dict(type="COMMAND_SW", tlmsid=tlmsid)
    if args:
        nsm_cmd["params"] = {"PITCH": args[0]}
    out = (nsm_cmd,)

    if tlmsid == "ACPCSFSU":
        out += ({"type": "COMMAND_SW", "tlmsid": "CSELFMT5"},)  # Telemetry format 5

    out += cmd_set_end_vehicle()
    out += cmd_set_scs107(date=date)
    out += cmd_set_dither("OFF", date=date)
    return out


def cmd_set_nsm(*args, date=None):
    return _cmd_set_nsm_or_safe_mode(*args, tlmsid="AONSMSAF", date=date)


def cmd_set_safe_mode(*args, date=None):
    out = _cmd_set_nsm_or_safe_mode(*args, tlmsid="ACPCSFSU", date=date)
    return out


def cmd_set_load_not_run(load_name, date=None):
    cmd = {
        "type": "LOAD_EVENT",
        "params": {"EVENT_TYPE": "LOAD_NOT_RUN", "LOAD": load_name},
    }
    return (cmd,)


def cmd_set_observing_not_run(load_name, date=None):
    cmd = {
        "type": "LOAD_EVENT",
        "params": {"EVENT_TYPE": "OBSERVING_NOT_RUN", "LOAD": load_name},
    }
    return (cmd,)


def cmd_set_hrc_not_run(load_name, date=None):
    cmds = (
        {"type": "COMMAND_HW", "tlmsid": "215PCAOF"},  # 15 V off
        {"type": "COMMAND_HW", "tlmsid": "224PCAOF"},  # 24 V off
        {"type": "COMMAND_HW", "tlmsid": "2IMHVOF"},  # HRC-I off
        {"type": "COMMAND_HW", "tlmsid": "2SPHVOF"},  # HRC-S off
    )
    return cmds


def cmd_set_command(*args, date=None):
    """Parse Command or Command not run params string and return a command dict.

    The format follows Backstop ``"<cmd_type> | PARAM1=VAL1, PARAM2=VAL2, .."``.
    This code follows the key steps in parse_cm.backstop.read_backstop_as_list().
    """
    params_str = args[0].strip().replace(" ", "").upper()

    cmd_type, args_str = params_str.split("|", 1)
    params = parse_backstop_params(args_str)
    tlmsid = params.pop("tlmsid", "None")
    cmd = {"type": cmd_type, "tlmsid": tlmsid, "params": params}

    return (cmd,)


def cmd_set_end_scs(*args, date=None):
    cmd = {"type": "COMMAND_SW", "tlmsid": "CODISASX"}
    cmd["params"] = {
        "MSID": "CODISASX",
        "CODISAS1": args[0],
    }
    return (cmd,)


def cmd_set_command_not_run(*args, date=None):
    (cmd,) = cmd_set_command(*args, date=date)
    # Save original type which gets used later in CommandTable.remove_not_run_cmds().
    cmd["params"]["__type__"] = cmd["type"]
    cmd["type"] = "NOT_RUN"
    return (cmd,)


def get_cmds_from_event(date, event, params_str):
    r"""
    Return a predefined cmd_set ``name`` corresponding to the given event.

    :param date: str, command event date
    :param event: str, command event name (e.g. 'Obsid', 'SCS-107', 'Maneuver')
    :param params_str: str, command event parameters string
    :returns: ``CommandTable`` with commands
    """
    event_func_name = event.lower().replace(" ", "_").replace("-", "")
    event_func = globals().get("cmd_set_" + event_func_name)
    if event_func is None:
        raise ValueError(f"unknown event {event!r}")

    if event in ("RTS", "Command", "Command not run"):
        # Delegate parsing to cmd_set_command
        args = [params_str]
    else:
        params_str = params_str.upper().split()
        args = [convert_to_int_float_str(p) for p in params_str]

    cmds = event_func(*args, date=date)

    # Load event does not generate commands
    if cmds is None:
        return None

    cmd_date = CxoTime(date)
    outs = []
    event_text = event.replace(" ", "_")
    event_date = CxoTime(date).date[:17]

    # Loop through commands. This could be a list of dict which is a relative
    # time specification of the commands (with 'dur' keys to indicate timing) or
    # a CommandTable, which has a date for each command.
    for step, cmd in enumerate(cmds):
        # Force CommandTable row to be a dict
        if not isinstance(cmd, dict):
            cmd = {name: cmd[name] for name in cmd.colnames}  # noqa: PLW2901

        # Get command duration (if any). If the cmd is only {'dur': <dt>} then
        # it is a pure delay so skip subsequent processing.
        dur = cmd.pop("dur", None)
        if dur and not cmd:
            cmd_date += dur * u.s
            continue

        date = cmd.pop("date", cmd_date.date)
        tlmsid = cmd.pop("tlmsid", None)
        cmd_type = cmd.pop("type")

        params = {}
        params["event"] = event_text
        params["event_date"] = event_date
        for key, val in cmd.pop("params", {}).items():
            params[key.lower()] = val
        # Allow for params to be included in cmd dict directly as well as within
        # the 'params' key.
        for key, val in cmd.items():
            params[key.lower()] = val
        scs = params.pop("scs", 0)

        out = {
            "idx": -1,
            "date": date,
            "type": cmd_type,
            "tlmsid": tlmsid,
            "scs": scs,
            "step": step,
            "time": CxoTime(date).secs,
            "source": "CMD_EVT",
            "vcdu": -1,
            "params": params,
        }
        outs.append(out)

        if dur is not None:
            cmd_date += dur * u.s

    out = CommandTable(outs)
    return out
