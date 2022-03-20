# Licensed under a 3-clause BSD style license - see LICENSE.rst
import re
from pathlib import Path

import astropy.units as u

from Quaternion import Quat
from cxotime import CxoTime
from parse_cm.common import _coerce_type as coerce_type
from kadi.commands.core import CommandTable

RTS_PATH = Path('FOT/configuration/products/rts')


def cmd_set_rts(*args, date=None):
    from parse_cm.csd import csd_cmd_gen

    rts = 'SCS_CATEGORY,OBSERVING\n' + '\n'.join(args).upper()
    cmds = csd_cmd_gen(rts, date=date)
    return cmds


def cmd_set_obsid(obs_id, date=None):
    """Return a command set that initiates a maneuver to the given attitude
    ``att``.

    :param obsid: obsid
    :returns: list of command defs suitable for generate_cmds()
    """
    return (dict(type='MP_OBSID',
                 tlmsid='COAOSQID',
                 params=dict(ID=obs_id)),)


def cmd_set_maneuver(*args, date=None):
    """Return a command set that initiates a maneuver to the given attitude
    ``att``.

    :param att: attitude compatible with Quat() initializer
    :returns: list of command defs suitable for generate_cmds()
    """
    att = Quat(args)
    return (dict(type='COMMAND_SW',
                 tlmsid='AONMMODE',
                 msid='AONMMODE',
                 dur=0.25625),
            dict(type='COMMAND_SW',
                 tlmsid='AONM2NPE',
                 msid='AONM2NPE',
                 dur=4.1),
            dict(type='MP_TARGQUAT',
                 tlmsid='AOUPTARQ',
                 params=dict(Q1=att.q[0], Q2=att.q[1],
                             Q3=att.q[2], Q4=att.q[3]),
                 dur=5.894),
            dict(type='COMMAND_SW',
                 tlmsid='AOMANUVR',
                 msid='AOMANUVR'),
            )


def cmd_set_aciscti(date=None):
    return (dict(type='ACISPKT',
                 tlmsid='WSVIDALLDN',
                 dur=1.025),
            dict(type='ACISPKT',
                 tlmsid='WSPOW0CF3F',
                 dur=1.025),
            dict(type='ACISPKT',
                 tlmsid='WT00216024',
                 dur=67),
            dict(type='ACISPKT',
                 tlmsid='XTZ0000005'))


def cmd_set_scs107(date=None):
    # SCS-106 (which is called by 107) was patched around 2021-Jun-08
    if date is not None and date > CxoTime('2021-06-08').date:
        pow_cmd = 'WSPOW0002A'  # 3-FEPS
    else:
        pow_cmd = 'WSPOW00000'  # 0-FEPS

    return (dict(type='COMMAND_SW',
                 dur=1.025,
                 tlmsid='OORMPDS'),
            dict(type='COMMAND_HW',
                 # dur=1.025,
                 tlmsid='AFIDP',
                 msid='AFLCRSET'),
            dict(type='SIMTRANS',
                 params=dict(POS=-99616),
                 dur=65.66),
            dict(type='ACISPKT',
                 tlmsid='AA00000000',
                 dur=1.025),
            dict(type='ACISPKT',
                 tlmsid='AA00000000',
                 dur=10.25),
            dict(type='ACISPKT',
                 tlmsid=pow_cmd),
            )


def cmd_set_dither(state, date=None):
    if state not in ('ON', 'OFF'):
        raise ValueError(f'Invalid dither state {state!r}')
    enab = 'EN' if state == 'ON' else 'DS'
    return (dict(type='COMMAND_SW',
                 tlmsid=f'AO{enab}DITH',
                 ),
            )


def cmd_set_bright_star_hold(date=None):
    out = (cmd_set_scs107(date=date)
           + cmd_set_dither('OFF', date=date)
           )
    return out


def cmd_set_nsm(date=None):
    nsm_cmd = dict(type='COMMAND_SW',
                   tlmsid='AONSMSAF')
    out = ((nsm_cmd,)
           + cmd_set_scs107(date=date)
           + cmd_set_dither('OFF', date=date)
           )
    return out


def cmd_set_safe_mode(date=None):
    safe_mode_cmds = (dict(type='COMMAND_SW',
                           tlmsid='ACPCSFSU'),  # CPE set pcad mode to safe sun
                      dict(type='COMMAND_SW',
                           tlmsid='CSELFMT5')  # Format 5 (programmable) select
                      )
    # This is a little lazy, but put the NSM commands on top of safe mode.
    # Even though not 100% accurate this does the right thing for commands and
    # state processing. Otherwise need to put stuff into commands_v2 and states
    # to handle the ACPCSFSU command.
    out = (safe_mode_cmds
           + cmd_set_nsm(date=date))
    return out


def cmd_set_load_not_run(load_name, date=None):
    return None


def cmd_set_observing_not_run(load_name, date=None):
    return None


def cmd_set_command(*args, date=None):
    params_str = args[0]
    cmd_type, args_str = params_str.split('|', 1)
    cmd = {'type': cmd_type.strip().upper()}

    # Strip spaces around equals signs and uppercase args (note that later the
    # keys are lowercased).
    args_str = re.sub(r'\s*=\s*', '=', args_str).upper()

    params = {}
    for param in args_str.split():
        key, val = param.split('=')
        if key == 'TLMSID':
            cmd['tlmsid'] = val
        else:
            params[key] = coerce_type(val)
    cmd['params'] = params

    return (cmd,)


def cmd_set_command_not_run(*args, date=None):
    cmd, = cmd_set_command(*args, date=date)
    cmd['type'] = 'NOT_RUN'
    return (cmd,)


def get_cmds_from_event(date, event, params_str):
    r"""
    Return a predefined cmd_set ``name`` generated with \*args.

    :param name: cmd set name (manvr|scs107|nsm)
    :param \*args: optional args
    :returns: list of dict with commands
    """
    event_func_name = event.lower().replace(' ', '_').replace('-', '')
    event_func = globals().get('cmd_set_' + event_func_name)
    if event_func is None:
        raise ValueError(f'unknown event {event!r}')

    if isinstance(params_str, str):
        if event == 'RTS':
            args = [params_str]
        elif event == 'Command':
            # Delegate parsing to cmd_set_command
            args = [params_str]
        else:
            params_str = params_str.upper().split()
            args = [coerce_type(p) for p in params_str]
    else:
        # Empty value means no args and implies params_str = np.ma.masked
        args = ()
    cmds = event_func(*args, date=date)

    # Load event does not generate commands
    if cmds is None:
        return None

    cmd_date = CxoTime(date)
    outs = []
    event_text = event.replace(' ', '_')
    event_date = CxoTime(date).date[:17]

    # Loop through commands. This could be a list of dict which is a relative
    # time specification of the commands (with 'dur' keys to indicate timing) or
    # a CommandTable, which has a date for each command.
    for step, cmd in enumerate(cmds):
        # Force CommandTable row to be a dict
        if not isinstance(cmd, dict):
            cmd = {name: cmd[name] for name in cmd.colnames}

        # Get command duration (if any). If the cmd is only {'dur': <dt>} then
        # it is a pure delay so skip subsequent processing.
        dur = cmd.pop('dur', None)
        if dur and not cmd:
            cmd_date += dur * u.s
            continue

        date = cmd.pop('date', cmd_date.date)
        tlmsid = cmd.pop('tlmsid', None)
        cmd_type = cmd.pop('type')

        params = {}
        params['event'] = event_text
        params['event_date'] = event_date
        for key, val in cmd.pop('params', {}).items():
            params[key.lower()] = val
        # Allow for params to be included in cmd dict directly as well as within
        # the 'params' key.
        for key, val in cmd.items():
            params[key.lower()] = val
        scs = params.pop('scs', 0)

        out = {'idx': -1,
               'date': date,
               'type': cmd_type,
               'tlmsid': tlmsid,
               'scs': scs,
               'step': step,
               'time': CxoTime(date).secs,
               'source': 'CMD_EVT',
               'vcdu': -1,
               'params': params}
        outs.append(out)

        if dur is not None:
            cmd_date += dur * u.s

    out = CommandTable(outs)
    return out
