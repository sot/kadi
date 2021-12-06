# Licensed under a 3-clause BSD style license - see LICENSE.rst
import re
from pathlib import Path

import astropy.units as u

from Quaternion import Quat
from cxotime import CxoTime
from parse_cm.common import _coerce_type as coerce_type
from kadi.commands.core import CommandTable

RTS_PATH = Path('FOT/configuration/products/rts')


def cmd_set_rts(date, *args):
    from parse_cm.csd import csd_cmd_gen

    rts = 'SCS_CATEGORY,OBSERVING\n' + ''.join(args).upper()
    cmds = csd_cmd_gen(rts, date=date)
    return cmds


def cmd_set_obsid(obs_id):
    """Return a command set that initiates a maneuver to the given attitude
    ``att``.

    :param obsid: obsid
    :returns: list of command defs suitable for generate_cmds()
    """
    return (dict(type='MP_OBSID',
                 tlmsid='COAOSQID',
                 params=dict(ID=obs_id)),)


def cmd_set_maneuver(*args):
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


def cmd_set_aciscti():
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


def cmd_set_scs107():
    return (dict(type='COMMAND_SW',
                 dur=1.025,
                 tlmsid='OORMPDS'),
            dict(type='COMMAND_HW',
                 dur=1.025,
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
                 tlmsid='WSPOW0002A'),
            )


def cmd_set_dither(state):
    if state not in ('ON', 'OFF'):
        raise ValueError(f'Invalid dither state {state!r}')
    enab = 'EN' if state == 'ON' else 'DS'
    return (dict(type='COMMAND_SW',
                 tlmsid=f'AO{enab}DITH',
                 ),
            )


def cmd_set_nsm():
    nsm_cmd = dict(type='COMMAND_SW',
                   tlmsid='AONSMSAF')
    out = ((nsm_cmd,)
           + cmd_set_scs107()
           + cmd_set_dither('OFF')
           )
    return out


def cmd_set_load_not_run(load_name):
    return None


def cmd_set_observing_not_run(load_name):
    return None


def cmd_set_command(*args):
    cmd = {'type': args[0]}

    params = {}
    for param in args[1:]:
        if param == '|':
            continue
        param = re.sub(r'\s+', '', param)
        key, val = param.split('=')
        key = key.upper()
        if key == 'TLMSID':
            cmd['tlmsid'] = val
        else:
            params[key] = coerce_type(val)
    cmd['params'] = params

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
        params_str = params_str.upper().split()
        args = [coerce_type(p) for p in params_str]
        if event == 'RTS':
            # Ycky hack, better way?
            args = [date] + args
    else:
        # Empty value means no args and implies params_str = np.ma.masked
        args = ()
    cmds = event_func(*args)

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
    for cmd in cmds:
        # Force CommandTable row to be a dict
        if not isinstance(cmd, dict):
            cmd = {name: cmd[name] for name in cmd.colnames}

        args = {}
        args['event'] = event_text
        args['event_date'] = event_date
        for key, val in cmd.get('params', {}).items():
            args[key.lower()] = val
        scs = args.pop('scs', 0)
        date = cmd.get('date', cmd_date.date)
        out = {'idx': -1,
               'date': date,
               'type': cmd['type'],
               'tlmsid': cmd.get('tlmsid', 'None'),
               'scs': scs,
               'step': 0,
               'source': 'CMD_EVT',
               'vcdu': -1,
               'params': args}
        outs.append(out)

        if 'dur' in cmd:
            cmd_date += cmd['dur'] * u.s

    out = CommandTable(outs)
    return out
