"""
This module provides the core functions for creating, manipulating and updating
the Chandra commanded states database.
"""

import collections

import numpy as np

from . import cmds as commands

from Chandra.Time import DateTime
import Chandra.Maneuver
from Quaternion import Quat

REV_PARS_DICT = commands.rev_pars_dict

# Canonical state0 giving spacecraft state at beginning of timelines
# 2002:007:13 fetch --start 2002:007:13:00:00 --stop 2002:007:13:02:00 aoattqt1
# aoattqt2 aoattqt3 aoattqt4 cobsrqid aopcadmd tscpos
STATE0 = {'ccd_count': 5,
          'clocking': 0,
          'datestart': '2002:007:13:00:00.000',
          'datestop': '2099:001:00:00:00.000',
          'dec': -11.500,
          'fep_count': 0,
          'hetg': 'RETR',
          'letg': 'RETR',
          'obsid': 61358,
          'pcad_mode': 'NPNT',
          'pitch': 61.37,
          'power_cmd': 'AA00000000',
          'q1': -0.568062,
          'q2': 0.121674,
          'q3': 0.00114141,
          'q4': 0.813941,
          'ra': 352.000,
          'roll': 289.37,
          'si_mode': 'undef',
          'simfa_pos': -468,
          'simpos': -99616,
          'trans_keys': 'undef',
          'tstart': 127020624.552,
          'tstop': 3187296066.184,
          'vid_board': 0,
          'dither': 'None'}


def decode_power(mnem):
    """
    Decode number of chips and feps from a ACIS power command
    Return a dictionary with the number of chips and their identifiers

    Example::

     >>> decode_power("WSPOW08F3E")
     {'ccd_count': 5,
      'ccds': 'I0 I1 I2 I3 S3 ',
      'fep_count': 5,
      'feps': '1 2 3 4 5 '}

    :param mnem: power command string

    """
    # the hex for the commanding is after the WSPOW
    powstr = mnem[5:]
    if (len(powstr) != 5):
        raise ValueError("%s in unexpected format" % mnem)

    # convert the hex to decimal and "&" it with 63 (binary 111111)
    fepkey = int(powstr, 16) & 63
    fep_info = {'fep_count': 0,
                'ccd_count': 0,
                'feps': '',
                'ccds': ''}
    # count the true binary bits
    for bit in xrange(0, 6):
        if (fepkey & (1 << bit)):
            fep_info['fep_count'] = fep_info['fep_count'] + 1
            fep_info['feps'] = fep_info['feps'] + str(bit) + ' '

    # convert the hex to decimal and right shift by 8 places
    vidkey = int(powstr, 16) >> 8

    # count the true bits
    for bit in xrange(0, 10):
        if (vidkey & (1 << bit)):
            fep_info['ccd_count'] = fep_info['ccd_count'] + 1
            # position indicates I or S chip
            if (bit < 4):
                fep_info['ccds'] = fep_info['ccds'] + 'I' + str(bit) + ' '
            else:
                fep_info['ccds'] = fep_info['ccds'] + 'S' + str(bit - 4) + ' '

    return fep_info


def _make_add_trans(transitions, date, exclude):
    def add_trans(date=date, **kwargs):
        # if no key in kwargs is in the exclude set then update transition
        if not (exclude and set(exclude).intersection(kwargs)):
            transitions.setdefault(date, {}).update(kwargs)
    return add_trans


class BaseTransition(object):
    @classmethod
    def get_state_changing_commands(cls, cmds):
        """
        Get commands that match the required attributes for state changing commands.
        """
        ok = np.ones(len(cmds), dtype=bool)
        for attr, val in cls.command_attributes.items():
            ok = ok & (cmds[attr] == val)
        return cmds[ok]


class SingleFixedTransition(BaseTransition):
    @classmethod
    def set_transitions(cls, transitions, cmds):
        """
        Set transitions for a Table of commands ``cmds``.  This is the simplest
        case where there is a single fixed attribute that gets set to a fixed
        value, e.g. pcad_mode='NMAN' for NMM.
        """
        state_cmds = cls.get_state_changing_commands(cmds)
        attr = cls.transition_attr
        val = cls.transition_val
        for cmd in state_cmds:
            transitions[cmd['date']][attr] = val


class NMM_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': 'AONMMODE'}
    transition_attr = 'pcad_mode'
    transition_val = 'NMAN'


class NPM_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': 'AONPMODE'}
    transition_attr = 'pcad_mode'
    transition_val = 'NPNT'


class HETG_INSR_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': '4OHETGIN'}
    transition_attr = 'hetg'
    transition_val = 'INSR'


class HETG_RETR_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': '4OHETGRE'}
    transition_attr = 'hetg'
    transition_val = 'RETR'


class LETG_INSR_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': '4OLETGIN'}
    transition_attr = 'letg'
    transition_val = 'INSR'


class LETG_RETR_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': '4OLETGRE'}
    transition_attr = 'letg'
    transition_val = 'RETR'


class DitherEnableTransition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': 'AOENDITH'}
    transition_attr = 'dither'
    transition_val = 'ENAB'


class DitherDisableTransition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': 'AODSDITH'}
    transition_attr = 'dither'
    transition_val = 'DISA'


class ParamTransition(BaseTransition):
    @classmethod
    def set_transitions(cls, transitions, cmds):
        """
        Set transitions for a Table of commands ``cmds``.  This is the simplest
        case where there is a single fixed attribute that gets set to a fixed
        value, e.g. pcad_mode='NMAN' for NMM.
        """
        state_cmds = cls.get_state_changing_commands(cmds)
        attr = cls.transition_attr
        param_key = cls.transition_param_key
        for cmd in state_cmds:
            val = dict(REV_PARS_DICT[cmd['idx']])[param_key]
            transitions[cmd['date']][attr] = val


class ObsidTransition(ParamTransition):
    command_attributes = {'type': 'MP_OBSID'}
    transition_attr = 'obsid'
    transition_param_key = 'id'


class SimTscTransition(ParamTransition):
    command_attributes = {'type': 'SIMTRANS'}
    transition_attr = 'simpos'
    transition_param_key = 'pos'


class SimFocusTransition(ParamTransition):
    command_attributes = {'type': 'SIMFOCUS'}
    transition_attr = 'simfa_pos'
    transition_param_key = 'pos'


class ACISTransition(BaseTransition):
    command_attributes = {'type': 'ACISPKT'}
    transition_attr = 'acis'

    @classmethod
    def set_transitions(cls, transitions, cmds):
        state_cmds = cls.get_state_changing_commands(cmds)
        for cmd in state_cmds:
            tlmsid = cmd['tlmsid']
            date = cmd['date']

            if tlmsid.startswith('WSPOW'):
                pwr = decode_power(tlmsid)
                transitions[date].update(fep_count=pwr['fep_count'],
                                         ccd_count=pwr['ccd_count'],
                                         vid_board=1, clocking=0,
                                         power_cmd=tlmsid)

            elif tlmsid in ('XCZ0000005', 'XTZ0000005'):
                transitions[date].update(clocking=1, power_cmd=tlmsid)

            elif tlmsid == 'WSVIDALLDN':
                transitions[date].update(vid_board=0, power_cmd=tlmsid)

            elif tlmsid == 'AA00000000':
                transitions[date].update(clocking=0, power_cmd=tlmsid)

            elif tlmsid == 'WSFEPALLUP':
                transitions[date].update(fep_count=6, power_cmd=tlmsid)

            elif tlmsid.startswith('WC'):
                transitions[date].update(si_mode='CC_' + tlmsid[2:7])

            elif tlmsid.startswith('WT'):
                transitions[date].update(si_mode='TE_' + tlmsid[2:7])


def get_transition_classes(trans_type='all'):
    """
    Get all BaseTransition subclasses in this module
    """
    import inspect

    trans_classes = []
    for name, var in globals().items():
        if (inspect.isclass(var) and issubclass(var, BaseTransition)
                and hasattr(var, 'transition_attr')):
            transition_attr = getattr(var, 'transition_attr')
            # transition_attr can be either a single string or a list of strings
            if (trans_type == 'all'
                    or (isinstance(transition_attr, basestring) and transition_attr == trans_type)
                    or (trans_type in transition_attr)):
                trans_classes.append(var)

    return trans_classes


def get_transitions(cmds, trans_types=['all']):
    transitions = collections.defaultdict(dict)
    for trans_type in trans_types:
        for trans_class in get_transition_classes(trans_type):
            trans_class.set_transitions(transitions, cmds)
    return transitions
