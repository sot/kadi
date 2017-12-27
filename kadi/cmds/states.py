"""
This module provides the core functions for creating, manipulating and updating
the Chandra commanded states database.
"""

import collections
import itertools

import numpy as np
import six

from astropy.table import Table, Column

from . import cmds as commands

from Chandra.Time import DateTime
import Chandra.Maneuver
from Quaternion import Quat

REV_PARS_DICT = commands.rev_pars_dict

# Registry of Transition classes with state transition name(s) as key
TRANSITIONS = collections.defaultdict(list)
STATE_KEYS = []


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


class TransitionMeta(type):
    """
    Metaclass that adds the class to the TRANSITIONS registry.
    """
    def __new__(mcls, name, bases, members):
        cls = super(TransitionMeta, mcls).__new__(mcls, name, bases, members)

        # Register transition classes that have a `transition_name`.
        if 'transition_name' in members:
            if 'state_keys' not in members:
                cls.state_keys = [cls.transition_name]

            for state_key in cls.state_keys:
                if state_key not in STATE_KEYS:
                    STATE_KEYS.append(state_key)
                TRANSITIONS[state_key].append(cls)

        return cls


@six.add_metaclass(TransitionMeta)
class BaseTransition(object):
    @classmethod
    def get_state_changing_commands(cls, cmds):
        """
        Get commands that match the required attributes for state changing commands.
        """
        ok = np.ones(len(cmds), dtype=bool)
        for attr, val in cls.command_attributes.items():
            ok = ok & (np.asarray(cmds[attr]) == val)
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
        val = cls.transition_val
        attr = cls.transition_name

        for cmd in state_cmds:
            transitions[cmd['date']][attr] = val


class NMM_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': 'AONMMODE'}
    transition_name = 'pcad_mode'
    transition_val = 'NMAN'


class NPM_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': 'AONPMODE'}
    transition_name = 'pcad_mode'
    transition_val = 'NPNT'


# class ACISTransition(BaseTransition):
#     command_attributes = {'type': 'ACISPKT'}
#     transition_name = 'acis'

#     @classmethod
#     def set_transitions(cls, transitions, cmds):
#         state_cmds = cls.get_state_changing_commands(cmds)
#         for cmd in state_cmds:
#             tlmsid = cmd['tlmsid']
#             date = cmd['date']

# class NPM_AutoEnableTransition(BaseTransition):
#     command_attributes = {'type': 'COMMAND_SW',
#                           'tlmsid': 'AONM2NPE'}
#     transition_name = 'pcad_mode'

#     @classmethod
#     def set_transitions(cls, transitions, cmds):
#         state_cmds = cls.get_state_changing_commands(cmds)
#         for cmd in state_cmds:
#             date = cmd['date']
#             transitions[date].update({'auto_npnt': True})


class HETG_INSR_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': '4OHETGIN'}
    transition_name = 'hetg'
    transition_val = 'INSR'


class HETG_RETR_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': '4OHETGRE'}
    transition_name = 'hetg'
    transition_val = 'RETR'


class LETG_INSR_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': '4OLETGIN'}
    transition_name = 'letg'
    transition_val = 'INSR'


class LETG_RETR_Transition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': '4OLETGRE'}
    transition_name = 'letg'
    transition_val = 'RETR'


class DitherEnableTransition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': 'AOENDITH'}
    transition_name = 'dither'
    transition_val = 'ENAB'


class DitherDisableTransition(SingleFixedTransition):
    command_attributes = {'type': 'COMMAND_SW',
                          'tlmsid': 'AODSDITH'}
    transition_name = 'dither'
    transition_val = 'DISA'


class ParamTransition(BaseTransition):
    @classmethod
    def set_transitions(cls, transitions, cmds):
        """
        Set transitions for a Table of commands ``cmds``.  This is the simplest
        case where there is an attribute that gets set to a specified
        value in the command, e.g. MP_OBSID or SIMTRANS
        """
        state_cmds = cls.get_state_changing_commands(cmds)
        param_key = cls.transition_param_key
        name = cls.transition_name

        for cmd in state_cmds:
            val = dict(REV_PARS_DICT[cmd['idx']])[param_key]
            transitions[cmd['date']][name] = val


class ObsidTransition(ParamTransition):
    command_attributes = {'type': 'MP_OBSID'}
    transition_name = 'obsid'
    transition_param_key = 'id'


class SimTscTransition(ParamTransition):
    command_attributes = {'type': 'SIMTRANS'}
    transition_param_key = 'pos'
    transition_name = 'simpos'


class SimFocusTransition(ParamTransition):
    command_attributes = {'type': 'SIMFOCUS'}
    transition_param_key = 'pos'
    transition_name = 'simfa_pos'


class ACISTransition(BaseTransition):
    command_attributes = {'type': 'ACISPKT'}
    transition_name = 'acis'
    state_keys = ['clocking', 'power_cmd', 'vid_board', 'fep_count', 'si_mode', 'ccd_count']

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


def get_transition_classes(state_keys=None):
    """
    Get all BaseTransition subclasses in this module corresponding to
    state keys ``state_keys``.
    """
    if isinstance(state_keys, six.string_types):
        state_keys = [state_keys]

    if state_keys is None:
        # itertools.chain => concat list of lists
        trans_classes = set(itertools.chain.from_iterable(TRANSITIONS.values()))
    else:
        trans_classes = set(itertools.chain.from_iterable(
                classes for state_key, classes in TRANSITIONS.items()
                if state_key in state_keys))
    return trans_classes


def get_transitions(cmds, state_keys=None):
    transitions = collections.defaultdict(dict)

    for transition_class in get_transition_classes(state_keys):
        transition_class.set_transitions(transitions, cmds)

    return transitions


def get_states_for_cmds(cmds, state_keys=None):
    # Define complete list of column names for output table corresponding to
    # each state key.  Maintain original order and uniqueness of keys.
    if state_keys is None:
        state_keys = tuple(TRANSITIONS.keys())

    indexes = {key: index for index, key in enumerate(state_keys)}

    # Make a dict of lists to hold state values
    # states = {key: [None] for key in state_keys}
    # states['datestart'][0] = [None]
    states = [list(None for key in state_keys)]

    transitions = get_transitions(cmds, state_keys)
    transition_dates = sorted(transitions.keys())

    i_complete = 0
    datestarts = [None]
    states = [[None] * len(state_keys)]

    for i, date in enumerate(transition_dates):
        transition = transitions[date]

        state = list(states[-1])
        for key, value in transition.items():
            state[indexes[key]] = value

        if state == states[-1]:
            continue

        datestarts.append(date)
        states.append(state)

        # Find the first row where all state keys are defined
        if i_complete == 0 and all(x is not None for x in state):
            i_complete = i + 1

    if i_complete == 0:
        raise ValueError('not all state keys defined')

    states = Table(rows=states[i_complete:], names=state_keys)
    states.add_column(Column(datestarts[i_complete:], name='datestart'), 0)

    return states
