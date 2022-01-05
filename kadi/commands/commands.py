import functools
import os

from kadi.commands import conf


def get_cmds(start=None, stop=None, inclusive_stop=False, scenario=None, **kwargs):
    """
    Get commands beteween ``start`` and ``stop``.

    By default the interval is ``start`` <= date < ``stop``, but if
    ``inclusive_stop=True`` then the interval is ``start`` <= date <= ``stop``.

    Additional ``key=val`` pairs can be supplied to further filter the results.
    Both ``key`` and ``val`` are case insensitive.  In addition to the any of
    the command parameters such as TLMSID, MSID, SCS, STEP, or POS, the ``key``
    can be:

    type
      Command type e.g. COMMAND_SW, COMMAND_HW, ACISPKT, SIMTRANS
    date
      Exact date of command e.g. '2013:003:22:11:45.530'

    If ``date`` is provided then ``start`` and ``stop`` values are ignored.

    Examples::

      >>> from kadi import commands cmds = commands.get_cmds('2012:001',
      >>> '2012:030') cmds = commands.get_cmds('2012:001', '2012:030',
      >>> type='simtrans') cmds = commands.get_cmds(type='acispkt',
      >>> tlmsid='wsvidalldn') cmds = commands.get_cmds(msid='aflcrset')
      >>> print(cmds)

    :param start: DateTime format (optional) Start time, defaults to beginning
        of available commands (2002:001)
    :param stop: DateTime format (optional) Stop time, defaults to end of available
        commands
    :param inclusive_stop: bool, include commands at exactly ``stop`` if True.
    :param scenario: str, None
        Commands scenario (applicable only for V2 commands)
    :param kwargs: key=val keyword argument pairs for filtering

    :returns: :class:`~kadi.commands.commands.CommandTable` of commands
    """
    commands_version = os.environ.get('KADI_COMMANDS_VERSION',
                                      conf.commands_version)
    if commands_version == '2':
        from kadi.commands.commands_v2 import get_cmds as get_cmds_
        get_cmds_ = functools.partial(get_cmds_, scenario=scenario)
    else:
        from kadi.commands.commands_v1 import get_cmds as get_cmds_

    cmds = get_cmds_(start=start, stop=stop,
                     inclusive_stop=inclusive_stop,
                     **kwargs)
    return cmds


def clear_caches():
    """Clear all commands caches.

    This is useful for testing and in case upstream products like the Command
    Events sheet have changed during a session.
    """
    commands_version = os.environ.get('KADI_COMMANDS_VERSION',
                                      conf.commands_version)
    if commands_version == '2':
        from kadi.commands.commands_v2 import clear_caches as clear_caches_vN
    else:
        from kadi.commands.commands_v1 import clear_caches as clear_caches_vN

    clear_caches_vN()
