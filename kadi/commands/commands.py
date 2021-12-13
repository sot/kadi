import os

from kadi.commands import conf


def get_cmds(start=None, stop=None, inclusive_stop=False, **kwargs):
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
    :param version: int, None
        Version of commands archive to use (default=version 1)
    :param kwargs: key=val keyword argument pairs for filtering

    :returns: :class:`~kadi.commands.commands.CommandTable` of commands
    """
    commands_version = os.environ.get('KADI_COMMANDS_VERSION',
                                      conf.commands_version)
    if commands_version == '2':
        from kadi.commands.commands_v2 import get_cmds as get_cmds_vN
    else:
        from kadi.commands.commands_v1 import get_cmds as get_cmds_vN

    cmds = get_cmds_vN(start=start, stop=stop,
                       inclusive_stop=inclusive_stop,
                       **kwargs)
    return cmds
