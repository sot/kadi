import warnings
import weakref

from astropy.table import Column
from cxotime import CxoTime

from kadi.commands.core import LazyVal, _find, load_idx_cmds, load_pars_dict

# Warn about deprecation but use FutureWarning so it actually shows up (since
# DeprecationWarning is ignored by default)
warnings.warn("kadi commands v1 is deprecated, use v2 instead", FutureWarning)

# Globals that contain the entire commands table and the parameters index
# dictionary.
IDX_CMDS = LazyVal(load_idx_cmds)
PARS_DICT = LazyVal(load_pars_dict)
REV_PARS_DICT = LazyVal(lambda: {v: k for k, v in PARS_DICT.items()})


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

    Parameters
    ----------
    start : DateTime format (optional)
        Start time, defaults to beginning of available commands (2002:001)
    stop : DateTime format (optional)
        Stop time, defaults to end of available commands
    inclusive_stop : bool
        Include commands at exactly ``stop`` if True.
    **kwargs : dict
        key=val keyword argument pairs for filtering

    Returns
    -------
    :class:`~kadi.commands.commands.CommandTable` of commands
    """
    out = _find(start, stop, inclusive_stop, IDX_CMDS, PARS_DICT, **kwargs)
    out.rev_pars_dict = weakref.ref(REV_PARS_DICT)
    out["params"] = None if len(out) > 0 else Column([], dtype=object)

    out.add_column(CxoTime(out["date"], format="date").secs, name="time", index=6)
    out["time"].info.format = ".3f"

    # Convert 'date' from bytestring to unicode. This is for compatibility with
    # the legacy V1 API.
    out.convert_bytestring_to_unicode()

    return out
