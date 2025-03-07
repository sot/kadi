# Licensed under a 3-clause BSD style license - see LICENSE.rst
import calendar
import functools
import logging
import os
import pickle
import struct
import warnings
import weakref
from pathlib import Path
from typing import Callable

import numpy as np
import parse_cm.paths
import tables
from astropy.table import Column, Row, Table, TableAttribute, vstack
from cxotime import CxoTime
from ska_helpers import retry

from kadi.config import conf
from kadi.paths import IDX_CMDS_PATH, PARS_DICT_PATH

__all__ = [
    "read_backstop",
    "get_cmds_from_backstop",
    "CommandTable",
    "add_observations_to_cmds",
    "filter_cmd_events_by_event",
    "SCS107_EVENTS",
    "filter_scs107_events",
    "set_time_now",
]


# Events to filter for getting as-planned commands after SCS-107
SCS107_EVENTS = ["SCS-107", "Observing not run", "Obsid", "RTS"]

logger = logging.getLogger(__name__)


class LazyVal:
    def __init__(self, load_func):
        self._load_func = load_func

    def __getattribute__(self, name):
        try:
            val = object.__getattribute__(self, "_val")
        except AttributeError:
            val = object.__getattribute__(self, "_load_func")()
            self._val = val

        if name == "_val":
            return val
        else:
            return val.__getattribute__(name)

    def __getitem__(self, item):
        return self._val[item]

    def __setitem__(self, item, value):
        return self._val.__setitem__(item, value)

    def __repr__(self):
        return repr(self._val)

    def __str__(self):
        return str(self._val)

    def __len__(self):
        return self._val.__len__()


@retry.retry(tries=4, delay=0.5, backoff=4)
def load_idx_cmds(version=None, file=None):
    """Load the cmds.h5 file, trying up to 3 times

        It seems that one can occasionally get:

        File "tables/hdf5extension.pyx", line 492, in tables.hdf5extension.File._g_new
    tables.exceptions.HDF5ExtError: HDF5 error back trace
         ...
        File "H5FDsec2.c", line 941, in H5FD_sec2_lock
        unable to lock file, errno = 11, error message = 'Resource temporarily unavailable'
    """  # noqa: E501
    if file is None:
        file = IDX_CMDS_PATH(version)
    with tables.open_file(file, mode="r") as h5:
        idx_cmds = CommandTable(h5.root.data[:])
    logger.info(f"Loaded {file} with {len(idx_cmds)} commands")

    # For V2 add the params column here to make IDX_CMDS be same as regular cmds
    if version == 2:
        idx_cmds["params"] = None

    return idx_cmds


@retry.retry(tries=4, delay=0.5, backoff=4)
def load_pars_dict(version=None, file=None):
    if file is None:
        file = PARS_DICT_PATH(version)
    with open(file, "rb") as fh:
        pars_dict = pickle.load(fh, encoding="ascii")
    logger.info(f"Loaded {file} with {len(pars_dict)} pars")
    return pars_dict


@functools.lru_cache
def load_name_to_cxotime(name):
    """Convert load name to date"""
    mon = name[:3].capitalize()
    imon = list(calendar.month_abbr).index(mon)
    day = name[3:5]
    yr = name[5:7]
    if int(yr) > 50:
        year = f"19{yr}"
    else:
        year = f"20{yr}"
    out = CxoTime(f"{year}-{imon:02d}-{day}")
    out.format = "date"
    return out


def vstack_exact(tables):
    """Stack tables known to have identical types and columns and no metadata.

    Parameters
    ----------
    tables : list
        List of tables

    Returns
    -------
    Table
        Stacked table with same type as first table
    """
    new_cols = []
    table0 = tables[0]
    names0 = table0.colnames
    for table in tables[1:]:
        if set(table.colnames) != set(names0):
            raise ValueError(
                f"Tables have different column names: {names0} != {table.colnames}"
            )
    for name in table0.colnames:
        new_col = np.concatenate([t[name] for t in tables])
        new_cols.append(new_col)

    out = table0.__class__(new_cols, names=names0)
    return out


def read_backstop(
    backstop: str | Path | list[str],
    add_observations: bool = False,
    load_name=None,
) -> "CommandTable":
    """Read ``backstop`` and return a ``CommandTable``.

    The ``backstop`` argument can be either be a string or Path to a backstop file or
    a list of lines from a backstop file.

    If ``add_observations`` is ``True`` then add OBS commands (default=False) into the
    returned ``CommandTable``. This allows the returned ``CommandTable`` to be used in
    calls to ``get_observations`` and ``get_starcats``.

    If ``load_name`` is provided then this is used for the "source" column in the
    returned table.

    If ``load_name`` is not provided, the default is to use the load name if it can be
    determined from the backstop directory path. This path must follow the SOT MP
    naming convention where the directory path containing the backstop file looks like
    ``<prefix>/<YEAR>/<MON><DD><YY>/ofls<REV>/*.backstop``. An example would be
    ``/proj/sot/ska/data/mpcrit1/mplogs/2023/DEC1123/oflsa/CR344_2303.backstop``.

    If reading a backstop file from a local directory or from a list of lines then the
    load name should be provided explicitly.

    Examples
    --------
    >>> import kadi.commands as kc
    >>> import parse_cm.paths
    >>> path = parse_cm.paths.load_file_path('DEC1123A', "CR*.backstop", "backstop")
    >>> cmds = kc.read_backstop(path, add_observations=True)
    >>> obss = kc.get_observations(cmds=cmds)
    >>> starcats = kc.get_starcats(cmds=cmds)

    Parameters
    ----------
    backstop: str or Path or list of str
        Backstop file name or path, or backstop file contents as a list of str
    add_observations : bool
        Add OBS commands (default=False) to allow get_observations() and get_starcats()
    load_name : str, optional
        Load name to use for "source" column if provided.

    Returns
    -------
    :class:`~kadi.commands.core.CommandTable` of commands
    """

    # Get load name from backstop directory path
    if isinstance(backstop, str | Path) and load_name is None:
        backstop = Path(backstop)
        try:
            load_name = parse_cm.paths.load_name_from_load_dir(backstop.parent)
        except parse_cm.paths.ParseLoadNameError:
            pass

    if isinstance(backstop, Table):
        warnings.warn(
            "Passing a Table to read_backstop() is deprecated and will raise "
            "an exception in a future release.",
            FutureWarning,
        )

    cmds = get_cmds_from_backstop(backstop)

    if add_observations:
        cmds = add_observations_to_cmds(cmds, load_name)

    return cmds


def add_observations_to_cmds(
    cmds: "CommandTable",
    load_name: str | None = None,
) -> "CommandTable":
    """Add OBS commands to ``cmds`` and return new ``CommandTable``.

    This function adds ``tlmsid=OBS`` commands into a copy of ``cmds`` and returns the
    new ``CommandTable``. This allows the returned ``CommandTable`` to be used in calls
    to ``get_observations`` and ``get_starcats``.

    If ``load_name`` is provided then this is used for the "source" column in the
    returned table. Otherwise the "source" column in ``cmds`` is used (if it exists) or
    it is set to "None".

    Parameters
    ----------
    cmds : CommandTable
        Commands table to add observations.
    load_name : str, optional
        Load name to use for "source" column if provided.

    Returns
    -------
    CommandTable
        New commands table with observations added.
    """

    import kadi.commands.commands_v2 as kcc2
    import kadi.commands.states as kcs

    # Adding observations via add_obs_cmds() requires the "source" column to exist.
    if "source" not in cmds.colnames or load_name is not None:
        cmds = cmds.copy()
        cmds["source"] = str(load_name)

    # Get continuity for attitude, obsid, and SIM position. These are part of OBS cmd.
    rltt = cmds.get_rltt() or cmds["date"][0]
    cont = kcs.get_continuity(
        date=rltt, state_keys=["simpos", "obsid", "q1", "q2", "q3", "q4"]
    )
    prev_att = tuple(cont[f"q{ii}"] for ii in range(1, 5))

    # Add observations to cmds. This uses the global PARS_DICT and REV_PARS_DICT from
    # kadi.commands.commands_v2. The AOSTRCAT parameters are added there in-place. Since
    # those global dicts are persistent and cannot be reloaded, there is no concern
    # about collisions with a different version of the commands v2 database.
    cmds_obs = kcc2.add_obs_cmds(
        cmds,
        kcc2.PARS_DICT,
        kcc2.REV_PARS_DICT,
        prev_att=prev_att,
        prev_simpos=cont["simpos"],
        prev_obsid=cont["obsid"],
    )

    # General implementation note: Using a weakref means that it is not possible to
    # define local dicts to keep the starcat params "private" to the cmds object. These
    # local dicts will go out of scope and the weakref becomes undefined. When commands
    # V1 is removed then there is the possibility of removing the `rev_pars_dict`
    # attribute. In general it is fragile and basically only works with the global
    # REV_PARS_DICT anyway. For instance appending two command tables with different
    # REV_PARS_DICTs will not work as expected.

    # Add weakref to REV_PARS_DICT so that it can be used in CommandRow.__getitem__,
    # in particular for getting star catalogs via the `starcat_idx` key in OBS commands.
    cmds_obs.rev_pars_dict = weakref.ref(kcc2.REV_PARS_DICT)

    return cmds_obs


def get_cmds_from_backstop(backstop, remove_starcat=False):
    """
    Initialize a ``CommandTable`` from ``backstop``.

    ``backstop`` can either be a string file name or a backstop table from
    ``parse_cm.read_backstop``.

    Parameters
    ----------
    backstop
        str or Table
    remove_starcat
        remove star catalog command parameters (default=False)

    Returns
    -------
    :class:`~kadi.commands.core.CommandTable` of commands
    """
    if isinstance(backstop, Path):
        backstop = str(backstop)

    if isinstance(backstop, (str, list)):
        from parse_cm import read_backstop

        bs = read_backstop(backstop)
    elif isinstance(backstop, Table):
        bs = backstop
    else:
        raise ValueError("`backstop` arg must be a string filename or a backstop Table")

    n_bs = len(bs)
    out = {}
    # Set idx to -1 so it does not match any real idx
    out["idx"] = np.full(n_bs, fill_value=-1, dtype=np.int32)
    out["date"] = np.chararray.encode(bs["date"])
    out["type"] = np.chararray.encode(bs["type"])
    out["tlmsid"] = np.chararray.encode(bs["tlmsid"])
    out["scs"] = bs["scs"].astype(np.uint8)
    out["step"] = bs["step"].astype(np.uint16)
    out["time"] = CxoTime(bs["date"], format="date").secs
    # Set timeline_id to 0, does not match any real timeline id
    out["timeline_id"] = np.zeros(n_bs, dtype=np.uint32)
    out["vcdu"] = bs["vcdu"].astype(np.int32)
    out["params"] = bs["params"]

    if remove_starcat:
        # Remove the lengthy parameters in star catalog but leave the command
        for idx in np.flatnonzero(bs["type"] == "MP_STARCAT"):
            out["params"][idx] = {}

    # Backstop has redundant param keys, get rid of them here
    for params in out["params"]:
        for key in ("tlmsid", "step", "scs"):
            params.pop(key, None)

        # Match the hack in update_commands which swaps in "event_type" for "type"
        # for orbit event commands
        if "type" in params:
            params["event_type"] = params["type"]
            del params["type"]

    out = CommandTable(out)
    out["time"].info.format = ".3f"

    return out


def _find(
    start=None, stop=None, inclusive_stop=False, idx_cmds=None, pars_dict=None, **kwargs
):
    """
    Get commands beteween ``start`` and ``stop``.

    By default the interval is ``start`` <= date < ``stop``, but if
    ``inclusive_stop=True`` then the interval is ``start`` <= date <= ``stop``.

    Additional ``key=val`` pairs can be supplied to further filter the results.
    Both ``key`` and ``val`` are case insensitive.  In addition to the any of
    the command parameters such as TLMSID, MSID, SCS, STEP, or POS, the ``key``
    can be:

    date : Exact date of command e.g. '2013:003:22:11:45.530'
    type : Command type e.g. COMMAND_SW, COMMAND_HW, ACISPKT, SIMTRANS

    Examples::

      >>> from kadi import commands
      >>> cmds = commands._find('2012:001', '2012:030')
      >>> cmds = commands._find('2012:001', '2012:030', type='simtrans')
      >>> cmds = commands._find(type='acispkt', tlmsid='wsvidalldn')
      >>> cmds = commands._find(msid='aflcrset')

    Parameters
    ----------
    start : CxoTime format (optional)
        Start time, defaults to beginning of available commands (2002:001)
    stop : CxoTime format (optional)
        Stop time, defaults to end of available commands
    include_stop : bool (optional)
        If True, find commands with ``date <= stop``, otherwise ``date < stop``.
    idx_cmds : CommandTable-like
        Table of commands from the commands archive HDF5 file (e.g. ``cmds2.h5``).
        In reality this is a ``LazyVal`` which encapsulates a ``CommandTable``.
    pars_dict : dict-like
        Dict mapping a command parameters tuple to the index in the commands
        archive params pickle file.  This is a ``LazyVal`` which encapsulates a
        dict.
    **kwargs : dict
        Additional key=val keyword argument pairs to filter the results.

    Returns
    -------
    CommandTable
    """
    ok = np.ones(len(idx_cmds), dtype=bool)
    par_ok = np.zeros(len(idx_cmds), dtype=bool)

    date = kwargs.pop("date", None)
    if date:
        ok &= idx_cmds["date"] == CxoTime(date).date
    else:
        if start:
            ok &= idx_cmds["date"] >= CxoTime(start).date
        if stop:
            if inclusive_stop:
                ok &= idx_cmds["date"] <= CxoTime(stop).date
            else:
                ok &= idx_cmds["date"] < CxoTime(stop).date

    for key, val in kwargs.items():
        key = key.lower()  # noqa: PLW2901
        if isinstance(val, str):
            val = val.upper()  # noqa: PLW2901
        if key in idx_cmds.dtype.names:
            ok &= idx_cmds[key] == val
        else:
            par_ok[:] = False
            for pars_tuple, idx in pars_dict.items():
                if isinstance(pars_tuple, bytes):
                    # pars_dict values includes bytes strings for encoded
                    # starcat params. Just skip those.
                    continue
                pars = dict(pars_tuple)
                if pars.get(key) == val:
                    par_ok |= idx_cmds["idx"] == idx
            ok &= par_ok
    cmds = idx_cmds[ok]

    return cmds


def get_starcat_keys_types():
    cat_keys = [
        "dimdts",
        "imgsz",
        "imnum",
        "maxmag",
        "minmag",
        "restrk",
        "type",
        "yang",
        "zang",
    ]
    # Unsigned char (uint8) and float (np.float32)
    cat_types = ["B", "B", "B", "f", "f", "B", "B", "f", "f"]
    keys = ["cmds"]
    types = ["B"]
    for idx in range(1, 17):
        for cat_key, cat_type in zip(cat_keys, cat_types):
            keys.append(f"{cat_key}{idx}")
            types.append(cat_type)
    return keys, "".join(types)


STARCAT_KEYS, STARCAT_TYPES = get_starcat_keys_types()


def encode_starcat_params(params_dict):
    if set(params_dict.keys()) != set(STARCAT_KEYS):
        raise ValueError(
            f"Star catalog keys must be {STARCAT_KEYS}, got {set(params_dict.keys())}"
        )
    args = tuple(params_dict[key] for key in STARCAT_KEYS)
    return struct.pack(STARCAT_TYPES, *args)


def decode_starcat_params(params_bytes):
    vals = struct.unpack(STARCAT_TYPES, params_bytes)
    return dict(zip(STARCAT_KEYS, vals))


class CommandRow(Row):
    def __getitem__(self, item):
        if item == "params":
            out = super().__getitem__(item)
            if out is None:
                idx = super().__getitem__("idx")
                # self.parent.rev_pars_dict should be a weakref to the reverse
                # pars dict for this CommandTable. But it might not be defined,
                # in which case just leave it as None.
                if rev_pars_dict := self.table.rev_pars_dict:
                    parvals = rev_pars_dict()[idx]
                    if isinstance(parvals, bytes):
                        params = decode_starcat_params(parvals)
                    else:
                        params = dict(parvals)
                else:
                    raise KeyError(
                        "params cannot be mapped because the rev_pars_dict "
                        "attribute is not defined (needs to be a weakref.ref "
                        "of REV_PARS_DICT)"
                    )
                out = self["params"] = params
        elif item not in self.colnames:
            out = self["params"][item]
        else:
            out = super().__getitem__(item)
        return out

    def keys(self):
        out = [name for name in self.colnames if name != "params"]
        if "params" in self.colnames:
            params = [key.lower() for key in sorted(self["params"])]
        else:
            params = []

        return out + params

    def values(self):
        return [self[key] for key in self.keys()]  # noqa: SIM118

    def items(self):
        return list(zip(self.keys(), self.values()))

    def __repr__(self):
        out = f"<Cmd {str(self)}>"
        return out

    def __str__(self):
        keys = self.keys()
        keys.remove("date")
        keys.remove("type")
        if "idx" in keys:
            keys.remove("idx")

        out = "{} {} ".format(self["date"], self["type"]) + " ".join(
            "{}={}".format(key, self[key])
            for key in keys
            if key not in ("type", "date", "time")
        )
        return out

    def __sstr__(self):
        return str(self._table[self.index : self.index + 1])


class CommandTable(Table):
    """
    Astropy Table subclass that is specialized to handle commands.

    This uses a ``params`` column that is expected to be ``None`` or a dict of
    params.
    """

    rev_pars_dict = TableAttribute()

    COL_TYPES = {
        "idx": np.int32,
        "date": "S21",
        "type": "S12",
        "tlmsid": "S10",
        "scs": np.uint8,
        "step": np.uint16,
        "source": "S8",
        "timeline_id": np.uint32,
        "vcdu": np.int32,
        "time": np.float64,
        "params": object,
    }

    def _convert_data_to_col(self, *args, **kwargs):
        col = super()._convert_data_to_col(*args, **kwargs)
        if col.info.name in self.COL_TYPES:
            col = col.astype(self.COL_TYPES[col.info.name])
        return col

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.colnames:
                return self.columns[item]
            else:
                return Column([cmd["params"].get(item) for cmd in self], name=item)

        elif isinstance(item, (int, np.integer)):
            return CommandRow(self, item)

        elif isinstance(item, (tuple, list)) and all(x in self.colnames for x in item):
            from copy import deepcopy

            from astropy.table import groups

            out = self.__class__([self[x] for x in item], meta=deepcopy(self.meta))
            out._groups = groups.TableGroups(
                out, indices=self.groups._indices, keys=self.groups._keys
            )
            return out

        elif isinstance(item, (slice, np.ndarray, list)) or (
            isinstance(item, tuple) and all(isinstance(x, np.ndarray) for x in item)
        ):
            # here for the many ways to give a slice; a tuple of ndarray
            # is produced by np.where, as in t[np.where(t['a'] > 2)]
            # For all, a new table is constructed with slice of all columns
            return self._new_from_slice(item)

        else:
            raise ValueError(
                "Illegal type {0} for table item access".format(type(item))
            )

    def __str__(self):
        # Cut out index column for printing
        colnames = self.colnames
        if "idx" in colnames:
            colnames.remove("idx")
        return self.__repr__(colnames)

    def __repr__(self, colnames=None):
        if colnames is None:
            colnames = self.colnames

        # Nice repr of parameters.
        tmp_params = None
        if "params" in colnames:
            params_list = []
            for params in self["params"]:
                if params is None:
                    params_list.append("N/A")
                else:
                    param_strs = [
                        "{}={}".format(key, val) for key, val in params.items()
                    ]
                    params_str = " ".join(param_strs)
                    if len(params_str) > 40:
                        params_str = params_str[:40] + " ..."
                    params_list.append(params_str)
            tmp_params = params_list
            colnames.remove("params")

        tmp = self[colnames]
        if tmp_params:
            tmp["params"] = tmp_params

        lines = tmp.pformat(max_width=-1)
        return "\n".join(lines)

    def __bytes__(self):
        return str(self).encode("utf-8")

    def find_date(self, date, side="left"):
        """Find row in table corresponding to ``date``.

        This is a thin wrapper around np.searchsorted that converts ``date`` to
        a byte string before calling searchsorted. This is necessary because
        the ``date`` column is a byte string and the astropy unicode machinery
        ends up getting called a lot in a way that impacts performance badly.

        Parameters
        ----------
        date : str, sequence of str
            Date(s) to search for.
        side : {'left', 'right'}, optional
            If 'left', the index of the first suitable location found is given.
            If 'right', return the last such index.  If there is no suitable
            index, return either 0 or N (where N is the length of `a`).

        Returns
        -------
        int
            Index of row(s) corresponding to ``date``.
        """
        if isinstance(date, CxoTime):
            date = date.date
        date = np.char.encode(date, encoding="ascii")
        idxs = np.searchsorted(self["date"], date, side=side)
        return idxs

    def fetch_params(self):
        """
        Fetch all ``params`` for every row and force resolution of actual values.

        This is handy for printing a command table and seeing all the parameters
        at once.
        """
        if "params" not in self.colnames:
            self["params"] = None
        for cmd in self:
            cmd["params"]

    def get_rltt(self) -> str | None:
        """Return the first RLTT (Running Load Termination Time) command in table.

        This is a command of type LOAD_EVENT with
        event_type=RUNNING_LOAD_TERMINATION_TIME.

        Returns
        -------
        str or None
            RLTT as a date string (e.g. '2012:001:23:59:59.999') or None if
            there is no RLTT in the table.
        """
        for cmd in self:
            if (
                cmd["type"] == "LOAD_EVENT"
                and cmd["params"].get("event_type") == "RUNNING_LOAD_TERMINATION_TIME"
            ):
                return cmd["date"]

        return None

    def get_scheduled_stop_time(self) -> str | None:
        """Return the last scheduled stop time in table.

        This is a command of type LOAD_EVENT with event_type=SCHEDULED_STOP_TIME.

        Returns
        -------
        str or None
            Scheduled stop time as a date string (e.g. '2012:001:23:59:59.999')
            or None if there is no scheduled stop time in the table.
        """
        for idx in range(len(self), 0, -1):
            cmd = self[idx - 1]
            if (
                cmd["type"] == "LOAD_EVENT"
                and cmd["params"].get("event_type") == "SCHEDULED_STOP_TIME"
            ):
                return cmd["date"]

        return None

    def add_cmds(self, cmds: "CommandTable", rltt: str | None = None):
        """
        Add CommandTable ``cmds`` to self and return the new CommandTable.

        The commands table is maintained in order (date, step, scs).

        Parameters
        ----------
        cmds : CommandTable
            Commands to add.
        rltt : str, optional
            Clip existing commands to the RLTT (in "date" format) of the new commands.

        Returns
        -------
        CommandTable
        """
        orig = self
        if rltt is not None:
            ok = orig["date"] <= rltt
            orig = orig[ok]

        try:
            # Substantially faster than plain Table vstack (this is slow due to
            # checks for strings overflowing and other generalities)
            out = vstack_exact([orig, cmds])
        except ValueError:
            out = vstack([orig, cmds])
        out.sort_in_backstop_order()
        out.rev_pars_dict = self.rev_pars_dict

        return out

    def sort_in_backstop_order(self):
        """Sort table in order (date, step, scs)

        This matches the order in backstop.
        """
        # For V2 use stable sort just on date, preserving the existing order.
        # Copied verbatim from astropy.table.Table.sort except 'stable' sort.
        # (Astropy sort does not provide `kind` argument for .sort(), hopefully
        # fixed by astropy 5.1).
        indexes = self.argsort(["date"], kind="stable")

        with self.index_mode("freeze"):
            for col in self.columns.values():
                # Make a new sorted column.  This requires that take() also copies
                # relevant info attributes for mixin columns.
                new_col = col.take(indexes, axis=0)

                # First statement in try: will succeed if the column supports an
                # in-place update, and matches the legacy behavior of astropy
                # Table.  However, some mixin classes may not support this, so
                # in that case just drop in the entire new column. See #9553 and
                # #9536 for discussion.
                try:
                    col[:] = new_col
                except Exception:
                    # In-place update failed for some reason, exception class not
                    # predictable for arbitrary mixin.
                    self[col.info.name] = new_col

    def as_list_of_dict(self, ska_parsecm=False):
        """Convert CommandTable to a list of dict.

        The command ``params`` are embedded as a dict for each command.

        If ``ska_parsecm`` is True then the output is made more compatible with
        the legacy output from ``ska_parsecm.read_backstop()``, namely:

        - Add ``cmd`` key which is set to the ``type`` key
        - Make ``params`` keys uppercase.

        Parameters
        ----------
        ska_parsecm : bool
            Make output more ska_parsecm compatible

        Returns
        -------
        list of dict
        """
        self.fetch_params()

        names = self.colnames
        cmds_list = [{name: cmd[name] for name in names} for cmd in self]

        if ska_parsecm:
            for cmd in cmds_list:
                if "type" in cmd:
                    cmd["cmd"] = cmd["type"]
                if "params" in cmd:
                    cmd["params"] = {
                        key.upper(): val for key, val in cmd["params"].items()
                    }

        return cmds_list

    def pformat_like_backstop(
        self,
        show_source=True,
        show_nonload_meta=True,
        sort_orbit_events=False,
        max_params_width=80,
    ):
        """Format the table in a human-readable format that is similar to backstop.

        Parameters
        ----------
        show_source : bool, optional
            Show the source (load name) of each command (default=True)
        show_nonload_meta : bool, optional
            Show event and event_date for non-load commands (default=True)
        sort_orbit_events : bool, optional
            Sort orbit events at same date by event_type (default=False, mostly
            for testing)
        max_params_width : int, optional
            Maximum width of parameter values string (default=80)

        Returns
        -------
        list
            List of lines.
        """
        lines = []
        has_params = "params" in self.colnames

        if has_params and sort_orbit_events:
            cmds = self.copy()
            cmds["cmd_idx"] = np.arange(len(cmds))
            orb_cmds = cmds[cmds["type"] == "ORBPOINT"]
            cg = orb_cmds.group_by("date")
            for i0, i1 in zip(cg.groups.indices[:-1], cg.groups.indices[1:]):
                # For multiple orbit events at the same date, pull them out and
                # sort the event_type and index and update in place.
                if i1 - i0 > 1:
                    vals = sorted(
                        [cg[ii]["params"]["event_type"] for ii in range(i0, i1)]
                    )
                    idxs = sorted(cg["cmd_idx"][i0:i1])
                    for idx, val in zip(idxs, vals):
                        cmds[idx]["params"]["event_type"] = val
        else:
            cmds = self

        for cmd in cmds:
            if has_params:
                # Make a single string of params like POS= 75624, SCS= 130, STEP= 9
                fmtvals = []
                keys = cmd["params"]
                for key in keys:
                    if not show_nonload_meta and key in (
                        "nonload_id",
                        "event",
                        "event_date",
                    ):
                        continue
                    val = cmd["params"][key]
                    if key == "aoperige":
                        fmt = "{}={:.13e}"
                    elif isinstance(val, float):
                        fmt = "{}={:.8e}"
                    elif key == "packet(40)":
                        continue
                    elif key.startswith(("aopcads", "co", "afl", "2s1s", "2s2s")):
                        fmt = "{}={:d} "
                    else:
                        fmt = "{}={}"
                    fmtvals.append(fmt.format(key, val))

                fmtvals.append(f"scs={cmd['scs']}")

                params_str = ", ".join(fmtvals)
            else:
                params_str = "N/A"

            if max_params_width is not None:
                params_str = params_str[:max_params_width]

            fmts = ("{}", "{:16s}", "{:10s}")
            args = (cmd["date"], cmd["type"], cmd["tlmsid"])
            if show_source:
                if "source" in self.colnames:
                    fmts += ("{:8s}",)
                    args += (cmd["source"],)
                elif "timeline_id" in self.colnames:
                    fmts += ("{:8d}",)
                    args += (cmd["timeline_id"],)
            fmts += ("{}",)
            args += (params_str,)
            fmt = " | ".join(fmts)
            lines.append(fmt.format(*args))

        return lines

    def pprint_like_backstop(self, *, logger_func=None, logger_text="", **kwargs):
        """Format the table in a human-readable format that is similar to backstop.

        Parameters
        ----------
        logger_func : function, optional
            Function to call with the formatted lines (default is print)
        show_source : bool, optional
            Show the source (load name) of each command (default=True)
        show_nonload_meta : bool, optional
            Show event and event_date for non-load commands (default=True)
        sort_orbit_events : bool, optional
            Sort orbit events at same date by event_type (default=False, mostly
            for testing)
        max_params_width : int, optional
            Maximum width of parameter values string (default=80)

        Returns
        -------
        list
            List of formatted lines.
        """
        lines = self.pformat_like_backstop(**kwargs)
        if logger_func is None:
            for line in lines:
                print(line)
        else:
            logger_func(logger_text + "\n" + "\n".join(lines) + "\n")

    def deduplicate_orbit_cmds(self):
        """Remove duplicate orbit commands (ORBPOINT type) in place.

        In the event of a load stops and a replan, there can be multiple cmds
        that describe the same orbit event.  Since the detailed timing might
        change between schedule runs, cmds are considered the same if the date
        is within 3 minutes. This code chooses the cmd from the latest loads in
        this case.
        """
        idxs = np.where(self["type"] == "ORBPOINT")[0]
        orbit_cmds = self[idxs]
        orbit_cmds["idx"] = idxs
        orbit_cmds["time"] = CxoTime(orbit_cmds["date"]).secs

        # Turn into a list of Row's and sort by (event_type, date)
        orbit_cmds = list(orbit_cmds)
        orbit_cmds.sort(key=lambda y: (y["params"]["event_type"], y["date"]))

        uniq_cmds = [orbit_cmds[0]]
        # Step through one at a time and add to uniq_cmds only if the candidate is
        # "different" from uniq_cmds[-1].
        for cmd in orbit_cmds:
            last_cmd = uniq_cmds[-1]
            if (
                cmd["params"]["event_type"] == last_cmd["params"]["event_type"]
                and abs(cmd["time"] - last_cmd["time"]) < 180
            ):
                # Same event as last (even if date is a bit different).  Now if
                # this one has a larger timeline_id that means it is from a more
                # recent schedule, so use that one.
                load_date = load_name_to_cxotime(cmd["source"])
                load_date_last = load_name_to_cxotime(last_cmd["source"])
                if load_date > load_date_last:
                    uniq_cmds[-1] = cmd
            else:
                uniq_cmds.append(cmd)

        uniq_idxs = [uniq_cmd["idx"] for uniq_cmd in uniq_cmds]
        remove_idxs = set(idxs) - set(uniq_idxs)
        self.remove_rows(list(remove_idxs))

    def remove_not_run_cmds(self):
        """Remove commands with type=NOT_RUN from the table.

        This looks for type=NOT_RUN commands and then removes those and any
        commands with the same date and same TLMSID. These are excluded via
        the "Command not run" event in the Command Events sheet, e.g. the
        LETG retract command in the loads after the LETG insert anomaly.
        """
        idxs_not_run = np.where(self["type"] == "NOT_RUN")[0]
        if len(idxs_not_run) == 0:
            return

        idxs_remove = set()
        for idx in idxs_not_run:
            cmd_not_run = self[idx]
            ok = (
                (self["date"] == cmd_not_run["date"])
                & (self["type"] == cmd_not_run["__type__"])
                & (self["tlmsid"] == cmd_not_run["tlmsid"])
            )
            for key in ("scs", "step"):
                if cmd_not_run[key] != 0:
                    ok &= self[key] == cmd_not_run[key]

            # Indexes of self commands that *might* match cmd_not_run.
            idxs = np.arange(len(self))[ok].tolist()

            # Now check that the params match.
            idxs_match = []
            for idx_check in idxs:
                # Get the intersection of the keys in cmd_not_run["params"] and self["params"][idx]
                self_params = self["params"][idx_check]
                cmd_not_run_params = cmd_not_run["params"]
                keys = set(cmd_not_run_params) & set(self_params)

                # Check that the values match for all common keys
                match = all(cmd_not_run_params[key] == self_params[key] for key in keys)
                if match:
                    idxs_match.append(idx_check)

            idxs_remove.update(idxs_match)

        logger.info(f"Removing {len(idxs_remove)} NOT_RUN cmds")
        for idx in sorted(idxs_remove, reverse=True):
            logger.debug(f"  {self[idx]}")
        self.remove_rows(list(idxs_remove) + list(idxs_not_run))


def get_par_idx_update_pars_dict(pars_dict, cmd, params=None, rev_pars_dict=None):
    """Get par_idx representing index into pars tuples dict.

    This is used internally in updating the commands H5 and commands PARS_DICT
    pickle files. The ``pars_dict`` input is updated in place.

    This code was factored out verbatim from kadi.update_cmds.py.

    Parameters
    ----------
    pars_dict
        dict of pars tuples
    cmd : dict or CommandRow
        Command for updated par_idx
    pars : dict, optional
        If provided, this is used instead of cmd['params']
    rev_pars_dict : dict, optional
        If provided, also update the reverse dict.

    Returns
    -------
    int
        Params index (value of corresponding pars tuple dict key)
    """
    # Define a consistently ordered tuple that has all command parameter information
    if params is None:
        params = cmd["params"]
    keys = set(params.keys()) - {"SCS", "STEP", "TLMSID"}

    if cmd["tlmsid"] == "AOSTRCAT":
        pars_tup = encode_starcat_params(params) if params else ()
    else:
        if cmd["tlmsid"] == "OBS":
            # Re-order parameters to a priority order.
            new_keys = ["obsid", "simpos", "obs_stop", "manvr_start", "targ_att"]
            for key in sorted(cmd["params"]):
                if key not in new_keys:
                    new_keys.append(key)
            keys = new_keys
        else:
            # Maintain original order of keys for OBS command but sort the rest.
            # This is done so the OBS command displays more nicely.
            keys = sorted(keys)
        pars_tup = tuple((key.lower(), params[key]) for key in keys)

    try:
        par_idx = pars_dict[pars_tup]
    except KeyError:
        # Along with transition to 32-bit idx in #190, ensure that idx=65535
        # never gets used. Prior to #190 this value was being used by
        # get_cmds_from_backstop() assuming that it will never occur as a
        # key in the pars_dict. Adding 65536 allows older versions to work
        # with the new cmds.pkl pars_dict.
        par_idx = len(pars_dict) + 65536
        pars_dict[pars_tup] = par_idx
        if rev_pars_dict is not None:
            rev_pars_dict[par_idx] = pars_tup

    return par_idx


def ska_load_dir(load_name: str) -> Path:
    """Return a load directory path in $SKA/data from a load name like "DEC1123B"."""

    warnings.warn(
        "ska_load_dir() is deprecated. Use parse_cm.paths.load_dir_from_load_name()",
        FutureWarning,
        stacklevel=2,
    )
    from parse_cm.paths import load_dir_from_load_name

    return load_dir_from_load_name(load_name)


def get_cxotime_now() -> str | None:
    """Get the value of CXOTIME_NOW env var (or legacy proxy) for kadi commands.

    This returns the value of the CXOTIME_NOW environment variable if set,
    otherwise the value of the KADI_COMMANDS_DEFAULT_STOP environment variable,
    otherwise None.
    """

    cxotime_now = os.environ.get(
        "CXOTIME_NOW", kadi_stop := os.environ.get("KADI_COMMANDS_DEFAULT_STOP")
    )

    if kadi_stop:
        warnings.warn(
            "Setting KADI_COMMANDS_DEFAULT_STOP env var is deprecated, use CXOTIME_NOW",
            UserWarning,
            stacklevel=2,
        )

    return cxotime_now


def filter_cmd_events_date_stop(date_stop):
    """
    Returns an event filter function to remove events with ``Date > date_stop``.

    The returned function can be used as an ``event_filter`` argument to ``get_cmds()``
    and other related functions.

    Parameters
    ----------
    date_stop : CxoTimeLike
        Date string in CxoTime-compatible format.

    Returns
    -------
    Callable
        Function that takes a Table of command events and returns a boolean numpy array
        of the same length as the table. The array is a mask and ``True`` entries are
        kept.
    """

    def func(cmd_events):
        stop = CxoTime(date_stop)
        # Filter table based on stop date. Need to use CxoTime on each event separately
        # because the date format could be inconsistent.
        ok = [CxoTime(cmd_event["Date"]).date <= stop.date for cmd_event in cmd_events]
        logger.debug(
            f"Filtering cmd_events to stop date {stop.date} "
            f"({np.count_nonzero(ok)} vs {len(cmd_events)})"
        )
        return ok

    return func


@functools.lru_cache()
def filter_cmd_events_by_event(*args: tuple[str]) -> Callable:
    """
    Returns a function that filters command events based on a list of event names.

    Example::

      >>> import kadi.commands as kc
      >>> filter = kc.filter_cmd_events_by_event("SCS-107", "RTS", "Obsid", "Observing not run")
      >>> with kc.set_time_now("2025:015"):
      ...     cmds = kc.get_cmds("2025:001", "2025:015", event_filter=filter)

    See also: :func:`~kadi.commands.core.filter_scs107_events`.

    Parameters
    ----------
    *args : str
        Event names (e.g., "SCS-107" or "RTS") to filter out from the command events.

    Returns
    -------
    Callable
        Function that takes a Table of command events and returns a boolean numpy array.


    """

    def func(cmd_events: Table) -> np.ndarray[bool]:
        ok = ~np.isin(cmd_events["Event"], args)
        logger.info(
            f"Filtering cmd_events['Event'] that match {args} "
            f"({np.count_nonzero(ok)} vs {len(cmd_events)})"
        )
        return ok

    return func


def filter_scs107_events(cmd_events: Table) -> np.ndarray[bool]:
    """Filter SCS-107 related events from command history.

    This filters out the following command event types:
    - "SCS-107"
    - "Observing not run"
    - "Obsid"
    - "RTS"

    Example::

      >>> import kadi.commands as kc
      >>> with kc.set_time_now("2025:015"):
      ...     cmds = kc.get_cmds("2025:001", "2025:015", event_filter=kc.filter_scs107_events)

    Parameters
    ----------
    cmd_events : Table
        Command events, where each event has an "Event" attribute.

    Returns
    -------
    numpy.ndarray
        A boolean array indicating which command events are not SCS-107 related.
    """
    filter_func = filter_cmd_events_by_event(*SCS107_EVENTS)
    return filter_func(cmd_events)


def filter_cmd_events_state(cmd_events: Table) -> np.ndarray[bool]:
    """
    Filters command events based on State.

    Parameters
    ----------
    cmd_events : Table
        Command events, where each event has a "State" attribute.

    Returns
    -------
    numpy.ndarray
        A boolean array indicating which command events have allowed states.
    """
    allowed_states = ["Predictive", "Definitive"]
    # In-work can be used to validate the new event vs telemetry prior to make it
    # operational.
    if conf.include_in_work_command_events:
        allowed_states.append("In-work")
    ok = np.isin(cmd_events["State"], allowed_states)
    return ok


def set_time_now(date):
    """Context manager to temporarily set CXOTIME_NOW to ``date``.

    This temporarily sets the CXOTIME_NOW environment variable to ``date`` within the
    context block. This is a very thin wrapper around ska_helpers.utils.temp_env_var.
    This effectively makes ``date`` serve as the current time and is useful for testing.

    In this example we get the observation history as if the 2025:012:14:37:04 SCS-107
    run never happened::

      import kadi.commands as kc
      start = "2025:012"
      stop = "2025:018"
      with kc.set_time_now(stop):
          obss_as_planned = kc.get_observations(
              start, stop, event_filter=kc.filter_scs107_events
          )

    Parameters
    ----------
    date : CxoTimeLike
        Date in CxoTime-compatible format.
    """
    from ska_helpers.utils import temp_env_var

    return temp_env_var("CXOTIME_NOW", CxoTime(date).date)
