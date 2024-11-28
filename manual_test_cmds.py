#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Test cmds archive.
"""

import os

import numpy as np
import ska_file
from chandra_time import DateTime

from kadi import update_cmds


def test_ingest():
    """Test ingest one day at a time.

    Test that doing the ingest a day at a time (which is the normal operational
    scenario) gives the same result as a one-time bulk ingest.  The latter is
    "easier" because there are no intermediate deletes and re-inserts.
    """

    # Some funkiness to be able to use two different data root values
    # within the same python session.  Inject the correct PATH and make
    # sure the LazyVals will re-read.
    data_root_tmp = ska_file.TempDir()
    data_root = data_root_tmp.name
    update_cmds.main(
        [
            "--start",
            "2011:340:12:00:00",
            "--stop",
            "2012:100:12:00:00",
            "--data-root",
            data_root,
        ]
    )

    # Some chicken-n-egg
    os.environ["KADI"] = os.path.abspath(data_root)
    from kadi.cmds import cmds

    # Reset
    if hasattr(cmds.idx_cmds, "_val"):
        del cmds.idx_cmds._val
    if hasattr(cmds.pars_dict, "_val"):
        del cmds.pars_dict._val

    cmds_at_once = cmds.filter("2012:010:12:00:00", "2012:090:12:00:00")
    pars_at_once = {v: k for k, v in cmds.pars_dict.items()}

    # Second part of funkiness to be able to use two different data root values
    # within the same python session.
    data_root_tmp = ska_file.TempDir()
    data_root = data_root_tmp.name
    os.environ["KADI"] = os.path.abspath(data_root)

    if hasattr(cmds.idx_cmds, "_val"):
        del cmds.idx_cmds._val
    if hasattr(cmds.pars_dict, "_val"):
        del cmds.pars_dict._val

    stop0 = DateTime("2012:001:12:00:00")
    for dt in range(0, 100, 1):
        update_cmds.main(["--stop", (stop0 + dt).date, "--data-root", data_root])

    cmds_per_day = cmds.filter("2012:010:12:00:00", "2012:090:12:00:00")
    pars_per_day = {v: k for k, v in cmds.pars_dict.items()}

    for name in ("date", "type", "tlmsid", "scs", "step", "timeline_id"):
        assert np.all(cmds_at_once[name] == cmds_per_day[name])
    for idx_at_once, idx_per_day in zip(cmds_at_once["idx"], cmds_per_day["idx"]):
        assert pars_at_once[idx_at_once] == pars_per_day[idx_per_day]

    assert len(cmds_per_day) == 12628  # Regression test of ingest process
