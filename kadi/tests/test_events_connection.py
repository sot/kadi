# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Test that kadi.events queries recover when the events database file is
replaced on disk (e.g. by the kadi_events cron task doing an atomic swap)
while a long-lived process holds an open database connection.

The stale-file tests run kadi in a subprocess with ``KADI`` pointing at a
temporary directory, since the events database path and Django setup are
frozen at import time.  On a local filesystem the old file handle keeps
reading the unlinked inode (no "disk I/O error" as on NFS), so the tests
assert the observable mechanism: after the swap, queries must see the new
file's contents, which requires dropping the stale connection.

The stale-connection handling is active only on Linux (the production
platform), so the subprocess scripts fake ``platform.system()`` to "Linux"
to exercise the mechanism on any OS (a no-op when actually on Linux).
"""

import os
import sqlite3
import subprocess
import sys
import textwrap

import pytest

# These tests atomically replace the database file while a sqlite connection
# has it open. Windows does not allow renaming over an open sqlite database
# (no FILE_SHARE_DELETE), so the scenario under test cannot occur there.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="requires POSIX rename-over-open-file semantics"
)

OBSID_TABLE_SQL = (
    'CREATE TABLE "events_obsid" ("start" varchar(21) NOT NULL PRIMARY KEY, '
    '"stop" varchar(21) NOT NULL, "tstart" real NOT NULL, "tstop" real NOT NULL, '
    '"dur" real NOT NULL, "obsid" integer NOT NULL)'
)


def make_events_db(path, obsids):
    """Create a minimal events database at ``path`` with one obsid event per
    entry in ``obsids``."""
    with sqlite3.connect(path) as con:
        con.execute(OBSID_TABLE_SQL)
        for ii, obsid in enumerate(obsids):
            start = f"2020:{ii + 1:03d}:00:00:00.000"
            stop = f"2020:{ii + 1:03d}:01:00:00.000"
            con.execute(
                "INSERT INTO events_obsid VALUES (?, ?, ?, ?, ?, ?)",
                (start, stop, 0.0, 3600.0, 3600.0, obsid),
            )


def run_kadi_subprocess(script, tmp_path):
    """Run ``script`` with python in a subprocess with KADI=tmp_path."""
    env = os.environ.copy()
    env["KADI"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_query_recovers_after_db_file_replaced(tmp_path):
    """After events3.db3 is atomically replaced (new inode), the next query
    must see the new file's contents instead of serving the stale handle."""
    make_events_db(tmp_path / "events3.db3", obsids=[1])
    stdout = run_kadi_subprocess(
        """
        import os
        import platform

        platform.system = lambda: "Linux"

        from kadi import events
        from kadi.tests.test_events_connection import make_events_db

        live = os.path.join(os.environ["KADI"], "events3.db3")

        def n_obsid(obsid):
            return len(events.obsids.filter(obsid__exact=obsid))

        n_before = n_obsid(1)
        # Atomic swap with a new file containing an additional obsid=2 event
        make_events_db(live + ".new", obsids=[1, 2])
        os.replace(live + ".new", live)
        n_after = n_obsid(2)
        print("RESULTS", [n_before, n_after])
        """,
        tmp_path,
    )
    assert "RESULTS [1, 1]" in stdout


def test_each_thread_recovers_after_db_file_replaced(tmp_path):
    """Every thread holds its own Django connection, so every thread must
    independently detect the file swap and drop its own stale connection."""
    make_events_db(tmp_path / "events3.db3", obsids=[1])
    stdout = run_kadi_subprocess(
        """
        import os
        import platform
        import threading

        platform.system = lambda: "Linux"

        from kadi import events
        from kadi.tests.test_events_connection import make_events_db

        live = os.path.join(os.environ["KADI"], "events3.db3")

        def n_obsid(obsid):
            return len(events.obsids.filter(obsid__exact=obsid))

        results = {}
        barrier = threading.Barrier(3)

        def worker(name):
            results[name, "before"] = n_obsid(1)
            barrier.wait()  # all threads have live connections
            barrier.wait()  # main thread has swapped the file
            results[name, "after"] = n_obsid(2)

        threads = [threading.Thread(target=worker, args=(ii,)) for ii in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        make_events_db(live + ".new", obsids=[1, 2])
        os.replace(live + ".new", live)
        barrier.wait()
        for thread in threads:
            thread.join()
        print("RESULTS", sorted(results.items()))
        """,
        tmp_path,
    )
    expected = [
        ((0, "after"), 1),
        ((0, "before"), 1),
        ((1, "after"), 1),
        ((1, "before"), 1),
    ]
    assert f"RESULTS {expected}" in stdout
