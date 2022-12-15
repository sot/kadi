#######################
Kadi events maintenance
#######################

Add a new event type or modify an existing type
###############################################

See https://github.com/sot/kadi/pull/34 for canonical example of file updates.

See also https://stackoverflow.com/questions/24311993
(how-to-add-a-new-field-to-a-model-with-new-django-migrations)

Setup
-----

Copy the current flight events into the local dir and set up kadi to use it::

    cd ~/git/kadi

    export KADI=$PWD
    cp $SKA/kadi/events3.db3 ./

If necessary switch to master or the latest flight release that DOES NOT HAVE
the new model. We need this for the migrations step to get django to set up the
events3.db3 database file properly.

To update the format of an existing event type, drop table(s)::

    sqlite3 events3.db3
    .tables  # to get table names
    drop table events_<model_class_name> ;
    # Note possibility of coupled tables like (manvr, manvrseq),
    # (orbit, orbitpoint), (darkcal, darkcalreplica).  In these cases
    # drop both.
    select * from events_update;
    delete from events_update where name='Manvr'  # for example
    delete from events_update where name='ManvrSeq'  # if needed

Migrations
----------

Remove any existing migrations and remake from the CURRENT RELEASE version of code.

    rm -rf kadi/events/migrations
    ./manage.py makemigrations events
    ./manage.py migrate --fake events

Switch to a new branch and edit or add new model in kadi/events/models.py

With the new or updated model class definition, apply the migration process. This should
indicate adding the new/updated event type::

    ./manage.py makemigrations events
    ./manage.py migrate events

Populate database
-----------------

Now add the new event data to the database::

    export ModelClassName=<model_class_name>
    python -m kadi.update_events --start=1999:200 --stop=2001:001 --model=${ModelClassName}

Update early events first and look for warnings.  Also confirm that the first event
matches what is in the current database unless a change is intended.
Some events may need a later start data to be fully sampled.  Manvr, for example,
should use a start of "1999:251"  (1999:230 + 21 days lookback).
Probably not needed for events that rely on only one event MSID.

Now populate the rest of the table events::

    python -m kadi.update_events --start=2001:001 --model=${ModelClassName}

Test
----
::

    ipython
    >>> from kadi import events
    >>> evts = events.<model_class_name>.filter()
    >>> evts

Update docs
-----------
::

    cd docs
    ipython
    >>> import kadi.events
    >>> kadi.events.__file__

    >>> run make_field_tables

    >>> update_models_docstrings(outfile='models_test.py')
    >>> exit()

    diff models_test.py ../kadi/events/models.py
    mv models_test.py ../kadi/events/models.py

    ipython
    >>> run make_field_tables

    # Update event_descriptions.rst in place
    >>> make_event_descriptions_section('event_descriptions.rst')
    >>> make_events_tables()

Copy the three tables there to the appropriate sections in index.rst
and ``kadi/events/__init__.py``.

.. Note:
   This makes tables that have one vertical space separating columns
   while the baseline files have two.  Leave at two.  Need to fix code
   or just do this manually.

Backward compatibility
----------------------

Use the current flight kadi code and the test database and check that kadi tests pass.
This should still be from the ``docs`` directory with ``KADI`` set to the local dir.
::

    ipython
    >>> import kadi
    >>> from kadi import paths
    >>> kadi.__version__  # Flight
    >>> paths.EVENTS_DB_PATH()  # Local
    >>> kadi.test('-v', '-k', 'test_events')

Prepare for release
-------------------

The new local database ``events3.db3`` should be generated within two weeks of
the expected release (the minimum model lookback is 21 days).

- Copy local database to ``/proj/sot/ska/data/kadi/rc/events3.db3`` as
  ``aca`` on HEAD.
- Add a note put into the release PR that this file needs to be
  moved to ``/proj/sot/ska/data/kadi/events3.db3`` post-install of the new
  release.


Reprocess kadi events over a specified interval
===============================================

If the kadi events.db3 database gets corrupted (e.g. problems related to the
2017:020 clock correlation issue), then simply reprocess as follows::

  cd ~/git/kadi
  # Possibly check out current flight release if needed, else master

  cp $SKA/data/kadi/events.db3 ./

  # Update from 2017:017 (for example) to present
  python -m kadi.update_events --start=2017:017 --delete-from-start

  cp events.db3 /proj/sot/ska/data/kadi/

Rebuild Events from scratch
###########################
::

    cd ~/git/kadi
    export KADI=$PWD
    rm -f events3.db3
    rm -rf kadi/events/migrations
    ./manage.py makemigrations events
    ./manage.py migrate

    # First line is just to see that every model works.  One can just drop the
    # --stop=2000:001 if you are sure it will work.
    # Note: use kadi_update_events for the installed version.
    python -m kadi.scripts.update_events --start=1999:001 --stop=2000:001
    python -m kadi.scripts.update_events --start=2000:001

Remove bad momentum dump (bad telem just before safe mode)
##########################################################
Search slack 'momentum "dump" on DOY 2020:145 that is an artifact'::

    cd $SKA/data/kadi  # As needed
    sqlite3 events3.db3
    sqlite> delete from events_dump where start='2020:145:14:17:22.641';

Re-build single events table
############################
::

    cd ~/git/kadi
    export KADI=$PWD
    cp /proj/sot/ska/data/kadi/events.db3 ./
    python -m kadi.scripts.update_events --start=1999:001 --model=CAP --delete-from-start
