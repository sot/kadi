.. _Chandra Command Events: https://docs.google.com/spreadsheets/d/19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ

Commands archive details
===========================

Concept overview
----------------

The key concept underlying the commands archive is that it uses web resources to always
provide the most current set of executed and planned load and non-load commands.
This is true even in rapidly changing circumstances such as anomaly recovery or
a fast TOO. The code provides correct results without need for the user to worry
about syncing any files. Working without a network is possible, see the
`Flight scenario (no network)`_ section for details.

The `Chandra Command Events`_
Google sheet is the foundation of this infrastructure. It provides a centralized
repository which contains information about "command events" that impact the
as-run commanding on Chandra. This document is viewable by anyone with the link
but can be edited only by FOT mission planning, Flight Directors and a small set
of managers. This spreadsheet will be maintained by FOT MP in a timely manner
(typically within one hour during anomalies) following a `defined process
<https://occweb.cfa.harvard.edu/twiki/bin/view/MissionPlanning/CommandEvents>`_.

The other key web resource is the OCCweb `FOT mission planning approved load products
<https://occweb.cfa.harvard.edu/occweb/FOT/mission_planning/PRODUCTS/APPR_LOADS/>`_
directory tree. This is used to automatically find all recent approved loads
and incorporate them into the load commands archive.

Scenarios
---------

A scenario is an specific version of events that you like to evaluate. The
default scenario is the `Chandra Command Events`_ sheet.

Providing for alternate scenarios is a key feature of the commands archive.
An example is checking for thermal propagation for assuming an ACIS CTI using
either 3-chips or 4-chips, or no CTI at all. Such scenarios are considered
"custom" scenarios and can be created and easily manipulated by the user.
Examples include:

- Different CTI options
- Different "cold attitude" options
- What if's, like what if we didn't manage to start the maneuver to 135 pitch in a NSM recovery?

One special scenario is the "flight" scenario, discussed below.

Flight scenario (no network)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the case of production applications that are running on the HEAD or GRETA
networks or where it is not possible or desirable to access the network to
update the local loads, the special ``"flight"`` scenario can be used.

The impact of selecting ``scenario="flight"`` in a commands query is that it
disables access to network resources (Google sheets and OCCweb). This means that
results will depend strictly on the production commands archive files
``${SKA}/data/kadi/cmds2.h5`` and ``${SKA}/data/kadi/cmds2.pkl``. On the HEAD
network these files are brought up to date each 10 minutes by a cron jobs, so
using ``"flight"`` in this case is a reliable way to eliminate dependence on the
kadi external web resources.

Using the ``"flight"`` scenario is also recommended for use on some GRETA
workstations if they cannot access the Chandra Command Events Google sheet.

Custom scenarios
^^^^^^^^^^^^^^^^

Custom events sheet
"""""""""""""""""""
The custom Google events sheet scenario provides a convenient built-in way to evaluate
"what-if" situations, including adding command loads that are in-work but not yet
approved. This is particularly useful to evaluate different operational scenarios or
recovery options.

The custom events sheet has the same format as the flight Chandra Command Events sheet.
The default custom sheet is the `FOTMP Command Events
<https://docs.google.com/spreadsheets/d/11p7_WRfOzuOMwASRGTdv1gjF-Kc-vm6zN59ZcYC5Lzo/edit?gid=0#gid=0>`_
sheet. This is maintained by FOT MP and only the FOT planners should modify this file.
The default custom sheet ID is specified by the ``cmd_events_custom_id`` configuration
parameter.

The custom sheet can be used by itself or (most commonly) as an additive supplement the
flight sheet. Just as the ``"flight"`` scenario has a special meaning, now the
``"flight+custom"`` or ``"custom"`` scenarios are special-case scenario names. Here is a
programmatic example of getting states starting from 7 days before now::

    >>> import kadi.commands.states as kcs

    # Use events from custom sheet and the flight sheet to get states
    >>> states = kcs.get_states("-7d", scenario="flight+custom")

     # Use only custom sheet events (mostly just for testing)
    >>> states = kcs.get_states("-7d", scenario="custom")

For use in existing application codes that do not explicitly set the kadi scenario, you
can alternatively set the ``KADI_SCENARIO`` environment variable accordingly.

Using a non-default custom events sheet
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

To use a non-default custom sheet, do the following:

1. **Create a copy** of the main `FOTMP Command Events <https://docs.google.com/spreadsheets/d/11p7_WRfOzuOMwASRGTdv1gjF-Kc-vm6zN59ZcYC5Lzo/edit?gid=0#gid=0>`_ sheet and add your events.
2. **Get the sheet ID** from the URL (the long alphanumeric string in the URL between
   ``spreadsheets/d/`` and ``/edit``).
3. **Configure access** by either:

   - Modifying the configuration file ``~/.kadi/config/kadi.cfg``::

       cmd_events_custom_id = <your_custom_sheet_id>

   - Setting the custom sheet ID programmatically::

       import kadi.commands as kc
       kc.conf.cmd_events_custom_id = <your_custom_sheet_id>

With this, calls to ``get_cmds`` or ``get_states`` with a scenario of ``"custom"`` or
``"flight+custom"`` will access your non-default custom events sheet.

Local scenario
""""""""""""""
You can create a custom scenario by making an appropriate local directory and making
a CSV file that provides the events.

This example shows the steps to programmatically add an ACIS CTI in the midst
of the 2021:296 NSM recovery using a local scenario directory::


    >>> from kadi import paths
    >>> from kadi.commands import conf, get_cmds

    >>> cmds = get_cmds(start='2022:001')  # Ensure local cmd_events.csv is up to date

    >>> path_flight = paths.CMD_EVENTS_PATH()
    >>> path_flight
    PosixPath('/Users/aldcroft/.kadi/cmd_events.csv')

    >>> path_cti = paths.CMD_EVENTS_PATH(scenario='nsm-cti')
    >>> path_cti
    PosixPath('/Users/aldcroft/.kadi/nsm-cti/cmd_events.csv')
    >>> path_cti.parent.mkdir(exist_ok=True, parents=True)

    >>> from astropy.table import Table
    >>> events_flight = Table.read(path_flight)
    >>> events_flight.colnames
    ['State', 'Date', 'Event', 'Params', 'Author', 'Reviewer', 'Comment']

    >>> cti_event = {'State': 'definitive', 'Date': '2021:297:13:00:00',
    ...              'Event': 'RTS', 'Params': 'RTSLOAD,1_CTI06,NUM_HOURS=12:00:00,SCS_NUM=135',
    ...              'Author': 'Tom Aldcroft', 'Reviewer': 'John Scott', 'Comment': ''}

    >>> events_cti = events_flight.copy()
    >>> events_cti.add_row(cti_event)
    >>> events_cti.write(path_cti, overwrite=True)

    >>> import os
    >>> os.environ['CXOTIME_NOW'] = '2021:299'

    >>> cmds = get_cmds('2021:296:10:35:00', '2021:298:01:58:00', scenario='nsm-cti')
    >>> cmds[cmds['event'] == 'RTS'].pprint_like_backstop()
    2021:297:13:00:00.000 | COMMAND_SW       | OORMPEN    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, msid=OORMPEN, scs=135
    2021:297:13:00:01.000 | ACISPKT          | WSVIDALLDN | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
    2021:297:13:00:02.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, 2s2sthv2=0 , msid=2S2STHV, scs=135
    2021:297:13:00:03.000 | COMMAND_HW       | 2S2HVON    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, msid=2S2HVON, scs=135
    2021:297:13:00:13.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, 2s2sthv2=4 , msid=2S2STHV, scs=135
    2021:297:13:00:23.000 | COMMAND_HW       | 2S2STHV    | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, 2s2sthv2=8 , msid=2S2STHV, scs=135
    2021:297:13:00:24.000 | ACISPKT          | WSPOW0CF3F | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
    2021:297:13:01:27.000 | ACISPKT          | WT007AC024 | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
    2021:297:13:01:31.000 | ACISPKT          | XTZ0000005 | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
    2021:297:13:01:35.000 | ACISPKT          | RS_0000001 | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135
    2021:297:13:01:39.000 | ACISPKT          | RH_0000001 | CMD_EVT  | event=RTS, event_date=2021:297:13:00:00, scs=135

Then from the bash command line::

    $ export KADI_SCENARIO=nsm-cti
    $ dpa_check \
        --outdir=out-cti \
        --oflsdir=DAWG-demo/OCT2521/oflsb \
        --state-builder=sql \
        --run-start=2021:296:18:00:00

.. _configuration-options:

Configuration options
---------------------

The kadi commands configuration options are stored in the file
``~/.kadi/config/kadi.cfg``. The location of this file is fixed.

The available options with the default settings are as follows::

    [commands]
    ## Default lookback for previous approved loads (days).
    default_lookback = 30

    ## Cache backstop downloads in the astropy cache. Should typically be False,
    ## but useful during development to avoid re-downloading backstops.
    cache_loads_in_astropy_cache = False

    ## Clean backstop loads (like APR1421B.pkl.gz) in the loads directory that are
    ## older than the default lookback. Most users will want this to be True, but
    ## for development or if you always want a copy of the loads set to False.
    clean_loads_dir = True

    ## Directory where command loads and command events are stored after
    ## downloading from Google Sheets and OCCweb.
    commands_dir = ~/.kadi

    ## Google Sheet ID for command events (flight scenario).
    cmd_events_flight_id = 19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ

    ## Half-width box size of star ID match for get_starcats() (arcsec).
    star_id_match_halfwidth = 5

    ## Half-width box size of fid ID match for get_starcats() (arcsec).
    fid_id_match_halfwidth = 40

Modify options
^^^^^^^^^^^^^^

To modify a configuration there a few options. First is programmatically within
Python to change a parameter for all subsequent code::

    >>> from kadi.commands import conf, get_cmds
    >>> conf.default_lookback
    30
    >>> conf.default_lookback = 60

You can also temporarily change an option within a context manager::

    >>> with conf.set_temp('include_in_work_command_events', True):
    ...     cmds_in_work = get_cmds('2022:001', '2022:002')  # Use Commands In-work events
    >>> cmds_flight = get_cmds('2022:001', '2022:002')  # Use only Predictive or Definitive

For an even-more permanent solution you can write out the configuration file
to disk and then edit it. Be wary of "temporarily" changing an option and  then
forgetting to revert it later.

    >>> import kadi
    >>> import kadi.events.models  # Due to a strange interaction with Django
    >>> status = kadi.create_config_file()
    INFO: The configuration file has been successfully written to
    ~/.kadi/config/kadi.cfg [astropy.config.configuration]


Environment variables
---------------------

``CXOTIME_NOW``
  For testing and demonstration purposes, this environment variable can be set to make
  the code believe that ``CXOTIME_NOW`` is the current time. See the
  `Mocking the current time`_ section for more details.

``KADI``
  Override the default location of kadi flight data files ``cmds2.h5`` and
  ``cmds2.pkl``.

``KADI_SCENARIO``
  Set the default scenario. This can be used to set the scenario in an
  application that is not aware of kadi scenarios, effectively a back door to
  override the flight commands.

``KADI_CMDS_VERSION``
  Set the kadi commands archive version to use. By default, kadi uses the highest
  version found in the data directory. The version is the integer value in the kadi
  ``cmds<N>.h5`` and ``cmds<N>.pkl`` files. This environment variable can be set to an
  integer value (e.g. ``2`` or ``3``) to force kadi to use a specific version.

Mocking the current time
------------------------
Setting the ``CXOTIME_NOW`` environment variable allows you to pretend that the current
time is a different time. Any calls to ``CxoTime`` or ``DateTime`` that normally return
the current time will now return a time object corresponding to ``CXOTIME_NOW``.

Many Ska functions have a ``stop`` argument that defaults to the current time, so setting
``CXOTIME_NOW`` will change the behavior of these functions.

For kadi commands functions, the situation is a bit more complex. Because kadi commands
are *predictive*, the default ``stop`` used by :func:`~kadi.commands.get_cmds` is the
current time plus one year. If ``CXOTIME_NOW`` is set, the "current time" will be the
value of ``CXOTIME_NOW`` and the default ``stop`` will be one year after that.

Setting ``CXOTIME_NOW`` also impacts the behavior of commands ingest and generation:

- Only Chandra Command Events that are before this time will be included.
- Only weekly approved loads with a load date before this time will be included. The
  load date of a weekly load set is midnight on date of the load name, so JAN2025A
  has a load date of 2025:020:00:00:00.
- Kadi commands dynamically regenerates recent commands (normally within 30 days of
  the current time) from the weekly approved loads and the Chandra Command Events.
  This is done to ensure that the commands are up-to-date with the current state of
  the spacecraft. If ``CXOTIME_NOW`` is set, the commands will be regenerated from
  the weekly approved loads and Chandra Command Events that are 30 days before this
  time.

Data files and resources
------------------------

Flight archive files
^^^^^^^^^^^^^^^^^^^^

The flight archive of commands and associated parameters are stored in the two
files listed below. These files are kept up to date each 10 minutes on the
HEAD server and must synced at least once each 3 weeks to GRETA and other
computers using either ``ska_sync`` or by other means.

``${SKA}/data/kadi/cmds2.h5``
  HDF5 table of commands

``${SKA}/data/kadi/cmds2.pkl``
  Python pickle file containing a dict of command parameters. Since the command
  parameters are often the same this significantly reduces the same of the
  archive data files.

Local archive files
^^^^^^^^^^^^^^^^^^^

The local archive is maintain by using the `Web resources`_ below. These files
are stored in ``~/.kadi`` by default but the location is configurable.

``cmd_events.csv``
  Local copy of the Chandra Command Events Google sheet as a CSV file.

``loads.csv``
  CSV file with information about recent approved loads that have been retrieved
  from OCCweb. This includes the command start and stop times, interrupt times,
  and the RLTT, scheduled stop time.

``loads.dat``
  Same as ``loads.csv`` but in a fixed-width human-readable format.

``loads/``
  Directory containing backstop commands for recent approved loads stored as a
  Python pickle file, e.g. ``MAR0722A.pkl.gz``.

``<scenario>/``
  Directory containing files for a custom scenario. The files are
  ``cmd_events.csv``, ``loads.csv``, ``loads.dat``. Note that the ``loads/``
  directory is not specific to a scenario and so the top-level version is used.

Web resources
^^^^^^^^^^^^^

`Chandra Command Events`_ Google sheet
  Centralized repository which contains information about "command events" that
  impact the as-run commanding on Chandra. This document is viewable by anyone
  with the link but can be edited only by FOT mission planning, Flight Directors
  and a small set of managers. This spreadsheet is maintained by FOT MP in
  a timely manner (typically within one hour during anomalies) following a
  `defined process
  <https://occweb.cfa.harvard.edu/twiki/bin/view/MissionPlanning/CommandEvents>`_.

`FOT mission planning approved load products <https://occweb.cfa.harvard.edu/occweb/FOT/mission_planning/PRODUCTS/APPR_LOADS/>`_
  This is used to automatically find all recent approved loads
  and incorporate them into the load commands archive.

Configuration and other files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These files are in the user home directory ``~/.kadi``. This directory location
is not configurable as they are set by the `astropy configuration sub-package
<https://docs.astropy.org/en/stable/config/index.html>`_.

``~/.kadi/config/kadi.cfg``
  Kadi configuration file.

``~/.kadi/cache``
  Cache download files. This can be removed at any time if needed.
