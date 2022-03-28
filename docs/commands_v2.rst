.. _Chandra Command Events: https://docs.google.com/spreadsheets/d/19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ

Commands archive v2 details
===========================

Concept overview
----------------

The key concept underlying version 2 is that it uses web resources to always
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

Differences from v1
-------------------

Apart from the fundamental change in data sources mentioned above, some key
changes from v1 are as follows:

- Commands table includes a ``source`` column that defines the source of the
  command. Most commonly this is a weekly load name, but it can also indicate
  a non-load command event for which further details are provided in the command
  parameters.
- Information about each distinct observation is embedded into the command
  archive as ``LOAD_EVENT`` pseudo-commands. The
  :func:`~kadi.commands.observations.get_observations` provides a fast and
  convenient way to find observations, both past and planned. See the
  :ref:`getting-observations` section for more details.
- Information about each ACA star catalog is stored in the command
  archive. The :func:`~kadi.commands.observations.get_observations` provides a
  convenient way to find ACA star catalogs, both past and planned. See the
  :ref:`getting-star-catalogs` section for more details.
- There are configuration options which can be set programmatically or in a fixed
  configuration file to control behavior of the package. See the
  `Configuration options`_ section for more details.

Scenarios
---------

A scenario is an specific version of events that you like to evaluate. The
default scenario is the `Chandra Command Events`_ sheet.

Providing for alternate scenarios is a key feature of the commands archive v2.
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

Using the ``"flight"`` scenario is also recommended for use on GRETA
workstations since they cannot access the Chandra Command Events Google sheet.

Custom scenario example
^^^^^^^^^^^^^^^^^^^^^^^

This example shows the steps to programmatically add an ACIS CTI in the midst
of the 2021:296 NSM recovery::


    >>> from kadi import paths
    >>> from kadi.commands import conf, get_cmds
    >>> conf.commands_version = '2'

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
    >>> os.environ['KADI_COMMANDS_DEFAULT_STOP'] = '2021:299'

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
    $ export PYTHONPATH=$HOME/git/kadi:$HOME/git/parse_cm  # for Ska3 2022.2
    $ dpa_check \
        --outdir=out-cti \
        --oflsdir=DAWG-demo/OCT2521/oflsb \
        --state-builder=sql \
        --run-start=2021:296:18:00:00

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

    ## Default version of kadi commands ("1" or "2").  Overridden by
    ## KADI_COMMANDS_VERSION environment variable.
    commands_version = 1

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

    >>> with conf.set_temp('commands_version', '2'):
    ...     cmds2 = get_cmds('2022:001', '2022:002')  # Use commands v2
    >>> cmds1 = get_cmds('2022:001', '2022:002')  # Use commands v1

For an even-more permanent solution you can write out the configuration file
to disk and then edit it. This could be a good option if you want to always
use commands version v2 for testing purposes.

    >>> import kadi
    >>> status = kadi.create_config_file()
    INFO: The configuration file has been successfully written to
    ~/.kadi/config/kadi.cfg [astropy.config.configuration]


Environment variables
---------------------

``KADI``
  Override the default location of kadi flight data files ``cmds2.h5`` and
  ``cmds2.pkl``.

``KADI_COMMANDS_VERSION``
  Override the default kadi commands version. In order to use the commands
  archive v2 you should set this to ``2``.

``KADI_COMMANDS_DEFAULT_STOP``
  For testing and demonstration purposes, this environment variable can be set
  to a date which is used as the default stop time for commands. In effect this
  makes the code believe that this is the current time and that there are no
  command loads available after this time.

``KADI_SCENARIO``
  Set the default scenario. This can be used to set the scenario in an
  application that is not aware of kadi scenarios, effectively a back door to
  override the flight commands.

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
