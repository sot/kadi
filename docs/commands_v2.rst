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

The `Chandra Command Events
<https://docs.google.com/spreadsheets/d/19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ/edit#gid=0>`_
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
  `Getting observations` section for more details.
- Information about each ACA star catalog is stored in the command
  archive. The :func:`~kadi.commands.observations.get_observations` provides a
  convenient way to find ACA star catalogs, both past and planned. See the
  `Getting star catalogs`_ section for more details.
- There are configuration options which can be set programmatically or in a fixed
  configuration file to control behavior of the package. See the
  `Configuration options`_ section for more details.

Getting observations
--------------------

The commands archive includes special ``LOAD_EVENT`` commands (like the RLTT) that
contain information about observations in the loads. An "observation" is defined
as a dwell in normal point mode following a maneuver and includes most OR's and
ER's. These are most easily accessed via the
:func:`~kadi.commands.observations.get_observations()` function. For example::

    >>> from kadi.commands import get_observations
    >>> obss = get_observations(obsid=26330)
    >>> obss[0]
    {'obsid': 26330,
    'simpos': 73296,
    'obs_stop': '2022:075:19:10:35.734',
    'manvr_start': '2022:075:17:39:54.696',
    'targ_att': (0.105773397, -0.727314387, -0.579818109, 0.351620152),
    'npnt_enab': True,
    'obs_start': '2022:075:17:57:55.985',
    'prev_att': (0.113370245, -0.848288771, -0.329080853, 0.399072852),
    'starcat_idx': 211630,
    'source': 'MAR1422A'}

Notice that the command always returns a list of observations, even for a query
asking for a specific ObsID. The reason is that cases of multiple observations
with the same ObsID are relatively common, in particular after SCS-107 stops the
observing loads this will happen. The commands archive reflects the commands
that ran on-board, and since ObsID updates are in the observing loads those
commands no longer run after SCS-107.

For example, ObsID 65526 was manually commanded after the HRC B-side anomaly and
persisted for 64 distinct observations in the vehicle loads::

    >>> obss = get_observations(obsid=65526)  # ObsID after HRC B-side anomaly
    >>> len(obss)
    64

Getting all the observations covering years or more is reasonably fast,
typically seconds the first query and then << 1 sec for subsequent queries.

    >>> %time obss = get_observations(start='2020:001', stop='2022:001')
    CPU times: user 28.4 ms, sys: 952 µs, total: 29.4 ms
    Wall time: 28.7 ms

For a large number of observations like this you may find it convenient to turn
this into an astropy ``Table``::

    >>> from astropy.table import Table
    >>> obss = Table(obss)
    >>> obss
    <Table length=4018>
    obsid simpos        obs_stop            manvr_start      ...       obs_start               prev_att [4]         starcat_idx  source
    int64 int64          str21                 str21         ...         str21                   float64               int64      str8
    ----- ------ --------------------- --------------------- ... --------------------- ---------------------------- ----------- --------
    47575  75624 2020:001:06:51:13.985 2020:001:06:15:56.053 ... 2020:001:06:46:33.236   -0.51363412 .. 0.832139148      166529 DEC2319A
    23000  75624 2020:001:19:18:15.914 2020:001:06:51:24.236 ... 2020:001:07:17:16.164  -0.475120775 .. 0.693367999      166530 DEC2319A
    47574  75624 2020:001:19:48:58.024 2020:001:19:18:26.165 ... 2020:001:19:44:17.276  -0.221358674 .. 0.710485779      166531 DEC2319A
    ...    ...                   ...                   ... ...                   ...                          ...         ...      ...
    25803  75624 2021:365:13:38:31.497 2021:365:05:20:58.062 ... 2021:365:05:47:31.748 -0.488490169 .. 0.0261316575      170515 DEC3021A
    26264  75624 2021:365:18:39:19.983 2021:365:13:38:41.748 ... 2021:365:14:08:20.234 -0.0658948803 .. 0.350032226      170516 DEC3021A
    26247  75624 2022:001:05:21:28.604 2021:365:18:39:30.234 ... 2021:365:19:06:05.650  0.553670113 .. 0.0635036584      170517 DEC3021A

Under the hood
^^^^^^^^^^^^^^

The observation information is stored as ``LOAD_EVENT`` commands that can be viewed
directly::

    >>> from kadi.commands import get_cmds
    >>> cmds = get_cmds('2022:001', '2022:002', type='LOAD_EVENT')
    >>> print(cmds)
            date            type    tlmsid scs step      time      source  vcdu params
    --------------------- ---------- ------ --- ---- ------------- -------- ---- ------
    2022:001:05:48:44.808 LOAD_EVENT    OBS   0    0 757403393.992 DEC3021A   -1    N/A
    2022:001:09:42:05.439 LOAD_EVENT    OBS   0    0 757417394.623 DEC3021A   -1    N/A
    2022:001:11:37:20.405 LOAD_EVENT    OBS   0    0 757424309.589 DEC3021A   -1    N/A
    2022:001:15:03:39.654 LOAD_EVENT    OBS   0    0 757436688.838 DEC3021A   -1    N/A
    2022:001:15:29:26.255 LOAD_EVENT    OBS   0    0 757438235.439 DEC3021A   -1    N/A
    2022:001:17:33:53.255 LOAD_EVENT    OBS   0    0 757445702.439 DEC3021A   -1    N/A
    >>> cmds[0]['params']
    {'obsid': 45814,
    'simpos': -99616,
    'obs_stop': '2022:001:09:13:04.557',
    'manvr_start': '2022:001:05:21:38.855',
    'targ_att': (0.530730117, 0.556620885, 0.610704042, 0.188518716),
    'npnt_enab': True,
    'obs_start': '2022:001:05:48:44.808',
    'prev_att': (-0.0743435142, -0.559183412, -0.804323901, 0.186681591),
    'starcat_idx': 170518}

As with :func:`~kadi.commands.commands.get_cmds` in the v2 archive, you can provide a
``scenario`` keyword to :func:`~kadi.commands.observations.get_observations` to
select a custom or ``'flight'`` scenario.

Getting star catalogs
---------------------

The ACA star catalogs associated with observations can be retrieved using the
:func:`~kadi.commands.observations.get_starcats()` function. For example::

    >>> from kadi.commands import get_starcats
    >>> acas = get_starcats(obsid=26330)
    >>> acas[0]
    <ACATable length=11>
    slot  idx      id    type  sz    mag    maxmag   yang     zang    dim   res  halfw
    int64 int64   int64   str3 str3 float64 float64 float64  float64  int64 int64 int64
    ----- ----- --------- ---- ---- ------- ------- -------- -------- ----- ----- -----
        0     1         2  FID  8x8    7.00    8.00  -773.14 -1862.22     1     1    25
        1     2         4  FID  8x8    7.00    8.00  2140.38    46.50     1     1    25
        2     3         5  FID  8x8    7.00    8.00 -1826.24    40.03     1     1    25
        3     4 194257752  BOT  6x6    6.09    7.59 -2389.55 -1716.62    28     1   160
        4     5 264114816  BOT  6x6    8.64   10.19    -2.30 -2430.32    28     1   160
        5     6 194249696  BOT  6x6    8.76   10.27 -2129.92 -2447.14    28     1   160
        6     7 263198168  BOT  6x6    8.81   10.33  -375.16  2416.16    28     1   160
        0     8 263199776  ACQ  6x6   10.13   11.20  1385.78  1949.73     8     1    60
        1     9 264113448  ACQ  6x6   10.27   11.20  -187.29 -1336.05     8     1    60
        2    10 263201064  ACQ  6x6   10.44   11.20   529.26   324.30     8     1    60
        7    11 263196576  ACQ  6x6   10.56   11.20 -1046.29   258.97    16     1   100

.. Note::
   The ``ACATable`` objects that are returned can be plotted but they
   are not fully equivalent to the catalogs that ``proseco`` would return. The
   CCD temperatures are set to -20 C and the ``.acqs`` and ``.guides`` attributes
   are stubbed with empty tables.

Scenarios
---------

A scenario is an specific version of events that you like to evaluate. The
default scenario is the`Chandra Command Events
<https://docs.google.com/spreadsheets/d/19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ/edit#gid=0>`_.

Providing for alternate scenarios is a key feature of the commands archive v2.
An example is checking for thermal propagation for assuming an ACIS CTI using
either 3-chips or 4-chips, or no CTI at all. Such scenarios are considered
"custom" scenarios and can be created and easily manipulated by the user.

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

Custom scenarios
^^^^^^^^^^^^^^^^

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
  ``cmds.pkl``.

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

`Chandra Command Events <https://docs.google.com/spreadsheets/d/19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ/edit#gid=0>`_ Google sheet
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
is not configurable.

``~/.kadi/config/kadi.cfg``
  Kadi configuration file.

``~/.kadi/cache``
  Cache download files. This can be removed at any time if needed.
