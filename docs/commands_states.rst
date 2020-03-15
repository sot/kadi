.. |get_cmds| replace:: :func:`~kadi.commands.commands.get_cmds`
.. |get_continuity| replace:: :func:`~kadi.commands.states.get_continuity`
.. |get_states| replace:: :func:`~kadi.commands.states.get_states`
.. |CommandTable| replace:: :class:`~kadi.commands.commands.CommandTable`

Commands and states
===================

**Commands**

The `Commands archive`_ is a table of every load command that has been run, or is currently
approved to be run, on the spacecraft since 2002.  This archive accounts for load stoppages,
replans, and certain non-load commands like ACIS CTI runs or Normal Sun Mode transitions.

**States and continuity**

Coupled with the commands archive is functionality determine `Chandra states and continuity`_:

- **State** values of certain parameters of interest (obsid, SIM-Z position, commanded
  attitude, ACIS power configuration, etc) over an **interval of time** during which all
  parameters have been unaffected by commanding and are unchanged.  This provides a
  compact way to represent the impact of commanding on the spacecraft state over time and
  is used in `xija thermal model <http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/xija/>`_
  predictions.

- **Continuity** values of certain parameters of interest at a **particular time**.
  The continuity represents the state values at a moment in time and also includes the
  date of the last command which affected the state.

The `State keys`_ section lists all state parameters which are implemented in the
installed code.  Note that a key design feature is that is it straightforward for users
to implement their own states, often with just a few lines of code.  See the `User-defined states`_
section for details.

Commands archive
----------------

The basic way to select commands is with the |get_cmds| method.  For example you can find
load commands from early in 2013 with::

  >>> from kadi import commands
  >>> cmds = commands.get_cmds('2013:001:00:00:00', '2013:001:00:56:10')
  >>> print(cmds)
           date            type      tlmsid   scs step timeline_id params
  --------------------- ---------- ---------- --- ---- ----------- ------
  2013:001:00:37:37.653   ORBPOINT       None   0    0   426098988    N/A
  2013:001:00:53:07.181 COMMAND_SW   AOACRSTD 129 1524   426098990    N/A
  2013:001:00:54:07.181 COMMAND_SW   AOFUNCDS 129 1526   426098990    N/A
  2013:001:00:55:07.181 COMMAND_SW   AOFUNCDS 129 1528   426098990    N/A
  2013:001:00:56:07.181 COMMAND_SW   AONMMODE 129 1530   426098990    N/A
  2013:001:00:56:07.181    ACISPKT AA00000000 132 1620   426098991    N/A
  2013:001:00:56:07.181   SIMTRANS       None 132 1623   426098991    N/A
  2013:001:00:56:07.438 COMMAND_SW   AONM2NPE 129 1532   426098990    N/A

In the |get_cmds| method, commands are selected with ``start <= date < stop``, where each
of these are evaluated as a date string with millisec precision.  In order to get commands
at exactly a certain date you need to select with the ``date`` argument::

  >>> print(commands.get_cmds(date='2013:001:00:56:07.181'))
           date            type      tlmsid   scs step timeline_id params
  --------------------- ---------- ---------- --- ---- ----------- ------
  2013:001:00:56:07.181 COMMAND_SW   AONMMODE 129 1530   426098990    N/A
  2013:001:00:56:07.181    ACISPKT AA00000000 132 1620   426098991    N/A
  2013:001:00:56:07.181   SIMTRANS       None 132 1623   426098991    N/A

The output ``cmds`` is based on the astropy `Table
<http://docs.astropy.org/en/stable/table/index.html>`_ object with many powerful and handy
features built in.  For instance you could sort by ``type``, ``tlmsid`` and ``date``::

  >>> cmds_type = cmds.copy()
  >>> cmds_type.sort(['type', 'tlmsid', 'date'])
  >>> print(cmds_type)
           date            type      tlmsid   scs step timeline_id params
  --------------------- ---------- ---------- --- ---- ----------- ------
  2013:001:00:56:07.181    ACISPKT AA00000000 132 1620   426098991    N/A
  2013:001:00:53:07.181 COMMAND_SW   AOACRSTD 129 1524   426098990    N/A
  2013:001:00:54:07.181 COMMAND_SW   AOFUNCDS 129 1526   426098990    N/A
  2013:001:00:55:07.181 COMMAND_SW   AOFUNCDS 129 1528   426098990    N/A
  2013:001:00:56:07.438 COMMAND_SW   AONM2NPE 129 1532   426098990    N/A
  2013:001:00:56:07.181 COMMAND_SW   AONMMODE 129 1530   426098990    N/A
  2013:001:00:37:37.653   ORBPOINT       None   0    0   426098988    N/A
  2013:001:00:56:07.181   SIMTRANS       None 132 1623   426098991    N/A

You can print a single command and get all the information about it::

  >>> print(cmds[5])
  2013:001:00:56:07.181 ACISPKT tlmsid=AA00000000 scs=132 step=1620 timeline_id=426098991 cmds=3 packet(40)=D80000300030603001300 words=3

This command has a number of attributes like ``date`` or ``tlmsid`` (shown in the original table) as well as command *parameters*: ``cmds``, ``packet(40)``, and ``words``.  You can access any of the attributes or parameters like a dictionary::

  >>> print(cmds[5]['packet(40)'])
  D80000300030603001300

You probably noticed the first time we printed ``cmds`` that the command parameters
``params`` were all listed as ``N/A`` (Not Available).  What happens if we print the
table again:

  >>> print(cmds)
           date            type      tlmsid   scs step timeline_id                      params
  --------------------- ---------- ---------- --- ---- ----------- -----------------------------------------------
  2013:001:00:37:37.653   ORBPOINT       None   0    0   426098988                                             N/A
  2013:001:00:53:07.181 COMMAND_SW   AOACRSTD 129 1524   426098990                                             N/A
  2013:001:00:54:07.181 COMMAND_SW   AOFUNCDS 129 1526   426098990                                             N/A
  2013:001:00:55:07.181 COMMAND_SW   AOFUNCDS 129 1528   426098990                                             N/A
  2013:001:00:56:07.181 COMMAND_SW   AONMMODE 129 1530   426098990                                             N/A
  2013:001:00:56:07.181    ACISPKT AA00000000 132 1620   426098991 cmds=3 packet(40)=D80000300030603001300 words=3
  2013:001:00:56:07.181   SIMTRANS       None 132 1623   426098991                                             N/A
  2013:001:00:56:07.438 COMMAND_SW   AONM2NPE 129 1532   426098990                                             N/A

So what happened?  The answer is that for performance reasons the |CommandTable| class is
lazy about loading the command parameters, and only does so when you directly request the
parameter value (as we did with ``packet(40)``).  If you want to just fetch them all
at once you can do so with the ``fetch_params()`` method::

  >>> cmds.fetch_params()
  >>> print(cmds)
           date            type      tlmsid   scs step timeline_id                      params
  --------------------- ---------- ---------- --- ---- ----------- -----------------------------------------------
  2013:001:00:37:37.653   ORBPOINT       None   0    0   426098988                              event_type=EQF013M
  2013:001:00:53:07.181 COMMAND_SW   AOACRSTD 129 1524   426098990                       hex=8032000 msid=AOACRSTD
  2013:001:00:54:07.181 COMMAND_SW   AOFUNCDS 129 1526   426098990           aopcadsd=21 hex=8030215 msid=AOFUNCDS
  2013:001:00:55:07.181 COMMAND_SW   AOFUNCDS 129 1528   426098990           aopcadsd=32 hex=8030220 msid=AOFUNCDS
  2013:001:00:56:07.181 COMMAND_SW   AONMMODE 129 1530   426098990                       hex=8030402 msid=AONMMODE
  2013:001:00:56:07.181    ACISPKT AA00000000 132 1620   426098991 cmds=3 packet(40)=D80000300030603001300 words=3
  2013:001:00:56:07.181   SIMTRANS       None 132 1623   426098991                                      pos=-99616
  2013:001:00:56:07.438 COMMAND_SW   AONM2NPE 129 1532   426098990                       hex=8030601 msid=AONM2NPE

Finally, note that you can request the value of an attribute or parameter for the entire
command table.  Note that command rows without that parameter will have a ``None`` object::

  >>> print(cmds['msid'])
    msid
  --------
      None
  AOACRSTD
  AOFUNCDS
  AOFUNCDS
  AONMMODE
      None
      None
  AONM2NPE

Notes and caveats
^^^^^^^^^^^^^^^^^^

* The exact set of load commands relies on the `Chandra commanded states database
  <http://cxc.harvard.edu/mta/ASPECT/tool_doc/cmd_states>`_ to determine which command
  loads ran on-board and for what duration.  This information comes from a combination of
  the iFOT load segments database and SOT update procedures for load interrupts.  It has
  been used operationally since 2009 and has frequent validation checking in the course of
  thermal load review.  Nevertheless there are likely a few missing commands here and
  there, particularly associated with load stoppages and replans.

* The kadi commands archive includes all commands for approved loads.  Once loads have
  been ingested into the database and iFOT has been updated accordingly, then the kadi
  commands will reflect this update (within an hour).

* Conversely if there is a load interrupt (SCS-107 or anomaly) then this will be reflected
  in the commands archive within an hour after an on-call person runs a script to update
  the `Chandra commanded states database
  <http://cxc.harvard.edu/mta/ASPECT/tool_doc/cmd_states>`_.

* Each load command has an identifier that can be used to retrieve the exact set of mission
  planning products in which the command was generated.  This is valid even in the case
  of a re-open replan in which a command load effectively has two source directories.

* The archive includes a select set of non-load commands which result from either
  autonomous on-board commanding (e.g. SCS-107) or real-time ground commanding
  (e.g. anomaly recovery).  This list is not comprehensive but includes those
  commands which typically affect mission planning continuity and thermal modeling.

* The parameters for the ACA star catalog command ``AOSTRCAT`` are not included since this
  alone would dramatically increase the database file size.  However, the commands are
  included.

* The command archive is stored in a highly performant HDF5 file backed by a
  dictionary-based index file of unique command parameters.  As of 2018-Jan, the commands
  archive is stored in two files with a total size about 52 Mb.

.. note:: Would a command-line interface be useful?

Chandra states and continuity
------------------------------

To get started, import the ``kadi.commands.states`` module::

  >>> from kadi.commands import states

The fundamental idea of the ``states`` module is that one has a state quantity
such as ``obsid`` or ``si_mode`` which is impacted by commands that Chandra
runs.  By stepping through all commands and maintaining a state vector during
that process, one assembles the state history relevant to those commands.
The identifer for each of these quantities is referred to as a ``state_key``

States
^^^^^^

A commanded state is an interval of time over which certain parameters of interest (obsid,
SIM-Z position, commanded attitude, ACIS power configuration, etc) are unchanged.

The |get_states| function is the workhorse for dynamic commanded states.  This
function is fairly flexible and is roughly equivalent to the combination of the legacy
``Chandra.cmd_states`` functions :func:`~Chandra.cmd_states.cmd_states.get_states`
and  :func:`~Chandra.cmd_states.get_cmd_states.fetch_states`.

States over date range
""""""""""""""""""""""

To get the commanded states over a date range you can do the following, which internally
does a call to |get_cmds| in order to get commands over the ``start`` / ``stop`` date
range::

  >>> states.get_states('2017:001:21:00:00', '2017:002:11:29:00',
  ...                   state_keys=['obsid', 'simpos', 'clocking'])
  <Table length=9>
        datestart              datestop       obsid simpos clocking    trans_keys
          str21                 str21         int64 int64   int64        object
  --------------------- --------------------- ----- ------ -------- ---------------
  2017:001:21:00:00.000 2017:001:21:02:06.467 18140  75624        1
  2017:001:21:02:06.467 2017:001:21:05:06.467 18140  75624        0        clocking
  2017:001:21:05:06.467 2017:001:21:05:10.467 19973  75624        0  clocking,obsid
  2017:001:21:05:10.467 2017:001:21:05:14.467 19973  75624        0        clocking
  2017:001:21:05:14.467 2017:001:21:05:38.467 19973  75624        0        clocking
  2017:001:21:05:38.467 2017:001:21:06:45.467 19973  75624        0        clocking
  2017:001:21:06:45.467 2017:002:11:23:43.185 19973  75624        1        clocking
  2017:002:11:23:43.185 2017:002:11:26:43.185 19973 -99616        0 clocking,simpos
  2017:002:11:26:43.185 2017:002:11:29:00.000 50432 -99616        0           obsid

Each state has a start and a stop date, the values for the requested state keys, and a
column called ``trans_keys`` that specifies which keys had their values updated to *start*
this state.

The first thing to note is that ``datestop`` for a state is always the same as the
``datestart`` for the following state.  There is no gap, and strictly speaking the state
values apply for the date range ``datestart <= date < datestop``.  This is the same as for
getting commands.  Next note that the first ``datestart`` and final ``datestop`` match
exactly the input ``start`` and ``stop`` for the function call.  This reflects that we
only "know" the states over the time range for which commands were requested.

The astute reader will notice that the 3rd through 6th row says ``clocking`` was
updated, but looking at values they are all ``0``.  What's going on?  The answer is that,
by default, |get_states| breaks the state if the value was *commanded*, regardless of
whether the value actually changed.  So let's dig in to the commands at exactly the state
transition time of the 3rd row::

  >>> print(commands.get_cmds(date='2017:001:21:05:06.467'))
           date           type     tlmsid   scs step timeline_id params
  --------------------- -------- ---------- --- ---- ----------- ------
  2017:001:21:05:06.467 MP_OBSID   COAOSQID 131  400   426102266    N/A
  2017:001:21:05:06.467  ACISPKT AA00000000 131  403   426102266    N/A

So there was an ACIS stop science, which sets clocking to ``0`` even though it
was already ``0`` (from the previous stop science 3 minutes earlier).  If you are
getting states for thermal model computation then you don't care about these identical
states.  In this case specify ``merge_identical=True`` in the function call::

  >>> sts = states.get_states('2017:001:21:00:00', '2017:002:11:29:00',
  ...                         state_keys=['obsid', 'simpos', 'clocking'],
  ...                         merge_identical=True)
  >>> sts
  <Table length=6>
        datestart              datestop       obsid simpos clocking    trans_keys
          str21                 str21         int64 int64   int64        object
  --------------------- --------------------- ----- ------ -------- ---------------
  2017:001:21:00:00.000 2017:001:21:02:06.467 18140  75624        1
  2017:001:21:02:06.467 2017:001:21:05:06.467 18140  75624        0        clocking
  2017:001:21:05:06.467 2017:001:21:06:45.467 19973  75624        0           obsid
  2017:001:21:06:45.467 2017:002:11:23:43.185 19973  75624        1        clocking
  2017:002:11:23:43.185 2017:002:11:26:43.185 19973 -99616        0 clocking,simpos
  2017:002:11:26:43.185 2017:002:11:29:00.000 50432 -99616        0           obsid

As a side note, although the ``trans_keys`` column looks like a string, that is
a bit of trickery that happens when you print the states table.  In fact each row
entry is a Python ``set()`` object.  In order to see when ``obsid`` changed in the
above query you could do::

  >>> ['obsid' in row['trans_keys'] for row in sts]
  [False, False, True, False, False, True]

Command line interface
""""""""""""""""""""""

One can do the same thing as above from the command-line using the ``get_chandra_states``
command.  This outputs the table (sans trans_keys) in a space-delimited format
to the console or a specified file.
::

  $ get_chandra_states --start 2017:001:21:00:00 --stop 2017:002:11:29:00 \
                       --state-keys=obsid,simpos,clocking \
                       --merge-identical

               datestart               datestop  obsid  simpos  clocking
   2017:001:21:00:00.000  2017:001:21:02:06.467  18140   75624         1
   2017:001:21:02:06.467  2017:001:21:05:06.467  18140   75624         0
   2017:001:21:05:06.467  2017:001:21:06:45.467  19973   75624         0
   2017:001:21:06:45.467  2017:002:11:23:43.185  19973   75624         1
   2017:002:11:23:43.185  2017:002:11:26:43.185  19973  -99616         0
   2017:002:11:26:43.185  2017:002:11:29:00.000  50432  -99616         0

The available options are::

  $ get_chandra_states --help

  usage: get_chandra_states [-h] [--start START] [--stop STOP]
                            [--state-keys STATE_KEYS] [--merge-identical]
                            [--outfile OUTFILE]

  Ouput the Chandra commanded states over a date range as a space-delimited
  ASCII table.

  optional arguments:
    -h, --help            show this help message and exit
    --start START         Start date (default=Now-10 days)
    --stop STOP           Stop date (default=None)
    --state-keys STATE_KEYS
                          Comma-separated list of state keys
    --merge-identical     Merge adjacent states that have identical values
                          (default=False)
    --outfile OUTFILE     Output file (default=stdout)

Using the command line interface and a single state key, or a related set that change
due to a single command, one can replicate the information in a backstop history
file.  For instance here ::

  $ tail <...>/JAN0818/ofls/History/ATTITUDE.txt

  2018006.072916206 | -5.27899874e-01 -6.92042461e-01 -4.90427812e-01  4.33533892e-02
  2018006.103533882 |  4.51367966e-01  6.45077701e-01  6.14710906e-01  4.76678196e-02
  2018006.130755248 | -4.28324009e-01 -4.40000915e-01  3.57368959e-01  7.03722364e-01
  2018006.214420159 | -3.23403971e-01 -6.11564724e-01 -7.15954877e-01  9.38460120e-02
  2018007.024414705 | -4.16664564e-01 -6.83613678e-01 -5.86236582e-01  1.24055031e-01
  2018007.164807705 | -5.04030078e-01 -7.09485195e-01 -4.78304550e-01  1.17512532e-01

  $ get_chandra_states --start 2018:006:07:29:16.206 --stop 2018:007:16:50:00 \
                       --state-keys=targ_q1,targ_q2,targ_q3,targ_q4

               datestart               datestop       targ_q1       targ_q2       targ_q3       targ_q4
   2018:006:07:29:16.206  2018:006:10:35:33.882  -0.527899874  -0.692042461  -0.490427812  0.0433533892
   2018:006:10:35:33.882  2018:006:13:07:55.248   0.451367966   0.645077701   0.614710906  0.0476678196
   2018:006:13:07:55.248  2018:006:21:44:20.159  -0.428324009  -0.440000915   0.357368959   0.703722364
   2018:006:21:44:20.159  2018:007:02:44:14.705  -0.323403971  -0.611564724  -0.715954877   0.093846012
   2018:007:02:44:14.705  2018:007:16:48:07.705  -0.416664564  -0.683613678  -0.586236582   0.124055031
   2018:007:16:48:07.705  2018:007:16:50:00.000  -0.504030078  -0.709485195   -0.47830455   0.117512532

To see more examples of this look at the backstop history section of the testing file
`kadi/commands/tests/test_states.py
<https://github.com/sot/kadi/blob/6cc8d7a241/kadi/commands/tests/test_states.py#L402>`_.
All of the supported state keys that reproduce backstop history files are tested here.

States from commands
""""""""""""""""""""

Instead of relying on |get_states| to get the commands and continuity, you can do things
manually.  For example::

  >>> start, stop = ('2017:001:21:00:00', '2017:002:11:29:00')
  >>> state_keys=['obsid', 'simpos', 'clocking']
  >>> cmds = commands.get_cmds(start, stop)
  >>> continuity = states.get_continuity(start, state_keys)
  >>> states.get_states(cmds=cmds, continuity=continuity,
  ...                   state_keys=state_keys,
  ...                   merge_identical=True)
  <Table length=5>
        datestart              datestop       obsid simpos clocking    trans_keys
          str21                 str21         int64 int64   int64        object
  --------------------- --------------------- ----- ------ -------- ---------------
  2017:001:21:02:06.467 2017:001:21:05:06.467 18140  75624        0        clocking
  2017:001:21:05:06.467 2017:001:21:06:45.467 19973  75624        0           obsid
  2017:001:21:06:45.467 2017:002:11:23:43.185 19973  75624        1        clocking
  2017:002:11:23:43.185 2017:002:11:26:43.185 19973 -99616        0 clocking,simpos
  2017:002:11:26:43.185 2017:002:11:26:43.185 50432 -99616        0           obsid

In the call to |get_states|, if you omit the ``continuity`` argument it will be determined
internally using the first command date.

This manual process is normally what would be done in a load review code where one needs
to consider up to four different elements:

- Continuity value from some moment, for instance the time of last available telemetry
  for thermal model propagation.
- Commands from the continuity time until the start of loads.
- Non-load commands (e.g. a possible CTI run)
- Load commands

In this case the calling code is responsible for logic to assemble a single commands table
for the ``cmds`` argument as a :class:`~kadi.commands.commands.CommandTable` object.

.. note:: The plan is to provide convenience methods and documentation to make this
   process more straightforward.  E.g.::

     # Get commands for new loads
     bs_cmds = parse_cm.read_backstop(backstop_file)
     load_start = bs_cmds[0]['date']

     # Get last telem from Ska archive.  NOTE: we can and should allow for use
     # of MAUDE here to reduce propagation!
     last_tlm_date = fetch.get_time_range('1dpamzt', format='date')[1]

     # Get approved commands from available telemetry through start of new loads
     cmds = commands.get_cmds(last_tlm_date, load_start)

     # Get pseudo-node values by running thermal model between
     # last_tlm_date - 3 days to last_tlm_date, using estimate or
     # fixed value of pseudo-node.

     # Add backstop commands.  The ``add_commands`` method will sort, but up to
     # user to make sure there is no overlap.
     cmds.add_commands(bs_cmds)  # not yet implemented

     # Optionally insert any non-load commands, e.g. for a CTI that may or may not happen
     non_load_cmds = ...
     cmds.add_commands(non_load_cmds)

     sts = states.get_states(cmds=cmds, state_keys=[...])

Continuity
^^^^^^^^^^

To get the continuity state for a desired set of state keys at a certain time, use
|get_continuity|.  Before doing this, recall that in IPython one can always get
help on a function, class, or method with ``<something>?`` or ``help(<something>)``.
So here is how to get help on the |get_continuity|::

  >>> states.get_continuity?
  Signature: states.get_continuity(date=None, state_keys=None, lookbacks=(7, 30, 180, 1000))
  Docstring:
  Get the state and transition dates at ``date`` for ``state_keys``.

  This function finds the state at a particular date by fetching commands
  prior to that date and determine the states.  It returns dictionary
  ``continuity`` provides the state values. Included in this dict is a special
  key ``__dates__`` which provides the corresponding date at which the
  state-changing command occurred.

  Since some state keys like ``pitch`` change often (many times per day) while
  others like ``letg`` may not change for weeks, this function does dynamic
  lookbacks from ``date`` to find transitions for each key.  By default it
  will try looking back 7 days, then 30 days, then 180 days, and finally 1000
  days.  This lookback sequence can be controlled with the ``lookbacks``
  argument.

  If ``state_keys`` is ``None`` then the default keys ``states.DEFAULT_STATE_KEYS``
  is used.  This corresponds to the "classic" Chandra commanded states (obsid,
  ACIS, PCAD, and mechanisms).

  :param date: date (DateTime compatible, default=NOW)
  :param state_keys: list of state keys or str (one state key) or None
  :param lookbacks: list of lookback times in days (default=[7, 30, 180, 1000])

  :returns: dict of state values

So let's get the state of ``obsid`` and ``si_mode`` at ``2017:300:00:00:00``::

  >>> continuity = states.get_continuity('2017:300:00:00:00', ['obsid', 'si_mode'])
  >>> continuity
  {'__dates__': {'obsid': '2017:299:21:50:34.193',
                 'si_mode': '2017:299:22:02:41.439'},
   'obsid': 19385,
   'si_mode': 'TE_00A02'}

The return value is a ``dict`` which has key/value pairs for each of the
desired state keys.  It also has a ``__dates__`` item which has the
corresponding date when state key changed value because of a command.
To prove this, let's look at the commands exactly at the state transition time::

  >>> cmds = commands.get_cmds(date=continuity['__dates__']['obsid'])
  >>> cmds.fetch_params()
  >>> print(cmds)
           date           type    tlmsid  scs step timeline_id      params
  --------------------- -------- -------- --- ---- ----------- ---------------
  2017:299:21:50:34.193 MP_OBSID COAOSQID 131  495   426102876 cmds=3 id=19385

If no value is supplied for the ``state_keys`` argument then the default set of
state keys shown below is used::

  >>> states.DEFAULT_STATE_KEYS
  ('ccd_count',
   'clocking',
   'dec',
   'dither',
   'fep_count',
   'hetg',
   'letg',
   'obsid',
   'off_nom_roll',
   'pcad_mode',
   'pitch',
   'power_cmd',
   'q1',
   'q2',
   'q3',
   'q4',
   'ra',
   'roll',
   'si_mode',
   'simfa_pos',
   'simpos',
   'targ_q1',
   'targ_q2',
   'targ_q3',
   'targ_q4',
   'vid_board')

State keys
^^^^^^^^^^

The list below shows available state keys along with a list of the transition
classes which affect the keys.

.. Run kadi.commands.states.print_state_keys_transition_classes_docs() to generate this list.

``acisfp_temp``
  - :class:`~kadi.commands.states.ACISFP_TempTransition`

``aoephem1``, ``aoephem2``, ``aoratio``, ``aoargper``, ``aoeccent``, ``ao1minus``, ``ao1plus``, ``aomotion``, ``aoiterat``, ``aoorbang``, ``aoperige``, ``aoascend``, ``aosini``, ``aoslr``, ``aosqrtmu``
  - :class:`~kadi.commands.states.EphemerisTransition`

``clocking``, ``power_cmd``, ``vid_board``, ``fep_count``, ``si_mode``, ``ccd_count``
  - :class:`~kadi.commands.states.ACISTransition`

``dither``
  - :class:`~kadi.commands.states.DitherDisableTransition`
  - :class:`~kadi.commands.states.DitherEnableTransition`

``dither_phase_pitch``, ``dither_phase_yaw``, ``dither_ampl_pitch``, ``dither_ampl_yaw``, ``dither_period_pitch``, ``dither_period_yaw``
  - :class:`~kadi.commands.states.DitherParamsTransition`

``eclipse``
  - :class:`~kadi.commands.states.EclipsePenumbraEntryTransition`
  - :class:`~kadi.commands.states.EclipsePenumbraExitTransition`
  - :class:`~kadi.commands.states.EclipseUmbraEntryTransition`
  - :class:`~kadi.commands.states.EclipseUmbraExitTransition`

``eclipse_timer``
  - :class:`~kadi.commands.states.EclipseEntryTimerTransition`

``ephem_update``
  - :class:`~kadi.commands.states.EphemerisUpdateTransition`

``letg``, ``hetg``, ``grating``
  - :class:`~kadi.commands.states.HETG_INSR_Transition`
  - :class:`~kadi.commands.states.HETG_RETR_Transition`
  - :class:`~kadi.commands.states.LETG_INSR_Transition`
  - :class:`~kadi.commands.states.LETG_RETR_Transition`

``obsid``
  - :class:`~kadi.commands.states.ObsidTransition`

``orbit_point``
  - :class:`~kadi.commands.states.OrbitPointTransition`

``q1``, ``q2``, ``q3``, ``q4``, ``targ_q1``, ``targ_q2``, ``targ_q3``, ``targ_q4``, ``ra``, ``dec``, ``roll``, ``auto_npnt``, ``pcad_mode``, ``pitch``, ``off_nom_roll``
  - :class:`~kadi.commands.states.AutoNPMDisableTransition`
  - :class:`~kadi.commands.states.AutoNPMEnableTransition`
  - :class:`~kadi.commands.states.ManeuverTransition`
  - :class:`~kadi.commands.states.NMM_Transition`
  - :class:`~kadi.commands.states.NPM_Transition`
  - :class:`~kadi.commands.states.NormalSunTransition`
  - :class:`~kadi.commands.states.SunVectorTransition`
  - :class:`~kadi.commands.states.TargQuatTransition`

``radmon``
  - :class:`~kadi.commands.states.RadmonDisableTransition`
  - :class:`~kadi.commands.states.RadmonEnableTransition`

``scs84``
  - :class:`~kadi.commands.states.SCS84DisableTransition`
  - :class:`~kadi.commands.states.SCS84EnableTransition`

``scs98``
  - :class:`~kadi.commands.states.SCS98DisableTransition`
  - :class:`~kadi.commands.states.SCS98EnableTransition`

``simfa_pos``
  - :class:`~kadi.commands.states.SimFocusTransition`

``simpos``
  - :class:`~kadi.commands.states.SimTscTransition`

``sun_pos_mon``
  - :class:`~kadi.commands.states.SPMDisableTransition`
  - :class:`~kadi.commands.states.SPMEclipseEnableTransition`
  - :class:`~kadi.commands.states.SPMEnableTransition`


Implementation
^^^^^^^^^^^^^^^

Basic design concepts for transition classes:

- All state-key specific information is encapsulated in Transition classes.  These inherit
  from :class:`~kadi.commands.states.BaseTransition` and have attributes to specify:

  - Command that generates state change
  - State key(s) that require this transition

- Simple cases are handled using :class:`~kadi.commands.states.BaseTransition` sub-classes
  that need only define class attributes.

- Transition classes are never instantiated, they contain only class methods.

- Transition classes have two key methods:

  :func:`~kadi.commands.states.BaseTransition.get_state_changing_commands`
    Quickly get a list of applicable commands using (usually) numpy filtering
    instead of looping and if/elif through every command.  This is done in the
    method and allows for getting a year of states in < 10-20 seconds.  This
    requirement drives some other code complexity, in particular transition
    function callbacks, where a transition is specified as a function that gets
    called during state evaluation.

  ``set_transitions()``
    Given a table of applicable commands, generate corresponding state
    transitions as a dict of state key updates.

- Once all transition dicts have been collected into a time-ordered list they are
  evaluated in order to accumulate discrete states.  Transition callback
  functions can dynamically add downstream transitions during this process to
  handle events like a maneuver, where the current state is required to generate
  mid-maneuver attitudes and the NPM transition.

- The signature of a transition function callback is::

    def callback(cls, date, transitions, state, idx):

  It has access to the current state date, the complete list of transitions,
  the current state, the current index into the transitions list, and
  any other keyword args in the transition that were inserted when transitions
  were set.  This function can add downstream transitions or directly update
  the current state.


User-defined states
^^^^^^^^^^^^^^^^^^^^

One of the driving factors in the design of the commanded states module is making it easy
for users to create custom states with minimal effort.  With the available base classes,
and in particular :class:`~kadi.commands.states.FixedTransition` and
:class:`~kadi.commands.states.ParamTransition`, it is often just a few lines of code.

For example, if we were interested in the state of the IU mode select, we look at examples
of the relevant command, which in this case is ``CIMODESL``.

  >>> cmds = commands.get_cmds('2017:360', '2018:001', tlmsid='CIMODESL')
  >>> cmds[0]
  <Cmd 2017:360:14:05:00.000 COMMAND_HW tlmsid=CIMODESL scs=128 step=2 timeline_id=426102971 hex=7C063C0 msid=CIU1024T>

Here we see that the IU mode state value is captured in the ``msid`` parameter.  That tells
us this can be implemented as a :class:`~kadi.commands.states.ParamTransition` sub-class.
This class is documented, but in practice it is probably easiest to look through the available
classes in the code and find an example.  In this case the :class:`~kadi.commands.states.ObsidTransition` class::

  class ObsidTransition(ParamTransition):
      """Obsid update"""
      command_attributes = {'type': 'MP_OBSID'}
      state_keys = ['obsid']
      transition_key = 'obsid'
      cmd_param_key = 'id'

So we just adapt this::

  >>> from kadi.commands.states import ParamTransition
  >>> class IUModeSelectTransition(ParamTransition):
  ...     """IU mode select update"""
  ...     command_attributes = {'tlmsid': 'CIMODESL'}
  ...     state_keys = ['iu_mode_select']
  ...     transition_key = 'iu_mode_select'
  ...     cmd_param_key = 'msid'

Notes:

- Just by running this code, the state is *automatically registered* and becomes part of the system.
- The ``command_attributes`` class attribute selects which commands will cause this state transition.
  This attribute is relevant for most transition classes.
- The ``state_keys`` attribute indicates when this class will be included in processing.  In other
  words, when the user requests states or continuity, they generally provide a ``state_keys`` argument
  specifying which keys are desired.  If the user ``state_keys`` overlaps with the class ``state_keys``,
  then this class will be processed.
- The ``transition_key`` and ``cmd_param_key`` attributes are specific to the
  :class:`~kadi.commands.states.ParamTransition` base class and indicate the *name* of the
  state to update using the *value* of the  specified command parameter, respectively.

.. note:: The ``transition_key`` must be in the ``state_keys`` list, but not vica-versa.
   The subtlety here is that if you have multiple transition classes that affect multiple
   states as a group, all the transition classes need to have the **same** ``state_keys``
   attribute.  An example of this is in the grating transition classes, e.g.::

     class HETG_INSR_Transition(FixedTransition):
         """HETG insertion"""
         command_attributes = {'tlmsid': '4OHETGIN'} # Command (HETG insert)
         state_keys = ['letg', 'hetg', 'grating']    # Collective set of grating states
         transition_key = ['hetg', 'grating']        # States that *this* class sets
         transition_val = ['INSR', 'HETG']           # Corresponding fixed values

So now with our new ``IUModeSelectTransition`` class defined, we can use it!
::

  >>> states.get_continuity('2018:001', state_keys='iu_mode_select')
  {'__dates__': {'iu_mode_select': '2018:001:02:30:00.000'},
   'iu_mode_select': 'CIU1024T'}

  >>> states.get_states('2018:001', '2018:004', state_keys='iu_mode_select')
  <Table length=19>
        datestart              datestop       iu_mode_select   trans_keys
          str21                 str21              str8          object
  --------------------- --------------------- -------------- --------------
  2018:001:12:00:00.000 2018:001:12:45:00.000       CIU1024T
  2018:001:12:45:00.000 2018:001:19:45:00.000       CIU1024X iu_mode_select
  2018:001:19:45:00.000 2018:002:02:00:00.000       CIU1024X iu_mode_select
  2018:002:02:00:00.000 2018:002:11:20:00.000       CIU1024T iu_mode_select
  2018:002:11:20:00.000 2018:002:19:00:00.000       CIU1024X iu_mode_select
  2018:002:19:00:00.000 2018:002:19:12:00.000        CIU512T iu_mode_select
  2018:002:19:12:00.000 2018:002:19:21:50.000       CIMODESL iu_mode_select
  2018:002:19:21:50.000 2018:002:19:55:00.000        CIU512T iu_mode_select
  2018:002:19:55:00.000 2018:002:20:04:50.000       CIMODESL iu_mode_select
  2018:002:20:04:50.000 2018:002:20:38:00.000        CIU512T iu_mode_select
  2018:002:20:38:00.000 2018:002:20:47:50.000       CIMODESL iu_mode_select
  2018:002:20:47:50.000 2018:002:21:21:00.000        CIU512T iu_mode_select
  2018:002:21:21:00.000 2018:002:21:30:50.000       CIMODESL iu_mode_select
  2018:002:21:30:50.000 2018:002:22:04:00.000        CIU512T iu_mode_select
  2018:002:22:04:00.000 2018:002:22:13:50.000       CIMODESL iu_mode_select
  2018:002:22:13:50.000 2018:003:11:10:00.000        CIU512T iu_mode_select
  2018:003:11:10:00.000 2018:003:19:35:00.000       CIU1024X iu_mode_select
  2018:003:19:35:00.000 2018:004:01:00:00.000       CIU1024T iu_mode_select
  2018:004:01:00:00.000 2018:004:12:00:00.000       CIU1024T iu_mode_select

Sometimes the pre-defined base classes are not enough, and in these cases the main
challenge is typically defining the ``set_transitions()`` method and potentially defining
transition callback functions.  There are a number of examples of this in the kadi code
and this should serve as your starting point.  The Ska team will be happy to assist
you if this is not enough.
