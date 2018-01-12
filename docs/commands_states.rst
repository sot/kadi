.. |filter| replace:: :func:`~kadi.commands.commands.filter`
.. |get_state0| replace:: :func:`~kadi.commands.states.get_state0`

Chandra commands and states
============================

**Commands**

The `Commands archive`_ is a table of every load command that has been run on the spacecraft
since 2002.  This is stored in a highly performant HDF5 file backed by a dictionary-based
index file of unique command parameters.  As of 2018-Jan, the commands archive is stored
in two files with a total size about 52 Mb.

**States and continuity**

Coupled with the commands archive is functionality related to `Chandra states and continuity`_.  A "state"
has two meanings in this context:

- The set of values of certain parameters of interest (obsid, SIM-Z position, commanded
  attitude, ACIS power configuration, etc) at a **particular time**.  This is effectively
  the concept of load continuity, where one must know the state of spacecraft parameters
  at the exact time a new load begins.

- The set of values over an **interval of time** during which all parameters of interest
  have been unaffected by commanding and are invariant.  This provides a compact way to
  represent the impact of commanding on the spacecraft state over time and is used in
  `xija thermal model <http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/xija/>`_
  predictions.


Commands archive
----------------

As with event queries, the basic way to select commands is with the
|filter| method.  For example you can find load
commands from early in 2013 with::

  >>> from kadi import commands
  >>> cmds = commands.filter('2013:001:00:00:00', '2013:001:00:56:10')
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

.. note:: In the |filter| method, commands are selected with ``start <= date < stop``,
   where each of these are evaluated as a date string with millisec precision.  In order
   to get commands at exactly a certain time step you need to make ``stop`` be 1 msec
   after ``start``.  See the example in `Chandra states and continuity`_.

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

This command has a number of attributes like ``date`` or `tlmsid`` (shown in the original table) as well as command *parameters*: ``cmds``, ``packet(40)``, and ``words``.  You can access any of the attributes or parameters like a dictionary::

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

So what happened?  The answer is that for performance reasons ``CommandTable`` class is
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


Chandra states and continuity
------------------------------

To get started, import the ``kadi.commands.states`` module::

  >>> from kadi.commands import states

The fundamental idea of the ``states`` module is that one has a state quantity
such as ``obsid`` or ``si_mode`` which is impacted by commands that Chandra
runs.  By stepping through all commands and maintaining a state vector during
that process, one assembles the state history relevant to those commands.
The identifer for each of these quantities is referred to as a ``state_key``

The next important idea is that inherited Python classes are used to encapsulate the
commands which affect particular state keys and the exact details by which a particular
command affects each state change,

Continuity
^^^^^^^^^^

To get the state for a desired set of state keys at a certain time, use
|get_state0|.  This is equivalent to load continuity
at that time.  Before doing this, recall that in IPython one can always get
help on a function, class, or method with ``<something>?`` or ``help(<something>)``.
So here is how to get help on the |get_state0|:

  >>> states.get_state0?
  Signature: states.get_state0(date=None, state_keys=None, lookbacks=(7, 30, 180, 1000))
  Docstring:
  Get the state and transition dates at ``date`` for ``state_keys``.

  This function finds the state at a particular date by fetching commands
  prior to that date and determine the states.  It returns dictionary
  ``state0`` provides the state values. Included in this dict is a special
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

  >>> state0 = states.get_state0('2017:300:00:00:00', ['obsid', 'si_mode'])
  >>> state0
  {'__dates__': {'obsid': '2017:299:21:50:34.193',
                 'si_mode': '2017:299:22:02:41.439'},
   'obsid': 19385,
   'si_mode': 'TE_00A02'}

The return value is a ``dict`` which has key/value pairs for each of the
desired state keys.  It also has a ``__dates__`` item which has the
corresponding date when state key changed value because of a command.
To prove this, let's look at the commands exactly at the state transition time::

  >>> from Chandra.Time import DateTime
  >>> date0 = DateTime(state0['__dates__']['obsid'])
  >>> cmds = commands.filter(date0, date0 + 0.001 / 86400)  # 1 msec later
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
