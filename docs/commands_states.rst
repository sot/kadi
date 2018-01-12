Chandra commands and states
============================

**Commands**

The commands archive is a table of every load command that has been run on the spacecraft
since 2002.  This is stored in a highly performant HDF5 file backed by a dictionary-based
index file of unique command parameters.  As of 2018-Jan, the commands archive is stored
in two files with a total size about 52 Mb.

**States and continuity**

Coupled with the commands archive is functionality related to Chandra states.  A "state"
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


Commands
---------

Usage
^^^^^

As with event queries, the basic way to select commands is with the ``filter()`` method.
For example you find some load commands from early in 2013 with::

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

The output ``cmds`` is an astropy `Table <http://docs.astropy.org/en/stable/table/index.html>`_
object with many powerful and handy features built in.  For instance you can easily select
two of the columns and make a new table::

  >>> print(cmds['type', 'tlmsid'])
     type      tlmsid
  ---------- ----------
    ORBPOINT       None
  COMMAND_SW   AOACRSTD
  COMMAND_SW   AOFUNCDS
  COMMAND_SW   AOFUNCDS
  COMMAND_SW   AONMMODE
     ACISPKT AA00000000
    SIMTRANS       None
  COMMAND_SW   AONM2NPE

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


States and continuity
---------------------

TBD.
