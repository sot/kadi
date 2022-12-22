.. |get_cmds| replace:: :func:`~kadi.commands.commands.get_cmds`
.. |CommandTable| replace:: :class:`~kadi.commands.commands.CommandTable`

Commands archive v1
-------------------
In order to use the v1 version do the following::

  >>> from kadi import commands
  >>> commands.conf.commands_version = "1"  # must be the string "1" not int 1

An alternative is to set the ``KADI_COMMANDS_VERSION`` environment variable to
``1``. This will globally apply to all subsequent Python sessions that inherit
this environment. For example from a linux/Mac bash command shell you can
enter::

  $ export KADI_COMMANDS_VERSION=1

The basic way to select commands is with the |get_cmds| method.  For example you can find
load commands from early in 2013 with::

  >>> from kadi import commands
  >>> cmds = commands.get_cmds('2013:001:00:00:00', '2013:001:00:56:10')
  >>> print(cmds)
          date            type      tlmsid   scs step      time     timeline_id   vcdu  params
  --------------------- ---------- ---------- --- ---- ------------- ----------- ------- ------
  2013:001:00:37:37.653   ORBPOINT       None   0    0 473387924.837   426098988 5533112    N/A
  2013:001:00:53:07.181 COMMAND_SW   AOACRSTD 129 1524 473388854.365   426098990 5584176    N/A
  2013:001:00:54:07.181 COMMAND_SW   AOFUNCDS 129 1526 473388914.365   426098990 5584410    N/A
  2013:001:00:55:07.181 COMMAND_SW   AOFUNCDS 129 1528 473388974.365   426098990 5584644    N/A
  2013:001:00:56:07.181 COMMAND_SW   AONMMODE 129 1530 473389034.365   426098990 5584878    N/A
  2013:001:00:56:07.181    ACISPKT AA00000000 132 1620 473389034.365   426098991 5584878    N/A
  2013:001:00:56:07.181   SIMTRANS       None 132 1623 473389034.365   426098991 5584878    N/A
  2013:001:00:56:07.438 COMMAND_SW   AONM2NPE 129 1532 473389034.622   426098990 5584879    N/A


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
