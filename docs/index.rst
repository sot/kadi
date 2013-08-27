.. kadi documentation master file, created by
   sphinx-quickstart on Fri May 10 13:30:54 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |CAP| replace:: :class:`~kadi.events.models.CAP`
.. |DsnComm| replace:: :class:`~kadi.events.models.DsnComm`
.. |Dump| replace:: :class:`~kadi.events.models.Dump`
.. |Dwell| replace:: :class:`~kadi.events.models.Dwell`
.. |Eclipse| replace:: :class:`~kadi.events.models.Eclipse`
.. |FaMove| replace:: :class:`~kadi.events.models.FaMove`
.. |MajorEvent| replace:: :class:`~kadi.events.models.MajorEvent`
.. |Manvr| replace:: :class:`~kadi.events.models.Manvr`
.. |ManvrSeq| replace:: :class:`~kadi.events.models.ManvrSeq`
.. |NormalSun| replace:: :class:`~kadi.events.models.NormalSun`
.. |Obsid| replace:: :class:`~kadi.events.models.Obsid`
.. |Orbit| replace:: :class:`~kadi.events.models.Orbit`
.. |OrbitPoint| replace:: :class:`~kadi.events.models.OrbitPoint`
.. |RadZone| replace:: :class:`~kadi.events.models.RadZone`
.. |SafeSun| replace:: :class:`~kadi.events.models.SafeSun`
.. |Scs107| replace:: :class:`~kadi.events.models.Scs107`
.. |TscMove| replace:: :class:`~kadi.events.models.TscMove`

Kadi archive
================================

The Kadi archive consists of the following integrated elements which allow for
easy access and manipulation of events and commands related to the Chandra X-ray Observatory:

**Database of Chandra events**

- Events in telemetry such as maneuvers, NPM dwells, obsids, mech movements, momentum
  dumps, orbit events, etc.
- CAPs, DSN passes, dark cals, SCS107, safe modes, bright star hold, etc
- Chandra major events since launch

**Database of Chandra commands**

- Every load command run on-board since 2002, with a link to source load products

**Python API** for accessing events for analysis and using with the Ska engineering archive

**Python tools** to maintain the Kadi database on the HEAD and Greta networks

**Web site** for browsing events on the icxc site or by a localhost server on GRETA. This
  will use the Django web framework to provide query and administrative capabilities  (*Coming soon!*).

**RESTful web service API** on the icxc site  (*Coming soon!*).

Overview
----------

Chandra events
^^^^^^^^^^^^^^^

As shown in the `Event definitions`_ section, there are a number of different types of
Chandra events that are available within the Kadi archive.  Each type of event is
essentially a database table with a number of data fields, some of them common to all
event types and some of them unique.

Interval events
"""""""""""""""""

Most of the event types represent an **interval of time** with some defining characteristic,
for instance maneuvers (|Manvr|), radiation zones (|RadZone|) or SIM TSC translations
(|TscMove|).  The documentation for each event type contains the event defintion, for
example the |Eclipse| event is the interval where ``AOECLIPS = 'ECL'``.

These "interval" event types all share the following fields:

============  ====================================
Field name     Description
============  ====================================
``start``      Start date (YYYY:DOY:HH:MM:SS.sss)
``stop``       Stop date (YYYY:DOY:HH:MM:SS.sss)
``tstart``     Start time (CXC seconds)
``tstop``      Stop time (CXC seconds)
``dur``        Duration (seconds)
============  ====================================

Many of the event types have additional fields that are specific to that type.  For
example the |FaMove| event also provides the values of the FA step position
``3FAPOS`` before and after the SIM focus assembly translation, and the |DsnComm| event
provides a host of information about each comm pass:

=========== ========== ========================
   Field       Type          Description
=========== ========== ========================
   ifot_id    Integer
     start   Char(21)
      stop   Char(21)
    tstart      Float    Start time (CXC secs)
     tstop      Float     Stop time (CXC secs)
       dur      Float          Duration (secs)
       bot    Char(4)       Beginning of track
       eot    Char(4)             End of track
  activity   Char(30)     Activity description
    config   Char(10)            Configuration
 data_rate    Char(9)                Data rate
      site   Char(12)                 DSN site
       soe    Char(4)   DSN Sequence Of Events
   station    Char(6)              DSN station
=========== ========== ========================

Non-interval events
""""""""""""""""""""""

The event types |MajorEvent|, |ManvrSeq|, and |OrbitPoint| are a bit different in that
they refer to **moment in time** rather than an interval.  For these types the only field
they all have in common is ``date``.  For example the |MajorEvent| type has the following fields:

======== ========== =============================================
 Field      Type                     Description
======== ========== =============================================
    key   Char(24)                     Unique key for this event
  start    Char(8)      Event time to the nearest day (YYYY:DOY)
   date   Char(11)   Event time to the nearest day (YYYY-Mon-DD)
 tstart      Float       Event time to the nearest day (CXC sec)
  descr       Text                             Event description
   note       Text          Note (comments or CAP # or FSW PR #)
 source    Char(3)                     Event source (FDB or FOT)
======== ========== =============================================


Even though many Chandra Major Events (such as a safe mode) are really intervals in time,
the source databases do not encapsulate that information so the derived Kadi database is
likewise ignorant of the actual interval.  For the specific case of safe mode events the
appropriate event type is actually |SafeSun|, which *does* have a meaningful start and
stop time.

Chandra commands
^^^^^^^^^^^^^^^^^^^

The commands database is a table of every load command that has been run on the spacecraft
since 2002.  There are a few caveats:

* This accounting of load commands relies on the Ska commanded states database.  It is
  very accurate and has frequent validation checking in the course of thermal load
  review.  Nevertheless there are likely a few missing commands here and there,
  particularly associated with load stoppages and replans.
* Unlike the commanded states database, non-load commands associated with SCS107 runs and
  manual CTI measurements are *not* included.  That means using the Kadi commands database
  is not recommended for constructing a reliable state timeline (e.g. SIM translation
  position) since it will be incorrect immediately following SCS107 transitions.
* The ACA star catalog command ``AOSTRCAT`` is not included since this alone would
  increase the database file size by a factor of 10.


Getting started
----------------

To start using the Kadi archive you first need to fire up IPython in the Ska environment.
From a shell terminal window do::

  % skatest  # Greta network
  % ska      # HEAD network

  % ipython --pylab

Now enter the following statement.  This imports the interface
modules for the event and commands databases::

  >>> from kadi import events, cmds

.. note::

   The ``%`` is the standard indication for entering a command on the ``csh`` terminal
   window prompt.  The ``>>>`` is the standard indication of entering a command at the
   Python or IPython interactive prompt.  You do not actually type these characters
   (although it is actually allowed to copy the ``>>>`` in IPython).

Get events as a table
^^^^^^^^^^^^^^^^^^^^^^^

The most basic operation is to get some events from the Kadi archive.  So let's find all
the SIM TSC moves since 2012:001::

  >>> tsc_moves = events.tsc_moves.filter('2012:001').table

Let's break this statement down to understand what is happening.  Remember that in Python
everything is an object and you can access object attributes or methods by chaining them
together with the ``.`` (period).

The first bit is ``events.tsc_moves``, which accesses
the object that allows querying the database of TSC move events.  You can easily see
all the available event types for query by doing::

  >>> events.<TAB>  # type "events." and then press <TAB>
  events.caps             events.EventQuery       events.manvr_templates  events.rad_zones
  events.dsn_comms        events.fa_moves         events.models           events.safe_suns
  events.dumps            events.get_dates_vals   events.obsids           events.scs107s
  events.dwells           events.major_events     events.orbits           events.tsc_moves
  events.eclipses         events.manvrs           events.query

From ``events.tsc_moves`` we chain the ``filter('2012:001')`` method to select events that
occurred after ``2012:001``.  The ``filter()`` method is very powerful and can perform
complex filters based on all the available attributes of an event.  In this case to select
an inclusive time range you would supply both the start and stop date in that order,
e.g. ``filter('2012:001', '2013:001')``.

The last bit of the chain is the `.table` attribute, which says to convert the
``filter('2012:001')`` output from a QuerySet object (which is discussed later) into a an
astropy `Table <http://docs.astropy.org/en/stable/table/index.html>`_ that can be printed,
plotted, and used in computations.  Now let's look at what came out by printing the Table.
Before you do this make your terminal window plenty wide, there are a bunch of fields and
you want to see them all::

  >>> print tsc_moves
          start                  stop             tstart        tstop          dur      start_3tscpos stop_3tscpos start_det stop_det max_pwm
  --------------------- --------------------- ------------- ------------- ------------- ------------- ------------ --------- -------- -------
  2012:001:18:21:31.715 2012:001:18:22:04.515 441829357.899 441829390.699 32.8000017405         75624        92903    ACIS-S   ACIS-I      10
  2012:002:02:50:28.517 2012:002:02:54:50.917 441859894.701 441860157.101 262.400013864         92903       -99616    ACIS-I    HRC-S      10
  2012:002:12:20:06.119 2012:002:12:24:28.519 441894072.303 441894334.703 262.400013864        -99616        75623     HRC-S   ACIS-S       4
  2012:003:16:17:49.324 2012:003:16:18:22.124 441994735.508 441994768.308 32.8000017405         75623        92903    ACIS-S   ACIS-I       9
  2012:003:22:19:42.925 2012:003:22:20:15.725 442016449.109 442016481.909 32.8000017405         92903        75624    ACIS-I   ACIS-S      10
                    ...                   ...           ...           ...           ...           ...          ...       ...      ...     ...
  2013:169:06:34:32.856 2013:169:06:35:05.656  487924540.04  487924572.84 32.8000018597         92903        75624    ACIS-I   ACIS-S      10
  2013:169:12:56:07.258 2013:169:12:56:40.058 487947434.442 487947467.242 32.8000018001         75624        92903    ACIS-S   ACIS-I       5
  2013:170:06:44:51.261 2013:170:06:49:46.461 488011558.445 488011853.645 295.200016022         89824       -99616    ACIS-I    HRC-S       9
  2013:170:18:38:48.063 2013:170:18:43:43.263 488054395.247 488054690.447 295.200016022        -99616        92903     HRC-S   ACIS-I       4

This shows the fields in the TSC moves event type.  Many of the moves have been snipped in
order to make the printout it fit onto one screen.  Now plot the maximum pulse width
modulation for each SIM translation as a function of time::

  >>> from Ska.Matplotlib import plot_cxctime
  >>> plot_cxctime(tsc_moves['tstart'], tsc_moves['max_pwm'], '.')
  >>> grid()

.. image:: tsc_moves.png

.. Use figure(figsize=(5, 3.5))

Events and obsids
"""""""""""""""""

With ``kadi`` it's easy to find the obsid for a particular event using the
:func:`~kadi.events.models.BaseModel.get_obsid()` method of an event object that is
returned from a filter query.  The following example asks for all maneuvers within the
range 2013:001:12:00:00 to 2013:002:12:00:00.  It then prints the obsid for the first
matching maneuver, then prints the obsid for all matching maneuvers.  Note that in this
case you do *not* use the final ``.table`` attribute as shown in the previous examples.
::

  >>> manvrs = events.manvrs.filter('2013:001:12:00:00', '2013:002:12:00:00')
  >>> print manvrs[0].get_obsid()
  15046
  >>> for manvr in manvrs:
  ...    print manvr.start, 'Obsid :', manvr.get_obsid()
  2013:001:16:38:45.535 Obsid : 15046
  2013:001:21:16:45.361 Obsid : 15213

The converse of finding events that match an obsid is possible as well.  In this
case there may be zero, one or many matching events.  In the following example
we start by looking for all manuever events between 2013:100 and 2013:200 that have
exactly two Kalman dwells::

  >>> manvrs = events.manvrs.filter('2013:100', '2013:200', n_dwell__exact=2)
  >>> print manvrs
  <Manvr: start=2013:104:21:38:11.867 dur=1442 n_dwell=2 template=two_acq>
  <Manvr: start=2013:142:12:43:01.918 dur=1763 n_dwell=2 template=three_acq>
  <Manvr: start=2013:144:19:19:07.204 dur=1553 n_dwell=2 template=three_acq>

Now we get the obsid for the first matching maneuver and search for Kalman dwells with
that obsid::

  >>> print manvrs[0].get_obsid()
  15304
  >>> print events.dwells.filter(obsid=15304)
  <Dwell: start=2013:104:22:04:05.255 dur=5795>
  <Dwell: start=2013:104:23:41:44.155 dur=4290>

As expected this matches what comes from asking directly for the dwells associated
with the maneuver via the ``dwell_set`` attribute::

  >>> print manvrs[0].dwell_set.all()
  <Dwell: start=2013:104:22:04:05.255 dur=5795>
  <Dwell: start=2013:104:23:41:44.155 dur=4290>

.. Note::

   For most event types the event obsid (as returned by
   :func:`~kadi.events.models.BaseModel.get_obsid()`) is defined as the obsid at the start
   of the event.  The exception is maneuver events, for which it makes most sense to use
   the obsid at the *end* of the maneuver since that is the obsid for the corresponding
   Kalman dwell(s), star catalog and OR / ER.


Use events to filter telemetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another common operation is using event intervals to filter telemetry data.  A simple
example is looking at EPHIN E1300 rates only outside of rad zone.  A more complicated
example is plotting OBC rate data only during "quiet" periods, which means during normal
point mode with Kalman lock, at least 300 seconds away from any maneuvers, SIM motions,
grating motions, and momentum dumps.  With Kadi doing this sort of filtering is easy.

Here we show just the simple example above.  First we plot the E1300 rates over a 30 day
span, including rad zones::

  >>> from Ska.engarchive import fetch
  >>> figure()

  >>> e1300 = fetch.Msid("SCE1300", '2012:020', '2012:050')
  >>> e1300_log = np.log10(e1300.vals.clip(10))
  >>> plot_cxctime(e1300.times, e1300_log, '-b')
  >>> ylabel('log10(E1300)')
  >>> grid()

.. image:: e1300_rates1.png

Now we remove the intervals corresponding to the radiation zone and overplot the filtered
data in red::

  >>> e1300.remove_intervals(events.rad_zones)
  >>> e1300_log = np.log10(e1300.vals.clip(10))
  >>> plot_cxctime(e1300.times, e1300_log, '-r')

.. image:: e1300_rates2.png

The converse operation of *selecting* intervals is also available::

  >>> e1300.remove_intervals(events.rad_zones)

The event intervals within a particular time range can be accessed as a list of 
``(start, stop)`` with the ``intervals()`` method::

  >>> events.rad_zones.intervals('2012:020', '2012:030')
  [('2012:020:14:56:33.713', '2012:020:23:42:31.713'),
   ('2012:023:06:15:15.541', '2012:023:16:15:33.248'),
   ('2012:025:21:22:22.506', '2012:026:08:10:20.506'),
   ('2012:028:13:22:04.533', '2012:028:22:20:02.533')]

Boolean combinations
"""""""""""""""""""""

The real power of intervals is that they can be combined with boolean expressions and
can be padded on either end to make it easy to create complex time filters.  The
following example selects all Kalman mode dwells, but removes times that could
be affected by a SIM TSC move or a momentum dump.

  >>> dwells = events.dwells
  >>> dwells.interval_pad = -100    # Negative pad means make the interval smaller
  >>> tsc_moves = events.tsc_moves
  >>> tsc_moves.interval_pad = 300  # Pad TSC moves by 300 seconds on each side
  >>> dumps = events.dumps
  >>> dumps.interval_pad = (10, 500)  # Pad dumps 10 secs before and 500 secs after

  >>> good_times = dwells & ~(tsc_moves | dumps)
  >>> dat = fetch.Msid('aoattqt1', '2012:001', '2012:002')
  >>> dat.plot()
  >>> dat.select_intervals(good_times)
  >>> dat.plot('.r')
  >>> grid()

.. image:: complex_event_filter.png

Get commands
^^^^^^^^^^^^^^^^

As with event queries, the basic way to select commands is with the ``filter()`` method,
possibly followed by accessing the results in tabular format with the ``table``
attribute.  For example you find the load commands that occurred in the first hour of 2012
New Year with::

  >>> from kadi import cmds
  >>> new_year_cmds = cmds.filter('2012:001:00:00:00', '2012:001:01:00:00')
  >>> print new_year_cmds.table
           date            type      tlmsid   scs step timeline_id
  --------------------- ---------- ---------- --- ---- -----------
  2012:001:00:11:29.455    ACISPKT AA00000000 131 1615   426098423
  2012:001:00:11:33.455    ACISPKT AA00000000 131 1619   426098423
  2012:001:00:11:37.455    ACISPKT WSPOW00000 131 1623   426098423
  2012:001:00:12:01.455    ACISPKT WSPOW08002 131 1629   426098423
  2012:001:00:13:04.455    ACISPKT WT00910014 131 1635   426098423
  2012:001:00:13:08.455    ACISPKT XTZ0000005 131 1723   426098423
  2012:001:00:13:12.455    ACISPKT RS_0000001 131 1728   426098423
  2012:001:00:13:16.455    ACISPKT RH_0000001 131 1732   426098423
  2012:001:00:18:50.781 COMMAND_HW      CNOOP 128 1073   426098422
  2012:001:00:18:51.806 COMMAND_HW      CNOOP 128 1075   426098422

The output is an astropy `Table <http://docs.astropy.org/en/stable/table/index.html>`_
object with many powerful and handy features built in.  For instance you can easily select
two of the columns and make a new table::

  >>> print new_year_cmds.table['type', 'tlmsid']
     type      tlmsid
  ---------- ----------
     ACISPKT AA00000000
     ACISPKT AA00000000
     ACISPKT WSPOW00000
     ACISPKT WSPOW08002
     ACISPKT WT00910014
     ACISPKT XTZ0000005
     ACISPKT RS_0000001
     ACISPKT RH_0000001
  COMMAND_HW      CNOOP
  COMMAND_HW      CNOOP

Unlike event queries, the output of the ``filter()`` method is frequently useful even for
simple cases because it is the only way to access the parameters associated with the
command.  To see this just print the raw ``new_year_cmds`` without converting into a
table::

   >>> new_year_cmds
  2012:001:00:11:29.455 ACISPKT     tlmsid=AA00000000 scs=131 step=1615 timeline_id=426098423 cmds=3 packet(40)=D80000300030603001300 words=3
  2012:001:00:11:33.455 ACISPKT     tlmsid=AA00000000 scs=131 step=1619 timeline_id=426098423 cmds=3 packet(40)=D80000300030603001300 words=3
  2012:001:00:11:37.455 ACISPKT     tlmsid=WSPOW00000 scs=131 step=1623 timeline_id=426098423 cmds=5 packet(40)=D8000070007030500200000000000010000 words=7
  2012:001:00:12:01.455 ACISPKT     tlmsid=WSPOW08002 scs=131 step=1629 timeline_id=426098423 cmds=5 packet(40)=D80000700071A9D00200000008000010002 words=7
  2012:001:00:13:04.455 ACISPKT     tlmsid=WT00910014 scs=131 step=1635 timeline_id=426098423 cmds=87 packet(40)=D800096009623390009000471CF00140091AA7A0 words=150
  2012:001:00:13:08.455 ACISPKT     tlmsid=XTZ0000005 scs=131 step=1723 timeline_id=426098423 cmds=4 packet(40)=D8000040004003A000E000400000 words=4
  2012:001:00:13:12.455 ACISPKT     tlmsid=RS_0000001 scs=131 step=1728 timeline_id=426098423 cmds=3 packet(40)=D80000300030042002100 words=3
  2012:001:00:13:16.455 ACISPKT     tlmsid=RH_0000001 scs=131 step=1732 timeline_id=426098423 cmds=3 packet(40)=D80000300030044002300 words=3
  2012:001:00:18:50.781 COMMAND_HW  tlmsid=CNOOP scs=128 step=1073 timeline_id=426098422 hex=7E00000 msid=CNOOPLR
  2012:001:00:18:51.806 COMMAND_HW  tlmsid=CNOOP scs=128 step=1075 timeline_id=426098422
  hex=7E00000 msid=CNOOPLR

To access the details of a particular command, select that command with the correct row index
and then get the corresponding item using dictionary-style access::

  >>> new_year_cmds[1]
  2012:001:00:11:33.455 ACISPKT     tlmsid=AA00000000 scs=131 step=1619 timeline_id=426098423 cmds=3 packet(40)=D80000300030603001300 words=3

  >>> new_year_cmds[1]['cmds']
  3

  >>> new_year_cmds[1]['packet(40)']
  'D80000300030603001300'


Details
-----------

The advanced usage section is incomplete, however examples of advanced
usage are demonstrated in thie `Kadi demo IPython notebook
<http://nbviewer.ipython.org/url/cxc.harvard.edu/mta/ASPECT/ipynb/kadi_demo.ipynb>`_.
Please refer to this notebook and download and run it.


Event definitions
^^^^^^^^^^^^^^^^^^^

============= ======================== ================
 Event class        Description           Query name
============= ======================== ================
       |CAP|          :ref:`event_cap`         ``caps``
   |DsnComm|     :ref:`event_dsn_comm`    ``dsn_comms``
      |Dump|         :ref:`event_dump`        ``dumps``
     |Dwell|        :ref:`event_dwell`       ``dwells``
   |Eclipse|      :ref:`event_eclipse`     ``eclipses``
    |FaMove|      :ref:`event_fa_move`     ``fa_moves``
|MajorEvent|  :ref:`event_major_event` ``major_events``
     |Manvr|        :ref:`event_manvr`       ``manvrs``
  |ManvrSeq|    :ref:`event_manvr_seq`   ``manvr_seqs``
 |NormalSun|   :ref:`event_normal_sun`  ``normal_suns``
     |Obsid|        :ref:`event_obsid`       ``obsids``
     |Orbit|        :ref:`event_orbit`       ``orbits``
|OrbitPoint|  :ref:`event_orbit_point` ``orbit_points``
   |RadZone|     :ref:`event_rad_zone`    ``rad_zones``
   |SafeSun|     :ref:`event_safe_sun`    ``safe_suns``
    |Scs107|       :ref:`event_scs107`      ``scs107s``
   |TscMove|     :ref:`event_tsc_move`    ``tsc_moves``
============= ======================== ================



Event filtering
^^^^^^^^^^^^^^^^^^

*TBD*

Event intervals
^^^^^^^^^^^^^^^^^^^

The :class:`~kadi.events.query.EventQuery` class provides a powerful way to define
time intervals based on events or combinations of events.


*TBD*

Commands
^^^^^^^^^^^^

*TBD*

API and related docs
---------------------

.. toctree::
   :maxdepth: 1

   api
   event_descriptions
   maneuver_templates

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

