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
``3FAPOS`` before and after the SIM focus assembly translation.

Non-interval events
""""""""""""""""""""""

The event types |MajorEvent|, |ManvrSeq|, and |OrbitPoint| are a bit different in that
they refer to **moment in time** rather than an interval.  For these types the only field
they all have in common is ``date``.

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
``filter('2012:001')`` output from a QuerySet object (which is discussed later) into a
simple Table object that can be printed, plotted, and used in computations.  Now let's
look at what came out by printing the Table.  Before you do this make your terminal window
plenty wide, there are a bunch of fields and you want to see them all::

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

Use events to filter telemetry
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another common operation is using event intervals to filter telemetry data.  A simple
example is looking at EPHIN E1300 rates only outside of rad zone.  A more complicated
example is plotting OBC rate data only during "quiet" periods, which means during normal
point mode with Kalman lock, at least 300 seconds away from any maneuvers, SIM motions,
grating motions, and momentum dumps.  With Kadi doing this sort of filtering is easy.

Here we show just the simple example above.  First we plot the E1300 rates over a 30 day
span, including rad zones::

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


Get commands
^^^^^^^^^^^^^^^^



Chandra events
----------------




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
     |Obsid|        :ref:`event_obsid`       ``obsids``
     |Orbit|        :ref:`event_orbit`       ``orbits``
|OrbitPoint|  :ref:`event_orbit_point` ``orbit_points``
   |RadZone|     :ref:`event_rad_zone`    ``rad_zones``
   |SafeSun|     :ref:`event_safe_sun`    ``safe_suns``
    |Scs107|       :ref:`event_scs107`      ``scs107s``
   |TscMove|     :ref:`event_tsc_move`    ``tsc_moves``
============= ======================== ================


Find and filter
^^^^^^^^^^^^^^^^^^

Event intervals
^^^^^^^^^^^^^^^^^^^

The :class:`~kadi.events.query.EventQuery` class provides a powerful way to define
time intervals based on events or combinations of events.


Commands
------------

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

