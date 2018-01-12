.. |CAP| replace:: :class:`~kadi.events.models.CAP`
.. |DarkCal| replace:: :class:`~kadi.events.models.DarkCal`
.. |DarkCalReplica| replace:: :class:`~kadi.events.models.DarkCalReplica`
.. |DsnComm| replace:: :class:`~kadi.events.models.DsnComm`
.. |Dump| replace:: :class:`~kadi.events.models.Dump`
.. |Dwell| replace:: :class:`~kadi.events.models.Dwell`
.. |Eclipse| replace:: :class:`~kadi.events.models.Eclipse`
.. |FaMove| replace:: :class:`~kadi.events.models.FaMove`
.. |GratingMove| replace:: :class:`~kadi.events.models.GratingMove`
.. |LoadSegment| replace:: :class:`~kadi.events.models.LoadSegment`
.. |LttBad| replace:: :class:`~kadi.events.models.LttBad`
.. |MajorEvent| replace:: :class:`~kadi.events.models.MajorEvent`
.. |Manvr| replace:: :class:`~kadi.events.models.Manvr`
.. |ManvrSeq| replace:: :class:`~kadi.events.models.ManvrSeq`
.. |NormalSun| replace:: :class:`~kadi.events.models.NormalSun`
.. |Obsid| replace:: :class:`~kadi.events.models.Obsid`
.. |Orbit| replace:: :class:`~kadi.events.models.Orbit`
.. |OrbitPoint| replace:: :class:`~kadi.events.models.OrbitPoint`
.. |PassPlan| replace:: :class:`~kadi.events.models.PassPlan`
.. |RadZone| replace:: :class:`~kadi.events.models.RadZone`
.. |SafeSun| replace:: :class:`~kadi.events.models.SafeSun`
.. |Scs107| replace:: :class:`~kadi.events.models.Scs107`
.. |TscMove| replace:: :class:`~kadi.events.models.TscMove`

Chandra events
==============

Overview
^^^^^^^^^^^^^^^^

The kadi Chandra events database stores a variety of events that are useful in
performing analysis or establishing situational awareness (i.e. what was happening
around March 2017?).  Examples include:

- Events in telemetry such as maneuvers, NPM dwells, obsids, mech movements, momentum
  dumps, orbit events, etc.
- CAPs, DSN passes, dark cals, SCS107, safe modes, bright star hold, etc
- Chandra major events since launch

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
""""""""""""""""""""


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

Getting started
^^^^^^^^^^^^^^^^^^^^^^^

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

  >>> tsc_moves = events.tsc_moves.filter(start='2012:001').table

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

From ``events.tsc_moves`` we chain the ``filter(start='2012:001')`` method to select events that
occurred after ``2012:001``.  The ``filter()`` method is very powerful and can perform
complex filters based on all the available attributes of an event.  In this case to select
an inclusive time range you would supply both the start and stop date in that order,
e.g. ``filter(start='2012:001', stop='2013:001')``.

The last bit of the chain is the `.table` attribute, which says to convert the
``filter(start='2012:001')`` output from a QuerySet object (which is discussed later) into a an
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


Getting help
^^^^^^^^^^^^^^

There are a lot of event types in `kadi` and it's not easy to remember everything.
Here we will make use of the IPython ``?`` function to get help on objects.
Starting from the top, you can list the available event types with the following::

  >>> from kadi import events
  >>> events?
  Type:       module
  String Form:<module 'kadi.events' from 'kadi/events/__init__.py'>
  File:       /data/baffin/tom/git/kadi/kadi/events/__init__.py
  Docstring:
  Access and manipulate events related to the Chandra X-ray Observatory

  Available events are:

  =================  ====================================  ==============
      Query name                 Description                Event class
  =================  ====================================  ==============
               caps                CAP from iFOT database             CAP
          dark_cals    ACA dark current calibration event         DarkCal
  dark_cal_replicas  ACA dark current calibration replica  DarkCalReplica
          dsn_comms                       DSN comm period         DsnComm
              dumps        Ground commanded momentum dump            Dump
             dwells                  Dwell in Kalman mode           Dwell
           eclipses                               Eclipse         Eclipse
           fa_moves                    SIM FA translation          FaMove
      grating_moves       Grating movement (HETG or LETG)     GratingMove
      load_segments       Load segment from iFOT database     LoadSegment
           ltt_bads                     LTT bad intervals          LttBad
       major_events                           Major event      MajorEvent
             manvrs                              Maneuver           Manvr
         manvr_seqs               Maneuver sequence event        ManvrSeq
        normal_suns                 Normal sun mode event       NormalSun
             obsids                Observation identifier           Obsid
             orbits                                 Orbit           Orbit
       orbit_points                           Orbit point      OrbitPoint
         pass_plans                             Pass plan        PassPlan
          rad_zones                        Radiation zone         RadZone
          safe_suns                        Safe sun event         SafeSun
            scs107s                            SCS107 run          Scs107
          tsc_moves                   SIM TSC translation         TscMove
  =================  ====================================  ==============

  More help available at:

  - Getting started
      http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/kadi/#getting-started

  - Details (event definitions, filtering, intervals)
      http://cxc.cfa.harvard.edu/mta/ASPECT/tool_doc/kadi/#details

Next you may want to learn about the specific fields that are available
for an event type.  Use the same ``?`` strategy here::

  >>> events.orbits?
  Type:       EventQuery
  String Form:<kadi.events.query.EventQuery object at 0x358f5d0>
  File:       /data/baffin/tom/git/kadi/kadi/events/query.py
  Docstring:
  Orbit

  **Event definition**: single Chandra orbit starting from ascending node crossing

  Full orbit, with dates corresponding to start (ORBIT ASCENDING NODE CROSSING), stop,
  apogee, perigee, radzone start and radzone stop.  Radzone is defined as the time
  covering perigee when radmon is disabled by command.  This corresponds to the planned
  values and may differ from actual in the case of events that run SCS107 and
  prematurely disable RADMON.

  **Fields**

  ================== ========== ==================================================
        Field           Type                       Description
  ================== ========== ==================================================
              start   Char(21)         Start time (orbit ascending node crossing)
               stop   Char(21)     Stop time (next orbit ascending node crossing)
             tstart      Float         Start time (orbit ascending node crossing)
              tstop      Float     Stop time (next orbit ascending node crossing)
                dur      Float                               Orbit duration (sec)
          orbit_num    Integer                                       Orbit number
            perigee   Char(21)                                       Perigee time
             apogee   Char(21)                                        Apogee time
          t_perigee      Float                             Perigee time (CXC sec)
      start_radzone   Char(21)                             Start time of rad zone
       stop_radzone   Char(21)                              Stop time of rad zone
   dt_start_radzone      Float   Start time of rad zone relative to perigee (sec)
    dt_stop_radzone      Float    Stop time of rad zone relative to perigee (sec)
  ================== ========== ==================================================

  Class Docstring:
  High-level interface for handling event queries.

  This includes a few key methods:

  - filter() : filter events matching criteria and return Django query set
  - intervals(): return time intervals between event start/stop times

  A key feature is that EventQuery objects can be combined with boolean
  and, or, and not logic to generate composite EventQuery objects.  From
  there the intervals() output can be used to select or remove the intervals
  from Ska.engarchive fetch datasets.

Finally, the EventQuery ``filter`` method has detailed help available to remind
you of syntax here::

  >>> events.manvrs.filter?
  Type:       instancemethod
  String Form:<bound method EventQuery.filter of <kadi.events.query.EventQuery object at 0x309e8d0>>
  File:       /data/baffin/tom/git/kadi/kadi/events/query.py
  Definition: events.manvrs.filter(self, start=None, stop=None, obsid=None, subset=None, **kwargs)
  Docstring:
  Find events between ``start`` and ``stop``, or with the given ``obsid``, which
  match the filter attributes in subsequent keyword argumentse.  The matching events
  are returned as a Django query set [1].

  If ``start`` or ``stop`` are not supplied they default to the beginning / end of
  available data.  The optional ``subset`` arg must be a Python slice() object and
  allows slicing of the filtered output.

  This function allows for the powerful field lookups from the underlying
  Django model implementation.  A field lookup is similar to an SQL ``WHERE``
  clause with the form ``<field_name>__<filter_type>=<value>`` (with a double
  underscore between.  For instance ``n_dwell__lte=1`` would be the same as
  ``SELECT ... WHERE n_dwell <= 1``.  Common filter types are:

  - ``exact`` (exact match), ``contains`` (contains string)
  - ``startswith``, ``endswith`` (starts or ends with string)
  - ``gt``, ``gte``, ``lt``, ``lte`` (comparisons)
  - ``isnull`` (field value is missing, e.g. manvrs.aca_proc_act_start)

  For the common case of testing equality (``exact``) there is a shortcut where
  the ``__exact`` can be skipped, so for instance ``n_dwell=1`` selects
  maneuver events with one dwell.  The full list of field lookups is at [2].

  Examples::

    >>> from kadi import events
    >>> events.manvrs.filter('2011:001', '2012:001', n_dwell=1, angle__gte=140)
    >>> events.manvrs.filter('2011:001', '2012:001', subset=slice(None, 5))  # first 5
    >>> events.manvrs.filter(obsid=14305)

  [1]: https://docs.djangoproject.com/en/1.5/topics/db/queries/
  [2]: https://docs.djangoproject.com/en/1.5/ref/models/querysets/#field-lookups

  :param start: start time (DateTime compatible format)
  :param stop: stop time (DateTime compatible format)
  :param obsid: obsid for event
  :param subset: subset of matching events that are output

  :returns: Django query set with matching events


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

  >>> e1300.select_intervals(events.rad_zones)

Sometimes it is useful to select intervals in an MSID while retaining the
original full dataset.  You can do this with the ``copy=True`` option, which
returns a brand new MSID object and leaves the original untouched::

  >>> e1300 = fetch.Msid("SCE1300", '2012:020', '2012:050')
  >>> e1300_rad_zones = e1300.select_intervals(events.rad_zones, copy=True)

More on intervals and padding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The event intervals themselves within a particular time range can be accessed as a list of
``(start, stop)`` with the ``intervals()`` method::

  >>> events.rad_zones.intervals('2012:020', '2012:030')
  [('2012:020:14:56:33.713', '2012:020:23:42:31.713'),
   ('2012:023:06:15:15.541', '2012:023:16:15:33.248'),
   ('2012:025:21:22:22.506', '2012:026:08:10:20.506'),
   ('2012:028:13:22:04.533', '2012:028:22:20:02.533')]

Frequently you want to pad the intervals by a bit in each direction.  For instance if you
want stable spacecraft attitudes you would want to filter out events that disturb the
attitude, namely SIM motions, grating motions, and momentum dumps.  But the events cover
just the exact interval of a motion or dump, and you'd like to chop out times for 5 or 10
minutes afterward.  In addition, it's usually a good idea to cut out a bit before just to
be safe.

To pad the grating events out by 5 minutes in each direction, create a new EventQuery
object like this::

  >>> grating_moves = events.grating_moves(pad=300)  # pad time in seconds

Positive values always make the interval *bigger* in each direction, so
a pad of ``300`` seconds makes the interval a total of 10 minutes longer (5 minutes
on each side).  A pad of ``-300`` seconds makes the interval start 5 minutes
later and end 5 minutes earlier.

You can be more precise and specify the padding at the event start and stop by
providing two values, the start and stop pad time.  Again, the convention is that
positive padding is always making the interval longer in each direction.
::

  >>> grating_moves = events.grating_moves(pad=(100, 300))  # pad time in seconds

To finish the example, we would define similar padded events for the SIM moves and dumps::

  >>> tsc_moves = events.tsc_moves(pad=(100, 300))
  >>> dumps = events.dumps(pad=(100, 300))

You can examine or set the padding using the ``interval_pad`` attribute::

  >>> dumps.interval_pad
  <Pad start=100.0 stop=300.0 seconds>
  >>> dumps.interval_pad = 500
  >>> dumps.interval_pad
  <Pad start=500.0 stop=500.0 seconds>

You can also set the interval pad for an event type directly.  *This method is
discouraged*, but shown below for reference since this used to be the accepted method::

  >>> events.grating_moves.interval_pad = -100

Boolean combinations
"""""""""""""""""""""

The real power of intervals is that they can be combined with boolean expressions and can
be padded on either end to make it easy to create complex time filters.  Using the results
in the previous section, we can create a composite event which is when there is a grating
move *OR* a momentum dump *OR* a SIM (TSC) move::

  >>> disturbances = grating_moves | dumps | tsc_moves

The following example selects all Kalman mode dwells, but removes times that could be
affected by a SIM TSC move or a momentum dump.

  >>> dwells = events.dwells(pad=-100)
  >>> good_times = dwells & ~disturbances  # Dwells and NOT disturbances

You can examine the composite ``good_times`` event query and see how it is constructed of
boolean combinations of the underlying base event query objects::

  >>> good_times
  (<EventQuery: Dwell pad=-100.0> AND NOT ((<EventQuery: GratingMove pad=(100.0, 300.0)>
  OR <EventQuery: Dump pad=(100.0, 300.0)>) OR <EventQuery: TscMove pad=(100.0, 300.0)>))

Finally you can you this composite event query to define times for selecting telemetry
of interest::

  >>> dat = fetch.Msid('aoattqt1', '2012:001', '2012:002')
  >>> dat.plot()
  >>> dat_good = dat.select_intervals(good_times, copy=True)
  >>> dat_good.plot('.r')
  >>> grid()

.. image:: complex_event_filter.png

Filtering the interval events
""""""""""""""""""""""""""""""

The examples shown above share the feature that the selected intervals were defined
using *all* of the events for a particular type.  However, it is also possible to
select only a subset of the available events based on other filter criteria.  For
example, if you wanted to examine telemetry during HETG insertions.  This would be
a snap with the following, which defines a new event query which is the subset
of HETG insertion grating moves::

  >>> hetg_insert = events.grating_moves(pad=50, grating='HETG', direction='INSR')

This new event query object can be used just like the original ``events.grating_moves``
except that now it only has HETG insertion events.
::

  >>> events.grating_moves
  <EventQuery: GratingMove pad=0.0>
  >>> hetg_insert
  <EventQuery: GratingMove pad=0.0 direction='INSR' grating='HETG'>

To overplot the grating angle as a function of time since the grating move start you might
do::

  >>> intervals = hetg_insert.intervals('2010:001', '2010:030')
  >>> intervals
  >>> print intervals  # This is a list of (start, stop) pairs
  [('2010:002:15:25:39.725', '2010:002:15:28:15.013'),
   ('2010:004:09:41:29.708', '2010:004:09:44:04.996'),
               ...
   ('2010:018:13:36:14.745', '2010:018:13:38:50.033'),
   ('2010:021:04:47:00.207', '2010:021:04:49:35.494')]
  >>> for start, stop in intervals:
  ...     dat = fetch.Msid('4hposaro', start, stop)
  ...     plot(dat.times - dat.times[0], dat.vals)

LTT bad times
@@@@@@@@@@@@@@

Another example of practical interest is using the LTT bad times event to remove bad times
for long-term trending plots by MSID.  Before explaining more about what is going on, here is
an example of filtering out LTT bad times (for the impatient)::

  >>> dat = fetch.Msid('AIRU2BT', '2011:001', '2013:001', stat='daily')
  >>> dat_good = dat.remove_intervals(events.ltt_bads(msid='AIRU2BT'), copy=True)
  >>> dat.plot('r', label='All')
  >>> dat_good.plot(label='Good')
  >>> plt.ylim(96, 103)
  >>> plt.grid()
  >>> plt.legend(fontsize='small')
  >>> plt.title('AIRU2BT')

.. image:: ltt_bads_airu2bt.png

Now let's look more closely at the LTT bad times events, which are derived from a
FOT-supplied file that has the dates when particular trending items (the ``msid`` column)
are bad for some reason::

  >>> print events.ltt_bads.filter('2000:001', '2001:001').table
          start                  stop            tstart       tstop       dur         msid      flag
  --------------------- --------------------- ------------ ------------ ------- --------------- ----
  2000:001:00:00:00.000 2000:002:00:00:00.000 63072064.184 63158464.184 86400.0 PITCH_STAB_PERF    J
  2000:001:00:00:00.000 2000:002:00:00:00.000 63072064.184 63158464.184 86400.0   YAW_STAB_PERF    J
  2000:004:00:00:00.000 2000:005:00:00:00.000 63331264.184 63417664.184 86400.0      GCM_TSCACC    1
                    ...                   ...          ...          ...     ...             ...  ...
  2000:358:00:00:00.000 2000:359:00:00:00.000 93916864.184 94003264.184 86400.0         3SDM15V    1
  2000:358:00:00:00.000 2000:359:00:00:00.000 93916864.184 94003264.184 86400.0          3SDP5V    1
  2000:366:00:00:00.000 2001:001:00:00:00.000 94608064.184 94694464.184 86400.0         3SDM15V    1

Some of the ``msid`` values correspond to like-named MSIDs in the engineering archive, but
many (including all those shown here) do not.  You can find the non-matches with::

  >>> print sorted(set(x.msid for x in events.ltt_bads.filter('1999:001')
                   if not (x.msid in fetch.content or 'DP_' + x.msid in fetch.content)))
  [u'*', u'3SDAGV', u'3SDFATSV', u'3SDM15V', u'3SDP15V', u'3SDP5V', u'3SDTSTSV', u'5EHSE300', u'ABIASZ',
  ...
  u'ROLL_BIAS_DIFF', u'SAMYTEMDEL', u'SAPYTEMDEL', u'TFCAG', u'TFCDG', u'VECANGLE_DIFF',
  u'YAW_BIAS_DIFF', u'YAW_CTRL', u'YAW_STAB', u'YAW_STAB_PERF']

There is a special ``msid`` value of ``'*'`` which corresponds to times that are bad for ALL
MSIDs.  The bad intervals with ``msid == '*'`` are always included in query results::

  >>> events.ltt_bads(msid='AACCCDPT').all()
  <LttBad: start=1999:270:00:00:00.000 msid=* flag=*>
  <LttBad: start=1999:291:00:00:00.000 msid=* flag=*>
  <LttBad: start=1999:295:00:00:00.000 msid=* flag=*>
  ...
  <LttBad: start=2011:190:00:00:00.000 msid=* flag=*>
  <LttBad: start=2011:191:00:00:00.000 msid=AACCCDPT flag=A>
  <LttBad: start=2011:192:00:00:00.000 msid=AACCCDPT flag=A>
  ...
  <LttBad: start=2011:302:00:00:00.000 msid=AACCCDPT flag=A>
  <LttBad: start=2012:150:00:00:00.000 msid=* flag=*>
  <LttBad: start=2012:151:00:00:00.000 msid=* flag=*>


Selecting overlapping events
""""""""""""""""""""""""""""""

Frequently you are interested in getting events based on overlap with other events.  Examples include:

- Select maneuver events either inside or outside the radiation zone.
- Select SIM moves that occur during a maneuver.
- Select maneuvers that have a SIM or Grating move.

This is possible using the :func:`~kadi.events.models.BaseModel.QuerySet.select_overlapping`
method of a `QuerySet` object.  A `QuerySet` is the object that gets returned when you use
the :func:`~kadi.events.query.EventQuery.filter()` or :func:`~kadi.events.query.EventQuery.all()`
methods.

This is all easiest to see by example.  Start by getting a subset of maneuvers that
occurred between 2001:001:00:00:00 and 2001:003:00:00:00:

  >>> manvrs = events.manvrs.filter('2001:001:00:00:00', '2001:003:00:00:00')
  >>> manvrs
  <Manvr: start=2001:001:07:48:35.843 dur=2073 n_dwell=2 template=nman_dwell>
  <Manvr: start=2001:002:05:20:14.046 dur=1184 n_dwell=1 template=normal>
  <Manvr: start=2001:002:08:53:23.997 dur=241 n_dwell=1 template=normal>
  <Manvr: start=2001:002:14:55:24.773 dur=240 n_dwell=1 template=normal>
  <Manvr: start=2001:002:20:50:34.523 dur=1185 n_dwell=1 template=normal>
  >>> type(manvrs)
  <class 'kadi.events.models.QuerySet'>

Now let's find which of those had at least some portion inside the rad zone::

  >>> manvrs.select_overlapping(events.rad_zones)
  [<Manvr: start=2001:002:05:20:14.046 dur=1184 n_dwell=1 template=normal>,
   <Manvr: start=2001:002:08:53:23.997 dur=241 n_dwell=1 template=normal>,
   <Manvr: start=2001:002:14:55:24.773 dur=240 n_dwell=1 template=normal>]

This makes sense if you look at the rad zone times near then::

  >>> events.rad_zones.filter('2001:001:00:00:00', '2001:003:00:00:00')
  <RadZone: 194 2001:002:05:15:03 2001:002:20:50:23 dur=56.1 ksec>

We can turn it around and ask which maneuvers had some part outside of the rad zone,
remembering the boolean expression logic that is possible.  In fact any composite
``QueryEvent`` expression can be used as the argument:

  >>> manvrs.select_overlapping(~events.rad_zones)
  [<Manvr: start=2001:001:07:48:35.843 dur=2073 n_dwell=2 template=nman_dwell>,
   <Manvr: start=2001:002:20:50:34.523 dur=1185 n_dwell=1 template=normal>]

One subtlety here is that the original `manvrs` object is a ``QuerySet`` while the result
of ``select_overlapping`` is a **list** of event objects.  A ``QuerySet`` is more powerful
because you could apply further filtering and so on, while the list is just a regular
Python list.  In practice this is not a real limitation.

Now on to the example of finding SIM moves that occur during a maneuver::

  >>> tsc_moves = events.tsc_moves.filter('2010:191', '2010:192')
  >>> tsc_moves
  <TscMove: start=2010:191:18:34:00.978 dur=262 start_3tscpos=75623 stop_3tscpos=-99616>
  <TscMove: start=2010:192:01:42:36.179 dur=230 start_3tscpos=-99616 stop_3tscpos=75623>
  <TscMove: start=2010:192:03:27:33.779 dur=33 start_3tscpos=75623 stop_3tscpos=92903>
  >>> tsc_moves.select_overlapping(events.manvrs)
  [<TscMove: start=2010:191:18:34:00.978 dur=262 start_3tscpos=75623 stop_3tscpos=-99616>,
   <TscMove: start=2010:192:01:42:36.179 dur=230 start_3tscpos=-99616 stop_3tscpos=75623>,
   <TscMove: start=2010:192:03:27:33.779 dur=33 start_3tscpos=75623 stop_3tscpos=92903>]

All of these moves overlapped with a maneuver, but what if we want the moves that occurred
**entirely within** a maneuver.  To do this set the ``allow_partial=False`` to insist on
complete overlaps::

  >>> tsc_moves.select_overlapping(events.manvrs, allow_partial=False)
  [<TscMove: start=2010:192:01:42:36.179 dur=230 start_3tscpos=-99616 stop_3tscpos=75623>,
   <TscMove: start=2010:192:03:27:33.779 dur=33 start_3tscpos=75623 stop_3tscpos=92903>]

This makes sense if you look at the maneuvers in that time interval.  The first of the
three SIM moves started before the maneuver at ``191:18:36:37`` but lasted about two minutes
into the maneuver.

  >>> events.manvrs.filter('2010:191', '2010:192')
  <Manvr: start=2010:191:12:39:40.939 dur=240 n_dwell=1 template=normal>
  <Manvr: start=2010:191:18:36:37.290 dur=241 n_dwell=1 template=normal>
  <Manvr: start=2010:192:01:41:47.492 dur=1020 n_dwell=1 template=normal>
  <Manvr: start=2010:192:03:26:36.892 dur=2869 n_dwell=1 template=normal>

To find every SIM-Z move in 2011 that didn't overlap at all with a maneuver you would do::

  >>> tsc_moves = events.tsc_moves.filter('2011:001', '2012:001')
  >>> tsc_moves.select_overlapping(~events.manvrs, allow_partial=False)
  [<TscMove: start=2011:158:15:24:01.657 dur=262 start_3tscpos=92903 stop_3tscpos=-99616>,
   <TscMove: start=2011:216:07:03:21.245 dur=230 start_3tscpos=75623 stop_3tscpos=-99616>,
   <TscMove: start=2011:297:18:28:57.614 dur=262 start_3tscpos=92903 stop_3tscpos=-99616>,
   <TscMove: start=2011:299:05:05:28.208 dur=230 start_3tscpos=75623 stop_3tscpos=-99616>]

Details
^^^^^^^

The advanced usage section is incomplete, however examples of advanced
usage are demonstrated in thie `Kadi demo IPython notebook
<http://nbviewer.ipython.org/url/cxc.harvard.edu/mta/ASPECT/ipynb/kadi_demo.ipynb>`_.
Please refer to this notebook and download and run it.


Event definitions
"""""""""""""""""

================ ============================= =====================
  Event class             Description                Query name
================ ============================= =====================
           |CAP|              :ref:`event_cap`              ``caps``
       |DarkCal|         :ref:`event_dark_cal`         ``dark_cals``
|DarkCalReplica| :ref:`event_dark_cal_replica` ``dark_cal_replicas``
       |DsnComm|         :ref:`event_dsn_comm`         ``dsn_comms``
          |Dump|             :ref:`event_dump`             ``dumps``
         |Dwell|            :ref:`event_dwell`            ``dwells``
       |Eclipse|          :ref:`event_eclipse`          ``eclipses``
        |FaMove|          :ref:`event_fa_move`          ``fa_moves``
   |GratingMove|     :ref:`event_grating_move`     ``grating_moves``
   |LoadSegment|     :ref:`event_load_segment`     ``load_segments``
        |LttBad|          :ref:`event_ltt_bad`          ``ltt_bads``
    |MajorEvent|      :ref:`event_major_event`      ``major_events``
         |Manvr|            :ref:`event_manvr`            ``manvrs``
      |ManvrSeq|        :ref:`event_manvr_seq`        ``manvr_seqs``
     |NormalSun|       :ref:`event_normal_sun`       ``normal_suns``
         |Obsid|            :ref:`event_obsid`            ``obsids``
         |Orbit|            :ref:`event_orbit`            ``orbits``
    |OrbitPoint|      :ref:`event_orbit_point`      ``orbit_points``
      |PassPlan|        :ref:`event_pass_plan`        ``pass_plans``
       |RadZone|         :ref:`event_rad_zone`         ``rad_zones``
       |SafeSun|         :ref:`event_safe_sun`         ``safe_suns``
        |Scs107|           :ref:`event_scs107`           ``scs107s``
       |TscMove|         :ref:`event_tsc_move`         ``tsc_moves``
================ ============================= =====================


Event filtering
""""""""""""""""

The :class:`~kadi.events.query.EventQuery` :func:`~kadi.events.query.EventQuery.filter`
method allows you to find events between ``start`` and ``stop``, or with the given
``obsid``, which match the filter attributes in subsequent keyword arguments.  The
matching events are returned as a Django query set [1].

If ``start`` or ``stop`` are not supplied they default to the beginning / end of
available data.  The optional ``subset`` arg must be a Python slice() object and
allows slicing of the filtered output.

This method allows for the use of `Django field lookups
<https://docs.djangoproject.com/en/1.5/ref/models/querysets/#field-lookups>`_ from the
underlying Django model implementation.  A field lookup is similar to an SQL ``WHERE``
clause with the form ``<field_name>__<filter_type>=<value>`` (with a double underscore
between).  For instance ``n_dwell__lte=1`` would be the same as ``SELECT ... WHERE n_dwell
<= 1``.  Common filter types are:

- ``exact`` (exact match), ``contains`` (contains string)
- ``startswith``, ``endswith`` (starts or ends with string)
- ``gt``, ``gte``, ``lt``, ``lte`` (comparisons)
- ``isnull`` (field value is missing, e.g. manvrs.aca_proc_act_start)

For the common case of testing equality (``exact``) there is a shortcut where
the ``__exact`` can be skipped, so for instance ``n_dwell=1`` selects
maneuver events with one dwell.  For a full list of field lookups see the general
Django documentation on `Making queries <https://docs.djangoproject.com/en/1.5/topics/db/queries/>`_

Examples::

  >>> from kadi import events
  >>> events.manvrs.filter('2011:001', '2012:001', n_dwell=1, angle__gte=140)
  >>> events.manvrs.filter('2011:001', '2012:001')[:5]  # first 5
  >>> events.manvrs.filter(obsid=14305)  # Manveuver(s) where obsid=14305 at start

To get all of the events use the :func:`~kadi.events.query.EventQuery.all` method::

  >>> events.normal_suns.all()
  <NormalSun: start=2000:027:13:33:38.872 dur=1020>
  <NormalSun: start=2000:049:03:23:08.544 dur=126872>
     ...
  <NormalSun: start=2011:299:05:09:02.945 dur=81773>
  <NormalSun: start=2012:151:12:33:36.285 dur=48711>

Advanced filtering
@@@@@@@@@@@@@@@@@@@@

The result of a :func:`~kadi.events.query.EventQuery.filter` query is a Django
``QuerySet`` object.  From this starting point you can chain additional queries
or take advantage of any of the ``QuerySet`` functionality described in
`Making queries <https://docs.djangoproject.com/en/1.5/topics/db/queries/>`_.

For example if you wanted to find all maneuvers in 2011-Jan that were not "normal"
and did not have an intermediate attitude, you could do::

  >>> manvrs_2011 = events.manvrs.filter('2011-01-01T00:00:00', '2012-01-01T00:00:00')
  >>> manvrs_2011.exclude(template='normal').exclude(template='interm_att')
  <Manvr: start=2011:111:11:09:03.933 dur=1011 n_dwell=1 template=two_acq>
  <Manvr: start=2011:112:06:22:42.186 dur=1116 n_dwell=1 template=two_acq>
     ...
  <Manvr: start=2011:296:21:46:41.147 dur=1934 n_dwell=2 template=three_acq>
  <Manvr: start=2011:298:22:07:37.432 dur=492 n_dwell=3 template=unknown>

To go really crazy and allow for even more complex queries, like those requiring ``OR``
logic, you can use `Django Q objects <https://docs.djangoproject.com/en/1.5/topics/db/queries/#complex-lookups-with-q-objects>`_.
