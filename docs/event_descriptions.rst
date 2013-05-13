=======================
Event Descriptions
=======================

.. _event_cap:

CAP from iFOT database
----------------------

**Event definition**: CAP from iFOT database

**Fields**

========= =========== =================
  Field       Type       Description
========= =========== =================
 ifot_id     Integer
   start    Char(21)
    stop    Char(21)
     num    Char(15)        CAP number
   title        Text         CAP title
   descr        Text   CAP description
   notes        Text         CAP notes
    link   Char(250)          CAP link
========= =========== =================

.. _event_dsn_comm:

Scheduled DSN comm period
-------------------------

**Event definition**: DSN comm pass beginning of support to end of support (not
  beginning / end of track).

**Fields**

=========== ========== ========================
   Field       Type          Description
=========== ========== ========================
   ifot_id    Integer
     start   Char(21)
      stop   Char(21)
       bot    Char(4)       Beginning of track
       eot    Char(4)             End of track
  activity   Char(30)     Activity description
    config   Char(10)            Configuration
 data_rate    Char(9)                Data rate
      site   Char(12)                 DSN site
       soe    Char(4)   DSN Sequence Of Events
   station    Char(6)              DSN station
=========== ========== ========================

.. _event_dump:

Ground commanded momentum dump
------------------------------

**Event definition**: interval where ``AOUNLOAD = GRND``

**Fields**

======== ========== ================================
 Field      Type              Description
======== ========== ================================
  start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
   stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
 tstart      Float            Start time (CXC secs)
  tstop      Float             Stop time (CXC secs)
    dur      Float                  Duration (secs)
======== ========== ================================

.. _event_dwell:

Dwell in Kalman mode
--------------------

**Event definition**: contiguous interval of AOACASEQ = KALM between::

  Start: AOACASEQ ==> KALM  (transition from any state to KALM)
  Stop:  AOACASEQ ==> not KALM (transition to any state other than KALM)
                **or**
         AOPCADMD ==> NMAN

Short Kalman dwells that are less than 400 seconds long are *ignored* and
are not recorded in the database.  These are typically associated with monitor
window commanding or multiple acquisition attempts).

**Fields**

============ ============ ========================================
   Field         Type                   Description
============ ============ ========================================
      start     Char(21)           Start time (YYYY:DDD:HH:MM:SS)
       stop     Char(21)            Stop time (YYYY:DDD:HH:MM:SS)
     tstart        Float                    Start time (CXC secs)
      tstop        Float                     Stop time (CXC secs)
        dur        Float                          Duration (secs)
 rel_tstart        Float   Start time relative to manvr end (sec)
      manvr   ForeignKey        Maneuver that contains this dwell
         ra        Float                    Right ascension (deg)
        dec        Float                        Declination (deg)
       roll        Float                         Roll angle (deg)
============ ============ ========================================

.. _event_eclipse:

Eclipse
-------

**Event definition**: interval where ``AOECLIPS = ECL``

**Fields**

======== ========== ================================
 Field      Type              Description
======== ========== ================================
  start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
   stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
 tstart      Float            Start time (CXC secs)
  tstop      Float             Stop time (CXC secs)
    dur      Float                  Duration (secs)
======== ========== ================================

.. _event_fa_move:

SIM FA translation
------------------

**Event definition**: interval where ``3FAMOVE = MOVE``

**Fields**

============== ========== ================================
    Field         Type              Description
============== ========== ================================
        start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
         stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
       tstart      Float            Start time (CXC secs)
        tstop      Float             Stop time (CXC secs)
          dur      Float                  Duration (secs)
 start_3fapos    Integer        Start FA position (steps)
  stop_3fapos    Integer         Stop FA position (steps)
============== ========== ================================

.. _event_major_event:

Major event
-----------

**Event definition**: events from the two lists maintained by the FOT and
  the FDB (systems engineering).

Two lists of major event related to Chandra are available on OCCweb:

- http://occweb.cfa.harvard.edu/occweb/web/fot_web/eng/reports/Chandra_major_events.htm
- http://occweb.cfa.harvard.edu/occweb/web/fdb_web/Major_Events.html

These two event lists are scraped from OCCweb and merged into a single list with a
common structure.  Unlike most kadi event types, the MajorEvent class does not
represent an interval of time (``start`` and ``stop``) but only has ``start``
(YYYY:DOY) and ``date`` (YYYY-Mon-DD) attributes to indicate the time.

**Fields**

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

.. _event_manvr:

Maneuver
--------

**Event definition**: interval where ``AOFATTMD = MNVR`` (spacecraft actually maneuvering)

The maneuver event includes a number of attributes that give a detailed
characterization of the timing and nature of the maneuver and corresponding
star acquisitions and normal point model dwells.

The ``start`` and ``stop`` time attributes for a maneuver event correspond exactly to
the start and stop of the actual maneuver.  However, the full maneuver event
contains information covering a larger time span from the end of the previous maneuver
to the start of the next maneuver::

  Previous maneuver
                         <---- Start of included information
    Previous MANV end
    Previous NPNT start

    ==> Maneuver <==

    Star acquisition
    Transition to KALM
    Kalman dwell
      Optional: more dwells, star acq sequences, NMAN/NPNT

    Transition to NMAN
    Transition to MANV
                         <---- End of included information
  Next maneuver

**Fields**

==================== ========== ============================================================
       Field            Type                            Description
==================== ========== ============================================================
              start   Char(21)                               Start time (YYYY:DDD:HH:MM:SS)
               stop   Char(21)                                Stop time (YYYY:DDD:HH:MM:SS)
             tstart      Float                                        Start time (CXC secs)
              tstop      Float                                         Stop time (CXC secs)
                dur      Float                                              Duration (secs)
    prev_manvr_stop   Char(21)             Stop time of previous AOFATTMD=MNVR before manvr
    prev_npnt_start   Char(21)            Start time of previous AOPCADMD=NPNT before manvr
         nman_start   Char(21)                        Start time of AOPCADMD=NMAN for manvr
        manvr_start   Char(21)                        Start time of AOFATTMD=MNVR for manvr
         manvr_stop   Char(21)                         Stop time of AOFATTMD=MNVR for manvr
         npnt_start   Char(21)                      Start time of AOPCADMD=NPNT after manvr
          acq_start   Char(21)                      Start time of AOACASEQ=AQXN after manvr
        guide_start   Char(21)                      Start time of AOACASEQ=GUID after manvr
       kalman_start   Char(21)                      Start time of AOACASEQ=KALM after manvr
 aca_proc_act_start   Char(21)                       Start time of AOPSACPR=ACT after manvr
          npnt_stop   Char(21)                       Stop time of AOPCADMD=NPNT after manvr
    next_nman_start   Char(21)                 Start time of next AOPCADMD=NMAN after manvr
   next_manvr_start   Char(21)                 Start time of next AOFATTMD=MNVR after manvr
            n_dwell    Integer    Number of kalman dwells after manvr and before next manvr
              n_acq    Integer   Number of AQXN intervals after manvr and before next manvr
            n_guide    Integer   Number of GUID intervals after manvr and before next manvr
           n_kalman    Integer   Number of KALM intervals after manvr and before next manvr
          anomalous    Boolean                             Key MSID shows off-nominal value
           template   Char(16)                                    Matched maneuver template
           start_ra      Float                           Start right ascension before manvr
          start_dec      Float                               Start declination before manvr
         start_roll      Float                                Start roll angle before manvr
            stop_ra      Float                             Stop right ascension after manvr
           stop_dec      Float                                 Stop declination after manvr
          stop_roll      Float                                  Stop roll angle after manvr
              angle      Float                                         Maneuver angle (deg)
==================== ========== ============================================================

``n_acq``, ``n_guide``, and ``n_kalman``: these provide a count of the number of times
    after the maneuver ends that ``AOACASEQ`` changes value from anything to ``AQXN``,
    ``GUID``, and ``KALM`` respectively.

``anomalous``: this is ``True`` if the following MSIDs have values that are
    not in the list of nominal state values:

    ==========  ===========================
       MSID          Nominal state values
    ==========  ===========================
     AOPCADMD       NPNT NMAN
     AOACASEQ       GUID KALM AQXN
     AOFATTMD       MNVR STDY
     AOPSACPR       INIT INAC ACT
     AOUNLOAD       MON  GRND
    ==========  ===========================

``template``: this indicates which of the pre-defined maneuver sequence templates were
    matched by this maneuver.  For details see :ref:`maneuver_templates`.

.. _event_manvr_seq:

Maneuver sequence event
-----------------------

Each entry in this table corresponds to a state transition for an MSID
that is relevant to the sequence of events comprising a maneuver event.

**Fields**

=========== ============ ===========
   Field        Type     Description
=========== ============ ===========
     manvr   ForeignKey
      msid      Char(8)
  prev_val      Char(4)
       val      Char(4)
      date     Char(21)
        dt        Float
      time        Float
 prev_date     Char(21)
 prev_time        Float
=========== ============ ===========

.. _event_obsid:

Observation identifier
----------------------

**Event definition**: interval where ``COBSRQID`` is unchanged.

**Fields**

======== ========== ================================
 Field      Type              Description
======== ========== ================================
  start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
   stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
 tstart      Float            Start time (CXC secs)
  tstop      Float             Stop time (CXC secs)
    dur      Float                  Duration (secs)
  obsid    Integer        Observation ID (COBSRQID)
======== ========== ================================

.. _event_orbit:

Orbit
-----

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

.. _event_orbit_point:

Orbit point
-----------

**Fields**

=========== ============ ===========
   Field        Type     Description
=========== ============ ===========
     orbit   ForeignKey
      date     Char(21)
      name      Char(9)
 orbit_num      Integer
     descr     Char(50)
=========== ============ ===========

.. _event_rad_zone:

Radiation zone
--------------

**Fields**

=========== ============ ================================
   Field        Type               Description
=========== ============ ================================
     start     Char(21)   Start time (YYYY:DDD:HH:MM:SS)
      stop     Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
    tstart        Float            Start time (CXC secs)
     tstop        Float             Stop time (CXC secs)
       dur        Float                  Duration (secs)
     orbit   ForeignKey
 orbit_num      Integer
   perigee     Char(21)
=========== ============ ================================

.. _event_safe_sun:

Safe sun event
--------------

**Event definition**: interval when CPE PCAD mode ``61PSTS02 = SSM``

During a safing event and recovery this MSID can toggle to different values,
so SafeSun events within 24 hours of each other are merged.

**Fields**

======== ========== ================================
 Field      Type              Description
======== ========== ================================
  start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
   stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
 tstart      Float            Start time (CXC secs)
  tstop      Float             Stop time (CXC secs)
    dur      Float                  Duration (secs)
  notes       Text
======== ========== ================================

.. _event_scs107:

SCS107 run
----------

**Event definition**: interval with the following combination of state values::

  3TSCMOVE = MOVE
  AORWBIAS = DISA
  CORADMEN = DISA

These MSIDs are first sampled onto a common time sequence of 16.4 sec samples
so the start / stop times are accurate only to that resolution.

Early in the mission there were two SIM TSC translations during an SCS107 run.
By the above rules this would generate two SCS107 events, but instead any two
SCS107 events within 600 seconds are combined into a single event.

**Fields**

======== ========== ================================
 Field      Type              Description
======== ========== ================================
  start   Char(21)   Start time (YYYY:DDD:HH:MM:SS)
   stop   Char(21)    Stop time (YYYY:DDD:HH:MM:SS)
 tstart      Float            Start time (CXC secs)
  tstop      Float             Stop time (CXC secs)
    dur      Float                  Duration (secs)
  notes       Text               Supplemental notes
======== ========== ================================

.. _event_tsc_move:

SIM TSC translation
-------------------

**Event definition**: interval where ``3TSCMOVE = MOVE``

In addition to reporting the start and stop TSC position, these positions are also
converted to the corresponding science instrument detector name, one of ``ACIS-I``,
``ACIS-S``, ``HRC-I``, or ``HRC-S``.  The maximum PWM value ``3MRMMXMV`` (sampled at
the stop time + 66 seconds) is also included.

**Fields**

=============== ========== ============================================
     Field         Type                    Description
=============== ========== ============================================
         start   Char(21)               Start time (YYYY:DDD:HH:MM:SS)
          stop   Char(21)                Stop time (YYYY:DDD:HH:MM:SS)
        tstart      Float                        Start time (CXC secs)
         tstop      Float                         Stop time (CXC secs)
           dur      Float                              Duration (secs)
 start_3tscpos    Integer                   Start TSC position (steps)
  stop_3tscpos    Integer                    Stop TSC position (steps)
     start_det    Char(6)   Start detector (ACIS-I ACIS-S HRC-I HRC-S)
      stop_det    Char(6)    Stop detector (ACIS-I ACIS-S HRC-I HRC-S)
       max_pwm    Integer                   Max PWM during translation
=============== ========== ============================================

