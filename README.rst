Kadi
====

This package maintains and provides access to an online archive of many Chandra
operational events, including:

- Events in telemetry such as maneuvers, NPM dwells, obsids, mech movements,
  momentum dumps, orbit events, etc.
- CAPs, DSN passes, dark cals, SCS107, safe modes, bright star hold, etc
- Chandra major events since launch
- Every load command run on-board since 2002, with a link to source load products

The contents of the Kadi archive will be accessible on the HEAD and GRETA networks in any
of three ways:

- From Python using a query API for processing and analysis.  This is be
  based largely on the Django object relational model that provides an abstraction
  layer over the SQL database backend.
- Via a web browser application on the icxc site or by a localhost server on GRETA.  This
  will use the Django web framework to provide query and administrative capabilities.
- Directly via the SQL database using any convenient method.

Another possibility would be a RESTful web service API.


Telemetry Events
-----------------

http://occweb.cfa.harvard.edu/twiki/Aspect/StarWorkingGroupMeeting2012x03x21

Envision multiple tables (maneuvers, BSH, NSM, etc), each of which has "default"
times that can be used for filtering data.

Maneuver events
^^^^^^^^^^^^^^^^^
- Define maneuver characteristics.
- Define "acceptable" maneuvers (for use in IRU calibration).

===== ===========================================================================
Event Sequence for Maneuvers:
===== ===========================================================================
t0    previous AOFATTMD = STDY
t1    previous NMAN --> NPNT transition time
t2    NPNT --> NMAN transition time
tref1 AOFATTMD = MNVR (AOMANUVR time)
tref2 AOFATTMD = STDY
t3    NMAN --> NPNT transition time (first CEBR check passes or times out)
t4    AOPSACPR changes from INIT to ACT (second CEBR check passes or times out)
t5    AOACASEQ changes from ACQ to GUID / 1-shot (acquisition star success)
t6    AOACASEQ changes from GUID to KALM (guide star success)
t7    NPNT --> NMAN transition time
t8    AOFATTMD = MNVR (AOMANUVR time)
===== ===========================================================================

MSIDs of Interest
~~~~~~~~~~~~~~~~~~
- AOPCADMD
- AOFATTMD
- AOACASEQ
- AOPSACPR
- AOUNLOAD

Maneuver Attributes
~~~~~~~~~~~~~~~~~~~~~

- AOATTQT1..AOATTQT4
- AOTARQT1..AOTARQT4
- AOATUPQ1..AOATUPQ3
- AOATTER1..AOATTER3
- AOGYRCT1..AOGYRCT4
- AOGBIAS1..AOGBIAS4
- Maneuver angle
- Eigenaxis
- Maneuver type
- 1-shot magnitude
- Max acceleration?
- Max velocity?
- Flag for "unusual"?

Other telemetry event tables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- SSM

  - Time of trip
  - CTU swap
  - Back to OBC control / NSM transition
  - NPM transition (BSH)
  - NPM transition (on guide stars)
  - Back to science

- NSM

  - Time of trip
  - NPM transition (BSH)
  - NPM transition (on guide stars)
  - Back to science

- BSH

  - Time of trip
  - NPM transition (BSH)
  - NPM transition (on guide stars)
  - Back to science

- SCS 107 runs

  - SCS 107 execution start and stop time
  - Return to science time
  - Cause?  (may be too difficult)

    - Should this only include SCS 107 "only" safing events (e.g. Radmon)?  Or all (e.g. SSM, NSM)?

- Momentum dumps 

  - PCAD mode  (AOPCADMD)
  - 1-shot  (RSS of AOATTERx)
  - System momentum (AOSYMOMx)
  - Pulse counts (AOTHRSTx)
  - VDE used (AOVDESEL)
  - Thruster temps (PMxTHV1T and PMxTHV2T)
  - Slightly-more detailed calculations:

    - Thruster efficiency 
    - Duty cycles 
    - Warm start count
    - Fuel flow rate
    - Fuel used
    - Tank pressure
    - Flag for anomalous

- Eclipses

  - Penumbra start and stop time
  - Umbra start and stop time
  - Discharge current
  - Charge current
  - Relay status

- SIM motion

  - Start and stop time
  - Starting and ending position
  - Convert counts into instrument?

- Grating motion

  - Start and stop times
  - Starting and ending grating

- Dark Current Cals 

  - Start and stop time
  
    - But will split replicas show up as two?  Is this preferable, or should each replica be called out indiv?

- IRU calibration uplink

  - Time

- CCD set point temperature changes

  - Time
  - Set point

- Gyro holds

  - Start and stop times

- Meteor showers

  - Type (based on date)
  - Start and stop times (from Brent or strictly by date)


Events with iFOT heritage
-------------------------

Certain iFOT tables will be synced into the Kadi event archive:

- CAPs
- DSN passes
- Load segment

The information within many other iFOT tables will be available in Kadi,
but with an intent for a higher degree of completeness.  For instance
the iFOT radiation zone table is missing times when there was no RADMON
commanding, and the ACA dark calibration is missing events in 2005 and 2006.

Major Events
--------------------

There are two major event tables that are maintained and available on OCCweb:

- The mission event history available from the FDB page
- The major event table available within the FOT engineering area

Kadi merges these two into a common format and makes them available for query.

SCS load commands
-----------------

Using the history of load segment runtimes maintained in the Ska commanded states
processing, Kadi maintains a database of all commands that were run via weekly load
commanding since 2002.  (Prior to that the data for precisely which load segments were run
over which times is not available in the commanded states history).  This database
can be quickly loaded and searched by time-based or attribute-based queries.
