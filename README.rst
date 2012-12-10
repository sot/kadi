Telemetry Events
================

http://occweb.cfa.harvard.edu/twiki/Aspect/StarWorkingGroupMeeting2012x03x21

Envision multiple tables (maneuvers, BSH, NSM, etc), each of which has "default"
times that can be used for filtering data.

Maneuver events
----------------
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
^^^^^^^^^^^^^^^^^^
- AOPCADMD
- AOFATTMD
- AOACASEQ
- AOPSACPR
- AOUNLOAD

Maneuver Attributes
^^^^^^^^^^^^^^^^^^^^

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
-------------------------

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
  - return to science time
  - cause?  (may be too difficult)

    - should this only include SCS 107 "only" safing events (e.g. Radmon)?  Or all (e.g. SSM, NSM)?

- momentum dumps 

  - PCAD mode  (AOPCADMD)
  - 1-shot  (RSS of AOATTERx)
  - system momentum (AOSYMOMx)
  - pulse counts (AOTHRSTx)
  - VDE used (AOVDESEL)
  - thruster temps (PMxTHV1T and PMxTHV2T)
  - slightly-more detailed calculations:

    - thruster efficiency 
    - duty cycles 
    - warm start count
    - fuel flow rate
    - fuel used
    - tank pressure
    - flag for anomalous

- Eclipses

- SIM motion

- Grating motion

- Dark Current Cals 

  - Start and stop time
  
    - But will split replicas show up as two?  Is this preferable, or should each replica be called out indiv?

- IRU calibration uplink

  - time

- CCD set point temperature changes

  - time
  - set point

- gyro holds

  - start and stop times

- meteor showers
  
  - type (based on date)
  - start and stop times (from Brent or strictly by date)

- guideline changes


- Misc events

  - SOSA uplink
  - uplinked new dump parameters
  - swapped IRUS
  - solar array off-point (usually eclipses or meteor showers)