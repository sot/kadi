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

- Misc events (ideally this would be the only table with manual inputs)

  - SOSA uplink
  - Uplinked new dump parameters
  - Swapped IRUS
  - FSS-B turn-on
  - Solar array off-point (usually eclipses or meteor showers)
  - Guideline changes
  - Several more based on Paul's event table in quarterly report