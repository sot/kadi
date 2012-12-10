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

- NSM
- SSM

  - Time of trip
  - CTU swap
  - Back to OBC control
  - Back to science

- BSH
- Dark Current Cals 
- IRU calibration uplinks
- swapped IRUS
- meteor shower
- gyro hold
- momentum dumps (flag for anomalous)

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

- solar array off point
- SIM motion
- grating motion
- eclipses
- CCD set point temperature changes
- guideline changes
- uplinked new dump parameters
- SCS-107 runs
- SOSA
