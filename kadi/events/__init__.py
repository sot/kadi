"""\
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
"""

import os
import django

if 'DJANGO_SETTINGS_MODULE' not in os.environ:
    os.environ['DJANGO_SETTINGS_MODULE'] = 'kadi.settings'

# In django >= 1.8 an explicit call to setup is required for standalone.
# This attribute does not exist in 1.6.
# https://docs.djangoproject.com/en/1.10/topics/settings/#calling-django-setup-is-required-for-standalone-django-usage
try:
    django.setup()
except AttributeError:
    pass

# from .models import *
from .query import *
