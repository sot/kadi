# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
from django.conf import settings
from django.utils.functional import empty

# If we are running standalone then this ENV var is not set.  Need to explicitly
# do the setup and then import query module attributes for standalone.
# For WSGI server the env var is set in wsgi.py.
# For the dev server it is set in manage.py.

# In addition, set DJANGO_ALLOW_ASYNC_UNSAFE, to avoid exception seen running in
# Jupyter notebook: SynchronousOnlyOperation: You cannot call this from an async
# context. See: https://stackoverflow.com/questions/59119396

if settings._wrapped is empty:
    if 'DJANGO_SETTINGS_MODULE' not in os.environ:
        os.environ['DJANGO_SETTINGS_MODULE'] = 'kadi.settings'
    os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"
    django.setup()
    from .query import *  # noqa
