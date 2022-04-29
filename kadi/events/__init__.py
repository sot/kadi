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
import importlib

# In addition, set DJANGO_ALLOW_ASYNC_UNSAFE, to avoid exception seen running in
# Jupyter notebook: SynchronousOnlyOperation: You cannot call this from an async
# context. See: https://stackoverflow.com/questions/59119396

os.environ['DJANGO_SETTINGS_MODULE'] = 'kadi.settings'
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

# Below is a list of every event query function + the `kadi.events.models`
# module that is imported in `kadi.events.query`.
#
# There is a little subtlety here. See discussion in
# https://github.com/sot/kadi/pull/231, but the upshot is that we need
# `from kadi.events import models` to run `django.setup()`. This syntax is what
# gets used in `update_events.py`. By including 'models' in `__all__` that
# imports it via `query.py` and thus runs django setup.
#
# `manage.py` imports the same module but runs `django.setup()` before importing
# it. However, it appears to import this via the equivalent of
# `import kadi.events.models`, which does NOT go through `query.py`.
# With this trick of lazy loading, both pathways work!

__all__ = [
    'models',
    'obsids',
    'tsc_moves',
    'dark_cal_replicas',
    'dark_cals',
    'scs107s',
    'fa_moves',
    'grating_moves',
    'dumps',
    'eclipses',
    'manvrs',
    'dwells',
    'safe_suns',
    'normal_suns',
    'major_events',
    'caps',
    'load_segments',
    'pass_plans',
    'dsn_comms',
    'orbits',
    'rad_zones',
    'ltt_bads'
]


def __getattr__(name):
    """
    Get the attribute from the query module.
    """
    if name in __all__:
        query = importlib.import_module('kadi.events.query')
        out = globals()[name] = getattr(query, name)
        return out
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
