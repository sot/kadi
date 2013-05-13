.. kadi documentation master file, created by
   sphinx-quickstart on Fri May 10 13:30:54 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Kadi event archive
================================

The Kadi event archive consists of the following integrated elements which allow for
easy access and manipulation of events related to the Chandra X-ray Observatory:

**Database of Chandra events**
  - Events in telemetry such as maneuvers, NPM dwells, obsids, mech movements, momentum
    dumps, orbit events, etc.
  - CAPs, DSN passes, dark cals, SCS107, safe modes, bright star hold, etc
  - Chandra major events since launch
  - Every load command run on-board since 2002, with a link to source load products
  - The database is contained in easily transportable sqlite3 or HDF5 files.

**Python API** for accessing events for analysis and using with the Ska engineering archive

**Python tools** to maintain the Kadi database on the HEAD and Greta networks

**Web site** for browsing events on the icxc site or by a localhost server on GRETA. This
  will use the Django web framework to provide query and administrative capabilities  (*Coming soon!*).

**RESTful web service API** on the icxc site  (*Coming soon!*).


Events
--------

Event definitions
^^^^^^^^^^^^^^^^^^^

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

