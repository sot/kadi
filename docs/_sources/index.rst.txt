.. kadi documentation master file, created by
   sphinx-quickstart on Fri May 10 13:30:54 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Kadi archive
================================

The Kadi archive consists of the following integrated elements which allow for
easy access and manipulation of events and commands related to the Chandra X-ray Observatory:

**Database of Chandra events**

- Events in telemetry such as maneuvers, NPM dwells, obsids, mech movements, momentum
  dumps, orbit events, etc.
- CAPs, DSN passes, dark cals, SCS107, safe modes, bright star hold, etc
- Chandra major events since launch

**Archive of Chandra commands**

- Every load command run on-board since 2002, with a link to source load products.
- Select non-load commands which result from either autonomous on-board commanding
  (e.g. SCS-107) or real-time ground commanding (e.g. anomaly recovery).
- Framework to dynamically generate the commanded state(s) of Chandra (a.k.a. load
  continuity) based on those mission loads and non-load commands.

**Python API** for accessing events and commands for analysis and using with the Ska engineering archive.

**Python tools** to maintain the Kadi database on the HEAD and Greta networks.

**Web site** for browsing events on the icxc site. This
  uses the Django web framework to provide query and administrative capabilities.

Chandra events
---------------

.. toctree::
   :maxdepth: 2

   events.rst

Commands and states
--------------------

.. toctree::
   :maxdepth: 2

   commands_states.rst

API and related docs
---------------------

.. toctree::
   :maxdepth: 1

   api
   event_descriptions
   maneuver_templates
