.. kadi documentation master file, created by
   sphinx-quickstart on Fri May 10 13:30:54 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Kadi archive
============

The Kadi archive consists of the following two elements which allow for easy access and
manipulation of events and commands related to the Chandra X-ray Observatory:

**Chandra events from telemetry and web resources**

- Events in telemetry such as maneuvers, NPM dwells, obsids, mech movements, momentum
  dumps, orbit events, and so forth.
- Events from web resources, for instance CAPs, DSN passes, Load segments, and Chandra
  major events.

**Chandra commands and states**

- Every load command run on-board since 2002, with a link to source load products.
- Non-load commands which result from either autonomous on-board commanding
  (e.g. SCS-107) or real-time ground commanding (e.g. anomaly recovery).
- Commanded state(s) of Chandra based on mission loads and non-load commands.

.. toctree::
   :hidden:

   events/index
   commands_states
   api/index
