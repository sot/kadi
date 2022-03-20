Commands archive v2 details
===========================

Concept overview
----------------

The key concept underlying version 2 is that it uses web resources to always
provide the most current set of executed and planned load and non-load commands.
This is true even in rapidly changing circumstances such as anomaly recovery or
a fast TOO. The code provides correct results without need for the user to worry
about syncing any files.

The `Chandra Command Events
<https://docs.google.com/spreadsheets/d/19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ/edit#gid=0>`_
Google sheet is the foundation of this infrastructure. It provides a centralized
repository which contains information about "command events" that impact the
as-run commanding on Chandra. This document is viewable by anyone with the link
but can be edited only by FOT mission planning, Flight Directors and a small set
of managers. This spreadsheet will be maintained by FOT MP in a timely manner
(typically within one hour during anomalies) following a `defined process
<https://occweb.cfa.harvard.edu/twiki/bin/view/MissionPlanning/CommandEvents>`_.

The other key web resource is the OCCweb `FOT mission planning approved load products
<https://occweb.cfa.harvard.edu/occweb/FOT/mission_planning/PRODUCTS/APPR_LOADS/>`_
directory tree. This is used to automatically find all recent approved loads
and incorporate them into the load commands archive.

Differences from v1
-------------------

Apart from the fundamental change in data sources mentioned above, some key
differences from v1 are as follows:

- Commands table includes a ``source`` column that defines the source of the
  command. Most commonly this is a weekly load name, but it can also indicate
  a non-load command event for which further details are provided in the command
  parameters.
- Information about each distinct observation is embedded into the command
  archive as `LOAD_EVT` pseudo-commands. The
  :func:`~kadi.commands.observations.get_observations` provides a fast and
  convenient way to find observations, both past and planned. See the
  `Getting observations` section for more details.
- Information about each ACA star catalog is stored in the command
  archive. The :func:`~kadi.commands.observations.get_observations` provides a
  convenient way to find ACA star catalogs, both past and planned. See the
  `Getting star catalogs`_ section for more details.
- There are configuration options which can be set programmatically or in a fixed
  configuration file to control behavior of the package. See the
  `Configuration options`_ section for more details.

Getting observations
--------------------

Getting star catalogs
---------------------

Configuration options
---------------------
