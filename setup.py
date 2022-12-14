# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import sys

from setuptools import setup

foundation_files = [
    "static/foundation/css/*",
    "static/foundation/js/foundation/*",
    "static/foundation/js/vendor/*",
    "static/foundation/img/.gitkeep",
    "static/foundation/js/foundation.min.js",
    "static/foundation/humans.txt",
    "static/foundation/robots.txt",
]

try:
    import testr
    from testr.setup_helper import cmdclass

    print(testr.__file__)
except ImportError:
    cmdclass = {}

if "--user" not in sys.argv:
    share_path = os.path.join("share", "kadi")
    data_files = [
        (
            share_path,
            ["task_schedule_cmds.cfg", "task_schedule_events.cfg", "ltt_bads.dat"],
        )
    ]
else:
    data_files = None

entry_points = {
    "console_scripts": [
        "get_chandra_states = kadi.commands.states:get_chandra_states",
        "kadi_update_cmds = kadi.scripts.update_cmds_v1:main",
        "kadi_update_cmds_v2 = kadi.scripts.update_cmds_v2:main",
        "kadi_update_events = kadi.scripts.update_events:main",
        "kadi_validate_states = kadi.scripts.validate_states:main",
    ]
}

setup(
    name="kadi",
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    description="Kadi command and events archive",
    author="Tom Aldcroft",
    author_email="taldcroft@cfa.harvard.edu",
    url="http://cxc.harvard.edu/mta/ASPECT/tool_doc/kadi/",
    packages=[
        "kadi",
        "kadi.events",
        "kadi.cmds",
        "kadi.scripts",
        "kadi.tests",
        "kadi.commands",
        "kadi.commands.tests",
    ],
    # Temporarily install static data into site-packages
    package_data={
        "kadi.events": ["templates/*/*.html", "templates/*.html"],
        "kadi.commands": ["templates/*.html"],
        "kadi": foundation_files
        + [
            "templates/*/*.html",
            "templates/*.html",
            "static/images/*",
            "static/*.css",
            "GIT_VERSION",
        ],
        "kadi.commands.tests": ["data/*.ecsv.gz"],
    },
    tests_require=["pytest"],
    data_files=data_files,
    cmdclass=cmdclass,
    entry_points=entry_points,
)
