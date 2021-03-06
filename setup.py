# Licensed under a 3-clause BSD style license - see LICENSE.rst
from setuptools import setup
import os
import sys

foundation_files = ['static/foundation/css/*',
                    'static/foundation/js/foundation/*',
                    'static/foundation/js/vendor/*',
                    'static/foundation/img/.gitkeep',
                    'static/foundation/js/foundation.min.js',
                    'static/foundation/humans.txt',
                    'static/foundation/robots.txt']

try:
    from testr.setup_helper import cmdclass
    import testr
    print(testr.__file__)
except ImportError:
    cmdclass = {}

if "--user" not in sys.argv:
    share_path = os.path.join("share", "kadi")
    data_files = [(share_path, ['task_schedule_cmds.cfg',
                                'task_schedule_events.cfg',
                                'ltt_bads.dat'])]
else:
    data_files = None

entry_points = {'console_scripts': [
    'get_chandra_states = kadi.commands.states:get_chandra_states',
    'kadi_update_cmds = kadi.update_cmds:main',
    'kadi_update_events = kadi.update_events:main']}

setup(name='kadi',
      use_scm_version=True,
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      description='Kadi events archive',
      author='Tom Aldcroft',
      author_email='taldcroft@cfa.harvard.edu',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/kadi/',
      packages=['kadi', 'kadi.events', 'kadi.cmds', 'kadi.tests',
                'kadi.commands', 'kadi.commands.tests'],
      # Temporarily install static data into site-packages
      package_data={'kadi.events': ['templates/*/*.html', 'templates/*.html'],
                    'kadi': foundation_files + ['templates/*/*.html', 'templates/*.html',
                                                'static/images/*', 'static/*.css',
                                                'GIT_VERSION'],
                    'kadi.commands.tests': ['data/*.ecsv.gz']},
      tests_require=['pytest'],
      data_files=data_files,
      cmdclass=cmdclass,
      entry_points=entry_points,
      )
