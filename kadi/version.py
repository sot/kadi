# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Version numbering template.

This file is intended to be copied into a project and provide version information,
including git version values.  It is copied instead of used as a module to ensure
that projects can always run their setup.py with no external dependency.

The `major`, `minor`, and `bugfix` variables hold the respective parts
of the version number.  Basic usage is to copy this file into your package directory
alongside __init__.py.  Then use as follows::

  from package.version import version_object
  from package.version import __version__  # version_object.version

In addition to ``major``, ``minor``, ``bugfix``, and ``dev`` attributes, the object
``version_object`` has a number of useful attributes::

  version            Including git info if dev=True    0.5 or 0.5dev-r21-123acf1
  git_sha            Git 7-digit SHA tag               123acf1
  git_revs           Git revision count                21

See ``setup.py`` in the ``versioning`` package for an example of what needs to
be done there.  The key step is ``version_object.write_git_version_file()``
in order to create a package file ``git_version.py`` alongside ``version.py``
which retains the git information outside of the git repo.
"""

import os
import ska_helpers


class SemanticVersion(object):
    def __init__(self, package):
        """
        Semantic version object with support for git revisions

        :param package: Package name
        """
        self.version = ska_helpers.get_version(package)
        info = ska_helpers.version.parse_version(self.version)
        self.major = info['major']
        self.minor = info['minor']
        self.bugfix = info['patch']
        self.dev = info['date'] is not None
        self.git_revs = info['distance']
        self.git_sha = info['hash']
        self.version_dir = os.path.abspath(os.path.dirname(__file__))

    def write_git_version_file(self):
        """
        Make the full version with git hashtag and release from GIT_VERSION,
        typically during `python setup.py sdist`
        """
        git_version_filename = os.path.join(self.version_dir, 'GIT_VERSION')

        # Remove any existing GIT_VERSION file
        if os.path.exists(git_version_filename):
            os.unlink(git_version_filename)

        git_revs = self.git_revs
        git_sha = self.git_sha
        with open(git_version_filename, 'w') as fh:
            fh.write('{} {}'.format(git_revs, git_sha))


package_version = SemanticVersion(__package__)

__version__ = package_version.version
__git_version__ = package_version.version
VERSION = (package_version.major, package_version.minor, package_version.bugfix,
           package_version.dev)
version = __version__  # For back-compatibility with legacy version.py
