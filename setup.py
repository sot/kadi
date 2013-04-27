from distutils.core import setup

from kadi.version import version

setup(name='kadi',
      version=version,
      description='Kadi events archive',
      author='Tom Aldcroft',
      author_email='aldcroft@head.cfa.harvard.edu',
      url='http://www.python.org/',
      packages=['kadi', 'kadi.events', 'kadi.cmds'],
      )
