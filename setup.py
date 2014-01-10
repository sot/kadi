from distutils.core import setup

from kadi.version import version

setup(name='kadi',
      version=version,
      description='Kadi events archive',
      author='Tom Aldcroft',
      author_email='aldcroft@head.cfa.harvard.edu',
      url='http://www.python.org/',
      packages=['kadi', 'kadi.events', 'kadi.cmds'],
      # Temporarily install static data into site-packages
      package_data={'kadi.events': ['templates/*/*.html', 'templates/*.html'],
                    'kadi': ['static/images/*', 'static/*.css']},
      )
