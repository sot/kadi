from distutils.core import setup

# from this_package.version import package_version object
from kadi.version import package_version

# Write GIT revisions and SHA tag into <this_package/git_version.py>
# (same directory as version.py)
package_version.write_git_version_file()

foundation_files = ['static/foundation/css/*',
                    'static/foundation/js/foundation/*',
                    'static/foundation/js/vendor/*',
                    'static/foundation/img/.gitkeep',
                    'static/foundation/js/foundation.min.js',
                    'static/foundation/humans.txt',
                    'static/foundation/robots.txt']

setup(name='kadi',
      version=package_version.version,
      description='Kadi events archive',
      author='Tom Aldcroft',
      author_email='aldcroft@head.cfa.harvard.edu',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/kadi/',
      packages=['kadi', 'kadi.events', 'kadi.cmds'],
      # Temporarily install static data into site-packages
      package_data={'kadi.events': ['templates/*/*.html', 'templates/*.html'],
                    'kadi': foundation_files + ['templates/*/*.html', 'templates/*.html', 
                                                'static/images/*', 'static/*.css', 
                                                'GIT_VERSION']}
)
