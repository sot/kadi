from distutils.core import setup

# from this_package.version import package_version object
from kadi.version import package_version

# Write GIT revisions and SHA tag into <this_package/git_version.py>
# (same directory as version.py)
package_version.write_git_version_file()

setup(name='kadi',
      version=package_version.version,
      description='Kadi events archive',
      author='Tom Aldcroft',
      author_email='aldcroft@head.cfa.harvard.edu',
      url='http://cxc.harvard.edu/mta/ASPECT/tool_doc/kadi/',
      zip_safe=False,
      packages=['kadi', 'kadi.events', 'kadi.cmds'],
      # Temporarily install static data into site-packages
      package_data={'kadi.events': ['templates/*/*.html', 'templates/*.html'],
                    'kadi': ['static/images/*', 'static/*.css', 'GIT_VERSION',
'static/foundation/css/foundation.css',
'static/foundation/css/foundation.min.css',
'static/foundation/css/normalize.css',
'static/foundation/humans.txt',
'static/foundation/img/.gitkeep',
'static/foundation/index.html',
'static/foundation/js/foundation.min.js',
'static/foundation/js/foundation/foundation.abide.js',
'static/foundation/js/foundation/foundation.accordion.js',
'static/foundation/js/foundation/foundation.alert.js',
'static/foundation/js/foundation/foundation.clearing.js',
'static/foundation/js/foundation/foundation.dropdown.js',
'static/foundation/js/foundation/foundation.equalizer.js',
'static/foundation/js/foundation/foundation.interchange.js',
'static/foundation/js/foundation/foundation.joyride.js',
'static/foundation/js/foundation/foundation.js',
'static/foundation/js/foundation/foundation.magellan.js',
'static/foundation/js/foundation/foundation.offcanvas.js',
'static/foundation/js/foundation/foundation.orbit.js',
'static/foundation/js/foundation/foundation.reveal.js',
'static/foundation/js/foundation/foundation.slider.js',
'static/foundation/js/foundation/foundation.tab.js',
'static/foundation/js/foundation/foundation.tooltip.js',
'static/foundation/js/foundation/foundation.topbar.js',
'static/foundation/js/vendor/fastclick.js',
'static/foundation/js/vendor/jquery.cookie.js',
'static/foundation/js/vendor/jquery.js',
'static/foundation/js/vendor/modernizr.js',
'static/foundation/js/vendor/placeholder.js',
'static/foundation/robots.txt',
]},
      )
