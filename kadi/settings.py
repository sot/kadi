# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Django settings for kadi project.

For more information on this file, see
https://docs.djangoproject.com/en/1.6/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.6/ref/settings/
"""
from __future__ import print_function

# Build paths inside the project like this: join(BASE_DIR, ...)
import sys
import os
from os.path import join, dirname, realpath
os.environ.setdefault('SKA', '/proj/sot/ska')

# Check if this instance is running in the production Apache kadi web server.
# If so set XDG_CONFIG_HOME and friend to make sure that the astropy config is
# stable for production and is not coming from the config in the user home dir.
# This will put the config file in $SKA/data/kadi/config.
#
# This code relies on the fact that currently in this case (production) kadi is
# installed into /proj/web-kadi and conf/httpd.conf contains: WSGIPythonPath
# /proj/web-kadi/lib/python2.7/site-packages.
#
# See also
# https://stackoverflow.com/questions/26979579/django-mod-wsgi-set-os-environment-variable-from-apaches-setenv
# for useful commentary on problems just using SetEnv in the conf file.  Also
# search email for XDG_CONFIG_HOME for more discussion and motivation.

if any(pth.startswith('/proj/web-kadi') for pth in sys.path):
    os.environ.setdefault('XDG_CONFIG_HOME',
                          join(os.environ['SKA'], 'data', 'config'))
    os.environ.setdefault('XDG_CACHE_HOME', os.environ['XDG_CONFIG_HOME'])

BASE_DIR = dirname(dirname(realpath(__file__)))

# Data paths for kadi project
from .paths import EVENTS_DB_PATH, DATA_DIR

# Make sure there is an events database
if not os.path.exists(EVENTS_DB_PATH()):
    import warnings
    message = ('\n\n'
               '***************************************'
               '\n\n'
               'Events database file {} not found.  \n'
               'Most likely this is not what you want since no events\n'
               'will be found. If you are running in a test or standalone\n'
               'Ska environment then you may need to set the KADI environment variable\n'
               'to point to a directory like /proj/sot/ska/data/kadi that has an\n'
               'events.db3 file.\n\n'
               '***************************************'.format(EVENTS_DB_PATH()))
    warnings.warn(message)

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.6/howto/deployment/checklist/

_secret_file = join(DATA_DIR(), 'secret_key.txt')
try:
    with open(_secret_file) as fh:
        SECRET_KEY = fh.read().strip()

except IOError:
    import random
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)'
    SECRET_KEY = ''.join([random.SystemRandom().choice(chars) for i in range(50)])
    try:
        with open(_secret_file, 'w') as fh:
            fh.write(SECRET_KEY)
        print('Created secret key file {}'.format(_secret_file))

    except IOError:
        pass  # Running as a non-production instance, don't worry about secret key

    else:
        try:
            import stat
            os.chmod(_secret_file, stat.S_IRUSR)
            print('Changed file mode to owner read-only')
        except Exception:
            import warnings
            warnings.warn('Unable to change file mode permission!')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

TEMPLATE_DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = (
    'django.contrib.admin',
    # 'django.contrib.admindocs',  # ?? needed?  Lost in 1.6
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'kadi.events',
)

OPTIONAL_APPS = ('mica.web', 'find_attitude.web')
for app in OPTIONAL_APPS:
    try:
        __import__(app)
    except ImportError:
        pass
    else:
        INSTALLED_APPS += (app,)


MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
)

ROOT_URLCONF = 'kadi.urls'

WSGI_APPLICATION = 'kadi.wsgi.application'


# Database
# https://docs.djangoproject.com/en/1.6/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': EVENTS_DB_PATH(),
    }
}

# FORCE_SCRIPT_NAME = '/kadi'
APPEND_SLASH = True

# Internationalization
# https://docs.djangoproject.com/en/1.6/topics/i18n/

LANGUAGE_CODE = 'en-us'

# TIME_ZONE = 'UTC'
TIME_ZONE = 'America/New_York'

USE_I18N = True

USE_L10N = True

USE_TZ = False


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.6/howto/static-files/

STATIC_URL = '/static/'

# OTHER Kadi customizations

# Django admin static files are installed directly from the web-kadi repo via
#   make install_admin_static

STATICFILES_DIRS = (
    #  Put strings here, like "/home/html/static" or "C:/www/django/static".
    #  Always use forward slashes, even on Windows.
    #  Don't forget to use absolute paths, not relative paths.
    #
    #  NOTE: It's possible to not define any STATICFILES_DIRS if the static
    #        files are located in directories like kadi/events/static/.  Then
    #        the staticfiles app will find them.  (E.g. /static/kadi.css would
    #        be at kadi/events/static/kadi.css.)
    #
    #  The following is for project-wide static files in kadi/static/.
    join(BASE_DIR, 'kadi/static'),
)

TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
    join(BASE_DIR, 'kadi/templates'),
)
