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
import os
from os.path import join, dirname, realpath

BASE_DIR = dirname(dirname(realpath(__file__)))

# Data paths for kadi project
from kadi.paths import EVENTS_DB_PATH, DATA_DIR  # noqa

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

# Application definition
INSTALLED_APPS = (
    # 'django.contrib.admin',
    # 'django.contrib.auth',
    # 'django.contrib.contenttypes',
    # 'django.contrib.sessions',
    # 'django.contrib.messages',
    # 'django.contrib.staticfiles',
    'kadi.events',
    'mica.web',
    'find_attitude.web_find_attitude',  # app label (last module) must be unique
)

# Database
# https://docs.djangoproject.com/en/1.6/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': EVENTS_DB_PATH(),
    }
}
