"""
Django settings for kadi project.

For more information on this file, see
https://docs.djangoproject.com/en/1.6/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/1.6/ref/settings/
"""

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
import os
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# Data paths for kadi project
from .paths import EVENTS_DB_PATH

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.6/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'q8&0dj7olx+p)#2#$xe@jwrz$ds8=im$a815t67oce_z14rted'

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
    os.path.join(BASE_DIR, 'static'),
    )

# Make this unique, and don't share it with anybody.  Do I need this?
# SECRET_KEY = 'd7m#*urgc4#1gg5pq)j@8a__$+=5@h82s_31une+$&amp;fy16#0no'

# Is this needed?
# TEMPLATE_DIRS = (
#     # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
#     # Always use forward slashes, even on Windows.
#     # Don't forget to use absolute paths, not relative paths.
#     os.path.join(BASE_DIR, 'events/templates'),
# )
