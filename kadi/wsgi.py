# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
WSGI config for kadi project.

This module contains the WSGI application used by Django's development server
and any production WSGI deployments. It should expose a module-level variable
named ``application``. Django's ``runserver`` and ``runfcgi`` commands discover
this application via the ``WSGI_APPLICATION`` setting.

Usually you will have the standard Django WSGI application here, but it also
might make sense to replace the whole Django WSGI application with a custom one
that later delegates to the Django one. For example, you could introduce WSGI
middleware here, or combine a Django application with an application of another
framework.

"""
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kadi.settings")

# When running production web site there should be no HOME, but
# some packages (e.g. sherpa) assume its existence.
os.environ.setdefault("HOME", "/tmp")

# Need SYBASE for some apps like star history.
# TODO: remove dependence on sybase, use mica guide / acq stats instead.
os.environ.setdefault("SYBASE", "/soft/SYBASE15.7")

# This application object is used by any WSGI server configured to use this
# file. This includes Django's development server, if the WSGI_APPLICATION
# setting points here.
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# Apply WSGI middleware here.
# from helloworld.wsgi import HelloWorldApplication
# application = HelloWorldApplication(application)
