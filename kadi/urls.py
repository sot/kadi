# Licensed under a 3-clause BSD style license - see LICENSE.rst
from django.conf.urls import include, url
from django.contrib import admin

import kadi.views
import mica.web.views
import find_attitude.web_find_attitude.views

admin.autodiscover()

urlpatterns = [  # '',
    # url(r'^admin/', include(admin.site.urls)),
    url(r'^kadi/events/', include('kadi.events.urls')),
    url(r'^$', kadi.views.IndexView.as_view()),
    url(r'^mica/$', mica.web.views.IndexView.as_view()),
    url(r'^pcad_acq/$', mica.web.views.AcqView.as_view()),
    url(r'^star_hist/$', mica.web.views.StarHistView.as_view()),
    url(r'^find_attitude/$', find_attitude.web_find_attitude.views.index),
]
