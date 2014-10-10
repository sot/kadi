from django.conf.urls import patterns, include, url
from django.contrib import admin

import kadi.views
import mica.web.views

admin.autodiscover()

urlpatterns = patterns('',
                       url(r'^admin/doc/', include('django.contrib.admindocs.urls')),
                       url(r'^admin/', include(admin.site.urls)),
                       url(r'^kadi/events/', include('kadi.events.urls')),
                       url(r'^$', kadi.views.IndexView.as_view()),
                       url(r'^mica/$', mica.web.views.IndexView.as_view()),
                       )

# Another way to do this, corresponds to commented code in kadi/views.py
# urlpatterns = patterns('',
#                        url(r'^admin/doc/', include('django.contrib.admindocs.urls')),
#                        url(r'^admin/', include(admin.site.urls)),
#                        url(r'^mica/$', mica.web.views.IndexView.as_view()),
#                        url(r'^kadi/events/', include('kadi.events.urls')),

#                        # Eventually consider forcing the kadi page to be the default page if other
#                        # urls fail to match. For now keep out the ".*" for debug purposes.
#                        url(r'^$', kadi.views.index, name='index'),
#                        )
