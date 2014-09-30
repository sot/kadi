from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
import kadi.views
import events.views
import mica.web.views
admin.autodiscover()


urlpatterns = patterns('',
                       url(r'^admin/doc/', include('django.contrib.admindocs.urls')),
                       url(r'^admin/', include(admin.site.urls)),
                       url(r'^mica/$', mica.web.views.IndexView.as_view()),
                       url(r'^kadi/events/', include('events.urls')),

                       # Force the kadi page to be the default page if other urls fail to match.
                       url(r'^$', kadi.views.index, name='index'),
                       )
