from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
import kadi.views
import mica.web.views
admin.autodiscover()

urlpatterns = patterns('',
                       # Examples:
                       # url(r'^$', 'kadi.views.home', name='home'),
                       # url(r'^kadi/', include('kadi.foo.urls')),
                       # Uncomment the admin/doc line below to enable admin documentation:
                       url(r'^admin/doc/', include('django.contrib.admindocs.urls')),
                       # Uncomment the next line to enable the admin:
                       url(r'^admin/', include(admin.site.urls)),
                       url(r'^kadi/events/', include('kadi.events.urls')),
                       url(r'^$', kadi.views.IndexView.as_view()),
                       url(r'^mica/$', mica.web.views.IndexView.as_view()),
                       )
