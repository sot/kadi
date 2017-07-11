from django.conf.urls import patterns, include, url
from django.contrib import admin

import kadi.views

admin.autodiscover()

urlpatterns = patterns('',
                       url(r'^admin/doc/', include('django.contrib.admindocs.urls')),
                       url(r'^admin/', include(admin.site.urls)),
                       url(r'^kadi/events/', include('kadi.events.urls')),
                       url(r'^$', kadi.views.IndexView.as_view())
                       )

# Other apps from mica, find_attitude and eng_archive are optional
try:
    import mica.web.views
except ImportError:
    pass
else:
    urlpatterns += patterns('',
                            url(r'^mica/$', mica.web.views.IndexView.as_view()),
                            url(r'^pcad_acq/$', mica.web.views.AcqView.as_view()),
                            url(r'^star_hist/$', mica.web.views.StarHistView.as_view()))

try:
    import find_attitude.web.views
except ImportError:
    pass
else:
    urlpatterns += patterns('',
                            url(r'^find_attitude/$', find_attitude.web.views.index))

try:
    import Ska.engarchive.web
except ImportError:
    pass
else:
    urlpatterns += patterns('',
                            url(r'^eng_archive/remote_func/$', Ska.engarchive.web.remote_func))

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
