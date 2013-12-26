import inspect

from django.conf.urls import patterns, url

from . import views

# Consider using a dispatch function based on a pattern that matches model name.  But just
# get this stupid version working first.  This makes a url() object for each EventList view.
urls = []
for name, val in vars(views).items():
    if inspect.isclass(val) and issubclass(val, views.EventList):
        # val is an event ListView class.  Chop off the "List" at end to get model class name
        EventListView = val
        model_class_name = EventListView.__name__[:-4]
        try:
            urls.append(url('^{}/list/$'.format(views.MODEL_NAMES[model_class_name]),
                            EventListView.as_view()))
        except KeyError:
            pass

urlpatterns = patterns('',
                       url('^$', views.IndexView.as_view()),
                       *urls)
