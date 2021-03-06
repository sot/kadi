# Licensed under a 3-clause BSD style license - see LICENSE.rst
import urllib
import shlex
import re


# Create your views here.
from django.views.generic import ListView, TemplateView, DetailView
from . import models
from .. import __version__

# Provide translation from event model class names like DarkCal to the URL name like dark_cal
MODEL_NAMES = {m_class.__name__: m_name
               for m_name, m_class in models.get_event_models().items()}


class BaseView(object):
    reverse_sort = False  # Reverse the default sort by model primary key (e.g. CAPs)

    def get_sort(self):
        sort = self.request.GET.get('sort')
        if not sort:  # No sort explicitly set in request
            sort = self.model._meta.ordering[0]
            print(('model name {} meta ordering {}'.format(self.model.__name__,
                                                           self.model._meta.ordering)))
            if self.reverse_sort:
                sort = '-' + sort
        return sort

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(BaseView, self).get_context_data(**kwargs)
        context['kadi_version'] = __version__

        event_models = models.get_event_models()
        # Make a list of tuples [(description1, name1), (description2, name2), ...]
        descr_names = [(model.__doc__.strip().splitlines()[0], name)
                       for name, model in event_models.items()]
        context['event_models'] = [{'name': name, 'description': descr}
                                   for descr, name in sorted(descr_names)]
        return context


class IndexView(BaseView, TemplateView):
    template_name = 'events/index.html'

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(IndexView, self).get_context_data(**kwargs)
        return context


class EventView(BaseView):
    """
    Mixin for common stuff between EventDetail and EventList
    """
    filter_string_field = 'start'
    filter_string_op = 'startswith'

    def filter_bare_string(self, queryset, filter_string):
        """
        Filter the ``queryset`` using ``filter_string``.

        If ``filter_string`` starts with 4 digits that looks like a date then use
        start__startswith=``filter_string``.  Otherwise use the class attributes
        (which default to the same thing).
        """
        if re.match(r'[12]\d{3}', filter_string):
            key = 'start__startswith'
        else:
            key = '{}__{}'.format(self.filter_string_field, self.filter_string_op)
        return queryset.filter(**{key: filter_string})

    def get_context_data(self, **kwargs):
        context = super(EventView, self).get_context_data(**kwargs)
        context['model_description'] = self.model.__doc__.strip().splitlines()[0]
        context['model_name'] = MODEL_NAMES[self.model.__name__]
        context['filter'] = self.request.GET.get('filter', '')
        context['sort'] = self.get_sort()

        self.formats = {field.name: getattr(field, '_kadi_format', '{}')
                        for field in self.model.get_model_fields()}

        return context

    def get_queryset(self):
        self.filter = str(self.request.GET.get('filter', ''))
        queryset = self.model.objects.all()
        tokens = shlex.split(self.filter)
        for token in tokens:
            match = re.match(r'(\w+)(=|>|<|>=|<=)([^=><]+)$', token)
            if match:
                op = {'=': 'exact',
                      '>=': 'gte',
                      '<=': 'lte',
                      '<': 'lt',
                      '>': 'gt'}[match.group(2)]
                key = '{}__{}'.format(match.group(1), op)
                val = match.group(3)
                queryset = queryset.filter(**{key: val})
            else:
                queryset = self.filter_bare_string(queryset, token)

        sort = self.get_sort()
        if sort and sort != self.model._meta.ordering[0]:
            queryset = queryset.order_by(sort)

        self.queryset = queryset
        return queryset


class EventDetail(EventView, DetailView):
    template_name = 'events/event_detail.html'

    def get_context_data(self, **kwargs):
        context = super(EventDetail, self).get_context_data(**kwargs)
        event = context['object']
        fields = self.model.get_model_fields()
        names = [field.name for field in fields]
        formats = self.formats

        context['names_vals'] = [(name, formats[name].format(getattr(event, name)))
                                 for name in names]

        try:
            obsid = event.get_obsid()
            url = '/mica/?obsid_or_date={}'.format(obsid)
            mica_link = '<a href="{}" target="_blank">{}</a>'.format(url, obsid)
        except Exception:
            mica_link = 'Unknown'
        context['mica_link'] = mica_link

        # Copy list definition properties
        next_get_params = {}
        previous_get_params = {}
        for key in ('filter', 'sort'):
            val = context[key]
            if val:
                next_get_params[key] = val
                previous_get_params[key] = val

        # If this came from a sorted list view there can be an index into the queryset.
        index = self.request.GET.get('index')
        if index is not None:
            index = int(index)
            if index > 0:
                context['previous_event'] = self.queryset[index - 1]
                previous_get_params['index'] = index - 1
            if index + 1 < self.queryset.count():
                context['next_event'] = self.queryset[index + 1]
                next_get_params['index'] = index + 1
        else:
            # Else just use primary-key based navigation
            context['next_event'] = event.get_next(self.queryset)
            context['previous_event'] = event.get_previous(self.queryset)

        context['next_get_params'] = '?' + urllib.parse.urlencode(next_get_params)
        context['previous_get_params'] = '?' + urllib.parse.urlencode(previous_get_params)

        return context


class EventList(EventView, ListView):
    paginate_by = 30
    context_object_name = 'event_list'
    template_name = 'events/event_list.html'
    ignore_fields = ['tstart', 'tstop']
    filter_help = """
<strong>Filtering help</strong>
<p><p>
Enter one or more filter criteria in the form <tt>column-name operator value</tt>,
with NO SPACE between the <tt>column-name</tt> and the <tt>value</tt>.

<p>
Examples:
<pre>
  start>2013
  start>2013:001 stop<2014:001
  start<2001 dur<=1800 n_dwell=2   [Maneuver]
</pre>

<p><p>
For some event types like <tt>MajorEvent</tt> which have one or more key
text fields, you can just enter a single word to search on.

<p>
Examples:
<pre>
  safe                            [MajorEvent]
  start>2013 eclipse              [MajorEvent]
  sequencer start>2010 stop<2011  [CAP from iFOT]
</pre>
"""

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(EventList, self).get_context_data(**kwargs)

        fields = [field for field in self.model.get_model_fields()
                  if field.name not in self.ignore_fields]
        field_names = [field.name for field in fields]

        root_url = '/kadi/events/{}/list'.format(context['model_name'])
        filter_ = context['filter']
        sort = context['sort']

        sort_icons = []
        header_classes = []
        sort_name = sort.lstrip('-')
        for field_name in field_names:
            get_params = self.request.GET.copy()
            if field_name == sort_name:
                if sort.startswith('-'):
                    icon = '<img src="/static/images/asc.gif">'
                    get_params['sort'] = field_name
                else:
                    icon = '<img src="/static/images/desc.gif">'
                    get_params['sort'] = '-' + field_name
                header_class = 'class="SortBy"'
            else:
                icon = '<img src="/static/images/asc-desc.gif">'
                get_params['sort'] = field_name
                header_class = ''
            sort_icons.append('<a href="{root_url}?{get_params}">{icon}</a>'
                              .format(root_url=root_url,
                                      get_params=urllib.parse.urlencode(get_params),
                                      icon=icon))
            header_classes.append(header_class)

        context['headers'] = [dict(header_class=x[0], field_name=x[1], sort_icon=x[2])
                              for x in zip(header_classes, field_names, sort_icons)]
        event_list = context['event_list']
        page_obj = context['page_obj']
        indices = range(page_obj.start_index() - 1, page_obj.end_index())
        context['event_rows'] = [(index, [self.formats[name].format(getattr(event, name))
                                          for name in field_names])
                                 for index, event in zip(indices, event_list)]
        context['filter'] = filter_
        context['filter_help'] = self.filter_help

        return context


# Define list view classes for each event model
# Should probably do this with code and metaclasses, but this works for now

class ObsidList(EventList):
    model = models.Obsid


class TscMoveList(EventList):
    model = models.TscMove
    filter_string_field = 'start_det'


class DarkCalReplicaList(EventList):
    model = models.DarkCalReplica
    reverse_sort = True


class DarkCalList(EventList):
    model = models.DarkCal
    reverse_sort = True


class Scs107List(EventList):
    model = models.Scs107
    reverse_sort = True


class GratingMoveList(EventList):
    model = models.GratingMove


class LoadSegmentList(EventList):
    model = models.LoadSegment
    reverse_sort = True


class FaMoveList(EventList):
    model = models.FaMove


class DumpList(EventList):
    model = models.Dump


class EclipseList(EventList):
    model = models.Eclipse


class ManvrList(EventList):
    model = models.Manvr


class DwellList(EventList):
    model = models.Dwell


class ManvrSeqList(EventList):
    model = models.ManvrSeq


class SafeSunList(EventList):
    model = models.SafeSun


class NormalSunList(EventList):
    model = models.NormalSun


class MajorEventList(EventList):
    model = models.MajorEvent
    filter_string_field = 'descr'
    filter_string_op = 'icontains'
    reverse_sort = True


class CAPList(EventList):
    model = models.CAP
    filter_string_field = 'title'
    filter_string_op = 'icontains'
    ignore_fields = EventList.ignore_fields + ['descr', 'notes', 'link']
    reverse_sort = True


class DsnCommList(EventList):
    model = models.DsnComm
    filter_string_field = 'activity'
    filter_string_op = 'icontains'
    reverse_sort = True


class OrbitList(EventList):
    model = models.Orbit
    ignore_fields = EventList.ignore_fields + ['t_perigee']


class OrbitPointList(EventList):
    model = models.OrbitPoint


class RadZoneList(EventList):
    model = models.RadZone


class LttBadList(EventList):
    model = models.LttBad


####
class ObsidDetail(EventDetail):
    model = models.Obsid


class TscMoveDetail(EventDetail):
    model = models.TscMove
    filter_string_field = 'start_det'


class DarkCalReplicaDetail(EventDetail):
    model = models.DarkCalReplica


class DarkCalDetail(EventDetail):
    model = models.DarkCal


class Scs107Detail(EventDetail):
    model = models.Scs107


class GratingMoveDetail(EventDetail):
    model = models.GratingMove


class LoadSegmentDetail(EventDetail):
    model = models.LoadSegment


class FaMoveDetail(EventDetail):
    model = models.FaMove


class DumpDetail(EventDetail):
    model = models.Dump


class EclipseDetail(EventDetail):
    model = models.Eclipse


class ManvrDetail(EventDetail):
    model = models.Manvr


class DwellDetail(EventDetail):
    model = models.Dwell


class ManvrSeqDetail(EventDetail):
    model = models.ManvrSeq


class SafeSunDetail(EventDetail):
    model = models.SafeSun


class NormalSunDetail(EventDetail):
    model = models.NormalSun


class MajorEventDetail(EventDetail):
    model = models.MajorEvent
    filter_string_field = 'descr'
    filter_string_op = 'icontains'


class CAPDetail(EventDetail):
    model = models.CAP
    filter_string_field = 'title'
    filter_string_op = 'icontains'


class DsnCommDetail(EventDetail):
    model = models.DsnComm
    filter_string_field = 'activity'
    filter_string_op = 'icontains'


class OrbitDetail(EventDetail):
    model = models.Orbit


class OrbitPointDetail(EventDetail):
    model = models.OrbitPoint


class RadZoneDetail(EventDetail):
    model = models.RadZone


class PassPlanList(EventList):
    model = models.PassPlan
    reverse_sort = True


class PassPlanDetail(EventDetail):
    model = models.PassPlan


class LttBadDetail(EventDetail):
    model = models.LttBad
