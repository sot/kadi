import shlex
import re

# Create your views here.
from django.views.generic import ListView, TemplateView, DetailView
from . import models

# Provide translation from event model class names like DarkCal to the URL name like dark_cal
MODEL_NAMES = {m_class.__name__: m_name
               for m_name, m_class in models.get_event_models().items()}


class IndexView(TemplateView):
    template_name = 'events/index.html'

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(IndexView, self).get_context_data(**kwargs)

        event_models = models.get_event_models()
        # Make a list of tuples [(description1, name1), (description2, name2), ...]
        descr_names = [(model.__doc__.strip().splitlines()[0], name)
                       for name, model in event_models.items()]
        context['event_models'] = [{'name': name, 'description': descr}
                                   for descr, name in sorted(descr_names)]
        return context


class EventView(object):
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
        context['filter'] = self.filter

        self.formats = {field.name: getattr(field, '_kadi_format', '{}')
                        for field in self.model.get_model_fields()}

        return context

    def get_queryset(self):
        self.filter = str(self.request.GET.get('filter', ''))
        queryset = self.model.objects.all()
        tokens = shlex.split(self.filter)
        for token in tokens:
            match = re.match(r'(\w+)(=|>|<|>=|<=)(.+)$', token)
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
        context['next_event'] = event.get_next(self.queryset)
        context['previous_event'] = event.get_previous(self.queryset)

        return context


class EventList(EventView, ListView):
    paginate_by = 30
    context_object_name = 'event_list'
    template_name = 'events/event_list.html'
    ignore_fields = ['tstart', 'tstop']

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(EventList, self).get_context_data(**kwargs)

        fields = self.model.get_model_fields()
        context['field_names'] = [field.name for field in fields
                                  if field.name not in self.ignore_fields]
        event_list = context['event_list']
        formats = self.formats
        context['event_rows'] = [[formats[name].format(getattr(event, name))
                                  for name in context['field_names']]
                                 for event in event_list]
        context['filter'] = self.request.GET.get('filter', '')

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


class DarkCalList(EventList):
    model = models.DarkCal


class Scs107List(EventList):
    model = models.Scs107


class GratingMoveList(EventList):
    model = models.GratingMove


class LoadSegmentList(EventList):
    model = models.LoadSegment


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


class CAPList(EventList):
    model = models.CAP
    filter_string_field = 'title'
    filter_string_op = 'icontains'
    ignore_fields = EventList.ignore_fields + ['descr', 'notes', 'link']


class DsnCommList(EventList):
    model = models.DsnComm
    filter_string_field = 'activity'
    filter_string_op = 'icontains'


class OrbitList(EventList):
    model = models.Orbit
    ignore_fields = EventList.ignore_fields + ['t_perigee']


class OrbitPointList(EventList):
    model = models.OrbitPoint


class RadZoneList(EventList):
    model = models.RadZone

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
