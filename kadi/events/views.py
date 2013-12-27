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


class EventDetail(DetailView):
    template_name = 'events/event_detail.html'

    def get_context_data(self, **kwargs):
        context = super(EventDetail, self).get_context_data(**kwargs)
        event = context['object']
        fields = self.model._meta.fields
        names = [field.name for field in fields]
        formats = {field.name: getattr(field, '_kadi_format', '{}')
                   for field in fields}
        context['names_vals'] = [(name, formats[name].format(getattr(event, name)))
                                 for name in names]
        context['model_description'] = self.model.__doc__.strip().splitlines()[0]
        context['model_name'] = MODEL_NAMES[self.model.__name__]
        context['next_event'] = event.get_next(**self.filter_kwargs)
        context['previous_event'] = event.get_previous(**self.filter_kwargs)
        if self.filter_kwargs:
            context['filter_url'] = '&'.join('{}={}'.format(key, val)
                                             for key, val in self.filter_kwargs.items())

        return context

    def get_queryset(self):
        self.filter_kwargs = {key: val for key, val in self.request.GET.items()
                              if re.match(r'\w+__\w+$', key)}
        return self.model.objects.filter(**self.filter_kwargs)


class EventList(ListView):
    paginate_by = 20
    context_object_name = 'event_list'
    template_name = 'events/event_list.html'

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(EventList, self).get_context_data(**kwargs)

        context['field_names'] = [field.name for field in self.model._meta.fields
                                  if not hasattr(field, '_kadi_no_list_view')]
        context['model_description'] = self.model.__doc__.strip().splitlines()[0]
        context['model_name'] = MODEL_NAMES[self.model.__name__]
        event_list = context['event_list']
        formats = {field.name: getattr(field, '_kadi_format', '{}')
                   for field in self.model._meta.fields}
        context['event_rows'] = [[formats[name].format(getattr(event, name))
                                  for name in context['field_names']]
                                 for event in event_list]
        if self.filter_kwargs:
            context['filter_url'] = '&'.join('{}={}'.format(key, val)
                                             for key, val in self.filter_kwargs.items())

        return context

    def get_queryset(self):
        self.filter_kwargs = {key: val for key, val in self.request.GET.items()
                              if re.match(r'\w+__\w+$', key)}
        return self.model.objects.filter(**self.filter_kwargs)


# Define list view classes for each event model
# Should probably do this with code and metaclasses, but this works for now

class ObsidList(EventList):
    model = models.Obsid


class TscMoveList(EventList):
    model = models.TscMove


class DarkCalReplicaList(EventList):
    model = models.DarkCalReplica


class DarkCalList(EventList):
    model = models.DarkCal


class Scs107List(EventList):
    model = models.Scs107


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


class IFotEventList(EventList):
    model = models.IFotEvent


class CAPList(EventList):
    model = models.CAP


class DsnCommList(EventList):
    model = models.DsnComm


class OrbitList(EventList):
    model = models.Orbit


class OrbitPointList(EventList):
    model = models.OrbitPoint


class RadZoneList(EventList):
    model = models.RadZone

####

class ObsidDetail(EventDetail):
    model = models.Obsid


class TscMoveDetail(EventDetail):
    model = models.TscMove


class DarkCalReplicaDetail(EventDetail):
    model = models.DarkCalReplica


class DarkCalDetail(EventDetail):
    model = models.DarkCal


class Scs107Detail(EventDetail):
    model = models.Scs107


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


class IFotEventDetail(EventDetail):
    model = models.IFotEvent


class CAPDetail(EventDetail):
    model = models.CAP


class DsnCommDetail(EventDetail):
    model = models.DsnComm


class OrbitDetail(EventDetail):
    model = models.Orbit


class OrbitPointDetail(EventDetail):
    model = models.OrbitPoint


class RadZoneDetail(EventDetail):
    model = models.RadZone
