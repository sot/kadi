# Create your views here.
from django.views.generic import ListView, TemplateView
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


class EventList(ListView):
    paginate_by = 20
    context_object_name = 'event_list'
    template_name = 'events/event_list.html'

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(EventList, self).get_context_data(**kwargs)

        context['field_names'] = [field.name for field in self.model._meta.fields]
        context['model_description'] = self.model.__doc__.strip().splitlines()[0]
        context['model_name'] = MODEL_NAMES[self.model.__name__]
        event_list = context['event_list']
        context['event_rows'] = [[getattr(event, attr) for attr in context['field_names']]
                                 for event in event_list]
        return context


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
