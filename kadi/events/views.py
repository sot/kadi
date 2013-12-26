# Create your views here.
from django.views.generic import ListView
from . import models

MODEL_NAMES = {m_class: m_name for m_name, m_class in models.get_event_models().items()}


class EventList(ListView):
    paginate_by = 20
    context_object_name = 'event_list'
    template_name = 'events/event_list.html'

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(EventList, self).get_context_data(**kwargs)
        # Add in a QuerySet of all the books
        context['field_names'] = [field.name for field in self.model._meta.fields]
        context['model_description'] = self.model.__doc__.strip().splitlines()[0]
        context['model_name'] = MODEL_NAMES[self.model]
        event_list = context['event_list']
        context['event_rows'] = [[getattr(event, attr) for attr in context['field_names']]
                                 for event in event_list]
        return context


class ManvrList(EventList):
    model = models.Manvr
