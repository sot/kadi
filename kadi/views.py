from django.views.generic import TemplateView

from .events.views import BaseView


class IndexView(BaseView, TemplateView):
    template_name = 'top_index.html'

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(IndexView, self).get_context_data(**kwargs)
        return context
