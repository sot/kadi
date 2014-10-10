from django.views.generic import TemplateView
# from django.template import RequestContext
# from django.shortcuts import render_to_response

from .events.views import BaseView


class IndexView(BaseView, TemplateView):
    template_name = 'kadi/top_index.html'

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(IndexView, self).get_context_data(**kwargs)
        return context


# Another way to do this, corresponds to commented code in kadi/urls.py
# def index(request):
#     context = RequestContext(request)

#     context_dict = {}

#     return render_to_response('kadi/top_index.html', context_dict, context)