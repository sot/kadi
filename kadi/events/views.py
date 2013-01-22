# Create your views here.
from django.views.generic import ListView
from .models import Manvr


class ManvrList(ListView):
    model = Manvr
    paginate_by = 40
