import re
import inspect

from django.contrib import admin
from django.contrib.admin.views import main

from . import models as mdl


class MyChangeList(main.ChangeList):
    def get_query_set(self, request):
        # First, we collect all the declared list filters.
        (self.filter_specs, self.has_filters, remaining_lookup_params,
         use_distinct) = self.get_filters(request)

        # Then, we let every list filter modify the queryset to its liking.
        qs = self.root_query_set
        for filter_spec in self.filter_specs:
            new_qs = filter_spec.queryset(request, qs)
            if new_qs is not None:
                qs = new_qs

        try:
            # Finally, we apply the remaining lookup parameters from the query
            # string (i.e. those that haven't already been processed by the
            # filters).
            qs = qs.filter(**remaining_lookup_params)
        except (main.SuspiciousOperation, main.ImproperlyConfigured):
            # Allow certain types of errors to be re-raised as-is so that the
            # caller can treat them in a special way.
            raise
        except Exception, e:
            # Every other error is caught with a naked except, because we don't
            # have any other way of validating lookup parameters. They might be
            # invalid if the keyword arguments are incorrect, or if the values
            # are not in the correct type, so we might get FieldError,
            # ValueError, ValidationError, or ?.
            raise main.IncorrectLookupParameters(e)

        # Use select_related() if one of the list_display options is a field
        # with a relationship and the provided queryset doesn't already have
        # select_related defined.
        if not qs.query.select_related:
            if self.list_select_related:
                qs = qs.select_related()
            else:
                for field_name in self.list_display:
                    try:
                        field = self.lookup_opts.get_field(field_name)
                    except main.models.FieldDoesNotExist:
                        pass
                    else:
                        if isinstance(field.rel, main.models.ManyToOneRel):
                            qs = qs.select_related()
                            break

        # Set ordering.
        ordering = self.get_ordering(request, qs)
        qs = qs.order_by(*ordering)

        def coerce_type(val):
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
            return val

        if self.query:
            start = None
            for bit in self.query.split():
                m = re.match(r'(\w+)=(.+)', bit)
                print(m)
                if m:
                    field_name, field_value = m.groups()
                    qs = qs.filter(**{'{}__iexact'.format(field_name): coerce_type(field_value)})
                elif re.match(r'\d{4}', bit):
                    if start is None:
                        qs = qs.filter(start__gte=bit)
                        start = bit
                    else:
                        qs = qs.filter(start__lte=bit)
                else:
                    print('Bad filter item')  # Just drop it
        return qs


class ModelAdminBase(admin.ModelAdmin):
    search_fields = ('start',)

    def get_changelist(self, request, **kwargs):
        """
        Returns the ChangeList class for use on the changelist page.
        """
        return MyChangeList


class TscMoveAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'start_3tscpos', 'stop_3tscpos',
                    'start_det', 'stop_det', 'max_pwm')


class FaMoveAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'start_3fapos', 'stop_3fapos')


class ManvrAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'dur', 'template', 'n_dwell', 'n_kalman', 'next_nman_start')


class ManvrSeqAdmin(ModelAdminBase):
    list_display = ('date', 'msid', 'prev_val', 'val', 'dt', 'manvr')


class DumpAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'dur')


class EclipseAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'dur')


class SafeSunAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'dur', 'notes')


class Scs107Admin(ModelAdminBase):
    list_display = ('start', 'stop', 'dur', 'notes')


class MajorEventAdmin(ModelAdminBase):
    list_display = ('start', 'date', 'source', 'descr', 'note')


class CAP(ModelAdminBase):
    list_display = ('start', 'num', 'title', 'descr', 'notes')


class DsnCommAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'bot', 'eot', 'activity', 'site', 'soe', 'station')


class OrbitAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'orbit_num', 'perigee', 'apogee',
                    'dt_start_radzone', 'dt_stop_radzone')


class OrbitPointAdmin(ModelAdminBase):
    list_display = ('date', 'orbit_num', 'name', 'orbit', 'descr')


class DwellAdmin(ModelAdminBase):
    list_display = ('start', 'stop', 'rel_tstart', 'manvr')


for name, obj in vars().items():
    if name.endswith('Admin') and inspect.isclass(obj) and issubclass(obj, ModelAdminBase):
        mdl_cls_name = name[:-5]
        print('Registeering {} {}'.format(mdl_cls_name, name))
        admin.site.register(getattr(mdl, mdl_cls_name), obj)