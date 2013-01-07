from django.contrib import admin
from . import models as mdl


class TscMoveAdmin(admin.ModelAdmin):
    list_display = ('start', 'stop', 'start_3tscpos', 'stop_3tscpos', 'start_det', 'stop_det')
    search_fields = ('start',)


class FaMoveAdmin(admin.ModelAdmin):
    list_display = ('start', 'stop', 'start_3fapos', 'stop_3fapos')
    search_fields = ('start',)


class ManvrAdmin(admin.ModelAdmin):
    list_display = ('start', 'stop', 'dur', 'template', 'n_dwell', 'n_kalman', 'next_nman_start')
    search_fields = ('start',)


admin.site.register(mdl.Manvr, ManvrAdmin)
admin.site.register(mdl.TscMove, TscMoveAdmin)
admin.site.register(mdl.FaMove, FaMoveAdmin)
