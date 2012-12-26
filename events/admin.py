from django.contrib import admin
from events.models import TscMove, TscMoveTimes, FaMove

class TscMoveAdmin(admin.ModelAdmin):
    list_display = ('datestart', 'datestop', 'tscpos_start', 'tscpos_stop', 'det_start', 'det_stop')
    search_fields = ('datestart',)

admin.site.register(TscMove, TscMoveAdmin)
admin.site.register(TscMoveTimes)
admin.site.register(FaMove)
