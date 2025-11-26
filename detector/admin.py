from django.contrib import admin
from .models import Search


@admin.register(Search)
class SearchAdmin(admin.ModelAdmin):
    list_display = ('user', 'label', 'model', 'score', 'created_at')
    search_fields = ('user__username', 'text', 'model', 'label')
    list_filter = ('model', 'label', 'created_at')
