# performance/admin.py
from django.contrib import admin
from .models import TradingFile

@admin.register(TradingFile)
class TradingFileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "file", "status", "uploaded_at")
    list_filter = ("status", "uploaded_at")
    search_fields = ("user__username", "file")
