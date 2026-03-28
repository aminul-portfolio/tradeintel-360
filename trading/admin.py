from django.contrib import admin
from .models import Trade

@admin.register(Trade)
class TradeAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "symbol", "side", "status", "entry_price", "exit_price", "pnl", "open_time")
    list_filter = ("side", "status", "symbol")
    search_fields = ("user__username", "symbol", "tag", "notes")
from django.contrib import admin

# Register your models here.
