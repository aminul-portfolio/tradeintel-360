# marketdata/admin.py
from django.contrib import admin
from .models import Asset, PriceOHLC, PriceSnapshot


# ----- Inlines -----
class LatestSnapshotsInline(admin.TabularInline):
    """
    Read-only inline to show the last few snapshots on the Asset detail page.
    Avoids heavy editing inline; just a quick view.
    """
    model = PriceSnapshot
    extra = 0
    fields = ("ts", "price", "source")
    readonly_fields = ("ts", "price", "source")
    ordering = ("-ts",)
    can_delete = False
    show_change_link = False
    max_num = 0  # don't allow adding via inline

    def has_add_permission(self, request, obj=None):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        # Only show the most recent 10 for performance/readability
        return qs.order_by("-ts")[:10]


# ----- Admin registrations -----
@admin.register(Asset)
class AssetAdmin(admin.ModelAdmin):
    list_display = ("symbol", "name", "asset_type", "exchange", "currency", "tz", "is_active")
    list_filter = ("asset_type", "exchange", "currency", "is_active")
    search_fields = ("symbol", "name")
    ordering = ("symbol",)
    list_per_page = 50
    list_editable = ("is_active",)
    inlines = [LatestSnapshotsInline]


@admin.register(PriceOHLC)
class PriceOHLCAdmin(admin.ModelAdmin):
    """
    OHLC rows can be numerous; use raw_id_fields for speed and index-friendly filters.
    """
    raw_id_fields = ("asset",)
    list_display = ("asset", "timeframe", "ts", "open", "high", "low", "close", "volume", "source")
    list_filter = ("asset", "timeframe", "source")
    search_fields = ("asset__symbol",)
    date_hierarchy = "ts"
    ordering = ("-ts",)
    list_per_page = 100


@admin.register(PriceSnapshot)
class PriceSnapshotAdmin(admin.ModelAdmin):
    raw_id_fields = ("asset",)
    list_display = ("asset", "ts", "price", "source")
    list_filter = ("asset", "source")
    search_fields = ("asset__symbol",)
    date_hierarchy = "ts"
    ordering = ("-ts",)
    list_per_page = 100
