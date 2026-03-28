# marketdata/views.py
from django.db.models import Q, F
from django.utils import timezone
from django.views.generic import ListView
from django.shortcuts import get_object_or_404, render
from .models import Asset, PriceOHLC, PriceSnapshot, Timeframe


class AssetListView(ListView):
    model = Asset
    template_name = "marketdata/asset_list.html"
    context_object_name = "assets"
    paginate_by = 20  # default

    def get_paginate_by(self, queryset):
        ps = self.request.GET.get("page_size")
        allowed = {"20", "50", "100"}
        return int(ps) if ps in allowed else 20

    def get_queryset(self):
        qs = Asset.objects.all()

        # Filters
        q = self.request.GET.get("q")
        if q:
            qs = qs.filter(Q(symbol__icontains=q) | Q(name__icontains=q))

        asset_type = self.request.GET.get("type")
        if asset_type:
            qs = qs.filter(asset_type=asset_type)

        if self.request.GET.get("active") == "1":
            qs = qs.filter(is_active=True)

        # Whitelist ordering to existing fields ONLY
        order = (self.request.GET.get("order") or "symbol").strip()
        allowed_orders = {
            "symbol", "-symbol",
            "name", "-name",
            "asset_type", "-asset_type",
            "exchange", "-exchange",
            "currency", "-currency",
        }
        if order not in allowed_orders:
            order = "symbol"

        return qs.order_by(order)



def asset_detail(request, pk):
    asset = get_object_or_404(Asset, pk=pk)
    has_data = PriceOHLC.objects.filter(asset=asset, timeframe=Timeframe.D1).exists()
    return render(request, "marketdata/asset_detail.html", {"asset": asset, "has_data": has_data})


# (keep your other views)
def dashboard(request):
    symbols = list(
        Asset.objects.filter(is_active=True)
        .order_by("symbol")
        .values_list("symbol", flat=True)
    )
    ohlc = (
        PriceOHLC.objects.select_related("asset")
        .filter(timeframe=Timeframe.D1)
        .order_by("-ts")[:200]
        .annotate(timestamp=F("ts"))
    )
    ctx = {"now": timezone.now(), "symbols": symbols, "snapshots": [], "ohlc": ohlc,
           "open_trades_count": 0, "total_alerts": 0}
    return render(request, "marketdata/dashboard.html", ctx)


class OHLCListView(ListView):
    model = PriceOHLC
    template_name = "marketdata/ohlc_list.html"
    paginate_by = 50

    def get_queryset(self):
        qs = (PriceOHLC.objects
              .select_related("asset")
              .order_by("-ts")
              .annotate(timestamp=F("ts")))
        sym = self.request.GET.get("symbol")
        if sym:
            qs = qs.filter(asset__symbol__iexact=sym)
        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        paginator = ctx["paginator"]
        page_obj  = ctx["page_obj"]
        # Compute here; templates can't call it with args
        ctx["elided_pages"] = paginator.get_elided_page_range(number=page_obj.number)
        ctx["ELLIPSIS"] = paginator.ELLIPSIS  # convenient for template comparison
        return ctx


def index_us100(request):
    return render(request, "marketdata/index_us100.html", {
        "index_name": "US100 (NASDAQ-100)",
        "quote": "US100",
    })
