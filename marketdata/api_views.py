# tradeintel/marketdata/api_views.py
from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from typing import Optional

from django.db.models import F, OuterRef, Subquery
from django.http import JsonResponse, Http404
from django.utils.dateparse import parse_datetime
from django.views.decorators.http import require_GET

from .models import Asset, PriceOHLC, PriceSnapshot, Timeframe
from .utils import cc_histoday_range, yf_range, yf_latest_close


# -----------------------------
# Helpers
# -----------------------------
def _aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware in UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=dt_timezone.utc)

def _dt(s: Optional[str]) -> Optional[datetime]:
    """
    Parse a querystring datetime into an aware UTC datetime.
    Accepts ISO strings (e.g. 2024-01-01 or 2024-01-01T00:00:00Z).
    """
    if not s:
        return None
    d = parse_datetime(s)
    if d is None:
        try:
            # Handle bare dates and 'Z'
            if len(s) == 10 and "-" in s:  # YYYY-MM-DD
                d = datetime.fromisoformat(s)
            else:
                d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt_timezone.utc)
    else:
        d = d.astimezone(dt_timezone.utc)
    return d

def _limit(request, default=2000, max_cap=10000) -> int:
    try:
        val = int(request.GET.get("limit", default))
    except (TypeError, ValueError):
        val = default
    return max(1, min(val, max_cap))

def _timeframe(request) -> str:
    """
    Parse ?tf=... to one of your stored values (Timeframe.* values).
    Your choices store values like '1m','5m','15m','1h','1d','1wk','1mo'.
    Defaults to Timeframe.D1 ('1d').
    """
    tf_s = (request.GET.get("tf") or "D1").upper()
    name_to_choice = {
        "M1":  Timeframe.M1,
        "M5":  Timeframe.M5,
        "M15": Timeframe.M15,
        "H1":  Timeframe.H1,
        "D1":  Timeframe.D1,
        "W1":  Timeframe.W1,
        "MN1": Timeframe.MN1,
    }
    return name_to_choice.get(tf_s, Timeframe.D1)  # returns stored value like '1d'


# -----------------------------
# APIs used by dashboard.js
# -----------------------------
@require_GET
def api_prices(request):
    """
    Latest snapshot per asset.
    Response: { "snapshots": [ { "symbol": "...", "price": 123.45, "timestamp": "..." }, ... ] }
    """
    latest_ts = (
        PriceSnapshot.objects
        .filter(asset=OuterRef("asset"))
        .order_by("-ts")
        .values("ts")[:1]
    )

    rows = (
        PriceSnapshot.objects
        .filter(ts=Subquery(latest_ts))
        .select_related("asset")
        .annotate(symbol=F("asset__symbol"))
        .values("symbol", "price", "ts")
        .order_by("symbol")
    )

    snaps = [
        {
            "symbol": r["symbol"],
            "price": float(r["price"]),
            "timestamp": r["ts"].astimezone(dt_timezone.utc).isoformat(),
        }
        for r in rows
    ]
    return JsonResponse({"snapshots": snaps})


@require_GET
def api_ohlc(request):
    """
    OHLC series for a symbol. Accepts ?start=, ?end=, ?tf= (M1,M5,M15,H1,D1,W1,MN1), ?limit=
    Response: { "symbol": "AAPL", "ohlc": [ { "timestamp": "...", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ... }, ... ] }
    """
    symbol = request.GET.get("symbol")
    if not symbol:
        return JsonResponse({"error": "symbol required"}, status=400)

    try:
        asset = Asset.objects.get(symbol__iexact=symbol)
    except Asset.DoesNotExist:
        return JsonResponse({"symbol": symbol, "ohlc": []})

    start = _dt(request.GET.get("start"))
    end   = _dt(request.GET.get("end"))
    tf    = _timeframe(request)          # stored value e.g. '1d'
    limit = _limit(request)

    qs = (
        PriceOHLC.objects
        .filter(asset=asset, timeframe=tf)
        .order_by("ts")
        .values("ts", "open", "high", "low", "close", "volume")
    )
    if start:
        qs = qs.filter(ts__gte=start)
    if end:
        qs = qs.filter(ts__lte=end)

    qs = qs[:limit]

    ohlc = [
        {
            "timestamp": r["ts"].astimezone(dt_timezone.utc).isoformat(),
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low":  float(r["low"]),
            "close": float(r["close"]),
            "volume": int(r["volume"]) if r["volume"] is not None else 0,
        }
        for r in qs
    ]
    return JsonResponse({"symbol": asset.symbol, "ohlc": ohlc})


@require_GET
def api_line_price_data(request):
    """
    Close-only series for line chart. Accepts ?start=, ?end=, ?tf=, ?limit=
    Response: { "symbol": "AAPL", "data": [ { "timestamp": "...", "price": ... }, ... ] }
    """
    symbol = request.GET.get("symbol")
    if not symbol:
        return JsonResponse({"error": "symbol required"}, status=400)

    try:
        asset = Asset.objects.get(symbol__iexact=symbol)
    except Asset.DoesNotExist:
        return JsonResponse({"symbol": symbol, "data": []})

    start = _dt(request.GET.get("start"))
    end   = _dt(request.GET.get("end"))
    tf    = _timeframe(request)
    limit = _limit(request)

    qs = (
        PriceOHLC.objects
        .filter(asset=asset, timeframe=tf)
        .order_by("ts")
        .values("ts", "close")
    )
    if start:
        qs = qs.filter(ts__gte=start)
    if end:
        qs = qs.filter(ts__lte=end)

    qs = qs[:limit]

    data = [
        {
            "timestamp": r["ts"].astimezone(dt_timezone.utc).isoformat(),
            "price": float(r["close"]),
        }
        for r in qs
    ]
    return JsonResponse({"symbol": asset.symbol, "data": data})


# -----------------------------
# Crypto / Index helpers for index_us100 page
# -----------------------------
@require_GET
def api_custom_trend(request, quote: str | None = None):
    """
    Mixed source: Yahoo for indices (US100/^NDX, ^GSPC, ^DJI, ^IXIC, ^RUT, GC=F), CryptoCompare for others.
    Requires ?start=YYYY-MM-DD&end=YYYY-MM-DD
    """
    q = (quote or request.GET.get("quote", "BTC")).upper()
    start_s = request.GET.get("start")
    end_s   = request.GET.get("end")
    if not start_s or not end_s:
        return JsonResponse({"error": "start and end required"}, status=400)

    try:
        start = datetime.strptime(start_s, "%Y-%m-%d").replace(tzinfo=dt_timezone.utc)
        end   = datetime.strptime(end_s, "%Y-%m-%d").replace(tzinfo=dt_timezone.utc)
    except Exception:
        return JsonResponse({"error": "Invalid dates"}, status=400)

    if q in ("US100", "^GSPC", "^DJI", "^IXIC", "^RUT", "GC=F"):
        sym = "^NDX" if q == "US100" else q
        bars = yf_range(sym, start, end)
    else:
        bars = cc_histoday_range(q, start, end)

    if not bars:
        return JsonResponse({"error": "No data found"}, status=400)

    labels = [b.ts.strftime("%b %d") for b in bars]
    prices = [round(b.close, 2) for b in bars]
    change = prices[-1] - prices[0]
    pct    = (change / prices[0]) * 100 if prices[0] else 0.0

    return JsonResponse({
        "labels": labels,
        "prices": prices,
        "change": round(change, 2),
        "change_percent": round(pct, 2),
    })


@require_GET
def api_live_price_us100(request):
    """
    Returns most recent close for US100 (^NDX) and previous day for comparison.
    Response: { "quote": "US100", "current_price": 123.45, "yesterday_price": 122.00, "diff": 1.45, "diff_percent": 1.19 }
    """
    q = request.GET.get("quote", "US100").upper()
    if q != "US100":
        return JsonResponse({"error": "Only US100 supported here."}, status=400)

    latest, prev = yf_latest_close("^NDX")
    if latest is None or prev is None:
        return JsonResponse({"error": "No recent data"}, status=400)

    diff = latest - prev
    pct  = (diff / prev) * 100 if prev else 0.0
    return JsonResponse({
        "quote": q,
        "current_price": round(latest, 2),
        "yesterday_price": round(prev, 2),
        "diff": round(diff, 2),
        "diff_percent": round(pct, 2),
    })


# -----------------------------
# Stub until Alerts app is wired
# -----------------------------
@require_GET
def api_alerts_triggered_stub(request):
    return JsonResponse({"alerts": []})
