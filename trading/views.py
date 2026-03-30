from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render

from .forms import TradeForm
from .models import Trade


def _query_without(request, *keys):
    q = request.GET.copy()
    for key in keys:
        q.pop(key, None)
    return q.urlencode()


@login_required
def trade_list(request):
    """
    Post-trade review table with filtering, smart search, and pagination.
    """
    qs = Trade.objects.filter(user=request.user).order_by("-open_time")

    symbol = (request.GET.get("symbol") or "").strip()
    side   = (request.GET.get("side")   or "").strip()
    status = (request.GET.get("status") or "").strip()
    q      = (request.GET.get("q")      or "").strip()

    if symbol:
        qs = qs.filter(symbol__icontains=symbol)

    if side in ("LONG", "SHORT"):
        qs = qs.filter(side=side)

    if status in ("OPEN", "CLOSED"):
        qs = qs.filter(status=status)

    if q:
        qs = qs.filter(
            Q(symbol__icontains=q)
            | Q(tag__icontains=q)
            | Q(notes__icontains=q)
            | Q(status__icontains=q)
            | Q(side__icontains=q)
        )

    page_obj   = Paginator(qs, 10).get_page(request.GET.get("page"))
    page_query = _query_without(request, "page")

    return render(request, "trading/trade_list.html", {
        "page":       page_obj,
        "symbol":     symbol,
        "side":       side,
        "status":     status,
        "q":          q,
        "page_query": page_query,
    })


@login_required
def trade_detail(request, pk):
    trade = get_object_or_404(Trade, pk=pk, user=request.user)
    return render(request, "trading/trade_detail.html", {"trade": trade})


@login_required
def trade_create(request):
    """
    Optional secondary workflow: manual trade entry.
    Keep this secondary in the public TradeIntel reviewer path.
    """
    if request.method == "POST":
        form = TradeForm(request.POST)
        if form.is_valid():
            trade      = form.save(commit=False)
            trade.user = request.user
            trade.pnl  = trade.realized_pnl()
            trade.save()
            messages.success(request, "Trade created.")
            return redirect("trading:trade_detail", pk=trade.pk)
    else:
        form = TradeForm()

    return render(request, "trading/trade_form.html", {
        "form":  form,
        "trade": None,
    })


@login_required
def trade_update(request, pk):
    trade = get_object_or_404(Trade, pk=pk, user=request.user)

    if request.method == "POST":
        form = TradeForm(request.POST, instance=trade)
        if form.is_valid():
            trade     = form.save(commit=False)
            trade.pnl = trade.realized_pnl()
            trade.save()
            messages.success(request, "Trade updated.")
            return redirect("trading:trade_detail", pk=trade.pk)
    else:
        form = TradeForm(instance=trade)

    return render(request, "trading/trade_form.html", {
        "form":  form,
        "trade": trade,
    })


@login_required
def trade_delete(request, pk):
    trade = get_object_or_404(Trade, pk=pk, user=request.user)

    if request.method == "POST":
        trade.delete()
        messages.success(request, "Trade deleted.")
        return redirect("trading:trade_list")

    return render(request, "trading/trade_confirm_delete.html", {"trade": trade})


@login_required
def strategy_risk(request):
    """
    Strategy-based position sizing using a scaled Kelly fraction + ATR stop.
    Secondary support tool — inputs via GET, returns suggested risk % and lot size.
    """
    result   = None
    defaults = {
        "balance":     request.GET.get("balance",     "10000"),
        "win_rate":    request.GET.get("win_rate",    "55"),
        "rr":          request.GET.get("rr",          "1.8"),
        "atr_pips":    request.GET.get("atr_pips",    "20"),
        "atr_mult":    request.GET.get("atr_mult",    "1.5"),
        "kelly_scale": request.GET.get("kelly_scale", "0.25"),
        "max_risk_pct":request.GET.get("max_risk_pct","1.0"),
        "pip_value":   request.GET.get("pip_value",   "1"),
        "floor_pct":   request.GET.get("floor_pct",   "0.10"),
    }

    try:
        balance      = float(defaults["balance"])
        win_rate     = float(defaults["win_rate"]) / 100.0
        rr           = float(defaults["rr"])
        atr_pips     = float(defaults["atr_pips"])
        atr_mult     = float(defaults["atr_mult"])
        kelly_scale  = float(defaults["kelly_scale"])
        max_risk_pct = float(defaults["max_risk_pct"])
        pip_value    = float(defaults["pip_value"])
        floor_pct    = float(defaults["floor_pct"])

        if balance > 0 and 0 < win_rate < 1 and rr > 0 and atr_pips > 0 and atr_mult > 0 and pip_value > 0:
            raw_kelly = win_rate - ((1 - win_rate) / rr)
            if raw_kelly < 0:
                raw_kelly = floor_pct / 100.0

            scaled_pct = raw_kelly * kelly_scale * 100.0
            risk_pct   = max(floor_pct, min(scaled_pct, max_risk_pct))
            stop_pips  = atr_pips * atr_mult
            risk_amount = balance * (risk_pct / 100.0)
            lot = risk_amount / (stop_pips * pip_value) if stop_pips > 0 else None

            result = {
                "raw_kelly_pct": round(raw_kelly * 100.0, 3),
                "risk_pct":      round(risk_pct, 3),
                "risk_amount":   round(risk_amount, 2),
                "stop_pips":     round(stop_pips, 2),
                "lot":           round(lot, 2) if lot else None,
            }
    except Exception:
        result = None

    return render(request, "trading/strategy_risk.html", {
        "defaults": defaults,
        "result":   result,
    })