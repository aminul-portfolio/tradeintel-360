# trading/views.py
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render

from .forms import TradeForm
from .models import Trade

# ---- CRUD ----

@login_required
def trade_list(request):
    qs = Trade.objects.filter(user=request.user).order_by("-open_time")
    symbol = request.GET.get("symbol")
    side = request.GET.get("side")
    status = request.GET.get("status")

    if symbol:
        qs = qs.filter(symbol__icontains=symbol)
    if side in ("LONG", "SHORT"):
        qs = qs.filter(side=side)
    if status in ("OPEN", "CLOSED"):
        qs = qs.filter(status=status)

    page = Paginator(qs, 10).get_page(request.GET.get("page"))
    return render(request, "trading/trade_list.html", {"page": page})

@login_required
def trade_detail(request, pk):
    trade = get_object_or_404(Trade, pk=pk, user=request.user)
    return render(request, "trading/trade_detail.html", {"trade": trade})

@login_required
def trade_create(request):
    if request.method == "POST":
        form = TradeForm(request.POST)
        if form.is_valid():
            trade = form.save(commit=False)
            trade.user = request.user
            trade.pnl = trade.realized_pnl()
            trade.save()
            messages.success(request, "Trade created.")
            return redirect("trading:trade_detail", pk=trade.pk)
    else:
        form = TradeForm()
    return render(request, "trading/trade_form.html", {"form": form})

@login_required
def trade_update(request, pk):
    trade = get_object_or_404(Trade, pk=pk, user=request.user)
    if request.method == "POST":
        form = TradeForm(request.POST, instance=trade)
        if form.is_valid():
            trade = form.save(commit=False)
            trade.pnl = trade.realized_pnl()
            trade.save()
            messages.success(request, "Trade updated.")
            return redirect("trading:trade_detail", pk=trade.pk)
    else:
        form = TradeForm(instance=trade)
    return render(request, "trading/trade_form.html", {"form": form, "trade": trade})

@login_required
def trade_delete(request, pk):
    trade = get_object_or_404(Trade, pk=pk, user=request.user)
    if request.method == "POST":
        trade.delete()
        messages.success(request, "Trade deleted.")
        return redirect("trading:trade_list")
    return render(request, "trading/trade_confirm_delete.html", {"trade": trade})

# ---- Calculators ----

@login_required
def risk_calculator(request):
    result = None
    try:
        side = request.GET.get("side")  # LONG/SHORT
        entry = float(request.GET.get("entry") or 0)
        sl = float(request.GET.get("sl") or 0)
        tp = float(request.GET.get("tp") or 0)

        if side in ("LONG", "SHORT") and entry and sl and tp:
            risk = (entry - sl) if side == "LONG" else (sl - entry)
            reward = (tp - entry) if side == "LONG" else (entry - tp)
            rr = reward / risk if risk > 0 else None
            result = {"risk": round(risk, 5), "reward": round(reward, 5), "rr": round(rr, 2) if rr else None}
    except Exception:
        result = None

    return render(request, "trading/risk_calculator.html", {"result": result})

@login_required
def lot_size(request):
    result = None
    try:
        balance = float(request.GET.get("balance") or 0)
        risk_pct = float(request.GET.get("risk_pct") or 0)
        stop_pips = float(request.GET.get("stop_pips") or 0)
        pip_value = float(request.GET.get("pip_value") or 1)
        risk_amount = balance * (risk_pct / 100.0)
        lot = risk_amount / (stop_pips * pip_value) if stop_pips > 0 and pip_value > 0 else None
        result = {"risk_amount": round(risk_amount, 2), "lot": round(lot, 2) if lot else None}
    except Exception:
        result = None
    return render(request, "trading/lot_size.html", {"result": result})

@login_required
def strategy_risk(request):
    return render(request, "trading/strategy_risk.html")

@login_required
def risk_per_trade(request):
    result = None
    try:
        balance = float(request.GET.get("balance") or 0)
        risk_pct = float(request.GET.get("risk_pct") or 0)
        risk_amount = balance * (risk_pct / 100.0)
        result = {"risk_amount": round(risk_amount, 2)}
    except Exception:
        result = None
    return render(request, "trading/risk_per_trade.html", {"result": result})
from django.shortcuts import render

# Create your views here.
