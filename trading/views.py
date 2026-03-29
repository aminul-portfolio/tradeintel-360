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
    side = (request.GET.get("side") or "").strip()
    status = (request.GET.get("status") or "").strip()
    q = (request.GET.get("q") or "").strip()

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

    page_obj = Paginator(qs, 10).get_page(request.GET.get("page"))
    page_query = _query_without(request, "page")

    return render(
        request,
        "trading/trade_list.html",
        {
            "page": page_obj,
            "symbol": symbol,
            "side": side,
            "status": status,
            "q": q,
            "page_query": page_query,
        },
    )


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
            trade = form.save(commit=False)
            trade.user = request.user
            trade.pnl = trade.realized_pnl()
            trade.save()
            messages.success(request, "Trade created.")
            return redirect("trading:trade_detail", pk=trade.pk)
    else:
        form = TradeForm()

    return render(
        request,
        "trading/trade_form.html",
        {
            "form": form,
            "trade": None,
        },
    )


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

    return render(
        request,
        "trading/trade_form.html",
        {
            "form": form,
            "trade": trade,
        },
    )


@login_required
def trade_delete(request, pk):
    trade = get_object_or_404(Trade, pk=pk, user=request.user)

    if request.method == "POST":
        trade.delete()
        messages.success(request, "Trade deleted.")
        return redirect("trading:trade_list")

    return render(request, "trading/trade_confirm_delete.html", {"trade": trade})