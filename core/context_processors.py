from trading.models import Trade

def open_trade_count(request):
    if request.user.is_authenticated:
        count = Trade.objects.filter(user=request.user, exit_price__isnull=True).count()
    else:
        count = 0
    return {'open_trade_count': count}
def core_context(request):
    return {
        'app_name': 'TradeIntel',
        'user_is_authenticated': request.user.is_authenticated,
        'sidebar_sections': [...],  # dynamic list if needed
    }
