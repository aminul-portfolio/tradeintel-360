# trading/urls.py
from django.urls import path
from . import views

app_name = "trading"

urlpatterns = [
    # CRUD
    path("", views.trade_list, name="trade_list"),
    path("<int:pk>/", views.trade_detail, name="trade_detail"),
    path("create/", views.trade_create, name="trade_create"),
    path("<int:pk>/edit/", views.trade_update, name="trade_update"),
    path("<int:pk>/delete/", views.trade_delete, name="trade_delete"),

    # Calculators
    path("tools/risk-calculator/", views.risk_calculator, name="risk_calculator"),
    path("tools/lot-size/", views.lot_size, name="lot_size"),
    path("tools/strategy-risk/", views.strategy_risk, name="strategy_risk"),
    path("tools/risk-per-trade/", views.risk_per_trade, name="risk_per_trade"),
]
