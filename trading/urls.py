from django.urls import path
from . import views

app_name = "trading"

urlpatterns = [
    # Post-trade review / CRUD
    path("",                    views.trade_list,   name="trade_list"),
    path("<int:pk>/",           views.trade_detail, name="trade_detail"),
    path("create/",             views.trade_create, name="trade_create"),
    path("<int:pk>/edit/",      views.trade_update, name="trade_update"),
    path("<int:pk>/delete/",    views.trade_delete, name="trade_delete"),

    # Secondary support tools
    path("strategy-risk/",      views.strategy_risk, name="strategy_risk"),
]