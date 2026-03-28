# marketdata/urls.py
from django.urls import path
from .views import AssetListView, asset_detail, dashboard, OHLCListView, index_us100
from . import views

app_name = "marketdata"

urlpatterns = [
    path("assets/", AssetListView.as_view(), name="asset_list"),
    path("assets/<int:pk>/", views.asset_detail, name="asset_detail"),

    path("dashboard/", dashboard, name="dashboard"),
    path("ohlc/", OHLCListView.as_view(), name="ohlc_list"),
    path("index_us100/", index_us100, name="index_us100"),
]
