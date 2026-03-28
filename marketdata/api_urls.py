# tradeintel/marketdata/api_urls.py
from django.urls import path
from . import api_views

urlpatterns = [
    path("prices/", api_views.api_prices, name="api_prices"),
    path("ohlc/", api_views.api_ohlc, name="api_ohlc"),
    path("line-price-data/", api_views.api_line_price_data, name="api_line_price_data"),
    path("custom-trend/", api_views.api_custom_trend, name="api_custom_trend"),
    path("custom-trend/us100/", api_views.api_custom_trend, {"quote":"US100"}, name="api_custom_trend_us100"),
    path("live-price-api/", api_views.api_live_price_us100, name="api_live_price_us100"),
    path("alerts/triggered/", api_views.api_alerts_triggered_stub, name="api_alerts_triggered"),
]
