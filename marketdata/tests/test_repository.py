import datetime as dt
from django.test import TestCase
from marketdata.models import Asset, OHLC, Timeframe
from marketdata.services.repository import get_or_create_asset, upsert_daily_ohlc
from marketdata.services.clients import Bar

class RepoTests(TestCase):
    def test_upsert_daily(self):
        asset = get_or_create_asset("TEST")
        bars = [
            Bar(ts=dt.datetime(2024,1,1, tzinfo=dt.timezone.utc), open=1, high=2, low=0.5, close=1.5, adj_close=None, volume=100),
            Bar(ts=dt.datetime(2024,1,2, tzinfo=dt.timezone.utc), open=1.5, high=2.2, low=1.2, close=2.0, adj_close=None, volume=120),
        ]
        n = upsert_daily_ohlc(asset, bars, source="unit")
        self.assertEqual(n, 2)
        self.assertEqual(OHLC.objects.filter(asset=asset, timeframe=Timeframe.D1).count(), 2)
