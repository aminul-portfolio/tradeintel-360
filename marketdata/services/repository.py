from __future__ import annotations
from datetime import datetime
from django.db import transaction
from ..models import Asset, OHLC, Timeframe

def get_or_create_asset(symbol: str, **kwargs) -> Asset:
    obj, _ = Asset.objects.get_or_create(symbol=symbol.upper(), defaults=kwargs)
    return obj

@transaction.atomic
def upsert_daily_ohlc(asset: Asset, bars, source: str = "yahoo") -> int:
    # Bulk upsert (conflict on unique (asset, timeframe, ts))
    # Django 5 bulk_create with update_conflicts=True
    ohlc_rows = []
    for b in bars:
        ohlc_rows.append(OHLC(
            asset=asset,
            timeframe=Timeframe.D1,
            ts=b.ts,
            open=b.open, high=b.high, low=b.low, close=b.close,
            adj_close=b.adj_close, volume=b.volume,
            source=source,
        ))
    if not ohlc_rows:
        return 0
    created = OHLC.objects.bulk_create(
        ohlc_rows,
        ignore_conflicts=True
    )
    # For existing rows we could bulk_update if needed; keeping v1 simple.
    return len(created)

def most_recent_bar_ts(asset: Asset) -> datetime | None:
    last = OHLC.objects.filter(asset=asset, timeframe=Timeframe.D1).order_by("-ts").values_list("ts", flat=True).first()
    return last
