# marketdata/management/commands/marketdata_load.py
import logging
from datetime import datetime, timedelta, timezone as py_tz
from typing import Any, Iterable, List, Optional

from django.core.management.base import BaseCommand
from django.utils import timezone
from marketdata.models import Asset, PriceOHLC
from ...symbol_map import YF_MAP  # relative import is robust inside app
from marketdata.utils import yf_range, cc_histoday_range

logger = logging.getLogger(__name__)

INDEX_PREFIX = '^'


def normalize_symbol(s: str) -> str:
    return s.strip().upper()


def infer_asset_type(symbol: str, yf_symbol: Optional[str]) -> str:
    if yf_symbol and yf_symbol.startswith(INDEX_PREFIX):
        return 'INDEX'
    if '/' in symbol:
        base, quote = (symbol.split('/', 1) + ['', ''])[:2]
        if len(base) == 3 and len(quote) == 3:
            return 'FOREX'
        return 'CRYPTO'
    return 'STOCK'


def _to_utc(dt: Any) -> Optional[datetime]:
    """Coerce various timestamp shapes to aware UTC datetime."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return timezone.make_aware(dt, py_tz.utc)
        return dt.astimezone(py_tz.utc)
    if isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt, tz=py_tz.utc)
    if isinstance(dt, str):
        try:
            iso = dt.rstrip('Z')
            d = datetime.fromisoformat(iso)
            if d.tzinfo is None:
                return timezone.make_aware(d, py_tz.utc)
            return d.astimezone(py_tz.utc)
        except Exception:
            return None
    return None


def _get(e: Any, dict_keys: Iterable[str], attrs: Iterable[str]) -> Any:
    """Safely read from dict or object attributes."""
    if isinstance(e, dict):
        for k in dict_keys:
            if k in e and e[k] is not None:
                return e[k]
    for a in attrs:
        if hasattr(e, a):
            v = getattr(e, a)
            if v is not None:
                return v
    return None


def normalize_bars(data: Any) -> List[dict]:
    """
    Accepts:
      - iterable of dicts
      - iterable of objects with attributes (e.g., Bar objects)
      - pandas.DataFrame (with index as datetime)
    Returns list of dicts with keys: ts, open, high, low, close, adj_close, volume
    """
    rows: List[dict] = []

    # pandas DataFrame?
    if hasattr(data, "iterrows") and hasattr(data, "columns"):
        try:
            cols = {c.lower().replace(' ', ''): c for c in list(data.columns)}
            for idx, r in data.iterrows():
                ts = _to_utc(idx)
                open_ = r.get(cols.get('open')) if hasattr(r, 'get') else r[cols.get('open', 'Open')]
                high = r.get(cols.get('high')) if hasattr(r, 'get') else r[cols.get('high', 'High')]
                low = r.get(cols.get('low')) if hasattr(r, 'get') else r[cols.get('low', 'Low')]
                close = r.get(cols.get('close')) if hasattr(r, 'get') else r[cols.get('close', 'Close')]
                adj = None
                if 'adjclose' in cols:
                    adj = r.get(cols.get('adjclose')) if hasattr(r, 'get') else r[cols['adjclose']]
                volume = r.get(cols.get('volume')) if hasattr(r, 'get') else r.get('Volume', 0)
                rows.append({
                    'ts': ts, 'open': float(open_), 'high': float(high), 'low': float(low),
                    'close': float(close), 'adj_close': float(adj) if adj is not None else float(close),
                    'volume': int(volume or 0),
                })
            return [r for r in rows if r['ts'] is not None]
        except Exception:
            pass  # fall through to generic handling

    # Generic iterable
    try:
        iterator = iter(data)
    except TypeError:
        return []

    for e in iterator:
        ts = _get(e, ('ts', 'date', 'time', 'datetime', 'end', 'timestamp'), ('ts', 'date', 'time', 'datetime', 'end', 'timestamp'))
        o = _get(e, ('open', 'o'), ('open',))
        h = _get(e, ('high', 'h'), ('high',))
        l = _get(e, ('low', 'l'), ('low',))
        c = _get(e, ('close', 'c'), ('close',))
        adj = _get(e, ('adj_close', 'adjclose', 'adjusted_close', 'ac'), ('adj_close', 'adjclose', 'adjusted_close'))
        v = _get(e, ('volume', 'vol', 'v'), ('volume',))
        ts = _to_utc(ts)
        if ts is None or o is None or h is None or l is None or c is None:
            continue
        rows.append({
            'ts': ts,
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'adj_close': float(adj) if adj is not None else float(c),
            'volume': int(v or 0),
        })
    return rows


class Command(BaseCommand):
    help = "Load historical D1 OHLC for symbols via yfinance (mapped or plain) or CryptoCompare (unmapped cryptos)."

    def add_arguments(self, parser):
        parser.add_argument('--symbols', type=str, required=True, help='Comma-separated list of symbols')
        parser.add_argument('--days', type=int, default=365, help='Days of history (default 365)')

    def handle(self, *args, **opts):
        raw_symbols = opts['symbols'].split(',')
        symbols = [normalize_symbol(s) for s in raw_symbols]
        days = int(opts['days'])

        end_dt = timezone.now()
        start_dt = end_dt - timedelta(days=days)

        for sym in symbols:
            print(f"Loading {sym} ...")
            try:
                yf_sym = YF_MAP.get(sym)
                if yf_sym:
                    self._load_from_yf(sym, yf_sym, start_dt, end_dt)
                elif '/' in sym:
                    self._load_from_crypto(sym, start_dt, end_dt)
                else:
                    self._load_from_yf(sym, sym, start_dt, end_dt)
            except Exception as exc:
                logger.exception("Failed loading %s: %s", sym, exc)
                print(f"  {sym}: ERROR {exc}")

    def _get_or_create_asset(self, symbol: str, yf_sym: Optional[str]):
        asset_type = infer_asset_type(symbol, yf_sym)
        asset, _ = Asset.objects.get_or_create(
            symbol=symbol,
            defaults={
                'name': symbol,
                'asset_type': asset_type,
                'exchange': '',
                'currency': '',
                'tz': 'UTC',
                'is_active': True,
            }
        )
        return asset

    def _bulk_upsert_ohlc(self, asset: Asset, data: Any, source: str) -> int:
        norm = normalize_bars(data)
        if not norm:
            return 0

        objs = []
        tf_value = 'D1'  # ✅ your model doesn't expose PriceOHLC.Timeframe; store the string directly
        for r in norm:
            objs.append(PriceOHLC(
                asset=asset,
                timeframe=tf_value,
                ts=r['ts'],
                open=r['open'],
                high=r['high'],
                low=r['low'],
                close=r['close'],
                adj_close=r['adj_close'],
                volume=r['volume'],
                source=source
            ))
        PriceOHLC.objects.bulk_create(objs, ignore_conflicts=True)
        return len(objs)

    def _load_from_yf(self, symbol: str, yf_symbol: str, start_dt, end_dt):
        data = yf_range(yf_symbol, start_dt, end_dt, interval='1d')
        if not data:
            print(f"  No data returned for {symbol}")
            return
        asset = self._get_or_create_asset(symbol, yf_symbol)
        inserted = self._bulk_upsert_ohlc(asset, data, source='yfinance')
        if inserted:
            print(f"  {symbol}: inserted {inserted} bars")
        else:
            print(f"  No data returned for {symbol}")

    def _load_from_crypto(self, symbol: str, start_dt, end_dt):
        data = cc_histoday_range(symbol, start_dt, end_dt)  # expects "BTC/USD"
        if not data:
            print(f"  No data returned for {symbol}")
            return
        asset = self._get_or_create_asset(symbol, yf_sym=None)
        inserted = self._bulk_upsert_ohlc(asset, data, source='cryptocompare')
        if inserted:
            print(f"  {symbol}: inserted {inserted} bars")
        else:
            print(f"  No data returned for {symbol}")
