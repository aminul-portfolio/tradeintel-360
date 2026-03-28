# marketdata/management/commands/marketdata_snapshots.py
"""
Drop-in replacement: robust snapshots fetcher.

- --symbols optional (defaults to first 50 active assets)
- Uses YF_MAP when available; also supports plain tickers (AAPL) and auto-maps FOREX like EUR/USD -> EURUSD=X
- Cryptos:
    * If mapped (e.g., BTC/USD -> BTC-USD), fetch via yfinance
    * Otherwise fall back to CryptoCompare spot (USD) for pairs like DOGE/USD
- Writes ONE PriceSnapshot per symbol (price only; no (None, None) tuple issues)
"""
import logging
from decimal import Decimal, InvalidOperation
from typing import Optional, Iterable

import requests
from django.core.management.base import BaseCommand
from django.utils import timezone

from marketdata.models import Asset, PriceSnapshot
from ...symbol_map import YF_MAP  # relative import so it works in package context

# Optional: use yfinance directly for resilience instead of relying solely on utils
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

logger = logging.getLogger(__name__)


def normalize_symbol(s: str) -> str:
    return s.strip().upper()


def to_decimal(val) -> Optional[Decimal]:
    if val is None:
        return None
    try:
        return Decimal(str(val))
    except (InvalidOperation, TypeError, ValueError):
        return None


def yf_fetch_last_price(ticker: str) -> Optional[Decimal]:
    """
    Resilient last price from yfinance:
      1) fast_info (last_price / regular_market_price)
      2) history(period='1d', interval='1m' or '1d') last non-null Close
    Returns Decimal or None.
    """
    if yf is None:
        return None

    try:
        t = yf.Ticker(ticker)

        # 1) fast_info
        try:
            fi = getattr(t, "fast_info", None)
            if fi:
                # yfinance has used both snake_case and camelCase over time
                for key in ("last_price", "regular_market_price", "lastPrice", "regularMarketPrice"):
                    if hasattr(fi, key):
                        price = getattr(fi, key)
                        d = to_decimal(price)
                        if d is not None:
                            return d
                # fast_info may also be a dict-like
                if isinstance(fi, dict):
                    for key in ("last_price", "regular_market_price", "lastPrice", "regularMarketPrice"):
                        if key in fi:
                            d = to_decimal(fi[key])
                            if d is not None:
                                return d
        except Exception:
            pass  # fall back to history

        # 2) history minute-level (latest bar)
        try:
            hist = t.history(period="1d", interval="1m", prepost=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                # Take the last non-null close
                close = hist["Close"].dropna()
                if not close.empty:
                    return to_decimal(close.iloc[-1])
        except Exception:
            pass

        # 3) history daily close
        try:
            hist = t.history(period="5d", interval="1d", prepost=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                close = hist["Close"].dropna()
                if not close.empty:
                    return to_decimal(close.iloc[-1])
        except Exception:
            pass

        return None
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return None


def cc_spot_price(fsym: str, tsym: str) -> Optional[Decimal]:
    """CryptoCompare spot price for unmapped cryptos like DOGE/USD."""
    url = "https://min-api.cryptocompare.com/data/price"
    try:
        resp = requests.get(url, params={"fsym": fsym, "tsyms": tsym}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        val = data.get(tsym)
        return to_decimal(val)
    except Exception as e:
        logger.warning("CryptoCompare spot failed for %s/%s: %s", fsym, tsym, e)
        return None


def is_forex_pair(symbol: str) -> bool:
    if "/" not in symbol:
        return False
    base, quote = symbol.split("/", 1)
    return len(base) == 3 and len(quote) == 3


def forex_to_yf(symbol: str) -> str:
    """EUR/USD -> EURUSD=X"""
    base, quote = symbol.split("/", 1)
    return f"{base}{quote}=X"


class Command(BaseCommand):
    help = "Create PriceSnapshot rows for given symbols or the first 50 active assets."

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            required=False,
            help="Comma-separated symbols (e.g., AAPL,MSFT,BTC/USD,US100). "
                 "If omitted, uses up to 50 active assets.",
        )

    def handle(self, *args, **opts):
        if opts.get("symbols"):
            symbols = [normalize_symbol(s) for s in opts["symbols"].split(",")]
        else:
            symbols = list(
                Asset.objects.filter(is_active=True)
                .order_by("id")
                .values_list("symbol", flat=True)[:50]
            )
            symbols = [normalize_symbol(s) for s in symbols]

        created = 0
        for sym in symbols:
            try:
                created += self._snapshot_one(sym)
            except Exception as exc:
                logger.exception("Snapshot failed for %s: %s", sym, exc)

        self.stdout.write(self.style.SUCCESS(f"Total snapshots created: {created}"))

    def _snapshot_one(self, symbol: str) -> int:
        asset = Asset.objects.filter(symbol=symbol).first()
        if not asset:
            logger.warning("Asset not found for %s", symbol)
            print(f"Asset not found for {symbol}")
            return 0

        price: Optional[Decimal] = None
        source: Optional[str] = None

        # 1) If mapped -> yfinance mapped ticker
        mapped = YF_MAP.get(symbol)
        if mapped:
            price = yf_fetch_last_price(mapped)
            if price is not None:
                source = "yfinance"

        # 2) Plain tickers via yfinance (AAPL, MSFT, GC=F, BTC-USD, ^GSPC, etc.)
        if price is None and "/" not in symbol:
            price = yf_fetch_last_price(symbol)
            if price is not None:
                source = "yfinance"

        # 3) If FOREX like EUR/USD and not mapped, auto-map to EURUSD=X and try yfinance
        if price is None and is_forex_pair(symbol):
            yf_fx = forex_to_yf(symbol)
            price = yf_fetch_last_price(yf_fx)
            if price is not None:
                source = "yfinance"

        # 4) Unmapped crypto pairs like DOGE/USD -> CryptoCompare spot
        if price is None and "/" in symbol and not is_forex_pair(symbol):
            base, quote = symbol.split("/", 1)
            price = cc_spot_price(base, quote)
            if price is not None:
                source = "cryptocompare"

        if price is None:
            print(f"No price retrieved for {symbol}")
            return 0

        # Write ONE snapshot row (only required fields)
        PriceSnapshot.objects.create(
            asset=asset,
            ts=timezone.now(),
            price=price,
            source=source or "unknown",
        )
        print(f"Snapshot created for {symbol}: {price} ({source})")
        return 1
