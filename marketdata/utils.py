# tradeintel/marketdata/utils.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Optional, List

import requests
import yfinance as yf
import pandas as pd
from django.utils.timezone import make_aware

CRYPTOCOMPARE_BASE = "https://min-api.cryptocompare.com/data"

# Friendly -> Yahoo map (indices etc.)
YF_INDEX_MAP: dict[str, str] = {
    "US100": "^NDX",   # NASDAQ-100
    "SPX": "^GSPC",
    "DOW": "^DJI",
}

def normalize_yf_symbol(symbol: str) -> str:
    """
    Make sure we request the right Yahoo symbol:
    - US100 -> ^NDX (via map above)
    - BTCUSD -> BTC-USD (add dash)
    - ETHUSD -> ETH-USD
    """
    if not symbol:
        return symbol
    u = symbol.upper().strip()
    # map friendly index names
    if u in YF_INDEX_MAP:
        return YF_INDEX_MAP[u]
    # crypto shorthand without dash -> add -USD
    if u.endswith("USD") and "-" not in u and len(u) > 3:
        base = u[:-3]
        return f"{base}-USD"
    return symbol

@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    adj_close: Optional[float] = None

def _aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return make_aware(dt, dt_timezone.utc)
    return dt.astimezone(dt_timezone.utc)

def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=dt_timezone.utc)
    return dt.astimezone(dt_timezone.utc)

# ---------- CryptoCompare (daily) ----------
def cc_histoday(quote: str, tsym: str="USD", limit: int=7, to_ts: Optional[int]=None) -> List[Bar]:
    params = {"fsym": quote, "tsym": tsym, "limit": limit}
    if to_ts is not None:
        params["toTs"] = to_ts
    try:
        r = requests.get(f"{CRYPTOCOMPARE_BASE}/v2/histoday", params=params, timeout=12)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return []

    items = (payload.get("Data") or {}).get("Data") or []
    out: List[Bar] = []
    for d in items:
        ts = _aware(datetime.utcfromtimestamp(d["time"]))
        vol = d.get("volumeto", d.get("volume"))
        out.append(
            Bar(
                ts=ts,
                open=float(d["open"]),
                high=float(d["high"]),
                low=float(d["low"]),
                close=float(d["close"]),
                volume=float(vol) if vol is not None else None,
            )
        )
    return out

def cc_histoday_range(quote: str, start: datetime, end: datetime, tsym: str="USD") -> List[Bar]:
    start_ts = int(_as_utc(start).timestamp())
    end_ts   = int(_as_utc(end).timestamp())
    days = max(1, (end_ts - start_ts)//86400)
    return cc_histoday(quote, tsym=tsym, limit=days, to_ts=end_ts)

# ---------- Yahoo Finance ----------
def yf_range(symbol: str, start: datetime, end: datetime, interval: str="1d") -> List[Bar]:
    """
    Robust Yahoo fetch:
    - normalize symbol
    - try yf.download; if empty, fallback to Ticker.history
    - handle MultiIndex columns; normalize names
    """
    yf_symbol = normalize_yf_symbol(symbol)

    start_d = _as_utc(start).date()
    end_d   = (_as_utc(end) + timedelta(days=1)).date()   # end-exclusive

    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.droplevel(0, axis=1)
            except Exception:
                pass
        # rename columns to canonical names
        rename = {}
        for c in df.columns:
            lc = str(c).strip().lower()
            if lc == "open": rename[c] = "Open"
            elif lc == "high": rename[c] = "High"
            elif lc == "low": rename[c] = "Low"
            elif lc == "close": rename[c] = "Close"
            elif lc in ("adj close","adjclose","adjusted close"): rename[c] = "Adj Close"
            elif lc == "volume": rename[c] = "Volume"
        if rename:
            df = df.rename(columns=rename)
        return df

    # First try: yf.download
    try:
        df = yf.download(
            yf_symbol,
            start=start_d,
            end=end_d,
            interval=interval,
            progress=False,
            auto_adjust=False,
            actions=False,
            prepost=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    df = _normalize_df(df)

    # Fallback: Ticker.history
    if df.empty or not {"Open","High","Low","Close"}.issubset(df.columns):
        try:
            t = yf.Ticker(yf_symbol)
            df2 = t.history(start=start_d, end=end_d, interval=interval, auto_adjust=False)
            df2 = _normalize_df(df2)
            if not df2.empty and {"Open","High","Low","Close"}.issubset(df2.columns):
                df = df2
        except Exception:
            pass

    if df.empty or not {"Open","High","Low","Close"}.issubset(df.columns):
        return []

    df = df.dropna(subset=["Open","High","Low","Close"])

    out: List[Bar] = []
    for idx, row in df.iterrows():
        ts = _as_utc(idx.to_pydatetime())
        out.append(
            Bar(
                ts=ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]) if "Volume" in df.columns and pd.notna(row.get("Volume")) else None,
                adj_close=float(row["Adj Close"]) if "Adj Close" in df.columns and pd.notna(row.get("Adj Close")) else None,
            )
        )
    return out

def yf_latest_close(symbol: str) -> tuple[Optional[float], Optional[float]]:
    yf_symbol = normalize_yf_symbol(symbol)
    try:
        df = yf.download(
            yf_symbol,
            period="10d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            actions=False,
            prepost=False,
            threads=False,
        )
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty or ("Close" not in df.columns and not isinstance(df.columns, pd.MultiIndex)):
        # try fallback
        try:
            t = yf.Ticker(yf_symbol)
            df = t.history(period="10d", interval="1d", auto_adjust=False)
        except Exception:
            return None, None

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.droplevel(0, axis=1)
        except Exception:
            pass

    if "Close" not in df.columns:
        return None, None

    df = df.dropna(subset=["Close"])
    if len(df) < 2:
        return None, None
    return float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
