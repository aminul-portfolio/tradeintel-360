from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional
import importlib

@dataclass
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: Optional[float]
    volume: Optional[int]

class BaseMarketDataClient:
    source_name = "base"
    def daily_ohlc(self, symbol: str, start: datetime, end: Optional[datetime] = None) -> Iterable[Bar]:
        raise NotImplementedError
    def latest_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

class YahooClient(BaseMarketDataClient):
    source_name = "yahoo"
    def __init__(self):
        if importlib.util.find_spec("yfinance") is None:
            raise RuntimeError("yfinance not installed")
        import yfinance as yf
        self.yf = yf

    def daily_ohlc(self, symbol: str, start: datetime, end: Optional[datetime] = None) -> Iterable[Bar]:
        end = end or datetime.utcnow()
        df = self.yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty:
            return []
        df = df.rename(columns=str.lower)
        # Ensure index is UTC-aware datetimes
        df.index = df.index.tz_convert("UTC") if df.index.tz is not None else df.index.tz_localize("UTC")
        for idx, row in df.iterrows():
            yield Bar(
                ts=idx.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                adj_close=float(row["adj close"]) if "adj close" in row and not (row["adj close"] != row["adj close"]) else None,
                volume=int(row["volume"]) if "volume" in row and row["volume"] == row["volume"] else None,
            )

    def latest_price(self, symbol: str) -> Optional[float]:
        ticker = self.yf.Ticker(symbol)
        info = ticker.history(period="1d")
        if info is None or info.empty:
            return None
        return float(info["Close"].iloc[-1])

def get_client(name: str = "yahoo") -> BaseMarketDataClient:
    name = (name or "").lower()
    if name in ("yahoo", "yfinance", "yf"):
        return YahooClient()
    raise ValueError(f"Unknown client: {name}")
