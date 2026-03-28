# tradeintel/marketdata/models.py
from django.db import models
from django.utils import timezone

class AssetType(models.TextChoices):
    STOCK="STOCK","Stock"; ETF="ETF","ETF"; FOREX="FOREX","Forex"
    CRYPTO="CRYPTO","Crypto"; INDEX="INDEX","Index"; FUTURE="FUTURE","Future"; OTHER="OTHER","Other"

class Timeframe(models.TextChoices):
    M1="1m","1 minute"; M5="5m","5 minutes"; M15="15m","15 minutes"
    H1="1h","1 hour"; D1="1d","1 day"; W1="1wk","1 week"; MN1="1mo","1 month"

class Asset(models.Model):
    symbol = models.CharField(max_length=40, unique=True, db_index=True)
    name = models.CharField(max_length=120, blank=True)
    asset_type = models.CharField(max_length=10, choices=AssetType.choices, default=AssetType.STOCK)
    exchange = models.CharField(max_length=50, blank=True)
    currency = models.CharField(max_length=10, blank=True)
    tz = models.CharField(max_length=40, blank=True)
    is_active = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta: ordering = ["symbol"]
    def __str__(self): return self.symbol

class PriceOHLC(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="ohlc")
    timeframe = models.CharField(max_length=4, choices=Timeframe.choices, default=Timeframe.D1)
    ts = models.DateTimeField(help_text="UTC bar end")
    open  = models.DecimalField(max_digits=20, decimal_places=8)
    high  = models.DecimalField(max_digits=20, decimal_places=8)
    low   = models.DecimalField(max_digits=20, decimal_places=8)
    close = models.DecimalField(max_digits=20, decimal_places=8)
    adj_close = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    volume = models.BigIntegerField(null=True, blank=True)
    source = models.CharField(max_length=30, blank=True)

    class Meta:
        unique_together = (("asset","timeframe","ts"),)
        indexes = [models.Index(fields=["asset","timeframe","ts"]), models.Index(fields=["timeframe","ts"])]
        ordering = ["asset","timeframe","ts"]

class PriceSnapshot(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE, related_name="snapshots")
    ts = models.DateTimeField(default=timezone.now, db_index=True)
    price = models.DecimalField(max_digits=20, decimal_places=8)
    bid = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    ask = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    source = models.CharField(max_length=30, blank=True)

    class Meta:
        ordering = ["-ts"]
        indexes = [models.Index(fields=["asset","ts"])]
