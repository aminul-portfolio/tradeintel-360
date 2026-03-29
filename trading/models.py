from django.conf import settings
from django.db import models
from django.utils import timezone


class Trade(models.Model):
    SIDE_CHOICES = (
        ("LONG", "Long"),
        ("SHORT", "Short"),
    )

    STATUS_CHOICES = (
        ("OPEN", "Open"),
        ("CLOSED", "Closed"),
    )

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="trades",
    )

    # Standalone symbol field for TradeIntel 360 post-trade review.
    symbol = models.CharField(max_length=20)

    side = models.CharField(max_length=5, choices=SIDE_CHOICES)
    quantity = models.DecimalField(max_digits=20, decimal_places=8, help_text="Lots or units")
    entry_price = models.DecimalField(max_digits=20, decimal_places=8)
    exit_price = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)

    stop_loss = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)
    take_profit = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)

    open_time = models.DateTimeField(default=timezone.now)
    close_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=6, choices=STATUS_CHOICES, default="OPEN")

    fees = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    notes = models.TextField(blank=True)
    tag = models.CharField(max_length=50, blank=True)

    # Cached outcome for convenience in review workflows
    pnl = models.DecimalField(max_digits=20, decimal_places=8, null=True, blank=True)

    class Meta:
        ordering = ["-open_time"]
        indexes = [
            models.Index(fields=["user", "symbol"]),
            models.Index(fields=["status"]),
            models.Index(fields=["open_time"]),
        ]

    def __str__(self):
        return f"{self.user.username} {self.side} {self.symbol} @ {self.entry_price}"

    def rr(self):
        """
        Risk/reward ratio based on entry, stop loss, and take profit.
        Used as a helper for trade review; not a standalone calculator surface.
        """
        if not self.stop_loss or not self.take_profit:
            return None

        entry = float(self.entry_price)
        sl = float(self.stop_loss)
        tp = float(self.take_profit)

        if self.side == "LONG":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp

        if risk <= 0:
            return None

        return round(reward / risk, 2)

    def realized_pnl(self):
        """
        Realized PnL for closed trades only.
        """
        if self.status != "CLOSED" or not self.exit_price:
            return None

        entry = float(self.entry_price)
        exit_ = float(self.exit_price)
        qty = float(self.quantity)
        fees = float(self.fees or 0)

        diff = (exit_ - entry) if self.side == "LONG" else (entry - exit_)
        return round(diff * qty - fees, 8)

    def duration_minutes(self):
        if self.close_time and self.open_time:
            return int((self.close_time - self.open_time).total_seconds() / 60)
        return None