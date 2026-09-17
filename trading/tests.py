from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from .models import Trade

User = get_user_model()


class TradeModelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="tradeuser",
            password="test-password-123",
        )

    def create_trade(self, **overrides):
        values = {
            "user": self.user,
            "symbol": "XAUUSD",
            "side": "LONG",
            "quantity": Decimal("2"),
            "entry_price": Decimal("100"),
            "status": "OPEN",
            "fees": Decimal("0"),
        }
        values.update(overrides)
        return Trade.objects.create(**values)

    def test_trade_rr_long(self):
        trade = self.create_trade(
            side="LONG",
            entry_price=Decimal("100"),
            stop_loss=Decimal("90"),
            take_profit=Decimal("120"),
        )

        self.assertEqual(trade.rr(), 2.0)

    def test_trade_rr_short(self):
        trade = self.create_trade(
            side="SHORT",
            entry_price=Decimal("100"),
            stop_loss=Decimal("110"),
            take_profit=Decimal("80"),
        )

        self.assertEqual(trade.rr(), 2.0)

    def test_trade_rr_missing_sl_tp(self):
        trade = self.create_trade(
            stop_loss=None,
            take_profit=None,
        )

        self.assertIsNone(trade.rr())

    def test_trade_rr_invalid_risk_returns_none(self):
        trade = self.create_trade(
            side="LONG",
            entry_price=Decimal("100"),
            stop_loss=Decimal("110"),
            take_profit=Decimal("120"),
        )

        self.assertIsNone(trade.rr())

    def test_trade_realized_pnl_long(self):
        trade = self.create_trade(
            side="LONG",
            status="CLOSED",
            entry_price=Decimal("100"),
            exit_price=Decimal("110"),
            quantity=Decimal("2"),
            fees=Decimal("1"),
        )

        self.assertEqual(trade.realized_pnl(), 19.0)

    def test_trade_realized_pnl_short(self):
        trade = self.create_trade(
            side="SHORT",
            status="CLOSED",
            entry_price=Decimal("100"),
            exit_price=Decimal("90"),
            quantity=Decimal("2"),
            fees=Decimal("1"),
        )

        self.assertEqual(trade.realized_pnl(), 19.0)

    def test_trade_realized_pnl_open(self):
        trade = self.create_trade(
            side="LONG",
            status="OPEN",
            entry_price=Decimal("100"),
            exit_price=None,
        )

        self.assertIsNone(trade.realized_pnl())

    def test_trade_duration_minutes(self):
        open_time = timezone.now()
        close_time = open_time + timedelta(minutes=90)

        trade = self.create_trade(
            open_time=open_time,
            close_time=close_time,
        )

        self.assertEqual(trade.duration_minutes(), 90)


class TradeViewAuthTests(TestCase):
    def test_trade_list_requires_login(self):
        response = self.client.get(reverse("trading:trade_list"))

        self.assertEqual(response.status_code, 302)
# Create your tests here.
