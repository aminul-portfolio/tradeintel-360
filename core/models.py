from django.conf import settings
from django.db import models

class UserProfile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)

    # Display & contact
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    display_name = models.CharField(max_length=100, blank=True)
    timezone = models.CharField(max_length=64, default='UTC')
    bio = models.TextField(blank=True)

    # TradeIntel preferences
    preferred_assets = models.CharField(max_length=200, blank=True)  # e.g. "BTCUSD, XAUUSD"
    daily_risk_limit_pct = models.DecimalField(max_digits=5, decimal_places=2, default=2)
    notify_email = models.BooleanField(default=True)
    notify_sound = models.BooleanField(default=True)
    two_factor_enabled = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.display_name or self.user.username
