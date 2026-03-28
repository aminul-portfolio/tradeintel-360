from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "display_name", "timezone", "daily_risk_limit_pct", "notify_email")
    search_fields = ("user__username", "display_name")
