from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from .models import UserProfile

User = get_user_model()


class UserProfileTests(TestCase):
    def test_userprofile_str_display_name(self):
        user = User.objects.create_user(
            username="profileuser",
            password="test-password-123",
        )

        profile = user.userprofile
        profile.display_name = "Aminul Islam"
        profile.save()

        self.assertEqual(str(profile), "Aminul Islam")

    def test_userprofile_str_username_fallback(self):
        user = User.objects.create_user(
            username="fallbackuser",
            password="test-password-123",
        )

        profile = user.userprofile
        profile.display_name = ""
        profile.save()

        self.assertEqual(str(profile), "fallbackuser")

    def test_user_creation_creates_userprofile(self):
        user = User.objects.create_user(
            username="signaluser",
            password="test-password-123",
        )

        self.assertTrue(
            UserProfile.objects.filter(user=user).exists()
        )


class CoreViewTests(TestCase):
    def test_home_view_unauthenticated(self):
        response = self.client.get(reverse("core:home"))

        self.assertEqual(response.status_code, 200)

    def test_profile_view_requires_login(self):
        response = self.client.get(reverse("core:profile"))

        self.assertEqual(response.status_code, 302)
# Create your tests here.
