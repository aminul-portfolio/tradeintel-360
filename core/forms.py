from django import forms
from django.contrib.auth.models import User
from .models import UserProfile

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']
        labels = {
            'first_name': 'First Name',
            'last_name': 'Last Name',
            'email': 'Email Address'
        }



class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = [
            'avatar', 'display_name', 'timezone', 'bio',
            'preferred_assets', 'daily_risk_limit_pct',
            'notify_email', 'notify_sound', 'two_factor_enabled'
        ]
        widgets = {
            'display_name': forms.TextInput(attrs={'class': 'form-control'}),
            'timezone': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g. Europe/London'}),
            'bio': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'preferred_assets': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'BTCUSD, XAUUSD, US100'}),
            'daily_risk_limit_pct': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01'}),
            'notify_email': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'notify_sound': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'two_factor_enabled': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
