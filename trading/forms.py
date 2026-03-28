# trading/forms.py
from django import forms
from .models import Trade

class TradeForm(forms.ModelForm):
    class Meta:
        model = Trade
        fields = [
            "symbol", "side", "quantity", "entry_price", "exit_price",
            "stop_loss", "take_profit", "open_time", "close_time",
            "status", "fees", "tag", "notes"
        ]
        widgets = {
            "symbol": forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. XAUUSD"}),
            "side": forms.Select(attrs={"class": "form-select"}),
            "quantity": forms.NumberInput(attrs={"class": "form-control", "step":"0.01"}),
            "entry_price": forms.NumberInput(attrs={"class": "form-control", "step":"0.0001"}),
            "exit_price": forms.NumberInput(attrs={"class": "form-control", "step":"0.0001"}),
            "stop_loss": forms.NumberInput(attrs={"class": "form-control", "step":"0.0001"}),
            "take_profit": forms.NumberInput(attrs={"class": "form-control", "step":"0.0001"}),
            "open_time": forms.DateTimeInput(attrs={"class": "form-control", "type":"datetime-local"}),
            "close_time": forms.DateTimeInput(attrs={"class": "form-control", "type":"datetime-local"}),
            "status": forms.Select(attrs={"class": "form-select"}),
            "fees": forms.NumberInput(attrs={"class": "form-control", "step":"0.01"}),
            "tag": forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. Breakout"}),
            "notes": forms.Textarea(attrs={"class": "form-control", "rows": 3}),
        }
