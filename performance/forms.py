from django import forms
from .models import TradingFile
from django import forms
class TradingFileForm(forms.ModelForm):
    class Meta:
        model = TradingFile
        fields = ['file']


class FilterForm(forms.Form):
    start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    symbol = forms.CharField(required=False, max_length=50)
