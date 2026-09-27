from django import forms

from .market_data import MAX_MARKET_DATA_BYTES
from .models import TradingFile
from .time_basis import (
    TIME_BASIS_FIXED_OFFSET,
    TIME_BASIS_IANA,
    TIME_BASIS_UTC,
    TimeBasisValidationError,
    create_time_basis,
)

DECLARED_EXPORT_CTRADER_CBOT = "CTRADER_CBOT_BARS_OPENTIMES"


class TradingFileForm(forms.ModelForm):
    class Meta:
        model = TradingFile
        fields = ["file"]


class FilterForm(forms.Form):
    start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={"type": "date"}))
    symbol = forms.CharField(required=False, max_length=50)


class MarketDataUploadForm(forms.Form):
    market_file = forms.FileField(
        label="Uploaded cTrader M1 bars",
        help_text="CSV only. Application-level size limit uses the closed M1 contract.",
        widget=forms.ClearableFileInput(attrs={"accept": ".csv"}),
    )
    declared_source = forms.CharField(
        label="Declared source",
        max_length=120,
        help_text="User-declared source metadata only. This is not a verified broker origin.",
    )
    declared_export_method = forms.ChoiceField(
        label="Declared export method",
        choices=(
            (
                DECLARED_EXPORT_CTRADER_CBOT,
                "cTrader cBot - Bars.OpenTimes",
            ),
        ),
        help_text="Confirm the declared cTrader cBot Bars.OpenTimes export method.",
    )
    time_basis_kind = forms.ChoiceField(
        label="Journal time basis",
        choices=(
            (TIME_BASIS_UTC, "UTC"),
            (TIME_BASIS_FIXED_OFFSET, "Fixed offset"),
            (TIME_BASIS_IANA, "IANA zone"),
        ),
        help_text="Required user declaration. No timezone is inferred.",
    )
    offset_minutes = forms.IntegerField(
        required=False,
        label="Fixed offset (minutes)",
        help_text="Required for Fixed offset. Must be a 15-minute step from -720 to 840.",
    )
    iana_zone = forms.CharField(
        required=False,
        max_length=64,
        label="IANA zone",
        help_text="Required for IANA, for example Europe/London.",
    )

    def clean_market_file(self):
        uploaded = self.cleaned_data.get("market_file")
        if uploaded is not None and uploaded.size > MAX_MARKET_DATA_BYTES:
            raise forms.ValidationError(
                "The uploaded market file exceeds the permitted size."
            )
        return uploaded

    def clean_iana_zone(self):
        zone = self.cleaned_data.get("iana_zone")
        if zone is None:
            return None
        text = str(zone).strip()
        return text or None

    def clean(self):
        cleaned = super().clean()
        kind = cleaned.get("time_basis_kind")
        offset = cleaned.get("offset_minutes")
        zone = cleaned.get("iana_zone")
        try:
            time_basis = create_time_basis(
                kind,
                offset_minutes=offset,
                zone=zone,
            )
        except TimeBasisValidationError as exc:
            self.add_error(None, exc.reason)
            return cleaned
        cleaned["time_basis"] = time_basis
        return cleaned
