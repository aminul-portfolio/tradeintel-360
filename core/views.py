from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import render, redirect
from .forms import UserForm, UserProfileForm
from django.contrib.auth.decorators import login_required
from .forms import UserForm, UserProfileForm
from django.shortcuts import render, redirect

def home(request):
    return render(request, 'home.html')


@login_required
def profile(request):
    user = request.user
    profile = getattr(user, 'userprofile', None)  # signals create it, but be safe

    if request.method == 'POST':
        uform = UserForm(request.POST, instance=user)
        pform = UserProfileForm(request.POST, request.FILES, instance=profile)
        if uform.is_valid() and pform.is_valid():
            uform.save()
            pform.save()
            messages.success(request, 'Profile updated successfully.')
            return redirect('core:profile')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        uform = UserForm(instance=user)
        pform = UserProfileForm(instance=profile)

    return render(request, 'core/profile.html', {'uform': uform, 'pform': pform})
# trading/views.py

@login_required
def strategy_risk(request):
    """
    Strategy-based position sizing using a scaled Kelly fraction + ATR stop.
    Inputs via GET; returns suggested risk % and lot size.
    """
    result = None
    # Defaults for initial load
    defaults = {
        "balance": request.GET.get("balance", "10000"),
        "win_rate": request.GET.get("win_rate", "55"),
        "rr": request.GET.get("rr", "1.8"),
        "atr_pips": request.GET.get("atr_pips", "20"),
        "atr_mult": request.GET.get("atr_mult", "1.5"),
        "kelly_scale": request.GET.get("kelly_scale", "0.25"),
        "max_risk_pct": request.GET.get("max_risk_pct", "1.0"),
        "pip_value": request.GET.get("pip_value", "1"),
        "floor_pct": request.GET.get("floor_pct", "0.10"),  # minimum risk% floor
    }

    try:
        balance = float(defaults["balance"])
        win_rate = float(defaults["win_rate"]) / 100.0
        rr = float(defaults["rr"])
        atr_pips = float(defaults["atr_pips"])
        atr_mult = float(defaults["atr_mult"])
        kelly_scale = float(defaults["kelly_scale"])       # 0..1
        max_risk_pct = float(defaults["max_risk_pct"])
        pip_value = float(defaults["pip_value"])
        floor_pct = float(defaults["floor_pct"])

        # Only compute when enough inputs are provided
        if balance > 0 and 0 < win_rate < 1 and rr > 0 and atr_pips > 0 and atr_mult > 0 and pip_value > 0:
            # Kelly fraction
            raw_kelly = win_rate - ((1 - win_rate) / rr)
            # guard rails
            if raw_kelly < 0:
                raw_kelly = floor_pct / 100.0

            scaled_pct = raw_kelly * kelly_scale * 100.0
            risk_pct = max(floor_pct, min(scaled_pct, max_risk_pct))

            stop_pips = atr_pips * atr_mult
            risk_amount = balance * (risk_pct / 100.0)
            lot = risk_amount / (stop_pips * pip_value) if stop_pips > 0 else None

            result = {
                "raw_kelly_pct": round(raw_kelly * 100.0, 3),
                "risk_pct": round(risk_pct, 3),
                "risk_amount": round(risk_amount, 2),
                "stop_pips": round(stop_pips, 2),
                "lot": round(lot, 2) if lot else None,
            }
    except Exception:
        result = None

    context = {"defaults": defaults, "result": result}
    return render(request, "trading/strategy_risk.html", context)





@login_required
def edit_profile(request):
    user = request.user
    profile = user.userprofile

    if request.method == 'POST':
        user_form = UserForm(request.POST, instance=user)
        profile_form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, "Your profile was updated successfully.")
            return redirect('core:edit_profile')  # Redirect to same page or change to 'profile'
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        user_form = UserForm(instance=user)
        profile_form = UserProfileForm(instance=profile)

    return render(request, 'core/edit_profile.html', {
        'user_form': user_form,
        'profile_form': profile_form
    })
