from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect

from .forms import UserForm, UserProfileForm


def home(request):
    return render(request, 'home.html')


@login_required
def profile(request):
    user    = request.user
    profile = getattr(user, 'userprofile', None)

    if request.method == 'POST':
        uform = UserForm(request.POST, instance=user)
        pform = UserProfileForm(request.POST, request.FILES, instance=profile)
        if uform.is_valid() and pform.is_valid():
            uform.save()
            pform.save()
            messages.success(request, 'Profile updated successfully.')
            return redirect('core:profile')
        messages.error(request, 'Please correct the errors below.')
    else:
        uform = UserForm(instance=user)
        pform = UserProfileForm(instance=profile)

    return render(request, 'core/profile.html', {'uform': uform, 'pform': pform})


@login_required
def edit_profile(request):
    user    = request.user
    profile = user.userprofile

    if request.method == 'POST':
        user_form    = UserForm(request.POST, instance=user)
        profile_form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile was updated successfully.')
            return redirect('core:edit_profile')
        messages.error(request, 'Please correct the errors below.')
    else:
        user_form    = UserForm(instance=user)
        profile_form = UserProfileForm(instance=profile)

    return render(request, 'core/edit_profile.html', {
        'user_form':    user_form,
        'profile_form': profile_form,
    })