from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView, PasswordResetView, PasswordResetConfirmView
from django.urls import reverse_lazy
from .forms import UserRegisterForm, UserLoginForm, UserUpdateForm, ProfileUpdateForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm

def register(request):
    """View for user registration"""
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'accounts/register.html', {'form': form})

class CustomLoginView(LoginView):
    """Custom login view with custom form"""
    form_class = UserLoginForm
    template_name = 'accounts/login.html'

class CustomLogoutView(LogoutView):
    """Custom logout view"""
    next_page = 'home'

class CustomPasswordResetView(PasswordResetView):
    """Custom password reset view"""
    template_name = 'accounts/password_reset.html'
    email_template_name = 'accounts/password_reset_email.html'
    success_url = reverse_lazy('password_reset_done')

class CustomPasswordResetConfirmView(PasswordResetConfirmView):
    """Custom password reset confirm view"""
    template_name = 'accounts/password_reset_confirm.html'
    success_url = reverse_lazy('password_reset_complete')

@login_required
def profile(request):
    """View for user profile"""
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)
        p_form = ProfileUpdateForm(request.POST, instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, 'Your profile has been updated!')
            return redirect('profile')
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)
    
    context = {
        'u_form': u_form,
        'p_form': p_form
    }
    
    return render(request, 'accounts/profile.html', context)

def terms_and_conditions(request):
    """View for Terms and Conditions page"""
    return render(request, 'accounts/terms.html')

def privacy_policy(request):
    """View for Privacy Policy page"""
    return render(request, 'accounts/privacy.html')

def faq(request):
    """View for FAQ page"""
    return render(request, 'accounts/faq.html')

def login_view(request):
    # Check if there's a 'next' parameter in the URL
    next_url = request.GET.get('next', '')
    
    if next_url and 'prediction' in next_url:
        messages.info(request, "Please log in to access the prediction feature. This helps us save your results and provide personalized recommendations.")
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect(request.POST.get('next', 'home'))
    else:
        form = AuthenticationForm()
    
    return render(request, 'accounts/login.html', {'form': form, 'next': next_url})