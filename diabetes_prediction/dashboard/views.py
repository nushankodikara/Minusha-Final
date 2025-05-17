from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from datetime import timedelta
from django.utils import timezone
from django.contrib.auth.models import User
from django.db.models import Avg, Count

# Import models from prediction app
from prediction.models import PredictionResult
from prediction.models import SugarReport

@login_required
def dashboard_view(request):  # Renamed to match URL configuration
    """Dashboard view that shows user's predictions and reports summary"""
    # Get recent predictions for this user
    predictions = PredictionResult.objects.filter(
        user_input__user=request.user  # Using user_input__user assuming this is the relationship
    ).order_by('-created_at')[:5]
    
    # Calculate days active
    first_activity = PredictionResult.objects.filter(
        user_input__user=request.user
    ).order_by('created_at').first()
    
    days_active = 0
    if first_activity:
        delta = timezone.now() - first_activity.created_at
        days_active = delta.days
    
    # Count total predictions
    predictions_count = PredictionResult.objects.filter(user_input__user=request.user).count()
    
    # Get uploaded reports
    try:
        reports = SugarReport.objects.filter(user=request.user).order_by('-uploaded_at')[:3]
        uploads_count = SugarReport.objects.filter(user=request.user).count()
    except:
        reports = []
        uploads_count = 0
    
    context = {
        'predictions': predictions,
        'predictions_count': predictions_count,
        'reports': reports,
        'uploads_count': uploads_count,
        'days_active': days_active,
    }
    
    return render(request, 'dashboard/dashboard.html', context)

@login_required
def user_management_view(request):
    # Get all users (for admin only)
    if request.user.is_staff:
        users = User.objects.all()
    else:
        users = User.objects.filter(id=request.user.id)
    
    context = {
        'users': users,
    }
    
    return render(request, 'dashboard/user_management.html', context)
@login_required
def analytics_view(request):
    # Get prediction statistics
    total_predictions = PredictionResult.objects.filter(user_input__user=request.user).count()
    
    # Get average glucose level
    avg_glucose = PredictionResult.objects.filter(user_input__user=request.user).aggregate(Avg('glucose'))['glucose__avg']
    
    # Get average BMI
    avg_bmi = PredictionResult.objects.filter(user_input__user=request.user).aggregate(Avg('bmi'))['bmi__avg']
    
    # Get prediction distribution by month
    # This is a simplified version - in a real app, you'd use more complex queries
    monthly_predictions = PredictionResult.objects.filter(user_input__user=request.user).values('created_at__month').annotate(count=Count('id'))
    
    context = {
        'total_predictions': total_predictions,
        'avg_glucose': avg_glucose,
        'avg_bmi': avg_bmi,
        'monthly_predictions': monthly_predictions,
    }
    
    return render(request, 'dashboard/analytics.html', context)