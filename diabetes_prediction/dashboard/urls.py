from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('user-management/', views.user_management_view, name='user_management'),
    path('analytics/', views.analytics_view, name='analytics'),
]