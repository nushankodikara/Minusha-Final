from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('predict/', views.prediction_form, name='prediction_form'),
    path('results/<int:prediction_id>/', views.prediction_results, name='prediction_results'),
    path('recommendations/', views.recommendations, name='recommendations'),
    path('recommendations/<int:prediction_id>/', views.recommendations, name='prediction_recommendations'),
    path('upload-report/', views.upload_sugar_report, name='upload_report'),
    path('sugar-report/<int:report_id>/', views.sugar_report_detail, name='sugar_report_detail'),
    path('prediction-history/', views.prediction_history, name='prediction_history'),
    path('delete-prediction/<int:prediction_id>/', views.delete_prediction, name='delete_prediction'),
    path('delete-sugar-report/<int:report_id>/', views.delete_sugar_report, name='delete_sugar_report'),
    path('dashboard/', views.dashboard, name='dashboard'),
]