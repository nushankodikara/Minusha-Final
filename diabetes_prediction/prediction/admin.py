from django.contrib import admin
from .models import PredictionResult, SugarReport, Recommendation

class RecommendationInline(admin.TabularInline):
    model = Recommendation
    extra = 0

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at', 'gender', 'age', 'bmi', 'prediction', 'risk_level')
    list_filter = ('prediction', 'risk_level', 'gender', 'created_at')
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('bmi', 'prediction', 'probability', 'risk_level')
    inlines = [RecommendationInline]

@admin.register(SugarReport)
class SugarReportAdmin(admin.ModelAdmin):
    list_display = ('user', 'uploaded_at', 'processed', 'glucose_level', 'hba1c')
    list_filter = ('processed', 'uploaded_at')
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('processed', 'glucose_level', 'hba1c', 'fasting_glucose', 'postprandial_glucose')

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ('user', 'category', 'title', 'created_at')
    list_filter = ('category', 'created_at')
    search_fields = ('user__username', 'user__email', 'title', 'description')