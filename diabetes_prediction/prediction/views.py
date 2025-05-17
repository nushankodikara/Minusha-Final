from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator

from .forms import PredictionForm, SugarReportForm
from .models import PredictionResult, SugarReport, Recommendation
from .ml_models.predictor import DiabetesPredictor
from .ml_models.ocr_processor import OCRProcessor
from .ml_models.recommendation_generator import RecommendationGenerator

from prediction.models import Prediction

import os
import pickle

# MODEL_PATH = 'C:\\Users\\Minusha Attygala\\Downloads\\diabetes_prediction - Copy\\diabetes_prediction\\prediction\\ml_models\\diabetes_model.pkl'

# with open(MODEL_PATH, 'rb') as file:
#     model = pickle.load(file)

def predict_diabetes(request):
    if request.method == 'POST':
        # Get data from the form
        gender = int(request.POST['gender'])  # Example: 0 for female, 1 for male
        age = int(request.POST['age'])
        height = float(request.POST['height'])  # in cm
        weight = float(request.POST['weight'])  # in kg
        activity = int(request.POST['activity'])  # physical activity scale (custom)

        # Calculate BMI
        height_m = height / 100  # convert cm to meters
        bmi = weight / (height_m ** 2)

        # Form the input vector
        input_data = [[gender, age, bmi, activity]]

        # Make prediction
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = 'Positive for Diabetes'
        else:
            result = 'Negative for Diabetes'

        return render(request, 'result.html', {'result': result})

    return render(request, 'home.html')  # form page


def home(request):
    """Home page view"""
    return render(request, 'prediction/home.html')

def about(request):
    """About page view"""
    return render(request, 'prediction/about.html')

@login_required
def contact(request):
    """Contact page view"""
    return render(request, 'prediction/contact.html')

@login_required
def prediction_form(request):
    """View for the prediction form"""
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Create prediction object but don't save yet
            prediction = form.save(commit=False)
            prediction.user = request.user
            
            # Get form data
            user_data = {
                'gender': form.cleaned_data['gender'],
                'age': form.cleaned_data['age'],
                'height': form.cleaned_data['height'],
                'weight': form.cleaned_data['weight'],
                'exercise_habits': form.cleaned_data['exercise_habits']
            }
            
            # Calculate BMI
            bmi = user_data['weight'] / ((user_data['height'] / 100) ** 2)
            prediction.bmi = bmi
            user_data['bmi'] = bmi
            
            # Make prediction
            predictor = DiabetesPredictor()
            result = predictor.predict(user_data)
            
            # Update prediction object with results
            prediction.prediction = result['prediction']
            prediction.probability = result['probability']
            prediction.risk_level = result['risk_level']
            
            # Save prediction
            prediction.save()
            
            # Generate recommendations
            recommendation_generator = RecommendationGenerator()
            recommendations = recommendation_generator.generate_recommendations(result, user_data)
            
            # Save recommendations
            for rec in recommendations:
                Recommendation.objects.create(
                    user=request.user,
                    prediction=prediction,
                    category=rec['category'],
                    title=rec['title'],
                    description=rec['description']
                )
            
            # Redirect to results page
            return redirect('prediction_results', prediction_id=prediction.id)
    else:
        form = PredictionForm()
    
    return render(request, 'prediction/prediction_form.html', {'form': form})

# Example usage in a view function
@login_required
def make_prediction(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extract features from form
            features = [
                # Order these according to how your model was trained
                form.cleaned_data['age'],
                form.cleaned_data['bmi'],
                # Add other features your model expects
            ]
            
            # Get prediction
            predictor = DiabetesPredictor()
            result = predictor.predict(features)
            prediction = result['prediction']
            probability = result['probability']
            
            # Save prediction result
            result = PredictionResult(
                user=request.user,
                age=form.cleaned_data['age'],
                # Set other fields...
                prediction=prediction,
                probability=probability,
            )
            result.save()
            
            # Redirect or render response
            # ...

@login_required
def prediction_results(request, prediction_id):
    """View for displaying prediction results"""
    prediction = get_object_or_404(PredictionResult, id=prediction_id, user=request.user)
    recommendations = prediction.recommendations.all()
    
    # Calculate probability as a percentage for template display
    probability_percentage = 0
    if prediction.probability is not None:
        probability_percentage = prediction.probability * 100

    context = {
        'prediction': prediction,
        'recommendations': recommendations,
        'probability_percentage': probability_percentage  # Add this to the context
    }
    
    return render(request, 'prediction/results.html', context)

@login_required
def recommendations(request, prediction_id=None):
    """View for displaying recommendations"""
    if prediction_id:
        # Show recommendations for a specific prediction
        prediction = get_object_or_404(PredictionResult, id=prediction_id, user=request.user)
        recommendations = prediction.recommendations.all()
    else:
        # Show the latest recommendations
        latest_prediction = PredictionResult.objects.filter(user=request.user).order_by('-created_at').first()
        if latest_prediction:
            recommendations = latest_prediction.recommendations.all()
            prediction = latest_prediction
        else:
            recommendations = []
            prediction = None
    
    context = {
        'recommendations': recommendations,
        'prediction': prediction
    }
    
    return render(request, 'prediction/recommendations.html', context)

@login_required
def upload_sugar_report(request):
    """View for uploading sugar reports"""
    if request.method == 'POST':
        form = SugarReportForm(request.POST, request.FILES)
        if form.is_valid():
            # Create sugar report object but don't save yet
            sugar_report = form.save(commit=False)
            sugar_report.user = request.user
            sugar_report.save()
            
            # Process the report using OCR
            ocr_processor = OCRProcessor()
            data = ocr_processor.process_report(sugar_report.report_file.path)
            
            # Update sugar report with extracted data
            sugar_report.glucose_level = data.get('glucose_level')
            sugar_report.hba1c = data.get('hba1c')
            sugar_report.fasting_glucose = data.get('fasting_glucose')
            sugar_report.postprandial_glucose = data.get('postprandial_glucose')
            sugar_report.processed = True
            sugar_report.save()
            
            messages.success(request, 'Sugar report uploaded and processed successfully.')
            return redirect('sugar_report_detail', report_id=sugar_report.id)
    else:
        form = SugarReportForm()
    
    # Get user's previous reports
    user_reports = SugarReport.objects.filter(user=request.user).order_by('-uploaded_at')
    paginator = Paginator(user_reports, 5)  # Show 5 reports per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'form': form,
        'page_obj': page_obj
    }
    
    return render(request, 'prediction/upload_sugar_report.html', context)

@login_required
def sugar_report_detail(request, report_id):
    """View for displaying sugar report details"""
    report = get_object_or_404(SugarReport, id=report_id, user=request.user)
    
    context = {
        'report': report
    }
    
    return render(request, 'prediction/sugar_report_detail.html', context)

@login_required
def prediction_history(request):
    """View for displaying prediction history"""
    predictions = PredictionResult.objects.filter(user=request.user).order_by('-created_at')
    paginator = Paginator(predictions, 10)  # Show 10 predictions per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj
    }
    
    return render(request, 'prediction/prediction_history.html', context)

@require_POST
@login_required
def delete_prediction(request, prediction_id):
    """View for deleting a prediction"""
    prediction = get_object_or_404(PredictionResult, id=prediction_id, user=request.user)
    prediction.delete()
    messages.success(request, 'Prediction deleted successfully.')
    return redirect('prediction_history')

@require_POST
@login_required
def delete_sugar_report(request, report_id):
    """View for deleting a sugar report"""
    report = get_object_or_404(SugarReport, id=report_id, user=request.user)
    report.delete()
    messages.success(request, 'Sugar report deleted successfully.')
    return redirect('upload_sugar_report')

@login_required
def dashboard(request):
    # Get recent predictions
    recent_predictions = Prediction.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    # Get total predictions count
    total_predictions = Prediction.objects.filter(user=request.user).count()
    
    # Get positive predictions count
    positive_predictions = Prediction.objects.filter(user=request.user, outcome=True).count()
    
    context = {
        'recent_predictions': recent_predictions,
        'total_predictions': total_predictions,
        'positive_predictions': positive_predictions,
        'negative_predictions': total_predictions - positive_predictions,
    }
    
    return render(request, 'dashboard/dashboard.html', context)
