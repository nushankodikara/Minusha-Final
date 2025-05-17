from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

import os
import pickle
import numpy as np

class PredictionResult(models.Model):
    """Model to store diabetes prediction results"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female'), ('other', 'Other')])
    age = models.PositiveIntegerField()
    height = models.FloatField(help_text="Height in cm")
    weight = models.FloatField(help_text="Weight in kg")
    bmi = models.FloatField(null=True, blank=True)
    exercise_habits = models.CharField(
        max_length=20, 
        choices=[
            ('sedentary', 'Sedentary (little or no exercise)'),
            ('light', 'Light (exercise 1-3 times/week)'),
            ('moderate', 'Moderate (exercise 3-5 times/week)'),
            ('active', 'Active (exercise 6-7 times/week)'),
            ('very_active', 'Very Active (exercise multiple times/day)')
        ]
    )
    prediction = models.BooleanField(help_text="True if diabetes predicted, False otherwise")
    probability = models.FloatField(null=True, blank=True, help_text="Probability of diabetes")
    risk_level = models.CharField(
        max_length=20, 
        choices=[
            ('low', 'Low'),
            ('moderate', 'Moderate'),
            ('high', 'High'),
            ('very_high', 'Very High')
        ],
        null=True, 
        blank=True
    )
    created_at = models.DateTimeField(default=timezone.now)
    
    def save(self, *args, **kwargs):
        # Calculate BMI if not provided
        if not self.bmi and self.height and self.weight:
            self.bmi = self.weight / ((self.height / 100) ** 2)
        super().save(*args, **kwargs)
    
    def __str__(self):
        return f"Prediction for {self.user.username} on {self.created_at.strftime('%Y-%m-%d')}"
    
    class Meta:
        ordering = ['-created_at']

class SugarReport(models.Model):
    """Model to store uploaded sugar reports"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sugar_reports')
    report_file = models.FileField(upload_to='sugar_reports/')
    uploaded_at = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)
    
    # Extracted data fields
    glucose_level = models.FloatField(null=True, blank=True)
    hba1c = models.FloatField(null=True, blank=True)
    fasting_glucose = models.FloatField(null=True, blank=True)
    postprandial_glucose = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"Sugar report for {self.user.username} on {self.uploaded_at.strftime('%Y-%m-%d')}"
    
    class Meta:
        ordering = ['-uploaded_at']

class Recommendation(models.Model):
    """Model to store personalized recommendations"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recommendations')
    prediction = models.ForeignKey(PredictionResult, on_delete=models.CASCADE, related_name='recommendations')
    category = models.CharField(
        max_length=20, 
        choices=[
            ('diet', 'Diet'),
            ('exercise', 'Exercise'),
            ('lifestyle', 'Lifestyle'),
            ('medical', 'Medical')
        ]
    )
    title = models.CharField(max_length=100)
    description = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.category} recommendation for {self.user.username}"
    
    class Meta:
        ordering = ['-created_at']

class Prediction(models.Model):
    user_input = models.TextField()
    prediction_result = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction: {self.prediction_result}"

class DiabetesModel:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the serialized diabetes prediction model"""
        model_path = os.path.join(os.path.dirname(__file__), 'diabetes_model.pkl')
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, features):
        """
        Make diabetes prediction using the loaded model
        
        Args:
            features: Array or list of features in the correct order expected by model
            
        Returns:
            prediction: Boolean indicating diabetes prediction
            probability: Float indicating probability of positive class
        """
        if self.model is None:
            raise ValueError("Model not loaded")
            
        # Convert features to numpy array and reshape for single prediction
        features_array = np.array(features).reshape(1, -1)
        
        # Get prediction
        prediction = bool(self.model.predict(features_array)[0])
        
        # Get probability
        try:
            probability = float(self.model.predict_proba(features_array)[0][1] * 100)
        except:
            probability = None
            
        return prediction, probability


# Create singleton instance
diabetes_model = DiabetesModel()