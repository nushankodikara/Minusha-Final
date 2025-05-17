import os
import joblib
import numpy as np
import pandas as pd
from django.conf import settings

class DiabetesPredictor:
    """Class for making diabetes predictions using the trained model"""
    
    def __init__(self):
        # Path to the model files
        model_dir = os.path.join(settings.BASE_DIR, 'prediction', 'ml_models', 'models')
        
        # Load the model (adjust filename as needed)
        self.model_path = os.path.join(model_dir, 'tuned_xgboost_model.pkl')
        self.scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        # Check if model files exist
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Model files not found. Please train the model first.")
        
        # Load model and scaler
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
    
    def preprocess_input(self, user_data):
        """
        Preprocess user input data
        
        Parameters:
        user_data (dict): User input data
        
        Returns:
        array: Preprocessed features
        """
        # Convert user data to DataFrame
        df = pd.DataFrame([user_data])
        
        # Map exercise habits to numerical values
        exercise_mapping = {
            'sedentary': 0,
            'light': 1,
            'moderate': 2,
            'active': 3,
            'very_active': 4
        }
        
        if 'exercise_habits' in df.columns:
            df['exercise_numeric'] = df['exercise_habits'].map(exercise_mapping)
            df.drop('exercise_habits', axis=1, inplace=True)
        
        # Map gender to numerical values
        gender_mapping = {
            'male': 0,
            'female': 1,
            'other': 2
        }
        
        if 'gender' in df.columns:
            df['gender_numeric'] = df['gender'].map(gender_mapping)
            df.drop('gender', axis=1, inplace=True)
        
        # Calculate BMI if height and weight are provided
        if 'height' in df.columns and 'weight' in df.columns and 'bmi' not in df.columns:
            df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        
        # Create age groups
        if 'age' in df.columns:
            bins = [0, 30, 45, 60, 120]
            labels = ['Young', 'Middle-aged', 'Senior', 'Elderly']
            df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
            # One-hot encode age groups
            age_dummies = pd.get_dummies(df['age_group'], prefix='age')
            df = pd.concat([df, age_dummies], axis=1)
            df.drop('age_group', axis=1, inplace=True)
        
        # Select only the features used during training
        # This list should match the features used during model training
        features = [
            'age', 'gender_numeric', 'height', 'weight', 'bmi', 'exercise_numeric',
            'age_Young', 'age_Middle-aged', 'age_Senior', 'age_Elderly'
        ]
        
        # Keep only available features
        available_features = [f for f in features if f in df.columns]
        
        # Fill missing features with zeros
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        
        return df[features]
    
    def predict(self, user_data):
        """
        Predict diabetes risk based on user input
        
        Parameters:
        user_data (dict): User input data
        
        Returns:
        dict: Prediction result and probability
        """
        try:
            # Preprocess input
            processed_data = self.preprocess_input(user_data)
            
            # Scale the data
            scaled_data = self.scaler.transform(processed_data)
            
            # Make prediction
            prediction = self.model.predict(scaled_data)[0]
            
            # Get probability
            probability = self.model.predict_proba(scaled_data)[0][1]  # Probability of class 1 (diabetes)
            
            # Determine risk level
            risk_level = self.get_risk_level(probability)
            
            # Return result
            result = {
                'prediction': bool(prediction),  # True if diabetes predicted, False otherwise
                'probability': float(probability),
                'risk_level': risk_level
            }
            
            return result
        
        except Exception as e:
            # Log the error
            print(f"Prediction error: {str(e)}")
            # Return a default response
            return {
                'prediction': False,
                'probability': 0.0,
                'risk_level': 'unknown',
                'error': str(e)
            }
    
    def get_risk_level(self, probability):
        """
        Determine risk level based on probability
        
        Parameters:
        probability (float): Probability of diabetes
        
        Returns:
        str: Risk level
        """
        if probability < 0.2:
            return 'low'
        elif probability < 0.5:
            return 'moderate'
        elif probability < 0.8:
            return 'high'
        else:
            return 'very_high'