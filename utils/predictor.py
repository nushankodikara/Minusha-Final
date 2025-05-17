import joblib
import numpy as np
import os

# Load the trained model
model_path = os.path.join("model", "diabetes_model.pkl")
model = joblib.load(model_path)

# Prediction function
def predict_diabetes(input_data):
    # input_data = [age, gender, height, weight, physical_activity]
    # Calculate BMI
    height_m = input_data[2] / 100
    bmi = input_data[3] / (height_m ** 2)
    features = np.array([input_data[0], input_data[1], bmi, input_data[4]]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]
