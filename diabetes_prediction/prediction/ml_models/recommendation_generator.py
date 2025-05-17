class RecommendationGenerator:
    """Class for generating personalized recommendations based on prediction results"""
    
    def generate_recommendations(self, prediction_result, user_data):
        """
        Generate personalized recommendations based on prediction results
        
        Parameters:
        prediction_result (dict): Prediction result from the model
        user_data (dict): User input data
        
        Returns:
        list: List of recommendation dictionaries
        """
        recommendations = []
        
        # Extract prediction details
        has_diabetes = prediction_result.get('prediction', False)
        risk_level = prediction_result.get('risk_level', 'low')
        
        # Extract user data
        bmi = user_data.get('bmi', 0)
        age = user_data.get('age', 0)
        exercise_habits = user_data.get('exercise_habits', 'sedentary')
        
        # Generate diet recommendations
        diet_rec = self._generate_diet_recommendations(has_diabetes, risk_level, bmi)
        recommendations.append(diet_rec)
        
        # Generate exercise recommendations
        exercise_rec = self._generate_exercise_recommendations(has_diabetes, risk_level, exercise_habits, age)
        recommendations.append(exercise_rec)
        
        # Generate lifestyle recommendations
        lifestyle_rec = self._generate_lifestyle_recommendations(has_diabetes, risk_level)
        recommendations.append(lifestyle_rec)
        
        # Generate medical recommendations
        medical_rec = self._generate_medical_recommendations(has_diabetes, risk_level)
        recommendations.append(medical_rec)
        
        return recommendations
    
    def _generate_diet_recommendations(self, has_diabetes, risk_level, bmi):
        """Generate diet recommendations"""
        if has_diabetes:
            title = "Diabetes-Friendly Diet Plan"
            description = (
                "Follow a balanced diet rich in fiber and low in simple carbohydrates. "
                "Include plenty of vegetables, whole grains, lean proteins, and healthy fats. "
                "Limit intake of processed foods, sugary beverages, and high-glycemic foods. "
                "Monitor carbohydrate intake and distribute it evenly throughout the day. "
                "Consider working with a registered dietitian to create a personalized meal plan."
            )
        elif risk_level in ['high', 'very_high']:
            title = "Diet Plan to Reduce Diabetes Risk"
            description = (
                "Adopt a diet rich in whole foods, including vegetables, fruits, whole grains, and lean proteins. "
                "Limit processed foods, sugary beverages, and refined carbohydrates. "
                "Include foods with a low glycemic index to help maintain stable blood sugar levels. "
                "Consider the Mediterranean or DASH diet, which have been shown to reduce diabetes risk."
            )
        else:
            title = "Healthy Eating Guidelines"
            description = (
                "Maintain a balanced diet with plenty of vegetables, fruits, whole grains, and lean proteins. "
                "Limit processed foods, sugary beverages, and excessive alcohol consumption. "
                "Stay hydrated by drinking plenty of water throughout the day. "
                "Practice portion control to maintain a healthy weight."
            )
        
        # Add BMI-specific recommendations
        if bmi >= 30:
            description += (
                " Since your BMI indicates obesity, focus on gradual weight loss through calorie reduction "
                "and increased physical activity. Even a 5-10% reduction in weight can significantly improve health outcomes."
            )
        elif bmi >= 25:
            description += (
                " Since your BMI indicates overweight, consider moderate calorie reduction and increased physical activity "
                "to achieve a healthier weight. Focus on nutrient-dense foods that provide satiety."
            )
        
        return {
            'category': 'diet',
            'title': title,
            'description': description
        }
    
    def _generate_exercise_recommendations(self, has_diabetes, risk_level, exercise_habits, age):
        """Generate exercise recommendations"""
        base_description = ""
        
        if has_diabetes:
            title = "Exercise Plan for Diabetes Management"
            base_description = (
                "Regular physical activity is crucial for managing diabetes. "
                "Aim for at least 150 minutes of moderate-intensity aerobic activity per week, "
                "spread over at least 3 days with no more than 2 consecutive days without activity. "
                "Include resistance training 2-3 times per week. "
                "Always carry a source of fast-acting carbohydrates during exercise in case of hypoglycemia. "
                "Monitor blood glucose before, during, and after exercise, especially if taking insulin."
            )
        elif risk_level in ['high', 'very_high']:
            title = "Exercise Plan to Reduce Diabetes Risk"
            base_description = (
                "Regular physical activity can significantly reduce your risk of developing diabetes. "
                "Aim for at least 150 minutes of moderate-intensity aerobic activity per week. "
                "Include resistance training 2-3 times per week to improve insulin sensitivity. "
                "Find activities you enjoy to help maintain consistency. "
                "Consider working with a fitness professional to create a personalized exercise plan."
            )
        else:
            title = "General Exercise Guidelines"
            base_description = (
                "Regular physical activity is important for overall health. "
                "Aim for at least 150 minutes of moderate-intensity aerobic activity per week. "
                "Include strength training exercises at least twice a week to build muscle and improve metabolism. "
                "Find activities you enjoy to increase adherence and make exercise a sustainable habit. "
                "Remember that even small amounts of physical activity are beneficial."
            )
        
        # Add exercise habit-specific recommendations
        if exercise_habits == 'sedentary':
            base_description += (
                " Since you currently have a sedentary lifestyle, start with short, low-intensity activities like walking "
                "for 10-15 minutes per day and gradually increase duration and intensity. Consider activities like swimming "
                "or cycling that are gentle on the joints."
            )
        elif exercise_habits == 'light':
            base_description += (
                " Since you currently engage in light exercise, try to increase your activity level gradually. "
                "Add an extra day of exercise each week or extend your sessions by 10-15 minutes. "
                "Consider adding variety to your routine to work different muscle groups."
            )
        
        # Add age-specific recommendations
        if age >= 65:
            base_description += (
                " As an older adult, focus on exercises that improve balance and flexibility to prevent falls. "
                "Low-impact activities like walking, swimming, or tai chi are excellent options. "
                "Always consult with your healthcare provider before starting a new exercise program."
            )
        elif age >= 50:
            base_description += (
                " For your age group, include exercises that maintain bone density, such as weight-bearing activities. "
                "Pay attention to proper form to prevent injuries, and allow adequate recovery time between intense workouts."
            )
        
        return {
            'category': 'exercise',
            'title': title,
            'description': base_description
        }
    
    def _generate_lifestyle_recommendations(self, has_diabetes, risk_level):
        """Generate lifestyle recommendations"""
        if has_diabetes:
            title = "Lifestyle Management for Diabetes"
            description = (
                "Manage stress through techniques like meditation, deep breathing, or yoga, as stress can affect blood glucose levels. "
                "Prioritize quality sleep, aiming for 7-9 hours per night, as poor sleep can impact insulin sensitivity. "
                "Avoid smoking and limit alcohol consumption, as these can worsen diabetes complications. "
                "Establish a routine for taking medications, checking blood glucose, and eating meals. "
                "Join a diabetes support group to connect with others facing similar challenges."
            )
        elif risk_level in ['high', 'very_high']:
            title = "Lifestyle Changes to Reduce Diabetes Risk"
            description = (
                "Prioritize stress management through techniques like meditation, deep breathing, or regular physical activity. "
                "Aim for 7-9 hours of quality sleep per night to improve insulin sensitivity. "
                "Avoid smoking and limit alcohol consumption to reduce inflammation and improve overall health. "
                "Maintain a consistent eating schedule to help regulate blood sugar levels. "
                "Consider using a health tracking app to monitor your progress and stay motivated."
            )
        else:
            title = "Healthy Lifestyle Habits"
            description = (
                "Manage stress through regular physical activity, mindfulness practices, or hobbies you enjoy. "
                "Prioritize quality sleep by establishing a consistent sleep schedule and creating a restful environment. "
                "Avoid smoking and limit alcohol consumption to support overall health. "
                "Stay hydrated by drinking plenty of water throughout the day. "
                "Take breaks from prolonged sitting by standing or walking for a few minutes every hour."
            )
        
        return {
            'category': 'lifestyle',
            'title': title,
            'description': description
        }
    
    def _generate_medical_recommendations(self, has_diabetes, risk_level):
        """Generate medical recommendations"""
        if has_diabetes:
            title = "Medical Care for Diabetes"
            description = (
                "Schedule regular check-ups with your healthcare provider to monitor your diabetes management. "
                "Get recommended screenings for diabetes complications, including eye exams, foot exams, and kidney function tests. "
                "Monitor your blood glucose as recommended by your healthcare provider. "
                "Take medications as prescribed and discuss any concerns or side effects with your healthcare provider. "
                "Consider working with a certified diabetes educator to improve your self-management skills."
            )
        elif risk_level in ['high', 'very_high']:
            title = "Medical Monitoring for High Diabetes Risk"
            description = (
                "Schedule regular check-ups with your healthcare provider to monitor your risk factors. "
                "Get your blood glucose levels tested at least annually, or more frequently if recommended. "
                "Discuss with your healthcare provider whether preventive medications might be appropriate for you. "
                "Monitor other health metrics, such as blood pressure and cholesterol levels. "
                "Consider genetic testing if you have a family history of diabetes to better understand your risk."
            )
        else:
            title = "Preventive Healthcare"
            description = (
                "Schedule regular check-ups with your healthcare provider for preventive care. "
                "Get recommended health screenings based on your age, gender, and risk factors. "
                "Stay up-to-date on vaccinations to protect against preventable diseases. "
                "Discuss any health concerns or changes with your healthcare provider promptly. "
                "Maintain a record of your health information, including test results and medications."
            )
        
        return {
            'category': 'medical',
            'title': title,
            'description': description
        }