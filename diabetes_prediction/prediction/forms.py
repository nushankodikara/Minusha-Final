from django import forms
from .models import PredictionResult, SugarReport

class PredictionForm(forms.ModelForm):
    """Form for diabetes prediction input"""
    class Meta:
        model = PredictionResult
        fields = ['gender', 'age', 'height', 'weight', 'exercise_habits']
        widgets = {
            'gender': forms.Select(attrs={'class': 'form-select'}),
            'age': forms.NumberInput(attrs={'class': 'form-control', 'min': 1, 'max': 120}),
            'height': forms.NumberInput(attrs={'class': 'form-control', 'min': 50, 'max': 300, 'step': '0.1'}),
            'weight': forms.NumberInput(attrs={'class': 'form-control', 'min': 20, 'max': 500, 'step': '0.1'}),
            'exercise_habits': forms.Select(attrs={'class': 'form-select'}),
        }

class SugarReportForm(forms.ModelForm):
    """Form for uploading sugar reports"""
    class Meta:
        model = SugarReport
        fields = ['report_file']
        widgets = {
            'report_file': forms.FileInput(attrs={'class': 'form-control'}),
        }
        
    def clean_report_file(self):
        file = self.cleaned_data.get('report_file')
        if file:
            # Check file extension
            ext = file.name.split('.')[-1].lower()
            if ext not in ['pdf', 'jpg', 'jpeg', 'png']:
                raise forms.ValidationError("Only PDF and image files (JPG, JPEG, PNG) are allowed.")
            # Check file size (limit to 5MB)
            if file.size > 5 * 1024 * 1024:
                raise forms.ValidationError("File size must be less than 5MB.")
        return file