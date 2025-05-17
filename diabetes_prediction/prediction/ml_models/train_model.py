import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Ensure diabetes.csv is in the same directory as this script (ml_models/)
DATA_FILE_PATH = os.path.join(SCRIPT_DIR, 'diabetes.csv')
MODELS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'models') # Output: ml_models/models/

# Ensure output directory exists
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

SAVED_MODEL_PATH = os.path.join(MODELS_OUTPUT_DIR, 'tuned_xgboost_model.pkl')
SAVED_SCALER_PATH = os.path.join(MODELS_OUTPUT_DIR, 'scaler.pkl')

# This will be derived based on notebook logic (bmi -> weight_status -> diabetes_risk -> encoded)
DERIVED_TARGET_COLUMN = 'diabetes_risk_encoded'

# --- 1. Load Data ---
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"ERROR: Data file not found at {file_path}")
        print(f"Please ensure 'diabetes.csv' is in the '{SCRIPT_DIR}' directory.")
        return None
    try:
        # The notebook loads a CSV from a specific Windows path.
        # We assume 'diabetes.csv' has been placed in SCRIPT_DIR.
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- Helper functions from Notebook (for target creation) ---
def bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif 18.5 <= bmi <= 24.9: return 'Normal weight'
    elif 25.0 <= bmi <= 29.9: return 'Overweight'
    elif 30.0 <= bmi <= 34.9: return 'Obese Class 1'
    elif 35.0 <= bmi <= 39.9: return 'Obese Class 2'
    else: return 'Morbidly Obese'

def map_diabetes_risk(weight_status):
    mapping = {
        "Underweight": "Low risk", "Normal weight": "Lowest risk",
        "Overweight": "Moderate to high risk", "Obese Class 1": "High risk",
        "Obese Class 2": "Very high risk", "Morbidly Obese": "Extremely high risk"
    }
    return mapping.get(weight_status, "Unknown risk") # .get for safety

# --- 2. Preprocess Data (Aligning with predictor.py features, target from notebook) ---
def preprocess_data_for_training(df_original):
    print("Preprocessing data for training...")
    df = df_original.copy()

    # A. Derive Target Variable (as per notebook logic)
    # Ensure 'Height' and 'Weight' columns exist for BMI calculation, or 'bmi' directly
    if 'Weight' in df.columns and 'Height' in df.columns:
        # Ensure Height is not zero to avoid division by zero
        df['Height_safe'] = df['Height'].replace(0, np.nan) # Replace 0 with NaN
        df['bmi_calculated'] = df['Weight'] / (df['Height_safe']**2)
        df.drop('Height_safe', axis=1, inplace=True)
    elif 'bmi' in df.columns: # If 'bmi' column already exists
         df.rename(columns={'bmi': 'bmi_calculated'}, inplace=True) # Use a consistent name
    else:
        print("ERROR: 'Weight' and 'Height' columns (or a 'bmi' column) are required to derive target but not found.")
        return None, None

    if 'bmi_calculated' not in df.columns:
        print("ERROR: 'bmi_calculated' could not be created.")
        return None, None
        
    df['weight_status'] = df['bmi_calculated'].apply(bmi_category)
    df['diabetes_risk_text'] = df['weight_status'].apply(map_diabetes_risk)

    # Encode the text-based risk to a numerical target
    # The notebook uses .cat.codes which assigns integers. We'll do similarly.
    # We need to ensure this matches what the XGBoost model in the notebook was trained on.
    # The notebook's target `y` for XGBoost comes from `df['diabetes_risk'].astype('category').cat.codes`.
    # Let's assume the distinct values in 'diabetes_risk_text' should map to 0, 1, 2...
    # For binary classification (which XGBClassifier with 'binary:logistic' expects),
    # we usually need a 0/1 target. Let's check unique values.
    print(f"Unique values in 'diabetes_risk_text': {df['diabetes_risk_text'].unique()}")
    
    # Simplistic binary mapping for now: "High risk" categories vs others.
    # THIS IS A CRITICAL ASSUMPTION. The notebook trains on multi-class 'diabetes_risk'.
    # If your Django app's `predictor.py` expects a binary output (True/False for diabetes),
    # the training target *must* be binary.
    # The notebook's XGBoost uses `eval_metric='logloss'`, which is fine for binary/multiclass,
    # but `objective='binary:logistic'` implies a binary target is expected by that *specific* XGBoost setup.
    
    # Let's create a binary target for demonstration: 1 if High/Very High/Extremely High, else 0
    # This aligns with a typical "has diabetes" vs "doesn't have diabetes" scenario.
    high_risk_categories = ["High risk", "Very high risk", "Extremely high risk"]
    df[DERIVED_TARGET_COLUMN] = df['diabetes_risk_text'].apply(lambda x: 1 if x in high_risk_categories else 0)
    print(f"Target column '{DERIVED_TARGET_COLUMN}' value counts:\n{df[DERIVED_TARGET_COLUMN].value_counts(normalize=True)}")
    
    y_target = df[DERIVED_TARGET_COLUMN]

    # B. Prepare Features (as expected by predictor.py)
    # Rename columns if they exist with different capitalization to match 'age', 'height', 'weight'.
    rename_map = {'Age': 'age', 'Height': 'height', 'Weight': 'weight', 'Gender': 'gender', 'BMI': 'bmi_direct'}
    for csv_col, model_col in rename_map.items():
        if csv_col in df.columns and model_col not in df.columns:
            df.rename(columns={csv_col: model_col}, inplace=True)
            print(f"Renamed CSV column '{csv_col}' to '{model_col}'")

    # Use 'bmi_calculated' if 'bmi' (original name for predictor.py's BMI) isn't present
    if 'bmi' not in df.columns and 'bmi_calculated' in df.columns:
        df.rename(columns={'bmi_calculated':'bmi'}, inplace=True) # predictor.py expects 'bmi'
    elif 'bmi_direct' in df.columns and 'bmi' not in df.columns: # If CSV had BMI directly
        df.rename(columns={'bmi_direct':'bmi'}, inplace=True)
        
    # Map gender (as in DiabetesPredictor)
    gender_mapping = {'male': 0, 'female': 1, 'other': 2} # predictor.py uses this
    if 'gender' in df.columns: # 'gender' should exist after renaming 'Gender'
        df['gender_numeric'] = df['gender'].astype(str).str.lower().map(gender_mapping)
    else:
        print("Warning: 'gender' column not found. 'gender_numeric' will be NaN.")
        df['gender_numeric'] = np.nan

    # Map exercise habits (as in DiabetesPredictor) - This column might not be in your `diabetes.csv`
    exercise_mapping = {'sedentary': 0, 'light': 1, 'moderate': 2, 'active': 3, 'very_active': 4}
    if 'exercise_habits' in df.columns: # This is from predictor.py, might not be in the notebook's CSV
        df['exercise_numeric'] = df['exercise_habits'].map(exercise_mapping)
    else:
        print("Warning: 'exercise_habits' column not found in CSV. 'exercise_numeric' will be NaN.")
        df['exercise_numeric'] = np.nan

    # Create age groups (as in DiabetesPredictor)
    if 'age' in df.columns: # 'age' should exist after renaming 'Age'
        bins = [0, 30, 45, 60, 120]
        labels = ['Young', 'Middle-aged', 'Senior', 'Elderly']
        df['age_group_temp'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
        age_dummies = pd.get_dummies(df['age_group_temp'], prefix='age', dummy_na=False)
        df = pd.concat([df, age_dummies], axis=1)
        df.drop('age_group_temp', axis=1, inplace=True, errors='ignore')
    else:
        print("Warning: 'age' column not found. Age group features will be missing or NaN.")
        for label in ['Young', 'Middle-aged', 'Senior', 'Elderly']:
            df[f'age_{label}'] = 0 # Create with 0 if age column is missing


    # Define the list of features that the DiabetesPredictor's scaler expects
    # These must be present in the DataFrame 'df' before this step.
    # 'height' and 'weight' are used by predictor.py's preprocess_input, even if BMI is primary.
    training_features_list = [
        'age', 'gender_numeric', 'height', 'weight', 'bmi', 'exercise_numeric',
        'age_Young', 'age_Middle-aged', 'age_Senior', 'age_Elderly'
    ]

    X_features = pd.DataFrame()
    for feature in training_features_list:
        if feature in df.columns:
            X_features[feature] = df[feature]
        else:
            print(f"Info: Feature '{feature}' for training not found in processed DataFrame. Adding it as NaNs.")
            X_features[feature] = np.nan # Create it; will be filled by median/0

    # Robust NaN filling strategy for features
    print("Filling NaNs in features...")
    for col in X_features.columns:
        if X_features[col].isnull().any():
            if pd.api.types.is_numeric_dtype(X_features[col]):
                fill_value = X_features[col].median()
                if np.isnan(fill_value): # If median is also NaN (e.g. all values were NaN)
                    fill_value = 0 
                print(f"Filling NaNs in numeric feature '{col}' with {fill_value}")
                X_features[col] = X_features[col].fillna(fill_value)
            else: # Should not happen for these pre-defined numeric/mapped features.
                fill_value = 0 # Fallback
                print(f"Warning: Filling NaNs in non-numeric feature '{col}' with {fill_value}. This is unexpected.")
                X_features[col] = X_features[col].fillna(fill_value)
    
    if X_features.isnull().any().any():
        print("ERROR: NaNs are still present in features after attempting to fill. Columns with NaNs:")
        print(X_features.isnull().sum()[X_features.isnull().sum() > 0])
        return None, None

    print("Preprocessing completed.")
    print(f"Features shape: {X_features.shape}")
    print(f"Target shape: {y_target.shape}")
    return X_features, y_target

# --- 3. Train Model and Scaler ---
def train_model_and_scaler(X, y):
    if X is None or y is None or X.empty or y.empty:
        print("ERROR: Features or target is empty/None. Cannot train.")
        return None, None
        
    print("Splitting data and training model...")
    # Stratify by y if it's binary or multi-class with reasonable distribution
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
    except ValueError as e:
        print(f"Stratify failed (possibly due to single class in y or small sample size for a class): {e}. Splitting without stratify.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    print(f"Training data shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test data shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Fit and save scaler (as predictor.py uses one)
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, SAVED_SCALER_PATH)
    print(f"Scaler saved to {SAVED_SCALER_PATH}")

    # XGBoost model as per notebook: xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    # The notebook did not specify other hyperparameters for XGBoost, so we use these defaults.
    # Added random_state for reproducibility.
    model = XGBClassifier(
        objective='binary:logistic', # Matches the binary target created above
        use_label_encoder=False,     # Deprecated, set to False
        eval_metric='logloss',       # As in notebook
        random_state=42              # For reproducibility
    )
    print("Training XGBoost model with parameters from notebook (objective='binary:logistic', eval_metric='logloss')...")
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, SAVED_MODEL_PATH)
    print(f"Model saved to {SAVED_MODEL_PATH}")

    accuracy = model.score(X_test_scaled, y_test)
    print(f"Model accuracy on test set: {accuracy:.4f}")
    print("Model training completed.")
    return model, scaler

# --- Main execution ---
if __name__ == '__main__':
    print(f"Starting model training script ({os.path.basename(__file__)})...")
    dataframe_original = load_data(DATA_FILE_PATH)
    
    if dataframe_original is not None:
        X_prepared_features, y_prepared_target = preprocess_data_for_training(dataframe_original)
        if X_prepared_features is not None and y_prepared_target is not None:
            if X_prepared_features.empty:
                print("ERROR: No features were generated after preprocessing. Cannot train model.")
            else:
                train_model_and_scaler(X_prepared_features, y_prepared_target)
        else:
            print("ERROR: Preprocessing returned None for features or target. Cannot train model.")
    else:
        print("ERROR: Data loading failed. Cannot train model.")
    print("Model training script finished.") 