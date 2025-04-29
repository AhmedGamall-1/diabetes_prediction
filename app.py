import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_cleaning import (
    create_interaction_features,
    create_polynomial_features,
    create_ratio_features,
    bin_continuous_variables
)
import os

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="centered"
)

# Load the model and scaler
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join('models', 'best_model.pkl'))
    return model

# Load the data to get feature names and ranges
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes.csv')
    return df

def preprocess_input_data(input_data):
    """
    Preprocess the input data to match the features used in training
    """
    # Create interaction features
    input_data = create_interaction_features(input_data)
    
    # Create polynomial features
    input_data = create_polynomial_features(input_data, ['Age', 'BMI', 'Glucose'])
    
    # Create ratio features
    input_data = create_ratio_features(input_data)
    
    # Create binned features
    input_data = bin_continuous_variables(input_data)
    
    # Ensure all expected columns are present
    expected_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Age', 'Glucose_BMI',
        'Age_Glucose', 'BloodPressure_BMI', 'Age_squared', 'BMI_squared',
        'Glucose_squared', 'BMI_to_Age', 'Glucose_to_BMI', 'Insulin_to_Glucose',
        'Age_bin_VeryYoung', 'Age_bin_Young', 'Age_bin_Middle', 'Age_bin_Senior',
        'Age_bin_Elderly', 'BMI_bin_VeryLow', 'BMI_bin_Low', 'BMI_bin_Normal',
        'BMI_bin_High', 'BMI_bin_VeryHigh', 'Glucose_bin_VeryLow',
        'Glucose_bin_Low', 'Glucose_bin_Normal', 'Glucose_bin_High',
        'Glucose_bin_VeryHigh'
    ]
    
    # Add missing columns with zeros
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match training data
    input_data = input_data[expected_columns]
    
    return input_data

def main():
    st.title("Diabetes Prediction App")
    st.write("""
    This app predicts the likelihood of diabetes based on various health metrics.
    Please enter the patient's information below.
    """)
    
    # Load data for reference
    df = load_data()
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input('Number of Pregnancies', 0, 17, 0)
        glucose = st.number_input('Glucose Level (mg/dL)', 0, 200, 100)
        blood_pressure = st.number_input('Blood Pressure (mm Hg)', 0, 122, 70)
        skin_thickness = st.number_input('Skin Thickness (mm)', 0, 99, 20)
        
    with col2:
        insulin = st.number_input('Insulin Level (mu U/ml)', 0, 846, 80)
        bmi = st.number_input('BMI', 0.0, 67.1, 25.0)
        diabetes_pedigree = st.number_input('Diabetes Pedigree Function', 0.078, 2.42, 0.5)
        age = st.number_input('Age', 21, 81, 30)
    
    # Create a button for prediction
    if st.button('Predict Diabetes Risk'):
        # Create input data
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        })
        
        try:
            # Preprocess the input data
            input_data_processed = preprocess_input_data(input_data)
            
            # Load and use the model
            model = load_model()
            prediction = model.predict(input_data_processed)
            prediction_proba = model.predict_proba(input_data_processed)
            
            # Display results
            st.subheader('Prediction Results')
            
            if prediction[0] == 1:
                st.error('The model predicts that the patient is likely to have diabetes.')
                st.write(f'Probability of having diabetes: {prediction_proba[0][1]:.2%}')
            else:
                st.success('The model predicts that the patient is not likely to have diabetes.')
                st.write(f'Probability of not having diabetes: {prediction_proba[0][0]:.2%}')
            
            # Add some explanation
            st.write("""
            Note: This prediction is based on machine learning and should be used as a screening tool only.
            Please consult with a healthcare professional for proper diagnosis and treatment.
            """)
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Please try again with different input values.")
    
    # Add some information about the model
    st.sidebar.title("About the Model")
    st.sidebar.write("""
    This model is a Random Forest classifier trained on the Pima Indians Diabetes Dataset.
    It uses the following features:
    - Number of Pregnancies
    - Glucose Level
    - Blood Pressure
    - Skin Thickness
    - Insulin Level
    - BMI
    - Diabetes Pedigree Function
    - Age
    - Plus engineered features like interactions and ratios
    """)
    st.sidebar.write("Model Accuracy: 78.15%")

if __name__ == "__main__":
    main() 