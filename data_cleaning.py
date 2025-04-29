import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy import stats

def remove_outliers(df, columns, n_std=3):
    """
    Remove outliers using z-score method
    """
    df_clean = df.copy()
    for column in columns:
        z_scores = stats.zscore(df_clean[column])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < n_std)
        df_clean = df_clean[filtered_entries]
    return df_clean

def create_interaction_features(df):
    """
    Create interaction features between important variables
    """
    # BMI and Age interaction
    df['BMI_Age'] = df['BMI'] * df['Age']
    
    # Glucose and BMI interaction
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    
    # Age and Glucose interaction
    df['Age_Glucose'] = df['Age'] * df['Glucose']
    
    # Blood Pressure and BMI interaction
    df['BloodPressure_BMI'] = df['BloodPressure'] * df['BMI']
    
    return df

def create_polynomial_features(df, columns, degree=2):
    """
    Create polynomial features for specified columns
    """
    for column in columns:
        df[f'{column}_squared'] = df[column] ** 2
        if degree > 2:
            df[f'{column}_cubed'] = df[column] ** 3
    return df

def create_ratio_features(df):
    """
    Create ratio features between relevant variables
    """
    # BMI to Age ratio
    df['BMI_to_Age'] = df['BMI'] / df['Age']
    
    # Glucose to BMI ratio
    df['Glucose_to_BMI'] = df['Glucose'] / df['BMI']
    
    # Insulin to Glucose ratio (important for diabetes)
    df['Insulin_to_Glucose'] = df['Insulin'] / df['Glucose']
    
    return df

def bin_continuous_variables(df):
    """
    Create binned versions of continuous variables
    """
    # Define bin edges based on the training data ranges
    age_bins = [0, 25, 30, 40, 50, 100]
    bmi_bins = [0, 18.5, 25, 30, 35, 100]
    glucose_bins = [0, 70, 100, 125, 200, 300]
    
    # Create bins using cut instead of qcut
    df['Age_bin'] = pd.cut(df['Age'], bins=age_bins, labels=['VeryYoung', 'Young', 'Middle', 'Senior', 'Elderly'])
    df['BMI_bin'] = pd.cut(df['BMI'], bins=bmi_bins, labels=['VeryLow', 'Low', 'Normal', 'High', 'VeryHigh'])
    df['Glucose_bin'] = pd.cut(df['Glucose'], bins=glucose_bins, labels=['VeryLow', 'Low', 'Normal', 'High', 'VeryHigh'])
    
    # Convert to dummy variables
    df = pd.get_dummies(df, columns=['Age_bin', 'BMI_bin', 'Glucose_bin'])
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset with more sophisticated methods
    """
    # Replace zeros with NaN for columns where zero is not a valid value
    columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_check] = df[columns_to_check].replace(0, np.nan)
    
    # Fill missing values using more sophisticated methods
    for column in columns_to_check:
        if column in ['Glucose', 'BloodPressure']:
            # For vital signs, use median based on Age groups
            df[column] = df.groupby(pd.qcut(df['Age'], q=5))[column].transform(
                lambda x: x.fillna(x.median())
            )
        elif column == 'BMI':
            # For BMI, use median based on Age and Glucose groups
            df[column] = df.groupby([
                pd.qcut(df['Age'], q=3),
                pd.qcut(df['Glucose'], q=3)
            ])[column].transform(lambda x: x.fillna(x.median()))
        else:
            # For other measurements, use median based on Age and BMI groups
            df[column] = df.groupby([
                pd.qcut(df['Age'], q=3),
                pd.qcut(df['BMI'].fillna(df['BMI'].median()), q=3)
            ])[column].transform(lambda x: x.fillna(x.median()))
    
    return df

def preprocess_data(df):
    """
    Preprocess the data for modeling with enhanced feature engineering
    """
    # Handle missing values first
    df_processed = handle_missing_values(df)
    
    # Remove outliers
    columns_for_outlier_removal = ['Glucose', 'BloodPressure', 'BMI', 'Age']
    df_processed = remove_outliers(df_processed, columns_for_outlier_removal)
    
    # Create new features
    df_processed = create_interaction_features(df_processed)
    df_processed = create_polynomial_features(df_processed, ['Age', 'BMI', 'Glucose'])
    df_processed = create_ratio_features(df_processed)
    df_processed = bin_continuous_variables(df_processed)
    
    # Separate features and target
    X = df_processed.drop('Outcome', axis=1)
    y = df_processed['Outcome']
    
    # Scale the features using RobustScaler (more robust to outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

def get_feature_names(df):
    """
    Get feature names from the dataset
    """
    return df.drop('Outcome', axis=1).columns.tolist()

if __name__ == "__main__":
    # Test the functions
    df = pd.read_csv('diabetes.csv')
    df_cleaned = handle_missing_values(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df_cleaned)
    print("Data preprocessing completed successfully!")
    print(f"Number of features after engineering: {X_train.shape[1]}")
    print("\nFeature names:")
    for i, feature in enumerate(X_train.columns, 1):
        print(f"{i}. {feature}") 