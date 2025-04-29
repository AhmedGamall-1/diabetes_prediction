# Diabetes Prediction ML Project

This project implements an end-to-end machine learning pipeline for predicting diabetes using the Pima Indians Diabetes Dataset.

## Project Structure

- `data_loader.py`: Handles data loading and initial exploration
- `data_cleaning.py`: Contains functions for data cleaning and preprocessing
- `model_training.py`: Implements model training with hyperparameter tuning
- `model_evaluation.py`: Contains functions for model evaluation and comparison
- `main.py`: Main script that orchestrates the entire pipeline

## Features

- Data preprocessing and cleaning
- Feature scaling
- Multiple model implementations (Random Forest, Logistic Regression, SVM)
- Hyperparameter tuning using GridSearchCV
- Comprehensive model evaluation
- Visualization of results

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to execute the entire pipeline:
```bash
python main.py
```

## Output

The pipeline generates several output files:
- Model files (`.pkl`): Trained models
- Evaluation results (`model_evaluation_results.txt`)
- Visualization plots:
  - Correlation matrix
  - Target distribution
  - Confusion matrices for each model
  - Model comparison chart

## Model Evaluation

The project evaluates models using multiple metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score

## License

This project is open source and available under the MIT License. 