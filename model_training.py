from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    ensure_dir('models')
    joblib.dump(best_rf, os.path.join('models', 'random_forest_model.pkl'))
    
    return best_rf, grid_search.best_params_

def train_xgboost(X_train, y_train):
    """
    Train an XGBoost model with hyperparameter tuning
    """
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    xgb = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_xgb = grid_search.best_estimator_
    ensure_dir('models')
    joblib.dump(best_xgb, os.path.join('models', 'xgboost_model.pkl'))
    
    return best_xgb, grid_search.best_params_

def train_catboost(X_train, y_train):
    """
    Train a CatBoost model
    """
    catboost_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False
    )
    catboost_model.fit(X_train, y_train)
    ensure_dir('models')
    joblib.dump(catboost_model, os.path.join('models', 'catboost_model.pkl'))
    
    return catboost_model, {
        'iterations': 1000,
        'learning_rate': 0.1,
        'depth': 6
    }

def train_knn(X_train, y_train):
    """
    Train a KNN model with hyperparameter tuning
    """
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_knn = grid_search.best_estimator_
    ensure_dir('models')
    joblib.dump(best_knn, os.path.join('models', 'knn_model.pkl'))
    
    return best_knn, grid_search.best_params_

def train_svm(X_train, y_train):
    """
    Train an SVM model with hyperparameter tuning
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_svm = grid_search.best_estimator_
    ensure_dir('models')
    joblib.dump(best_svm, os.path.join('models', 'svm_model.pkl'))
    
    return best_svm, grid_search.best_params_

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model with hyperparameter tuning
    """
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_lr = grid_search.best_estimator_
    ensure_dir('models')
    joblib.dump(best_lr, os.path.join('models', 'logistic_regression_model.pkl'))
    
    return best_lr, grid_search.best_params_

def train_all_models(X_train, y_train):
    """
    Train all models and return their best versions
    """
    models = {}
    best_params = {}
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model, rf_params = train_random_forest(X_train, y_train)
    models['random_forest'] = rf_model
    best_params['random_forest'] = rf_params
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_model, xgb_params = train_xgboost(X_train, y_train)
    models['xgboost'] = xgb_model
    best_params['xgboost'] = xgb_params
    
    # Train CatBoost
    print("\nTraining CatBoost...")
    catboost_model, catboost_params = train_catboost(X_train, y_train)
    models['catboost'] = catboost_model
    best_params['catboost'] = catboost_params
    
    # Train KNN
    print("\nTraining KNN...")
    knn_model, knn_params = train_knn(X_train, y_train)
    models['knn'] = knn_model
    best_params['knn'] = knn_params
    
    # Train SVM
    print("\nTraining SVM...")
    svm_model, svm_params = train_svm(X_train, y_train)
    models['svm'] = svm_model
    best_params['svm'] = svm_params
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model, lr_params = train_logistic_regression(X_train, y_train)
    models['logistic_regression'] = lr_model
    best_params['logistic_regression'] = lr_params
    
    return models, best_params 