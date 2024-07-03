import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# load data
legacy_interpolation = pd.read_csv('multibeam_data.csv')
data = pd.read_csv('sampling_data.csv') 



# Create unlabeled data
xy_data = data[['x', 'y']]
filtered_legacy_interpolation = legacy_interpolation.merge(xy_data, on=['x', 'y'], how='left', indicator=True)
unlabeled_data = filtered_legacy_interpolation[filtered_legacy_interpolation['_merge'] == 'left_only']
unlabeled_data.drop(columns=['_merge'], inplace=True)
unlabeled_data


# Create a stratified index to split the data
X = data.drop(['x', 'y', 'Gravel', 'Sand', 'Silt', 'Clay','type', 'class', 'distance'], axis=1)
y = data[['Gravel', 'Sand', 'Silt', 'Clay']]

stratify_col = data['class']
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=stratify_col, test_size=0.2, random_state=42)



# train model
def train_predict_models(X_train, y_train, unlabel_scaled_1):
    models = {
        'RandomForest': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
        'XGBoost': MultiOutputRegressor(XGBRegressor(random_state=42)),
        'SVM': MultiOutputRegressor(SVR()),
        'LightGBM': MultiOutputRegressor(LGBMRegressor(random_state=42)),
        'DNN': Sequential([
            Dense(64, activation='relu', input_dim=X_train.shape[1]),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(y_train.shape[1], activation='softmax')
        ])
    }
    param_grids = {
        'RandomForest': {'estimator__n_estimators': [50, 100, 200], 'estimator__max_depth': [10, 20, 30], 'estimator__min_samples_split': [2, 5, 8, 9], 'estimator__min_samples_leaf': [3, 4, 5, 6, 7]},
        'XGBoost': {'estimator__n_estimators': [50, 100, 150, 200], 'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3], 'estimator__max_depth': [3, 4, 5, 6], 'estimator__subsample': [0.6, 0.7, 0.9, 1.0]},
        'SVM': {'estimator__C': [0.1, 2, 3, 4, 5], 'estimator__kernel': ['linear', 'rbf'], 'estimator__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7]},
        'LightGBM': {'estimator__num_leaves': [10, 20, 30, 40, 50], 'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4], 'estimator__n_estimators': [50, 100, 150, 200]}
    }
    if 'DNN' in models:
        models['DNN'].compile(optimizer='adam', loss='mse', metrics=['mae'])
        models['DNN'].fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    predictions = []
    
    for name, model in models.items():
        if name != 'DNN':
            grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = models['DNN']

        y_pred = best_model.predict(unlabel_scaled_1)
        predictions.append(y_pred)

    y_unlabeled_pred = np.mean(predictions, axis=0)
    return y_unlabeled_pred, models



# self-learning
def update_training_data(X_train, y_train, X_unlabeled, y_unlabeled_pred, top_percentage=0.01):
    # Calculate MSE for each prediction compared to the average
    mse_values = [mean_squared_error([y_train.iloc[i]], [pred]) for i, pred in enumerate(y_unlabeled_pred)]
    top_indices = np.argsort(mse_values)[:int(top_percentage * len(X_unlabeled))]

    # Add top data to training data
    X_train_updated = pd.concat([X_train, X_unlabeled.iloc[top_indices]], ignore_index=True)
    y_train_updated = pd.concat([y_train, pd.DataFrame(y_unlabeled_pred[top_indices], columns=y_train.columns)], ignore_index=True)

    # Remove the used data from unlabeled data
    X_unlabeled.drop(index=top_indices, inplace=True)

    return X_train_updated, y_train_updated, X_unlabeled



# self-learning iteration
def execute_learning_cycle(X_train, y_train, X_unlabeled, iterations=10, top_percentage=0.01):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_unlabeled_scaled = scaler.transform(X_unlabeled)
    unlabel_scaled_1 = scaler.transform(unlabeled_data)

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")
        
        y_unlabeled_pred, models = train_predict_models(X_train_scaled, y_train, X_unlabeled_scaled)
        
        X_train_scaled, y_train, X_unlabeled_scaled = update_training_data(
            X_train_scaled, y_train, X_unlabeled_scaled, y_unlabeled_pred, top_percentage
        )
        
        if len(X_unlabeled_scaled) == 0:
            print("No more unlabeled data to process.")
            break

    return X_train_scaled, y_train, models

X_train_final, y_train_final, trained_models = execute_learning_cycle(
    X_train, y_train, unlabeled_data, iterations=10, top_percentage=0.01
)



# result
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'RandomForest': MultiOutputRegressor(RandomForestRegressor(random_state=42)),
        'XGBoost': MultiOutputRegressor(XGBRegressor(random_state=42)),
        'SVM': MultiOutputRegressor(SVR()),
        'LightGBM': MultiOutputRegressor(LGBMRegressor(random_state=42)),
        'DNN': Sequential([
            Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
            Dense(128, activation='relu'),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(y_train.shape[1], activation='linear')  
        ])
    }

    param_grids = {
        'RandomForest': {'estimator__n_estimators': [50, 100, 200], 'estimator__max_depth': [10, 20, 30], 'estimator__min_samples_split': [2, 5, 8, 9], 'estimator__min_samples_leaf': [3, 4, 5, 6, 7]},
        'XGBoost': {'estimator__n_estimators': [50, 100, 150, 200], 'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3], 'estimator__max_depth': [3, 4, 5, 6], 'estimator__subsample': [0.6, 0.7, 0.9, 1.0]},
        'SVM': {'estimator__C': [0.1, 2, 3, 4, 5], 'estimator__kernel': ['linear', 'rbf'], 'estimator__epsilon': [0.1, 0.2, 0.5, 0.6, 0.7]},
        'LightGBM': {'estimator__num_leaves': [10, 20, 30, 40, 50], 'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.4], 'estimator__n_estimators': [50, 100, 150, 200]}
    }
    
    if 'DNN' in models:
        models['DNN'].compile(optimizer='adam', loss='mse', metrics=['mae'])
        models['DNN'].fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
    
    test_scores = {}
    
    for name, model in models.items():
        if name != 'DNN':
            grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = models['DNN']
        
        y_pred_test = best_model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_scores[name] = test_mse
        print(f"{name} Test MSE: {test_mse}")
    
    return test_scores

test_scores = train_and_evaluate_models(X_train_final, y_train_final, X_test, y_test)