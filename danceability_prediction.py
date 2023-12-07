#%%[markdown]
## Modeling to predict Danceability
#%%
# Import
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from Spotify_EDA import df_danceabiltiy # Import df data frame from Spotify_EDA to use the processed data for modeling

# %%
# %%

# Selecting relevant features and the target variable
features = ['energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'danceability'

# Handling missing values
spotify_data = df_danceabiltiy.dropna(subset=features + [target])

# Splitting the dataset into training and testing sets
X = spotify_data[features]
y = spotify_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predicting and evaluating the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}, R-squared: {r2}')
# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Selecting relevant features and the target variable
features = ['energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
target = 'danceability'

# Handling missing values
spotify_data = df_danceabiltiy.dropna(subset=features + [target])

# Splitting the dataset into training and testing sets
X = spotify_data[features]
y = spotify_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to evaluate a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Building and evaluating Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_mse, rf_r2 = evaluate_model(rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Building and evaluating Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)
gb_mse, gb_r2 = evaluate_model(gb_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Hyperparameter tuning for Random Forest using Grid Search
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid_rf, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train_scaled, y_train)

# Best parameters and evaluation
best_rf_model = grid_search_rf.best_estimator_
best_rf_mse, best_rf_r2 = evaluate_model(best_rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

# Print the results
print("Random Forest MSE:", rf_mse, "R2:", rf_r2)
print("Gradient Boosting MSE:", gb_mse, "R2:", gb_r2)
print("Tuned Random Forest MSE:", best_rf_mse, "R2:", best_rf_r2)
print("Best Random Forest Parameters:", grid_search_rf.best_params_)

# %%
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Hyperparameters to tune
param_dist_rf = {
    'n_estimators': randint(100, 300),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11)
}

# Randomized Search for hyperparameter tuning
random_search_rf = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_dist_rf, n_iter=50, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', random_state=42)
random_search_rf.fit(X_train_scaled, y_train)

# Best parameters and evaluation
best_rf_model = random_search_rf.best_estimator_
best_rf_mse_ex, best_rf_r2_ex = evaluate_model(best_rf_model, X_train_scaled, y_train, X_test_scaled, y_test)

print("Tuned Random Forest (Extended) MSE:", best_rf_mse_ex, "R2:", best_rf_r2_ex)
print("Best Random Forest (Extended) Parameters:", random_search_rf.best_params_)