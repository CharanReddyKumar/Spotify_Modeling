#%%[markdown]
##Modeling to predict popularity
#%%
# Import 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Spotify_EDA import df # Import df data frame from Spotify_EDA to use the processed data for modeling
from scipy.stats import skew
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Check for skewness
for col in numeric_cols:
    skewness = skew(df[col])
    print(f"Skewness of {col}: {skewness}")

# Assuming df is your DataFrame and numeric_cols contains the names of your numeric columns
for col in numeric_cols:
    skewness = skew(df[col])
    print(f"Skewness of {col}: {skewness}")

#%%[markdown]
# * duration_ms (10.81): Highly right-skewed. Consider applying a log transformation.
# * explicit (2.96): Moderately right-skewed. Investigate the distribution; if it's a boolean or binary feature, skewness might not be relevant.
# * loudness (-2.01): Moderately left-skewed. You might consider a square or cube root transformation.
# * speechiness (4.64): Highly right-skewed. A log transformation could be beneficial.
# * instrumentalness (1.74): Moderately right-skewed. Log transformation could be applied.
# * liveness (2.11): Moderately right-skewed. Log transformation is recommended.

# but we dont do transformation because we would not like to use linear model
#%%
from sklearn.model_selection import train_test_split

# Drop unnecessary columns
df = df.drop(['track_id','album_name', 'track_name'], axis=1)

# Define features and target variable
X = df.drop('popularity', axis=1)
y = df['popularity']
print(X.head(10))
print(y.head(10))

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now X_train and y_train can be used to train the model, 
# and X_test and y_test to evaluate its performance
#%%
print(X.columns)
#%%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize and train model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_linear = linear_model.predict(X_test)
print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_linear, squared=False))
print("Linear Regression R² Score:", r2_score(y_test, y_pred_linear))

#%%
from sklearn.ensemble import RandomForestRegressor

# Initialize and train model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf_model.predict(X_test)
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("Random Forest R² Score:", r2_score(y_test, y_pred_rf))

from xgboost import XGBRegressor

# Initialize and train model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("XGBoost R² Score:", r2_score(y_test, y_pred_xgb))

#%%
from sklearn.linear_model import Lasso

# Initialize and train model
lasso_model = Lasso(random_state=42)
lasso_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_lasso = lasso_model.predict(X_test)
print("Lasso Regression RMSE:", mean_squared_error(y_test, y_pred_lasso, squared=False))
print("Lasso Regression R² Score:", r2_score(y_test, y_pred_lasso))

# %%
from catboost import CatBoostRegressor

# Initialize and train model
cat_model = CatBoostRegressor(random_state=42, verbose=0)
cat_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_cat = cat_model.predict(X_test)
print("CatBoost RMSE:", mean_squared_error(y_test, y_pred_cat, squared=False))
print("CatBoost R² Score:", r2_score(y_test, y_pred_cat))
#%%
from sklearn.linear_model import Ridge

# Initialize and train model
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_ridge = ridge_model.predict(X_test)
print("Ridge Regression RMSE:", mean_squared_error(y_test, y_pred_ridge, squared=False))
print("Ridge Regression R² Score:", r2_score(y_test, y_pred_ridge))

#%%
from sklearn.svm import SVR

# Initialize and train model
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_svm = svm_model.predict(X_test)
print("SVM RMSE:", mean_squared_error(y_test, y_pred_svm, squared=False))
print("SVM R² Score:", r2_score(y_test, y_pred_svm))

#%%
from sklearn.ensemble import VotingRegressor

# Create the sub-models
estimators = [
    ('linear', linear_model),
    ('random_forest', rf_model),
    ('xgb', xgb_model),
    ('lasso', lasso_model),
    ('catboost', cat_model),
    ('ridge', ridge_model),
    ('svm', svm_model)
]

# Create the voting regressor
voting_model = VotingRegressor(estimators, weights=[2, 3, 3, 1, 3, 1, 1])  # Weights can be adjusted

# Fit the voting regressor to the training data
voting_model.fit(X_train, y_train)

# Predict and evaluate
y_pred_voting = voting_model.predict(X_test)
print("Voting Regressor RMSE:", mean_squared_error(y_test, y_pred_voting, squared=False))
print("Voting Regressor R² Score:", r2_score(y_test, y_pred_voting))

#%%
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Define base models
estimators = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgboost', XGBRegressor(n_estimators=100, random_state=42))
]

# Define final meta-learner model
final_estimator = Ridge()

# Create the Stacking Regressor
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=final_estimator
)

# Fit the model
stacking_regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred_stack = stacking_regressor.predict(X_test)
print("Stacking Regressor RMSE:", mean_squared_error(y_test, y_pred_stack, squared=False))
print("Stacking Regressor R² Score:", r2_score(y_test, y_pred_stack))

#%%

""" # Define the sub-models for VotingRegressor
estimators = [
    ('linear', linear_model),
    ('random_forest', rf_model),
    ('xgb', xgb_model),
    ('lasso', lasso_model),
    ('catboost', cat_model),
    ('ridge', ridge_model),
    ('svm', svm_model)
]

# Grid Search for Optimal Weights
from itertools import product

# Define a range of weights
weight_options = [1, 2, 3, 4, 5]

# Generate combinations of weights
weight_combinations = product(weight_options, repeat=len(estimators))

# Define a function to create a VotingRegressor with given weights
def get_voting_regressor(weights):
    return VotingRegressor(estimators, weights=weights)

# Grid search
best_score = float('inf')
best_weights = None

for weights in weight_combinations:
    model = get_voting_regressor(weights)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = mean_squared_error(y_test, y_pred, squared=False)
    if score < best_score:
        best_score = score
        best_weights = weights

print("Best Weights:", best_weights)
print("Best Score:", best_score)

# Create the Voting Regressor with the Best Weights
voting_model_optimized = VotingRegressor(estimators, weights=best_weights)
voting_model_optimized.fit(X_train, y_train)

# Evaluate the Optimized Model
y_pred_voting_optimized = voting_model_optimized.predict(X_test)
print("Optimized Voting Regressor RMSE:", mean_squared_error(y_test, y_pred_voting_optimized, squared=False))
print("Optimized Voting Regressor R² Score:", r2_score(y_test, y_pred_voting_optimized)) """

%%
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

# List of all models
models = [linear_model, rf_model, xgb_model, lasso_model, cat_model, ridge_model, svm_model, voting_model, stacking_regressor]
model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'Lasso', 'CatBoost', 'Ridge', 'SVM', 'Voting Regressor', 'Stacking Regressor']

# Evaluating all models
results = []
for model, name in zip(models, model_names):
    rmse, r2 = evaluate_model(model, X_test, y_test)
    results.append({'Model': name, 'RMSE': rmse, 'R² Score': r2})

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(results_df)
# %%



# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Set up the matplotlib figure
plt.figure(figsize=(14, 6))

# Plot RMSE
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='RMSE', data=results_df)
plt.title('Comparison of Model RMSE')
plt.xticks(rotation=45)
plt.ylabel('RMSE')
plt.xlabel('Model')

# Plot R² Score
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R² Score', data=results_df)
plt.title('Comparison of Model R² Score')
plt.xticks(rotation=45)
plt.ylabel('R² Score')
plt.xlabel('Model')

plt.tight_layout()
plt.show()



