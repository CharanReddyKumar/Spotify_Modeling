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
import seaborn as sns
from scipy.stats import skew
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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


