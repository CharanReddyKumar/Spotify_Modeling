#%%
#Linear Regressor

#import libraries needed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score



# Features (X) and Target Variable (y)
X = df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1)
y = df['instrumentalness']

# Feature scaling (optional but recommended for Linear Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
linear_reg = LinearRegression()


# Train the model
linear_reg.fit(X_train, y_train)

# Make predictions
y_pred = linear_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Linear Regression - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")

#*The Linear Regression model has an MSE of 0.0639, suggesting that, on average, the model's predictions deviate by this amount from the actual values. The R-squared value of 0.3163 indicates that the model explains about 31.63% of the variability in the target variable. 

# %%
#Decision Tree Regressor

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Features (X) and Target Variable (y)
X = df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1)
y = df['instrumentalness']

# Feature scaling 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Regressor model
decision_tree_reg = DecisionTreeRegressor(random_state=42)

# Train the model
decision_tree_reg.fit(X_train, y_train)



# Make predictions
y_pred = decision_tree_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Decision Tree Regressor - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")
# * The Decision Tree Regressor has a higher MSE compared to Linear Regression. In terms of MSE, lower values are preferred, so the Linear Regression model performed better in this regard.
# * The Decision Tree Regressor has a lower R² compared to Linear Regression. R² measures the proportion of variance explained, and a higher value is generally better. In this case, Linear Regression captured a larger proportion of the variance.
# * he Linear Regression model outperformed the Decision Tree Regressor in both MSE and R². It means that, based on the provided metrics, the Linear Regression model is preferable for this task.

#%%
#Gradient Boosting Modeling

from sklearn.ensemble import GradientBoostingRegressor


# Features (X) and Target Variable (y)
X = df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1)
y = df['instrumentalness']

# Feature scaling (optional but recommended for Gradient Boosting)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Gradient Boosting Regressor model
gradient_boosting_reg = GradientBoostingRegressor(random_state=42)

# Train the model
gradient_boosting_reg.fit(X_train, y_train)


# Make predictions
y_pred = gradient_boosting_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Gradient Boosting Regressor - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")

# * The Gradient Boosting Regressor has the lowest MSE among the three models. Lower MSE indicates better performance, so the Gradient Boosting Regressor outperforms both the Decision Tree Regressor and Linear Regression in terms of MSE.
# * The Gradient Boosting Regressor also has the highest R², indicating that it explains a larger proportion of the variance compared to the other models.
#The Gradient Boosting Regressor performed better than both the Decision Tree Regressor and Linear Regression in terms of both MSE and R².
#Gradient Boosting models are often powerful and can capture complex relationships in the data.
#%%

#Random forest Regressor

#Import library
from sklearn.ensemble import RandomForestRegressor


# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1))
y = df['instrumentalness']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
random_forest_reg = RandomForestRegressor(random_state=42)

# Train the model
random_forest_reg.fit(X_train, y_train)

# Make predictions
y_pred = random_forest_reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Random Forest Regressor - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")

#*The Random Forest Regressor has the lowest MSE among all the models, indicating the best performance in terms of minimizing prediction errors.
#*The Random Forest Regressor also has the highest R², indicating that it explains a larger proportion of the variance compared to the other models.
#*The Random Forest Regressor outperforms the XGBoost Regressor, Gradient Boosting Regressor, and Decision Tree Regressor in terms of both MSE and R².

#%%
#XGBoost

from xgboost import XGBRegressor

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(['instrumentalness', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre'], axis=1))
y = df['instrumentalness']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
xgb_reg = XGBRegressor(random_state=42)

# Train the model
xgb_reg.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_reg.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Print evaluation metrics for XGBoost Regressor
print(f"XGBoost Regressor - Mean Squared Error: {mse_xgb:.4f}, R-squared: {r2_xgb:.4f}")
#*The XGBoost Regressor has a lower MSE than the Decision Tree Regressor and Gradient Boosting Regressor but a slightly higher MSE than the Random Forest Regressor.
#*The XGBoost Regressor has a higher R² than the Decision Tree Regressor and Gradient Boosting Regressor but a slightly lower R² than the Random Forest Regressor.
#*The XGBoost Regressor performs well, providing a good balance between MSE and R².


# %%
# Calculate residuals
residuals = y_test - y_pred_random_forest

# Plotting the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.title('Distribution of Residuals for Random Forest Regressor')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

#*The skewness and kurtosis values indicate that the distribution is not normally distributed. This is not uncommon for random forest regressors, as they can produce residuals with a variety of distributions.
#*Overall, the histogram suggests that the random forest regressor is performing well. The residuals are centered at zero and have a relatively small standard deviation. However, the skewed distribution and outliers suggest that there may be some outliers in the data that are affecting the model's predictions.

# %%
import matplotlib.pyplot as plt

# Plotting Mean Squared Error
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for Mean Squared Error
ax1.bar(evaluation_df['Model'], evaluation_df['Mean Squared Error'], color='blue', alpha=0.7)
ax1.set_ylabel('Mean Squared Error')
ax1.set_title('Model Mean Squared Error')

# Rotating x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

# Plotting R-squared
fig, ax2 = plt.subplots(figsize=(10, 6))

# Line plot for R-squared
ax2.plot(evaluation_df['Model'], evaluation_df['R-squared'], marker='o', color='red')
ax2.set_ylabel('R-squared')
ax2.set_title('Model R-squared')


plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

#The bar graph shows the mean squared error (MSE) for four different machine learning models: linear regression, decision tree regressor, gradient boosting regressor, and random forest regressor. The MSE is a measure of how well a model's predictions match the actual values. A lower MSE indicates a better model fit.

#As you can see from the graph, the random forest regressor has the lowest MSE, followed by the gradient boosting regressor, the decision tree regressor, and finally, the linear regression model. This suggests that the random forest regressor is the best performing model out of the five.

#The Line graph shows that Random Forest Regressor is best fit compared to other model.A Higher R square indicates a better model fit.
#The random forest regressor is the best performing model out of the four, based on the MSE metric and R squared.