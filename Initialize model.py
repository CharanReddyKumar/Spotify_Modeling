#%%
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

