
#%%
#Import 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
# %%
df =pd.read_csv('dataset.csv')
df.head(5) #Read Data first 5 rows
#Check for null values
df.isnull().sum()
#Remove NA values
df = df.dropna()
df.isnull().sum()
df = df.drop("Unnamed: 0", axis=1) #remove the unnamed column
#checking for duplicates
df.duplicated().sum()
#removing duplicates
df = df.drop_duplicates() #remove all duplicates
df.duplicated().sum()
df_genres = df.copy()

#%%
#%% 
#PREDICT GENRES OVERALL
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Clean Data
music_data = df_genres
music_data.dropna(inplace=True)
music_data.drop_duplicates(inplace=True)

X = music_data.drop(columns = ["track_id", "artists", "album_name", "track_name", "track_genre"])
y = music_data["track_genre"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print(score)

#not so good

#%%
####SPECIFIC GENRES?#####

df_genres.head()
#check variable types prior to encoding, as we need to know
# the genre label to select the desired ones for analysis 
data_types = df_genres.dtypes
#track_genre is a string with several categories, lets examine further 
genre_counts = df_genres['track_genre'].value_counts()
# Count the number of genres with at least 1000 rows
num_genres_with_1000_rows = (genre_counts >= 1000).sum()
#33 genres have 1000+  records, lets see what those are: 
# Filter genres with at least 1000 rows
genres_1000 = genre_counts[genre_counts >= 1000].index

# Subset the dataframe based on selected genres
df_selected = df_genres[df_genres['track_genre'].isin(genres_1000)]

genres_list = ['disco', 'electronic', 'industrial', 'techno', 'synth-pop', 'funk']
selected_genres = df_selected[df_selected['track_genre'].isin(genres_list)]

#%% feature selection FOR SPECIFIC GENRES 
#Lets select the relevant columns. We won't be needing to encode aritst name etc
column_names = selected_genres.columns
print(column_names)
#lets select just the ones we need
# Assuming selected_genres is your DataFrame
columns_to_select = ['popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 
                     'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                     'valence', 'tempo', 'time_signature', 'track_genre']  # Replace with the actual column names you want to select

# Create a new DataFrame with selected columns
predict_genre = selected_genres[columns_to_select].copy()

#%%
#Identify numeric and categorical columns
numeric_cols = []
categorical_cols = []
for col in predict_genre.columns:
    if predict_genre[col].dtype == np.float64 or predict_genre[col].dtype == np.int64:
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print('numeric columns:', numeric_cols)
print('Categorical columns:', categorical_cols)
#%%
# Create a LabelEncoder object
label_encoder = LabelEncoder()

for col in categorical_cols:
    predict_genre[col] = label_encoder.fit_transform(predict_genre[col])

# Display the updated DataFrame
print(predict_genre.head())

#%% 
#Correlation analysis 
import pandas as pd

# Assuming df is your DataFrame with numeric features and dummy coded factor variables
correlation_matrix = predict_genre.corr()
target_correlation = correlation_matrix['track_genre'].abs()  # Replace 'genre_dummy' with your actual target variable

# Select features with high correlation
selected_features = target_correlation[target_correlation > 0.2].index  # Adjust the correlation threshold as needed

print(selected_features)

#from this we get: 'duration_ms', 'instrumentalness', 'valence'
#%%
# Recursive Feature Elimination (RFE):
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Assuming X is your feature matrix and y is the target variable
X = predict_genre.drop('track_genre', axis=1)  # Remove the target variable
y = predict_genre['track_genre']

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)  # Set the number of features you want to select
fit = rfe.fit(X, y)

selected_features2 = X.columns[fit.support_]
print(selected_features2)

#from this we get: 'popularity', 'duration_ms', 'key', 'loudness', 'tempo'
#%%
# Feature Importance from Tree-based Models:
from sklearn.ensemble import RandomForestClassifier

# Assuming X is your feature matrix and y is the target variable
X = predict_genre.drop('track_genre', axis=1)  # Remove the target variable
y = predict_genre['track_genre']

model = RandomForestClassifier()
model.fit(X, y)

feature_importance = model.feature_importances_
selected_features3 = X.columns[feature_importance > 0.10]  # Adjust the importance threshold as needed

print(selected_features3)
#from this we get: 'popularity', 'acousticness', 'instrumentalness'

#%%
# so our different kinds of feature selection method got us 3 different combos 
print("Correlation Matrix:", selected_features3)

print("RFE:", selected_features2)

print("Tree-based:", selected_features3)


#%%

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#simple model 
features_set1 = ['popularity', 'acousticness', 'instrumentalness']
X_set1 = predict_genre[features_set1]
y = predict_genre['track_genre']

# Split the data into training and testing sets
X_train_set1, X_test_set1, y_train, y_test = train_test_split(X_set1, y, test_size=0.2, random_state=42)

# Create and train the model
model_set1 = LogisticRegression()
model_set1.fit(X_train_set1, y_train)

# Make predictions
y_pred_set1 = model_set1.predict(X_test_set1)

# Evaluate the model
accuracy_set1 = accuracy_score(y_test, y_pred_set1)
print(f"Model with features set 1 Accuracy: {accuracy_set1:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_set1))

# Model with features ['popularity', 'duration_ms', 'key', 'loudness', 'tempo']
features_set2 = ['popularity', 'duration_ms', 'key', 'loudness', 'tempo']
X_set2 = predict_genre[features_set2]

# Split the data into training and testing sets
X_train_set2, X_test_set2 = train_test_split(X_set2, test_size=0.2, random_state=42)

# Create and train the model
model_set2 = LogisticRegression()
model_set2.fit(X_train_set2, y_train)

# Make predictions
y_pred_set2 = model_set2.predict(X_test_set2)

# Evaluate the model
accuracy_set2 = accuracy_score(y_test, y_pred_set2)
print(f"\nModel with features set 2 Accuracy: {accuracy_set2:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_set2))

#both are so bad?


#%% classification tree
#visualize this process 




#%%assessing model 
##%

#1. overall prediction 
#2. specific genres
#3. feature selection? Which ones are valuable?
#4. Lets look at a tree



#%%
## song recomendation based on genre ?

