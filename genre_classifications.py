
#%%
#Import 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
df_genres.head()
genre_counts = df_genres['track_genre'].value_counts()
num_genres_with_1000_rows = (genre_counts >= 1000).sum()
genres_1000 = genre_counts[genre_counts >= 1000].index
df_selected = df_genres[df_genres['track_genre'].isin(genres_1000)]


genres_list = ['disco', 'electronic', 'industrial', 'techno', 'synth-pop', 'funk']

selected_genres = df_selected[df_selected['track_genre'].isin(genres_list)]

music_data = selected_genres
music_data.dropna(inplace=True)
music_data.drop_duplicates(inplace=True)


features_set1 = ['popularity', 'acousticness', 'instrumentalness']


X = music_data[features_set1]
y = music_data['track_genre']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f'Decision Tree Classifier Accuracy: {dt_accuracy:.4f}')

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Classifier Accuracy: {rf_accuracy:.4f}')

# Support Vector Classifier (SVC)
svc_classifier = SVC()
svc_classifier.fit(X_train, y_train)
svc_predictions = svc_classifier.predict(X_test)
svc_accuracy = accuracy_score(y_test, svc_predictions)
print(f'Support Vector Classifier Accuracy: {svc_accuracy:.4f}')

# K-Nearest Neighbors Classifier (KNN)
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f'K-Nearest Neighbors Classifier Accuracy: {knn_accuracy:.4f}')

# %%
