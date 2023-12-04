
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
#Load Data
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
#########################
#       GENRE EDA       #
#########################
df_genres.head()
#check variable types prior to encoding, as we need to know
# the genre label to select the desired ones for analysis 
data_types = df_genres.dtypes
#track_genre is a string with several categories, lets examine further 
genre_counts = df_genres['track_genre'].value_counts()
# Count the number of genres with at least 1000 rows
num_genres_with_1000_rows = (genre_counts >= 1000).sum()
print(f"Number of genres with at least 1000 rows: {num_genres_with_1000_rows}")
#33 genres have 1000+  records, lets see what those are: 
# Filter genres with at least 1000 rows
genres_1000 = genre_counts[genre_counts >= 1000].index

# Subset the dataframe based on selected genres
df_selected = df_genres[df_genres['track_genre'].isin(genres_1000)]

genres_list = ['disco', 'electronic', 'industrial', 'techno', 'synth-pop', 'funk']
selected_genres = df_selected[df_selected['track_genre'].isin(genres_list)]

#%% feature selection 
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



#%%glm model


#%% classification tree


#%%assessing model 