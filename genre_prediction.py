
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


#%%glm model


#%% classification tree


#%%assessing model 