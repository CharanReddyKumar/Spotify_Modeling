
## READ IN DATA 
#%%
import pandas as pd
csv_file_path = '/Users/paulinemnev/Desktop/DataMining/Final_project/Spotify_Modeling/dataset.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the DataFrame
print(df)

# %%
## SELECT COLUMNS that i belive are relevant, this will change with analysis 
# Specify the columns you want to keep
columns_to_keep = ['track_id', 'track_name', 'popularity', 'duration_ms', 'explicit',
                   'danceability', 'energy', 'loudness', 'speechiness',
                   'acousticness', 'instrumentalness', 'valence', 'tempo', 'track_genre']

# Create a new DataFrame with only the specified columns
df_eda = df[columns_to_keep]

# Display the DataFrame with selected columns
print(df_eda)

#%% Lets check for NA values
missing_values = df_eda.isna().sum()
print(missing_values)
#just drop, theres only one 
df_eda = df_eda.dropna()

# Display the cleaned DataFrame
print(df_eda)

#%%
#Lets examine the variables 
df_eda.head()

data_types = df_eda.dtypes
print(data_types)
#track_id is a randomly generated sequence of numbers and letters. we don't need this. we will drop it.

# track_name will be helpful to look at results

#track_genre is a string with several categories, lets examine further 

#explicit is a boolean, lets examine further
# %%
#look at explicit 
# Display unique values in the 'explicit' column
unique_explicit_values = df_eda['explicit'].unique()
print(unique_explicit_values)

#lets change these to 0s and 1s for our modeling purposes, 0 beign false, 1 being true.

# Convert boolean values to integers (0 for False, 1 for True) in the 'explicit' column
df_eda['explicit'] = df_eda['explicit'].astype(int)

# Display the updated DataFrame
print(df_eda)
#%%
#lets look at a plot of these
import matplotlib.pyplot as plt

# Plot a bar chart for the 'explicit' column
df_eda['explicit'].value_counts().plot(kind='bar', rot=0, color=['skyblue', 'salmon'])

# Add labels and title
plt.xlabel('Explicit')
plt.ylabel('Count')
plt.title('Distribution of Explicit Values')

# Show the plot
plt.show()

#most are not explicit as we see here

# %%
# ok now lets look at track_genre the same way 

# Display unique values in the 'explicit' column
unique_explicit_values = df_eda['track_genre'].unique()
print(unique_explicit_values)

# Find the number of unique values in the 'track_genre' column
num_unique_genres = df_eda['track_genre'].nunique()
print(f'Number of unique genres: {num_unique_genres}')

#there are 114 unique genres... we might have to subset our data into predicting justa  few of these

# Count the occurrences of each genre in the 'track_genre' column
genre_counts = df_eda['track_genre'].value_counts()

# Display the top genres with the most rows
print(genre_counts)

# Count the occurrences of each genre in the 'track_genre' column
genre_counts = df_eda['track_genre'].value_counts()

# Count the number of genres with at least 1000 rows
num_genres_with_1000_rows = (genre_counts >= 1000).sum()

# Display the number of genres with at least 1000 rows
print(f"Number of genres with at least 1000 rows: {num_genres_with_1000_rows}")

#ok they also all have that many records... so we will have to decide for ourselves
# %%
#lets select  genres we would like to predict and focus on those. 

#perhaps different kinds of rock ? or different kinds of pop? or edm?
# #these would be interesting for me... but lets also see what the computer
# #identifies as the top 10 genres that are the most differnt from one another

# %%
