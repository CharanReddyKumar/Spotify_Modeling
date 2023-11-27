#%%[markdown]
## Column Description
#
# * track_id: The Spotify ID for the track
#
# * artists: The artists' names who performed the track. If there is more than one artist, they are separated by a ;
#
# * album_name: The album name in which the track appears
#
# * track_name: Name of the track
#
# * popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity.
#
# * duration_ms: The track length in milliseconds
#
# * explicit: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
#
# * danceability: Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable
#
# * energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale
#
# * key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
#
# * loudness: The overall loudness of a track in decibels (dB)
#
# * mode: Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0
#
# * speechiness: Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks
#
# * acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
#
# * instrumentalness: Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
#
# * liveness: Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live
#
# * valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry)
#
# * tempo:*The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration
#
# * time_signature: An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of 3/4, to 7/4.
#
# * track_genre: The genre in which the track belongs

#%%[markdown]
## EDA and Variable Selection

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
# %%
#Check for null values
df.isnull().sum()
# %%
#Since there are only 3 misisng values, we can just drop them 
#Remove NA values
df = df.dropna()
df.isnull().sum()

# %%
df = df.drop("Unnamed: 0", axis=1) #remove the unnamed column
# %%
#Previews dataframe
df.describe()

# %%
#checking for duplicates
df.duplicated().sum()
# %%
#removing duplicates
df = df.drop_duplicates() #remove all duplicates
df.duplicated().sum()
# %%
#View data type and info, inform us how to format variables later on for modeling
df.info()

df_genres = df.copy()

# %%
#Corelation matrix
corr=df.corr()
corr

# Correlation with 'popularity'
correlation_with_popularity = corr['popularity'].sort_values(ascending=False)

print(correlation_with_popularity)

# heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

#%%[markdwon]

# As one of our target variables is popularity we dont see any great significant corelation between any other feature variable, so we can use regression models if we 
# perform transformation if the data is not normally distributed or else we can try other modeling methods.
# %%
#Identify numeric and categorical columns

numeric_cols = []
categorical_cols = []
for col in df.columns:
    if df[col].dtype == np.float64 or df[col].dtype == np.int64:
        numeric_cols.append(col)
    else:
        categorical_cols.append(col)

print('numeric columns:', numeric_cols)
print('Categorical columns:', categorical_cols)

# %%
# Seperate out the discrete numeric columns
df.head()

discrete_numeric=[feature for feature in numeric_cols if df[feature].nunique()<20]
discrete_numeric

# List of discrete numeric columns
discrete_numeric_columns = discrete_numeric

# Create a LabelEncoder object
label_encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Display the updated DataFrame
print(df.head())


# %%[Markdown]
## Popularity EDA

#%%
# Distribution of Popularity
plt.figure(figsize=(10, 6))
sns.histplot(df['popularity'], bins=30, kde=True)
plt.title('Popularity Distribution')
plt.xlabel('Popularity')
plt.ylabel('Frequency')
plt.show()

# %%
for feature in numeric_cols:
    dataset=df.copy()
    print(feature, 'skewness is :', skew(dataset[feature]))
    sns.histplot(x=feature, data=dataset, bins=25, kde=True)
    plt.show()

#%%[markdown]

#We observe the distribution of continuous features:

# * Danceability, Valence, and Tempo:** These features exhibit an almost normal distribution.

# * Loudness: The distribution of loudness is left-skewed.

# * Other Features: The distributions of other features are right-skewed.

# Various tarnsformations can be performed to remove the skewness.

#%%
for feature in ['explicit', 'mode', 'time_signature', 'key']:
    dataset=df.copy()
    sns.barplot(x=feature, y=dataset['popularity'], data=dataset, estimator=np.median)
    plt.show()

#%%[markdown]
# We see that songs which contain explicit lyrics are more popular in comparision with songs that do not contain such lyrics.
#
# key : Popularity of the song is not much dependent on the key of the song
#
# Modes : The tracks in both the modes are equally popular.
#
# time_signature : Tracks having time_signature 0 and 4 are more popular than other.
#%%

# checking the outliers
features_continuous_numerical = ['popularity',
 'duration_ms',
 'danceability',
 'energy',
 'loudness',
 'speechiness',
 'acousticness',
 'instrumentalness',
 'liveness',
 'valence',
 'tempo']
for feature in features_continuous_numerical:
    dataset=df.copy()
    sns.boxplot(data=dataset[feature])
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()

#%%[markdown]
# We see that apart from energy, acousticness and valence, there are a lot of outliers in all other features
# %%
#########################
#       GENRE EDA       #
#########################
df_genres.head()
#check variable types prior to encoding, as we need to know
# the genre label to select the desired ones for analysis 
data_types = df_genres.dtypes
#track_genre is a string with several categories, lets examine further 

#All of the genres present in the dataset
unique_explicit_values = df_genres['track_genre'].unique()
print(unique_explicit_values)
num_unique_genres = df_genres['track_genre'].nunique()
print(f'Number of unique genres: {num_unique_genres}')

#there are 114 unique genres... we will subset our data into predicting just a few of these
#%%
#Do all genres have enough data to explore? 
#Count the occurrences of each genre in the 'track_genre' column
genre_counts = df_genres['track_genre'].value_counts()
# Count the number of genres with at least 1000 rows
num_genres_with_1000_rows = (genre_counts >= 1000).sum()
print(f"Number of genres with at least 1000 rows: {num_genres_with_1000_rows}")
#33 genres have 1000+  records, lets see what those are: 
# Filter genres with at least 1000 rows
genres_1000 = genre_counts[genre_counts >= 1000].index

# Subset the dataframe based on selected genres
df_selected = df_genres[df_genres['track_genre'].isin(genres_1000)]

# Display the selected genres
print("Selected genres with at least 1000 rows:")
print(genres_1000)
#Lets go with a subset of genres from those with at least 1000 rows so that 
#we can ensure enough training data 

#%% 
# selecting relevant genres:
# I would like to explore how spotify can tell the difference between
#similar genres, lets select electornic music that we might not know
#the specific differences between ourselves and see if the model can classify them

# Selecting relevant genres:
# I would like to explore how Spotify can tell the difference between
# similar genres. Let's select electronic music genres that we might not know
# the specific differences between ourselves and see if the model can classify them.
#Also add inrock-n-roll, it is upbeat but not edm to see if model picks up on those differences

genres_list = ['disco', 'electronic', 'industrial', 'techno', 'synth-pop', 'funk', 'rock-n-roll']
selected_genres = df_selected[df_selected['track_genre'].isin(genres_list)]
df_selected_genres_shape = selected_genres.shape
print(f"Shape of the selected genres DataFrame: {df_selected_genres_shape}")
selected_genres.head()

#%%
#now lets encode the relevant genres 
# Fit and transform the 'track_genre' column using .loc
selected_genres.loc[:, 'genre'] = label_encoder.fit_transform(selected_genres['track_genre'])

#%%
features_continuous_numerical
# Set up subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))
fig.suptitle('Distribution of Features by Genre')

# Plot histograms for each feature by genre
for i, feature in enumerate(features_continuous_numerical):
    row, col = divmod(i, 4)
    sns.histplot(data=selected_genres, x=feature, hue='track_genre', bins=30, kde=True, ax=axes[row, col], palette='viridis')
    axes[row, col].set_title(feature)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#%%
#examine categorical / factor variables 
for feature in ['explicit', 'mode', 'time_signature', 'key']:
    dataset=selected_genres
    sns.barplot(x=feature, y=selected_genres['genre'], data=selected_genres, estimator=np.median)
    plt.show()

# %%
