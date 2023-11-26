#%%[Markdown]
#EDA

#%%
#Import 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# %%
#Load Data
df =pd.read_csv('dataset.csv')
df.head(5) #Read Data first 5 rows
# %%
#Check for null values
df.isnull().sum()
# %%
#Remove NA values
df = df.dropna()
df.isnull().sum()

# %%
df = df.drop("Unnamed: 0", axis=1) #remove the unnamed column
# %%
df.describe()

# %%
#checking for duplicates
df.duplicated().sum()
# %%
#removing duplicates
df = df.drop_duplicates() #remove all duplicates
df.duplicated().sum()
# %%
df.info()
# %%
#Corelation matrix
corr=df.corr()
corr

# heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
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

# Apply label encoding to each discrete numeric column
for column in discrete_numeric_columns:
    df[column] = label_encoder.fit_transform(df[column])

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
# Popularity vs. Numeric Features
numeric_features = ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=feature, y='popularity', data=df)
    plt.title(f'Popularity vs. {feature}')
    plt.xlabel(feature)
    plt.ylabel('Popularity')
    plt.show()


# %%

# Popularity vs. Categorical Features
categorical_features = ['explicit', 'mode', 'time_signature']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=feature, y='popularity', data=df)
    plt.title(f'Popularity vs. {feature}')
    plt.xlabel(feature)
    plt.ylabel('Popularity')
    plt.show()


# %%
# %%
