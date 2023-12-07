#%%[markdown]
## Modeling to predict Danceability
#%%
# Import
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from Spotify_EDA import df_danceabiltiy # Import df data frame from Spotify_EDA to use the processed data for modeling

# %%
# %%