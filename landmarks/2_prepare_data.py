# %% coding=utf-8
import numpy as np
import pandas as pd


data = pd.read_excel('E:/data/SCUT-FBP5500_v2/All_Ratings.xlsx', None)
df_asian = data['Asian_Female']
df_data = df_asian.groupby('Filename').agg(np.mean)
df_data['file'] = df_data.index.astype(str)
df_label = df_data[['file', 'Rating']]

#df_label.drop(['Unnamed: 0'], axis=1, inplace=True)
df_features = pd.read_csv('data/face/features.csv')
df_features['file'] = [x.replace('E:/data/SCUT-FBP5500_v2/Images\\', '') for x in df_features['image_index']]
try:
    df_features.drop(['Unnamed: 0'], axis=1, inplace=True)
except:
    pass

try:
    df_features.drop(['image_index'], axis=1, inplace=True)
except:
    pass

df_input = pd.merge(df_label, df_features, on='file')
#df_input.rename({'':'label'}, inplace=True)
df_input.to_csv('data/face/df_input.csv')
