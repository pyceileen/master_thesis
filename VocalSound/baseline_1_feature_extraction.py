# -*- coding: utf-8 -*-
"""
Created on Wed May 13 17:27:05 2020

@author: Pei-yuChen
"""

import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from baseline_preprocessing import feature_extraction_all


labels_df_db = pd.read_csv('data\\label_df_EmoDB_4emo.csv')
# select columns
labels_df_db = labels_df_db[['speaker', 'emotion']]
audios_db = pickle.load(open('data\\audio_EmoDB_4emo_7.5s.pkl', 'rb'))

labels_df_ie = pd.read_csv('data\\label_df_IEMOCAP_4emo.csv')
# select columns
labels_df_ie = labels_df_ie[['speaker', 'emotion']]
audios_ie = pickle.load(open('data\\audio_IEMOCAP_4emo_7.5s.pkl', 'rb'))

labels_df_rv = pd.read_csv('data\\label_df_RAVDESS_4emo.csv')
# select columns
labels_df_rv = labels_df_rv[['speaker', 'emotion']]
audios_rv = pickle.load(open('data\\audio_RAVDESS_4emo_7.5s.pkl', 'rb'))


# split iemocap for ensemble
train_audios_ie, ensemble_audios_ie, train_labels_df_ie, ensemble_labels_df_ie = train_test_split(
    audios_ie, labels_df_ie, test_size=0.15, shuffle=True, random_state = 6)

labels_df = pd.concat([labels_df_db,train_labels_df_ie,labels_df_rv], ignore_index=True)
audios = audios_db+train_audios_ie+audios_rv

#%%
labels_df.to_csv('data\\baesline_label_df_4emo.csv', index=False)  

ensemble_labels_df_ie.to_csv('data\\baesline_label_df_4emo_ensemble.csv', index=False)  

#%%
# transform all the audio vectors to features
audio_vectors = pd.DataFrame()
for idx, audio in enumerate(audios): # this for loop took half a day to process
    vector = feature_extraction_all(audio, sr = 16000)
    audio_vectors = audio_vectors.append(vector, ignore_index=True)
    print(idx)
#%%
audio_vectors.to_csv('data\\baseline_features_all_4emo.csv', index=False)


#%%

audio_vectors = pd.DataFrame()
for idx, audio in enumerate(ensemble_audios_ie): # this for loop took half a day to process
    vector = feature_extraction_all(audio, sr = 16000) 
    audio_vectors = audio_vectors.append(vector, ignore_index=True)
    print(idx)

#%%
audio_vectors.to_csv('data\\baseline_features_all_4emo_ensemble.csv', index=False)









