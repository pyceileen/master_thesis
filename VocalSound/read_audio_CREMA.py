# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:16:09 2020

@author: Pei-yuChen
"""

import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import itertools

import librosa
from tqdm import tqdm
import pickle
import itertools
import librosa.display
import numpy as np
import pandas as pd


file_list = []
path = 'data\CREMA-D\AudioWAV'
for path, subdirs, files in os.walk(path):
    files[:] = [d for d in files if d.endswith('.wav')] 
    for name in files:
        file_list.append(os.path.join(path, name))
        
        
 #%%
audio_length = []
for i in tqdm(file_list):
    y, sr = librosa.load(i, sr=16000)
    length = librosa.get_duration(y=y, sr=16000)
    audio_length.append(length)

audio_mean = np.mean(audio_length)
audio_std = np.std(audio_length)
print(audio_mean+audio_std)

#%%
labels_df = pd.read_csv('data\label_df_CREMA.csv')

options = ['ang','hap','neu','sad']
labels_df_filtered = labels_df.loc[labels_df['emotion'].isin(options)]

filtered_inx = labels_df_filtered.index.values.astype(int)
file_list_filtered = [file_list[i] for i in filtered_inx]


audio_vectors = []
for i in tqdm(file_list_filtered):
    y, sr = librosa.load(i, sr=16000, duration = 7.5)
    audio_vectors.append(y)

#%%
    
# export
with open('data\\audio_CREMA_4emo_7.5s.pkl','wb') as f:
    pickle.dump(audio_vectors, f)


labels_df_filtered.to_csv('data\label_df_CREMA_4emo.csv', index=False)  
       
        
        
        
        
        
        
        
        
        
        