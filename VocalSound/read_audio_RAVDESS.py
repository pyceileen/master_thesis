# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:51:10 2020

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

#%%
file_list = []
for i in range(1,25):
    path = 'data\RAVDESS\Audio_Speech_Actors_01-24\Actor_{0:02d}'.format(i)
    session_file = []
    for path, subdirs, files in os.walk(path):
        files[:] = [d for d in files if d.endswith('.wav')] 
        for name in files:
            session_file.append(os.path.join(path, name))
    file_list.append(session_file)
file_list = list(itertools.chain(*file_list))

#%%
# get length
audio_length = []
for i in tqdm(file_list):
    y, sr = librosa.load(i, sr=16000)
    length = librosa.get_duration(y=y, sr=16000)
    audio_length.append(length)


audio_mean = np.mean(audio_length)
audio_std = np.std(audio_length)
print('mean + std length: ', audio_mean+audio_std)

#%%
import pandas as pd

label_df = pd.read_csv('data\label_df_RAVDESS.csv')

# select emotions
options = ['ang','hap','neu','sad']
label_df_filtered = label_df.loc[label_df['emotion'].isin(options)]

filtered_inx = label_df_filtered.index.values.astype(int)
file_list_filtered = [file_list[i] for i in filtered_inx]


audio_vectors = []
for i in tqdm(file_list_filtered):
    y, sr = librosa.load(i, sr=16000, duration=7.5) 
    audio_vectors.append(y)

#%%
with open('data\\audio_RAVDESS_4emo_7.5s.pkl','wb') as f:
    pickle.dump(audio_vectors, f)

label_df_filtered.to_csv('data\label_df_RAVDESS_4emo.csv', index=False)  








