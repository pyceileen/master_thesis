# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:23:39 2020

@author: Pei-yuChen
"""

import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import librosa
from tqdm import tqdm
import pickle
import numpy as np 
import pandas as pd 
import itertools

file_list_db = []
path = 'data\EmoDB\wav'
for path, subdirs, files in os.walk(path):
    files[:] = [d for d in files if d.endswith('.wav') and not d.startswith('.')] 
    for name in files:
        file_list_db.append(os.path.join(path, name))

file_list_ie = []
for i in range(1,6):
    path = 'data\IEMOCAP_full_release\Session{0}\sentences\wav'.format(i)
    session_file = []
    for path, subdirs, files in os.walk(path):
        files[:] = [d for d in files if d.endswith('.wav') and not d.startswith('.')] 
        for name in files:
            session_file.append(os.path.join(path, name))
    file_list_ie.append(session_file)
file_list_ie = list(itertools.chain(*file_list_ie))

#%%
file_list = file_list_db+file_list_ie
audio_length = []
for i in tqdm(file_list):
    y, sr = librosa.load(i, sr=16000)
    length = librosa.get_duration(y=y, sr=16000)
    audio_length.append(length)


audio_mean = np.mean(audio_length)
audio_std = np.std(audio_length)


#%%
audio_vectors = []
for i in tqdm(file_list):
    y, sr = librosa.load(i, sr=16000, duration = 7.5) 
    y = librosa.util.fix_length(y, 120000) #  zero padding 16000*7.5
    audio_vectors.append(y)


# export
with open('data\\audios_all.pkl','wb') as f:
    pickle.dump(audio_vectors, f)



#%%

labels_df_EmoDB = pd.read_csv('data\EmoDB_label_df.csv')
labels_df_IEMOCAP = pd.read_csv('data\IEMOCAP_label_df.csv')


labels_df_EmoDB = labels_df_EmoDB[['speaker', 'emotion']]
labels_df_IEMOCAP = labels_df_IEMOCAP[['speaker', 'emotion']]

labels_df_all = pd.concat([labels_df_EmoDB, labels_df_IEMOCAP])
 

labels_df_all.to_csv('data\labels_df_all.csv', index=False)  









