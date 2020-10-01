# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 19:38:19 2020

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

labels_df = pd.read_csv('data\labels_df_all.csv')

options = ['ang','hap','neu','sad']
labels_df_filtered = labels_df.loc[labels_df['emotion'].isin(options)]

filtered_inx = labels_df_filtered.index.values.astype(int)
file_list_filtered = [file_list[i] for i in filtered_inx]

audio_vectors = []
for i in tqdm(file_list_filtered):
    y, sr = librosa.load(i, sr=16000, duration = 4) 
    y = librosa.util.fix_length(y, 64000) #  zero padding 16000*7.5
    audio_vectors.append(y)

with open('data\\audios_all_4emo.pkl','wb') as f:
    pickle.dump(audio_vectors, f)

labels_df_filtered.to_csv('data\labels_df_all_4emo.csv', index=False)  


