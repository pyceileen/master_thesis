# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:07:23 2020

@author: Pei-yuChen
"""
import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

# adopt from: https://github.com/Demfier/multimodal-speech-emotion-recognition/blob/master/1_extract_emotion_labels.ipynb


import librosa
from tqdm import tqdm
import pickle
import itertools
import librosa.display
import numpy as np

 
file_list = []
for i in range(1,6):
    path = 'data\IEMOCAP_full_release\Session{0}\sentences\wav'.format(i)
    session_file = []
    for path, subdirs, files in os.walk(path):
        files[:] = [d for d in files if d.endswith('.wav') and not d.startswith('.')] 
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
print(audio_mean+audio_std)

#%%
import pandas as pd

label_df = pd.read_csv('data\label_df_IEMOCAP.csv')

# select only improvised 
label_df_filtered = label_df.loc[(label_df['wav_file'].str.contains("impro"))]

# select emotions
options = ['ang','hap','neu','sad']
label_df_filtered = label_df_filtered.loc[label_df_filtered['emotion'].isin(options)]

filtered_inx = label_df_filtered.index.values.astype(int)
file_list_filtered = [file_list[i] for i in filtered_inx]


audio_vectors = []
for i in tqdm(file_list_filtered):
    y, sr = librosa.load(i, sr=16000, duration=7.5) 
    audio_vectors.append(y)

#%%
# export
with open('data\\audio_IEMOCAP_4emo_7.5s.pkl','wb') as f:
    pickle.dump(audio_vectors, f)

label_df_filtered.to_csv('data\label_df_IEMOCAP_4emo.csv', index=False)  








