# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:19:44 2020

@author: Pei-yuChen
"""

import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

labels_df_db = pd.read_csv('data/label_df_EmoDB_4emo.csv')
# select columns
labels_df_db = labels_df_db[['speaker', 'emotion']]
audios_db = pickle.load(open('data/audio_EmoDB_4emo_7.5s.pkl', 'rb'))

audio_hap = []
audio_hap.append(audios_db[0])
audio_neu = []
audio_neu.append(audios_db[1])
audio_ang = []
audio_ang.append(audios_db[2])
audio_sad = []
audio_sad.append(audios_db[5])

#%%
from util.prepare_data import prepare_data_librosa

FEATURE = 'logmel'


hap_1, hap_2, hap_3 = prepare_data_librosa(audio_hap, 
                                           features=FEATURE, N_FEATURES=3, 
                                           scaled=True)

neu_1, neu_2, neu_3 = prepare_data_librosa(audio_neu, 
                                           features=FEATURE, N_FEATURES=3, 
                                           scaled=True)

ang_1, ang_2, ang_3 = prepare_data_librosa(audio_ang, 
                                           features=FEATURE, N_FEATURES=3, 
                                           scaled=True)
sad_1, sad_2, sad_3 = prepare_data_librosa(audio_sad, 
                                           features=FEATURE, N_FEATURES=3, 
                                           scaled=True)


#%%
import librosa
import librosa.display
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
sr=16000
S = librosa.feature.melspectrogram(y=audios_db[5], sr=sr, n_mels=128,
                                    fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=16000,
                         fmax=8000, ax=ax)

ax.set(title='Mel-frequency spectrogram')

