# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:27:36 2020

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
import pandas as pd
label_df = pd.DataFrame(columns=['wav_file', 'emotion','speaker'])

label_df['wav_file'] = [x[-24:] for x in file_list]
label_df['speaker'] = label_df.wav_file.apply(lambda x: 'RV'+x[-6:-4])
label_df['emotion'] = label_df.wav_file.apply(lambda x: x[6:8])

def num2en(row):#Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = 
                #sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    if row['emotion'] == '01':
        emo = 'neu'
    elif row['emotion'] == '02':
        emo = 'cal'
    elif row['emotion'] == '03':
        emo = 'hap'
    elif row['emotion'] == '04':
        emo = 'sad'
    elif row['emotion'] == '05':
        emo = 'ang'
    elif row['emotion'] == '06':
        emo = 'fea'
    elif row['emotion'] == '07':
        emo = 'dis'                    
    else:
        emo = 'sur'        
    return(emo)

label_df['emotion'] = label_df.apply(lambda x:num2en(x), axis=1) 

#%%
label_df.to_csv('data\label_df_RAVDESS.csv', index=False)




