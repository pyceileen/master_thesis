# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:40:30 2020

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
speakers = ['DC','JE','JK','KL']
for i in speakers:
    path = 'data\SAVEE\AudioData\{0}'.format(i)
    session_file = []
    for path, subdirs, files in os.walk(path):
        files[:] = [d for d in files if d.endswith('.wav')] 
        for name in files:
            session_file.append(os.path.join(path, name))
    file_list.append(session_file)
file_list = list(itertools.chain(*file_list))


import pandas as pd
import re

label_df = pd.DataFrame(columns=['wav_file', 'emotion','speaker'])

label_df['wav_file'] = [x[21:] for x in file_list]
label_df['speaker'] = label_df.wav_file.apply(lambda x: x[0:2])
label_df['emotion'] = label_df.wav_file.apply(lambda x: x[3:5])
# strip number
def strip_num(emotion):
    x = re.sub(r'\d+', '', emotion)
    return(x)

label_df['emotion'] = label_df['emotion'].apply(strip_num) 

def num2en(row): # 'a', 'd', 'f', 'h', 'n', 'sa' and 'su' 
                # 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise'. 
                # E.g., 'd3.txt' gives the annotation for the 3rd disgust sentence. 

    if row['emotion'] == 'a':
        emo = 'ang'
    elif row['emotion'] == 'd':
        emo = 'dis'
    elif row['emotion'] == 'f':
        emo = 'fea'
    elif row['emotion'] == 'h':
        emo = 'hap'
    elif row['emotion'] == 'n':
        emo = 'neu'
    elif row['emotion'] == 'sa':
        emo = 'sad'                
    else:
        emo = 'sur'        
    return(emo)

label_df['emotion'] = label_df.apply(lambda x:num2en(x), axis=1) 

#%%
label_df.to_csv('data\label_df_SAVEE.csv', index=False)
