# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:09:28 2020

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

label_df = pd.DataFrame(columns=['wav_file','speaker','emotion'])

label_df['wav_file'] = files
label_df['speaker'] = label_df.wav_file.apply(lambda x: 'CR'+x[0:4])
label_df['emotion'] = label_df.wav_file.apply(lambda x: x[9:12].lower())

#%%
label_df.to_csv('data\label_df_CREMA.csv', index=False)









