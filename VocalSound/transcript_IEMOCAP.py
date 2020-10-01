# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 07:19:14 2020

@author: Pei-yuChen
"""
import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#%%

df = pd.read_csv('data/label_df_IEMOCAP_4emo.csv')

#%% Build Text data files

import re
import os
import pickle
import itertools

useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)

file2transcriptions = {}

for sess in range(1, 6):
    transcripts_path = 'data/IEMOCAP_full_release/Session{}/dialog/transcriptions/'.format(sess)
    transcript_files = [f for f in os.listdir(transcripts_path) if not f.startswith('.')]
    for f in transcript_files:
        with open('{}{}'.format(transcripts_path, f), 'r') as f:
            all_lines = f.readlines()

        for l in all_lines:
            audio_code = useful_regex.match(l).group()
            transcription = l.split(':')[-1].strip()
            # assuming that all the keys would be unique and hence no `try`
            file2transcriptions[audio_code] = transcription
            
file2transcriptions = ({k:v for k,v in file2transcriptions.items() if 'Ses' in k})
#%%
# save dict
with open('data/audiocode2text.pkl', 'wb') as file:
    pickle.dump(file2transcriptions, file)
len(file2transcriptions)


#%%

audiocode2text = pickle.load(open('data/audiocode2text.pkl', 'rb'))

label_df = pd.read_csv('data\label_df_IEMOCAP_4emo.csv')


label_df['transcriptions'] = [audiocode2text[code] for code in label_df['wav_file']]

label_df.to_csv('data\label_df_IEMOCAP_4emo.csv', index=False)



















