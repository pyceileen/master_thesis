# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:17:05 2020

@author: Pei-yuChen
"""
import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

#%%
import librosa
from tqdm import tqdm
import pickle
import numpy as np 
import pandas as pd 
import itertools

###########
# IEMOCAP #
###########

# read all the files path to a list 
 
IEMOCAP_list = []
for i in range(1,6):
    path = 'data\IEMOCAP_full_release\Session{0}\sentences\wav'.format(i)
    session_file = []
    for path, subdirs, files in os.walk(path):
        files[:] = [d for d in files if d.endswith('.wav') and not d.startswith('.')] 
        for name in files:
            session_file.append(os.path.join(path, name))
    IEMOCAP_list.append(session_file)
IEMOCAP_list = list(itertools.chain(*IEMOCAP_list))


IEMOCAP_df = pd.read_csv('data\label_df_IEMOCAP.csv')

# select only improvised 
IEMOCAP_df_filtered = IEMOCAP_df.loc[(IEMOCAP_df['wav_file'].str.contains("impro"))]

# select emotions
options = ['ang','hap','neu','sad']
IEMOCAP_df_filtered = IEMOCAP_df_filtered.loc[IEMOCAP_df_filtered['emotion'].isin(options)]

IEMOCAP_inx = IEMOCAP_df_filtered.index.values.astype(int)
IEMOCAP_list_filtered = [IEMOCAP_list[i] for i in IEMOCAP_inx]


#%% train_test_split for training

import pandas as pd


labels_df_ie = pd.read_csv('data/label_df_IEMOCAP_4emo.csv')
# select columns
# labels_df_ie = labels_df_ie[['speaker', 'emotion', 'transcriptions']]
audios_ie = pickle.load(open('data/audio_IEMOCAP_4emo_7.5s.pkl', 'rb'))



from sklearn.model_selection import train_test_split
# split iemocap for ensemble
train_audios_ie, ensemble_audios_ie, train_labels_df_ie, ensemble_labels_df_ie = train_test_split(
    audios_ie, labels_df_ie, test_size=0.15, shuffle=True, random_state = 6)




ensemble_index = ensemble_labels_df_ie.index
ensemble_list = [IEMOCAP_list_filtered[i] for i in ensemble_index]

# append string
string = r'C:\\Users\\Pei-yuChen\\Desktop\\Programming\\Model\\VocalSound\\'
ensemble_list = [string + x for x in ensemble_list]

#%%
import pickle

with open('data/ensemble_list', 'wb') as fp:
    pickle.dump(ensemble_list, fp)
    
ensemble_labels_df_ie.to_csv('data/ensemble_IEMOCAP_df.csv', index=False)






