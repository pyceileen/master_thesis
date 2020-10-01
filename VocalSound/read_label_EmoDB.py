# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 13:06:35 2020

@author: Pei-yuChen
"""
import pandas as pd
import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)


file_list = []
path = 'data\EmoDB\wav'
for path, subdirs, files in os.walk(path):
    files[:] = [d for d in files if d.endswith('.wav')] 
    for name in files:
        file_list.append(os.path.join(path, name))

#%%        
label_df = pd.DataFrame(columns=['wav_file','speaker','emotion'])

label_df['wav_file'] = files
label_df['speaker'] = label_df.wav_file.apply(lambda x: 'DB'+x[0:2])
label_df['emotion'] = label_df.wav_file.apply(lambda x: x[5])


def db2en(row):
    if row['emotion'] == 'W':
        emo = 'ang'
    elif row['emotion'] == 'L':
        emo = 'bor'
    elif row['emotion'] == 'E':
        emo = 'dis'
    elif row['emotion'] == 'A':
        emo = 'fea'
    elif row['emotion'] == 'F':
        emo = 'hap'
    elif row['emotion'] == 'T':
        emo = 'sad'
    else:
        emo = 'neu'        
    return(emo)

label_df['emotion'] = label_df.apply(lambda x:db2en(x), axis=1) 

#%%
label_df.to_csv('data\label_df_EmoDB.csv', index=False)
