# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:20:28 2020

@author: Pei-yuChen
"""
# adopt from: https://github.com/Demfier/multimodal-speech-emotion-recognition/blob/master/1_extract_emotion_labels.ipynb

import os 
os.chdir(r"C:\Users\Pei-yuChen\Desktop\Programming\Model\VocalSound")

import re
import pandas as pd

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

start_times, end_times, wav_file_names = [], [], []
emotions, V, A, D, speakers = [], [], [], [], []

for sess in range(1, 6):
    emo_evaluation_dir = 'data\IEMOCAP_full_release\Session{}\dialog\EmoEvaluation\\'.format(sess)
    evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
    for file in evaluation_files:
        with open(emo_evaluation_dir + file) as f:
            content = f.read()
        info_lines = re.findall(info_line, content)
        for line in info_lines[1:]:  # the first line is a header
            start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
            start_time, end_time = start_end_time[1:-1].split('-')
            val, act, dom = val_act_dom[1:-1].split(',')
            val, act, dom = float(val), float(act), float(dom)
            start_time, end_time = float(start_time), float(end_time)
            speaker = wav_file_name[0:5]+wav_file_name[-4]
            start_times.append(start_time)
            end_times.append(end_time)
            wav_file_names.append(wav_file_name)
            emotions.append(emotion)
            V.append(val)
            A.append(act)
            D.append(dom)
            speakers.append(speaker)

#%%
df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 
                                   'emotion', 'V', 'A', 'D', 'speaker'])

df_iemocap['start_time'] = start_times
df_iemocap['end_time'] = end_times
df_iemocap['wav_file'] = wav_file_names
df_iemocap['emotion'] = emotions
df_iemocap['V'] = V
df_iemocap['A'] = A
df_iemocap['D'] = D
df_iemocap['speaker'] = speakers

df_iemocap.tail()

#%%
df_iemocap.to_csv('data\IEMOCAP_label_df.csv', index=False)








