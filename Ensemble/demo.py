# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 17:44:56 2020

@author: Pei-yuChen
"""

# prepare data
import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)


import pickle
with open ('IEMOCAP_list', 'rb') as fp:
    ensemble_list = pickle.load(fp)

import pandas as pd
label_df = pd.read_csv('label_df_IEMOCAP_4emo.csv')


#%% load models

from util.load_audio import load_all_models, stacked_dataset, audio2wave
from joblib import load
from util.prepare_data import prepare_data_librosa
  
# load audio models  
n_members = 10
all_models = load_all_models(n_members, model_type = '3_categories')
print('Loaded %d submodels!' % len(all_models))
audio_logistic = load('models\\audio_logistic.joblib') 
print('Loaded audio logistic model!')

# load text model
from util.load_text import load_BERT, predict_text
bert_model = load_BERT(model_type='3_categories')

# load ensemble moedel
final_logistic = load('models\\final_logistic.joblib') 



#%% predictions


##### speech to text

from util.speech2text import speech2text
import string
import random


ran = random.randint(0, len(ensemble_list)-1)
file = ensemble_list[ran]
text = speech2text(file)
text_true = label_df['transcriptions'][ran]

print('Speech preditction:', text)
# remove punctuation and lowercase

print('True transcription:', text_true.lower()) 
#text_true.translate(str.maketrans('', '', string.punctuation)).lower()

from scipy.special import softmax
import numpy as np

index_map = {'neg': 0, 'neu': 1, 'pos': 2}
rev = { v:k for k,v in index_map.items()}


##### audio
# convert audio into feature
audio = audio2wave(file)
audio_X = prepare_data_librosa(audio,
                               features='logmel',
                               scaled=True)

stackedX_test = stacked_dataset(all_models, audio_X)
audio_pred = audio_logistic.predict(stackedX_test)  
print('Audio prediction:', [rev[item] for item in audio_pred])

##### text
text_pred = predict_text(bert_model,text)
print('Text prediction:', [rev[item] for item in np.array([np.argmax(text_pred)])])

##### ensemble
ensemble_text_test = softmax(text_pred)
ensemble_audio_test = audio_logistic.predict_proba(stackedX_test)

stack_test = np.dstack((ensemble_text_test, ensemble_audio_test))
stack_test = stack_test.reshape((stack_test.shape[0], stack_test.shape[1]*stack_test.shape[2]))

ensemble_pred = final_logistic.predict(stack_test)
print('Ensemble prediction:', [rev[item] for item in ensemble_pred])

def cate(emo):
    if emo == 'hap':
        i = 'pos'
    elif emo == 'neu':
        i = 'neu'
    else:
        i = 'neg'    
    return (i)

print('Correct Answer:', [cate(label_df['emotion'][ran])])

































