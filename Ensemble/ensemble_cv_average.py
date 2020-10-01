# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 09:51:48 2020

@author: Pei-yuChen
"""

import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)


import pickle
with open ('ensemble_list', 'rb') as fp:
    ensemble_list = pickle.load(fp)

# text data
import pandas as pd
import numpy as np
ensemble_IEMOCAP_df = pd.read_csv('ensemble_IEMOCAP_df.csv')
text_X = ensemble_IEMOCAP_df['transcriptions']

### audio data################################################################
import librosa
from tqdm import tqdm

ensemble_audio = []
for i in tqdm(ensemble_list):
    y, sr = librosa.load(i, sr=16000, duration=7.5) 
    ensemble_audio.append(y)

from util.prepare_data import prepare_data_librosa
FEATURE = 'logmel' #logmel or mfcc
scaled = True # True or False
    
audio_X = prepare_data_librosa(ensemble_audio,
                               features=FEATURE,
                               scaled=scaled)


def cate(array):
    new_array = []
    for i in array:
        if i == 'hap':
            i = 'pos'
        elif i == 'neu':
            i = 'neu'
        else:
            i = 'neg'
        new_array.append(i)        
    return (new_array)


audio_y = cate(ensemble_IEMOCAP_df['emotion'])
audio_y = np.array(audio_y)
##############################################################################

#%% load text model

# split for training and testing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from util.load_text import load_BERT, predict_text
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix    
from util.load_audio import load_all_models, stacked_dataset, fit_logistic
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

kf = KFold(n_splits=5, shuffle = True, random_state = 31)

#load text model and predict
bert_model = load_BERT(model_type='3_categories')

#load audio models
n_members = 10
all_models = load_all_models(n_members, model_type = '3_categories')
print('Loaded %d models!' % len(all_models))
print('')

for num, indices in enumerate(kf.split(audio_y)):
    
    print('===== Start CV {} out of 5 ====='.format(num+1))
    print('.')
    
    train_index = indices[0]
    test_index = indices[1]

    trainX_text, testX_text = text_X[train_index], text_X[test_index]
    trainy_text, testy_text = audio_y[train_index], audio_y[test_index]

    trainX_audio, testX_audio = audio_X[train_index], audio_X[test_index]
    trainy_audio, testy_audio = audio_y[train_index], audio_y[test_index]


    trainX_text = trainX_text.reset_index(drop=True)
    testX_text = testX_text.reset_index(drop=True)


    
    train_data = trainX_text
    test_data = testX_text
    
    text_results_train = []
    for i in range(len(train_data)): 
        result = predict_text(bert_model,train_data[i])
        text_results_train.append(result)
    
    text_results_test = []
    for i in range(len(test_data)): 
        result = predict_text(bert_model,test_data[i])
        text_results_test.append(result)
    
    #text results


    trainy_text_pred = np.argmax(text_results_train, axis=1) 
    testy_text_pred = np.argmax(text_results_test, axis=1) 
    
    index_map = {'neg': 0, 'neu': 1, 'pos': 2}
    rev = { v:k for k,v in index_map.items()}
    
    trainy_text_pred = [rev[item] for item in trainy_text_pred]
    testy_text_pred = [rev[item] for item in testy_text_pred]
    
    
    
    print('acc: ', accuracy_score(trainy_text, trainy_text_pred)) 

    print('acc: ', accuracy_score(testy_text, testy_text_pred)) 


    
    # fit audio ensemble
    trainy_audio = pd.Series(trainy_audio).factorize(sort=True)[0]
    
    audio_logistic = fit_logistic(all_models, trainX_audio, trainy_audio)


    # evaluate standalone models on test dataset

    
    for individual_model in all_models:
      _, acc = individual_model.evaluate(trainX_audio, 
                                         to_categorical(trainy_audio), verbose=0)
      #print('Single Model Accuracy (train): %.3f' % acc)
    
    stackedX_train = stacked_dataset(all_models, trainX_audio)
    trainy_audio_pred = audio_logistic.predict(stackedX_train)
    
    print('Stacked Model Accuracy (train): ', accuracy_score(trainy_audio, trainy_audio_pred)) 

    #############
    testy_audio = pd.Series(testy_audio).factorize(sort=True)[0]
    for individual_model in all_models:
      _, acc = individual_model.evaluate(testX_audio, 
                                         to_categorical(testy_audio), verbose=0)
      #print('Single Model Accuracy (test): %.3f' % acc)
      
    stackedX_test = stacked_dataset(all_models, testX_audio)
    testy_audio_pred = audio_logistic.predict(stackedX_test)  
    
    print('Stacked Model Accuracy (test): ', accuracy_score(testy_audio, testy_audio_pred)) 


    #stacking: 3 emotions + 3 emotions (treat audio models as one)

    #ensemble average (treat all audio models as one)
    
    from scipy.special import softmax
    
    ensemble_text = np.array(text_results_test)
    ensemble_text = softmax(ensemble_text, axis=1)
    
    ensemble_audio = audio_logistic.predict_proba(stackedX_test)
    
    summed = np.add(ensemble_text, ensemble_audio)
    ensemble_comb = np.argmax(summed, axis=1)
    
    print('acc: ', accuracy_score(testy_audio, ensemble_comb)) 




#%% ensemble handcraft

text_predict = np.argmax(ensemble_text, axis=1)
audio_predict = np.argmax(ensemble_audio, axis=1)

comb_predict = np.add(text_predict, audio_predict)

com = []

for index, i in enumerate(comb_predict):
    if i == 2:
        if text_predict[index] == 1 and audio_predict[index] ==1:
            ans = 'neu'
        else:
            ans = 'neu'       
    elif i >= 3:
        ans = 'pos'
    else:
        ans = 'neg'
    com.append(ans)
    

print('acc: ', accuracy_score(testy_text, com)) 
print(confusion_matrix(testy_text, com))















