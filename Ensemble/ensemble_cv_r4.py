# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 08:12:07 2020

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
N_EMOTIONS = 3 # 3 or 4
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

def split(array, cutoff1, cutoff2):
    new_array = []
    for i in array:
        if i < cutoff1:
            i = 'neg'
        elif i > cutoff2:
            i = 'pos'
        else:
            i = 'neu'
        new_array.append(i)
        
    return (new_array)


audio_y = cate(ensemble_IEMOCAP_df['emotion'])
audio_y = np.array(audio_y)

##############################################################################
#%%

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
from sklearn.preprocessing import StandardScaler

from util.load_text import load_BERT, predict_text

bert_model = load_BERT(model_type='regression')
    
from util.load_audio import load_all_models, stacked_dataset, fit_logistic
    
n_members = 10
all_models = load_all_models(n_members, model_type = '4_categories')

    
kf = KFold(n_splits=5, shuffle = True, random_state = 31)

test_text_acc = []
test_audio_acc = []
ensemble_acc = []

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
    
    
    
    #load text model and predict
    

    
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
    
    # text results
    # unlist
    text_results_train_cat = np.concatenate(text_results_train, axis=0)
    text_results_test_cat = np.concatenate(text_results_test, axis=0 )
    
    text_results_train_cat = split(text_results_train_cat, 2.86,3.59)
    text_results_test_cat = split(text_results_test_cat, 2.86,3.59)
    
    print('acc: ', accuracy_score(trainy_text, text_results_train_cat))

    print('acc: ', accuracy_score(testy_text, text_results_test_cat))
    

    
    # load audio models

   
    # fit audio ensemble
    trainy_audio = pd.Series(trainy_audio).factorize(sort=True)[0]
    
    audio_logistic = fit_logistic(all_models, trainX_audio, trainy_audio)
    
    #
    # evaluate standalone models on test dataset
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    stackedX_train = stacked_dataset(all_models, trainX_audio)
    trainy_audio_pred = audio_logistic.predict(stackedX_train)
    
    print('Stacked Model Accuracy (train): ', accuracy_score(trainy_audio, trainy_audio_pred)) 

    #############
    testy_audio = pd.Series(testy_audio).factorize(sort=True)[0]
      
    stackedX_test = stacked_dataset(all_models, testX_audio)
    testy_audio_pred = audio_logistic.predict(stackedX_test)  
    
    print('Stacked Model Accuracy (test): ', accuracy_score(testy_audio, testy_audio_pred)) 

    # stacking: regression + 4 emotions (treat audio models as one)
    from sklearn.linear_model import LogisticRegression
    
    # train
    ensemble_text_train = np.array(text_results_train)
    ensemble_audio_train = audio_logistic.predict_proba(stackedX_train)
    
    stack_train = np.concatenate((ensemble_text_train,ensemble_text_train,ensemble_text_train,
                                  ensemble_audio_train),axis=1)
        
    final_logistic = LogisticRegression()
    final_logistic.fit(stack_train, trainy_audio)
    
    # test
    ensemble_text_test = np.array(text_results_test)
    ensemble_audio_test = audio_logistic.predict_proba(stackedX_test)
    
    stack_test = np.concatenate((ensemble_text_test,ensemble_text_test,ensemble_text_test,
                                 ensemble_audio_test),axis=1)
    
    
    ensemble_test = final_logistic.predict(stack_test)
    
    print('Accuracy (test): ', accuracy_score(testy_audio, ensemble_test)) 
    ensemble_acc.append(accuracy_score(testy_audio, ensemble_test))
