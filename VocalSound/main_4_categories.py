# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:02:11 2020

@author: Pei-yuChen
"""
import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

labels_df_db = pd.read_csv('data/label_df_EmoDB_4emo.csv')
# select columns
labels_df_db = labels_df_db[['speaker', 'emotion']]
audios_db = pickle.load(open('data/audio_EmoDB_4emo_7.5s.pkl', 'rb'))


labels_df_ie = pd.read_csv('data/label_df_IEMOCAP_4emo.csv')
# select columns
labels_df_ie = labels_df_ie[['speaker', 'emotion']]
audios_ie = pickle.load(open('data/audio_IEMOCAP_4emo_7.5s.pkl', 'rb'))

labels_df_rv = pd.read_csv('data/label_df_RAVDESS_4emo.csv')
# select columns
labels_df_rv = labels_df_rv[['speaker', 'emotion']]
audios_rv = pickle.load(open('data/audio_RAVDESS_4emo_7.5s.pkl', 'rb'))

#split iemocap for ensemble
train_audios_ie, ensemble_audios_ie, train_labels_df_ie, ensemble_labels_df_ie = train_test_split(
    audios_ie, labels_df_ie, test_size=0.15, shuffle=True, random_state = 6)


labels_df = pd.concat([labels_df_db,train_labels_df_ie,labels_df_rv], ignore_index=True)
audios = audios_db+train_audios_ie+audios_rv

print(len(labels_df))
labels_df

#%%

import numpy as np
from util.prepare_data import prepare_data_librosa

FEATURE = 'logmel' #logmel or mfcc
scaled = True # True or False
    
audio_X = prepare_data_librosa(audios,
                               features=FEATURE,
                               scaled=scaled)

ensemble_X = prepare_data_librosa(ensemble_audios_ie,
                               features=FEATURE,
                               scaled=scaled)

y = labels_df['emotion']

ensemble_y = ensemble_labels_df_ie['emotion']
ensemble_y = ensemble_y.factorize(sort=True)[0]

print(len(audio_X))
print(len(ensemble_X))

#%%

def fit_model(input_shape, N_EMOTIONS, train_X, train_y, val_X, val_y, N_EPOCH, es, mc):
	# define model
  model = module.create_model(input_shape, N_EMOTIONS)
  model.fit(train_X, 
            to_categorical(train_y), 
            validation_data=(val_X, to_categorical(val_y)),
            batch_size=32,
            epochs=N_EPOCH, verbose=1, 
            shuffle=True,
            callbacks=[es, mc]
            ) 
  return(model)

#%%

from sklearn.model_selection import LeavePGroupsOut, StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from util.miscellaneous import get_sample_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from itertools import islice 
from importlib import import_module
from sklearn.metrics import f1_score,precision_score,recall_score



N_EPOCH = 200
N_TRAIN = 10
N_EMOTIONS = 4

acc = []
recall = []
precision = []
f1 = []
module = import_module('ACRNN2D')
for i in range(N_TRAIN):
    print('======== START TRAINING {} OUT OF {} TIMES ========'.format(i+1, N_TRAIN))
    
    # save a proportion for ensemble    

    train_X, val_X, train_y, val_y = train_test_split(
        audio_X, y, test_size=0.2, shuffle=True, random_state = i+1) # 0.9 x 0.2 = 0.18

    train_y = train_y.factorize(sort=True)[0]
    val_y = val_y.factorize(sort=True)[0]
  
                
    input_shape = train_X.shape[1:]

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint('/content/gdrive/My Drive/VocalSound/models/audio_final_4cat_' + str(i + 1) + '.h5', 
                        monitor='val_acc', mode='max', 
                        verbose=1, save_best_only=True)
                         
    # fit model
    model = fit_model(input_shape, N_EMOTIONS, train_X, train_y, val_X, val_y, N_EPOCH, es, mc)
    

    # save model
    filename = '/content/gdrive/My Drive/VocalSound/models/audio_final_4cat_' + str(i + 1) + '.h5'
    print('')
    print('>>> %s Saved!' % filename)
    print('')

    saved_model = load_model('/content/gdrive/My Drive/VocalSound/models/audio_final_4cat_' + str(i + 1) + '.h5', 
                             custom_objects={'Attention':module.Attention}
                             )

    
    y_pred = saved_model.predict(ensemble_X)
    y_pred = np.argmax(y_pred,axis=1)

    y_true = ensemble_y
    
    
    acc.append(accuracy_score(y_true,y_pred))
    recall.append(recall_score(y_true,y_pred, average='macro'))
    precision.append(precision_score(y_true,y_pred, average='macro'))
    f1.append(f1_score(y_true,y_pred, average='macro'))
    print("    TEST acc:", accuracy_score(y_true,y_pred))
    
    print('======== FINISH TRAINING {} OUT OF {} TIMES ========'.format(i+1,N_TRAIN))
    print('')
    
    
    print('目前結果（{}／{}）：'.format(i+1,N_TRAIN))
    for j in range(len(acc)):
        print('acc_{}: '.format(j+1), acc[j])
    for j in range(len(recall)):
        print('recall_{}: '.format(j+1), recall[j])
    for j in range(len(precision)):
        print('precision_{}: '.format(j+1), precision[j])
    for j in range(len(f1)):
        print('f1_{}: '.format(j+1), f1[j])                        
    print('')

















