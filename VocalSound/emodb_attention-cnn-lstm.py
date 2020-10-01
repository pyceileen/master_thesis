# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:50:50 2020

@author: Pei-yuChen
"""


import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import pandas as pd
import numpy as np
import pickle


labels_df = pd.read_csv('data\EmoDB_label_df.csv')
audios = pickle.load(open('data\\EmoDB_audio_vectors.pkl', 'rb'))
audios = np.vstack(audios) # list of array to array

# normalization to xero mean and unit variance
from sklearn import preprocessing
audios_scaled = preprocessing.scale(audios, axis = 1)
print(np.mean((audios_scaled.mean(axis=1))))
print(np.mean((audios_scaled.std(axis=1))))

# only four emotions
options = ['happiness','neutral','anger','sadness']
labels_df_filtered = labels_df.loc[labels_df['emotions'].isin(options)]

filtered_inx = labels_df_filtered.index.values.astype(int)
audios_filtered = audios[filtered_inx]

#%%
import librosa

X = []
for i in range(len(audios_filtered)):
    mfcc = librosa.feature.mfcc(y=audios_filtered[i], sr=16000, n_mfcc=40)
    X.append(mfcc)
X = np.stack(X, axis=0)

y = labels_df_filtered['emotions']

#%%
X_reshape = []
for i in range(len(X)):
    #split into 2 time_steps
    split_one_data = np.hsplit(X[i],2)
    #flatten and reshape: (40,126) -> (2,40,63)
    # 2 = 2 time_steps
    flatten_one_data = np.concatenate(split_one_data).ravel().reshape(2,40,63) 
    X_reshape.append(flatten_one_data)
    
# flatten list of arrays and reshape    
X_reshape = np.concatenate(X_reshape).ravel().reshape(339,2,40,63) 

X_reshape = X_reshape.reshape(X_reshape.shape + (1,))  
#%%
# =============================================================================

# import random
# 
# # randomly decide val and test speakers
# val_test = random.sample(list(np.unique(labels_df_filtered['speakers'])), 2)
# val = random.choice(val_test) #which speaker is val
# test = list(set(val_test) - set([val]))[0] #which speaker is test
# train = np.setdiff1d(list(labels_df_filtered['speakers']),val_test)
# # yields the elements in `list_2` that are NOT in `list_1`
# 
# train_label_df = labels_df_filtered.loc[labels_df_filtered['speakers'].isin(train)]
# val_label_df = labels_df_filtered.loc[labels_df_filtered['speakers'] == val]
# test_label_df = labels_df_filtered.loc[labels_df_filtered['speakers'] == test]
# 
# 
# 
# train_inx = train_label_df.index.values.astype(int)
# val_inx = val_label_df.index.values.astype(int)
# test_inx = test_label_df.index.values.astype(int)
# 
# # get the index
# train_audios = audios[train_inx]
# val_audios = audios[val_inx]
# test_audios = audios[test_inx]
# 

#%%
# import librosa
# 
# 
# train_X = []
# for i in range(len(train_audios)):
#     mfcc = librosa.feature.mfcc(y=train_audios[i], sr=16000, n_mfcc=40)
#     train_X.append(mfcc)
# train_X = np.stack(train_X, axis=0)
# train_y = train_label_df['emotions']
# 
# val_X = []
# for i in range(len(val_audios)):
#     mfcc = librosa.feature.mfcc(y=val_audios[i], sr=16000, n_mfcc=40)
#     val_X.append(mfcc)
# val_X = np.stack(val_X, axis=0)
# val_y = val_label_df['emotions']
# 
# test_X = []
# for i in range(len(test_audios)):
#     mfcc = librosa.feature.mfcc(y=test_audios[i], sr=16000, n_mfcc=40)
#     test_X.append(mfcc)
# test_X = np.stack(test_X, axis=0)
# test_y = test_label_df['emotions']
# =============================================================================
#%%
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import backend as K

class Attention(Layer):
	def __init__(self, regularizer=None, **kwargs):
		super(Attention, self).__init__(**kwargs)
		self.regularizer = regularizer
		self.supports_masking = True

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.context = self.add_weight(name='context', 
									   shape=(input_shape[-1], 1),
									   initializer=initializers.RandomNormal(
									   		mean=0.0, stddev=0.05, seed=None),
									   regularizer=self.regularizer,
									   trainable=True)
		super(Attention, self).build(input_shape)

	def call(self, x, mask=None):
		attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
		attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1)

		if mask is not None:
			# use only the inputs specified by the mask
			# import pdb; pdb.set_trace()
			attention = attention*K.cast(mask, 'float32')

		weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
		return weighted_sum

	def compute_output_shape(self, input_shape):
		print(input_shape)
		return (input_shape[0], input_shape[-1])
    
	def get_config(self):
		config = super(Attention, self).get_config()
		return config


#%%
from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers

def create_model(input_shape):
    l2_reg = regularizers.l2(1e-3)
    model = Sequential() 
    model.add(TimeDistributed(Conv2D(128, (3, 5), activation='relu'),
                              input_shape=input_shape)) 
    model.add(TimeDistributed(MaxPooling2D(pool_size =(2, 2)))) 

    model.add(TimeDistributed(Conv2D(256, (3, 5), activation='relu'))) 
    model.add(TimeDistributed(Conv2D(256, (3, 5), activation='relu'))) 
    model.add(TimeDistributed(Conv2D(256, (3, 5), activation='relu'))) 
         
    model.add(TimeDistributed(Flatten())) 

    model.add(Bidirectional(LSTM(128, 
                                 return_sequences=True,
                                 kernel_regularizer=l2_reg)))

    model.add(Attention(regularizer=l2_reg))
    
    model.add(Dense(4, kernel_initializer='he_normal',
                   activation='softmax'))
    
    model.compile(loss= 'categorical_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['accuracy'])
    return model


#%%
from sklearn.model_selection import LeavePGroupsOut
from tensorflow.keras.utils import to_categorical
from util.miscellaneous import get_sample_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from itertools import islice 


N_GROUPS = 1
N_EPOCH = 1
N_TRAIN = 2
START_SPEAKER = 0
loop = []
for i in range(N_TRAIN):
    print('======== START TRAINING {} OUT OF {} TIMES ========'.format(i+1, N_TRAIN))

    lpgo = LeavePGroupsOut(n_groups=N_GROUPS) #Number of groups to leave out in the test split.
    groups = np.array(labels_df_filtered['speakers'])
    
    cvscores = []
    gen = lpgo.split(X_reshape, y, groups)
    for num, indices in islice(enumerate(gen,1), # index from 1 instead of 0
                               START_SPEAKER, None): # loop from 5th onwards
        train_idx = indices[0]
        val_idx = indices[1]
        
        print('   ===== Fitting CV {} out of {} ====='.format(num, lpgo.get_n_splits(groups=groups)))
        print("     TRAIN:", np.unique(groups[train_idx]))
        print("       VAL:", np.unique(groups[val_idx]))
    
        
        train_X = X_reshape[train_idx]
        train_y = y.iloc[train_idx]
        
        val_X = X_reshape[val_idx]
        val_y = y.iloc[val_idx]
      
        
        train_y = train_y.factorize(sort=True)[0]
        val_y = val_y.factorize(sort=True)[0]
        
        train_weight = get_sample_weight(train_y)    
        val_weight = get_sample_weight(val_y)
        
        input_shape = train_X.shape[1:]
        model = create_model(input_shape)
        history = model.fit(train_X, 
                            to_categorical(train_y), 
                            #sample_weight=train_weight,
                            batch_size=16,
                            epochs=N_EPOCH, verbose=1, 
                            shuffle=True
                            ) 
        y_pred = model.predict(val_X)
        y_pred = np.argmax(y_pred,axis=1)
    
        y_true = val_y
        
    
        cvscores.append(accuracy_score(y_true,y_pred))
        print("    VAL acc:", accuracy_score(y_true,y_pred))
        print('   ===== Finished Fitting CV {} out of {} ====='.format(num, lpgo.get_n_splits(groups=groups)))
        print('')    
    
    print('======== FINISH TRAINING {} OUT OF {} TIMES ========'.format(i+1,N_TRAIN))
    print('')
    
    loop.append(cvscores)
    
    print('目前結果（{}／{}）：'.format(i,N_TRAIN))
    for j in range(len(loop)):
        print('acc_{}: '.format(j), loop[j])
    print('')

#%%
import csv
with open('output\\EmoDB_.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(loop)    
#%%
# =============================================================================
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
# 
# datagen.fit(train_X)
# =============================================================================


#%%

saved_model = load_model('cnn-lstm.h5')

test_X = test_X.reshape(test_X.shape+(1,))


y_pred = saved_model.predict(test_X)
y_pred = np.argmax(y_pred,axis=1)


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

test_y = test_y.factorize(sort=True)[0]
y_true = test_y

print(accuracy_score(y_true,y_pred))
print(balanced_accuracy_score(y_true,y_pred))
print(classification_report(y_true, y_pred))



























