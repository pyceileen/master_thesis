# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:27:26 2020

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

# select emotions
N_EMOTIONS = 4
if N_EMOTIONS ==4:
    options = ['anger','happiness','neutral','sadness']
else:
    options = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral','sadness']

labels_df_filtered = labels_df.loc[labels_df['emotions'].isin(options)]

filtered_inx = labels_df_filtered.index.values.astype(int)
audios_filtered = audios_scaled[filtered_inx]

#%%
import librosa

# logMel
X1 = []
for i in range(len(audios_filtered)):
    mel = librosa.feature.mfcc(y=audios_filtered[i], sr=16000, n_mels=40,n_mfcc=40, 
                                         n_fft=512, win_length=int(0.025*16000), hop_length=int(0.01*16000),
                                         fmin=300, fmax=8000                                      
                                         )
    #logmel= librosa.power_to_db(mel,ref=np.max)
    X1.append(mel)
X1 = np.stack(X1, axis=0)

# delete the last column: 401 -> 400
#X1 = np.delete(X1, X1.shape[2]-1, axis=2) #delete(array, which_to_delete, axis)


# logMel delta
X2 = []
for i in range(len(X1)):
    delta = librosa.feature.delta(X1[i], order=2)
    X2.append(delta)
X2 = np.stack(X2, axis=0)

# logMel delta delta
X3 = []
for i in range(len(X2)):
    delta = librosa.feature.delta(X2[i], order=2)
    X3.append(delta)
X3 = np.stack(X3, axis=0)


y = labels_df_filtered['emotions']


#%%
N_STEP = 2
N_DATA = X1.shape[0]
N_COEF = X1.shape[1] # number of coefficients (40)
N_FRAMES = X1.shape[2] # 126


X1_reshape = []
for i in range(len(X1)): # loop through each data entry
    # split into time_steps
    split_one_data = np.hsplit(X1[i],N_STEP)
    
    # flatten and reshape: (40,126) -> (time_steps,40,126//time_steps)
    # 40 MFCC
    flatten = np.concatenate(split_one_data).ravel()
    reshape_one_data = flatten.reshape(N_STEP,N_COEF,N_FRAMES//N_STEP)
    X1_reshape.append(reshape_one_data)    
# flatten list of arrays and reshape    
X1_reshape = np.concatenate(X1_reshape).ravel()
X1_reshape = X1_reshape.reshape(N_DATA,N_STEP,N_COEF,N_FRAMES//N_STEP) 

X2_reshape = []
for i in range(len(X2)): 
    split_one_data = np.hsplit(X2[i],N_STEP)
    flatten = np.concatenate(split_one_data).ravel()
    reshape_one_data = flatten.reshape(N_STEP,N_COEF,N_FRAMES//N_STEP)
    X2_reshape.append(reshape_one_data)      
X2_reshape = np.concatenate(X2_reshape).ravel()
X2_reshape = X2_reshape.reshape(N_DATA,N_STEP,N_COEF,N_FRAMES//N_STEP) 

X3_reshape = []
for i in range(len(X3)):  
    split_one_data = np.hsplit(X3[i],N_STEP)
    flatten = np.concatenate(split_one_data).ravel()
    reshape_one_data = flatten.reshape(N_STEP,N_COEF,N_FRAMES//N_STEP)
    X3_reshape.append(reshape_one_data)      
X3_reshape = np.concatenate(X3_reshape).ravel()
X3_reshape = X3_reshape.reshape(N_DATA,N_STEP,N_COEF,N_FRAMES//N_STEP) 
#%%
# combine features
N_FEATURES = 3
X_all = []
for i in range(len(X1_reshape)): # loop through each data entry
    one_dat = []
    for j in range(len(X1_reshape[i])): #aka range(N_STEP)
        X1_inx = X1_reshape[i][j]
        X2_inx = X2_reshape[i][j]
        X3_inx = X3_reshape[i][j]
        X1X2X3_inx = np.dstack((X1_inx, X2_inx, X3_inx))
        one_dat.append(X1X2X3_inx)
    
    one_data =  np.concatenate(one_dat).ravel()
    reshape_one_data = one_data.reshape(N_STEP,N_COEF,N_FRAMES//N_STEP,N_FEATURES)
    X_all.append(reshape_one_data)
    
X_all_reshape = np.concatenate(X_all).ravel()
X_all_reshape = X_all_reshape.reshape(N_DATA,N_STEP,N_COEF,N_FRAMES//N_STEP,N_FEATURES) 

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
from tensorflow.keras.layers import LeakyReLU


def create_model(input_shape):
    l2_reg = regularizers.l2(1e-3)
    model = Sequential() 
    model.add(TimeDistributed(Conv2D(128, (3, 3)),
                              input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 4)))) 

    model.add(TimeDistributed(Conv2D(128, (3, 3))))
    model.add(LeakyReLU(alpha=0.1))
    model.add(TimeDistributed(Conv2D(128, (3, 3))))
    model.add(LeakyReLU(alpha=0.1))

         
    model.add(TimeDistributed(Flatten())) 

    model.add(Bidirectional(LSTM(128, 
                                 return_sequences=True,
                                 kernel_regularizer=l2_reg)))

    model.add(Attention(regularizer=l2_reg))
    
    model.add(Dense(N_EMOTIONS, kernel_initializer='he_normal',
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



























