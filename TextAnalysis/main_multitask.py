# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:07:07 2020

@author: Pei-yuChen
"""

import os 
os.chdir(r"C:\Users\Pei-yuChen\Desktop\Programming\Model\TextAnalysis")

import numpy as np
import tensorflow as tf
np.random.seed(1237)
tf.random.set_seed(1237)

from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from util.read_dataset import load_data
from util.embedding import get_max_length, make_embedding_layer, make_padded_doc
from util.miscellaneous import draw_regression

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv
from importlib import import_module

DATA_PATH = 'emobank.csv'
EMBED_PATH = 'util\glove.6B.300d.txt'


(train_X, train_yV,train_yA,train_yD), (val_X, val_yV,val_yA,val_yD), (test_X, test_yV,test_yA,test_yD) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=False, multitask=True)
#%%
from sklearn.preprocessing import MinMaxScaler

# reshape 1d arrays to 2d arrays
train_yV = train_yV.reshape(len(train_yV), 1)
val_yV = val_yV.reshape(len(val_yV), 1)
test_yV = test_yV.reshape(len(test_yV), 1)

train_yA = train_yA.reshape(len(train_yA), 1)
val_yA = val_yA.reshape(len(val_yA), 1)
test_yA = test_yA.reshape(len(test_yA), 1)

train_yD = train_yD.reshape(len(train_yD), 1)
val_yD = val_yD.reshape(len(val_yD), 1)
test_yD = test_yD.reshape(len(test_yD), 1)

scaler = MinMaxScaler()

train_yV = scaler.fit_transform(train_yV)
val_yV = scaler.transform(val_yV)
test_yV = scaler.transform(test_yV)

train_yA = scaler.fit_transform(train_yA)
val_yA = scaler.transform(val_yA)
test_yA = scaler.transform(test_yA)

train_yD = scaler.fit_transform(train_yD)
val_yD = scaler.transform(val_yD)
test_yD = scaler.transform(test_yD)
#%%
##### for regression #####
make_token = Tokenizer()
make_token.fit_on_texts(train_X)

max_length = get_max_length(train_X)
embedding_layer = make_embedding_layer(train_X, max_length, make_token, trainable=False)

padded_train_X = make_padded_doc(train_X, max_length, make_token)
padded_val_X = make_padded_doc(val_X, max_length, make_token)
padded_test_X = make_padded_doc(test_X, max_length, make_token)

#%%
WHICH_MODEL = 'multitask_biLSTM_reg'
module = import_module(WHICH_MODEL)

model = module.create_model(max_length,embedding_layer)
history = model.fit(padded_train_X, [train_yV,train_yA,train_yD],
                    validation_data=(padded_val_X, [val_yV,val_yA,val_yD]),
                    batch_size=64, epochs=40, verbose=1, 
                    shuffle=True
                    )
#%%
y_trueV = test_yV.flatten()
y_trueA = test_yA.flatten()
y_trueD = test_yD.flatten()

y_predV, y_predA, y_predD = model.predict(padded_test_X)
y_predV = np.round(y_predV, 3).flatten()
y_predA = np.round(y_predA, 3).flatten()
y_predD = np.round(y_predD, 3).flatten()

mae_woV = mean_absolute_error(y_trueV, y_predV)
mae_woA = mean_absolute_error(y_trueA, y_predA)
mae_woD = mean_absolute_error(y_trueD, y_predD)

mse_woV = mean_squared_error(y_trueV, y_predV)
mse_woA = mean_squared_error(y_trueA, y_predA)
mse_woD = mean_squared_error(y_trueA, y_predA)

rmse_woV = mean_squared_error(y_trueV, y_predV, squared=False)
rmse_woA = mean_squared_error(y_trueA, y_predA, squared=False)
rmse_woD = mean_squared_error(y_trueD, y_predD, squared=False)

corr_woV = pearsonr(y_trueV, y_predV)[0]
corr_woA = pearsonr(y_trueA, y_predA)[0]
corr_woD = pearsonr(y_trueD, y_predD)[0]

#%% ensemble
from keras.models import Model

def get_intermediate_output(X, model, layer_name):
    inter_modelL = Model(inputs=model.input,
                         outputs = model.get_layer(layer_name).output)
    inter_outputL = inter_modelL.predict(X)
    return(inter_outputL)

trainC = get_intermediate_output(padded_train_X, modelC, "shared")
trainL = get_intermediate_output(padded_train_X, modelL, "shared")
trainG = get_intermediate_output(padded_train_X, modelG, "shared")

valC = get_intermediate_output(padded_val_X, valC, "shared")
valL = get_intermediate_output(padded_val_X, valL, "shared")
valG = get_intermediate_output(padded_val_X, valG, "shared")



























