# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:07:07 2020

@author: Pei-yuChen
"""

import os 
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

import numpy as np
import tensorflow as tf
#np.random.seed(1237)
#tf.random.set_seed(1237)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from util.read_dataset import load_data
from util.embedding import get_max_length, make_embedding_layer 
from util.embedding import make_padded_doc, get_embedding_matrix
from util.miscellaneous import draw_regression

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv
from importlib import import_module

DATA_PATH = '/data/emobank.csv'
EMBED_PATH = '/util/glove.6B.300d.txt'


(train_X, train_yV,train_yA,train_yD), (val_X, val_yV,val_yA,val_yD), (test_X, test_yV,test_yA,test_yD) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=False, multitask=True)




from sklearn.preprocessing import MinMaxScaler

# reshape 1d arrays to 2d arrays
train_yV_trans = train_yV.reshape(len(train_yV), 1)
val_yV_trans = val_yV.reshape(len(val_yV), 1)
test_yV_trans = test_yV.reshape(len(test_yV), 1)

train_yA_trans = train_yA.reshape(len(train_yA), 1)
val_yA_trans = val_yA.reshape(len(val_yA), 1)
test_yA_trans = test_yA.reshape(len(test_yA), 1)

train_yD_trans = train_yD.reshape(len(train_yD), 1)
val_yD_trans = val_yD.reshape(len(val_yD), 1)
test_yD_trans = test_yD.reshape(len(test_yD), 1)

scalerV = MinMaxScaler()
train_yV_trans = scalerV.fit_transform(train_yV_trans)
val_yV_trans = scalerV.transform(val_yV_trans)
test_yV_trans = scalerV.transform(test_yV_trans)

scalerA = MinMaxScaler()
train_yA_trans = scalerA.fit_transform(train_yA_trans)
val_yA_trans = scalerA.transform(val_yA_trans)
test_yA_trans = scalerA.transform(test_yA_trans)

scalerD = MinMaxScaler()
train_yD_trans = scalerD.fit_transform(train_yD_trans)
val_yD_trans = scalerD.transform(val_yD_trans)
test_yD_trans = scalerD.transform(test_yD_trans)





make_token = Tokenizer()
make_token.fit_on_texts(train_X)

max_length = get_max_length(train_X)
vocab_size = len(make_token.word_index) +1
embedding_weights = get_embedding_matrix(EMBED_PATH, vocab_size, 300, make_token)

padded_train_X = make_padded_doc(train_X, max_length, make_token)
padded_val_X = make_padded_doc(val_X, max_length, make_token)
padded_test_X = make_padded_doc(test_X, max_length, make_token)




from tensorflow.keras.models import Model

def get_intermediate_output(X, model, layer_name):
    inter_modelL = Model(inputs=model.input,
                         outputs = model.get_layer(layer_name).output)
    inter_outputL = inter_modelL.predict(X)
    return(inter_outputL)




y_trueV = test_yV

experiments = ['multitask_biLSTM_reg',
               'multitask_CNN_reg',
               'multitask_biGRU_reg']

manual_mae_wo = []
manual_mse_wo = []
manual_rmse_wo = []
manual_corr_wo = []  
for i in range(1):
    
    for j in experiments:
        WHICH_MODEL = j
        print('')
        print('===== Fitting {0} model the {1} time ====='.format(WHICH_MODEL, i))
        
                
        #with tf.device('/gpu:0'):
        module = import_module(WHICH_MODEL)
        model = module.create_model(vocab_size,max_length,embedding_weights)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, 
                   patience=15)
        mc = ModelCheckpoint('/content/gdrive/My Drive/TextAnalysis/best_'+WHICH_MODEL+'.h5', 
                     monitor='val_outputV_mae', mode='min', verbose=0, save_best_only=True)
        
        history = model.fit(padded_train_X, [train_yV_trans,train_yA_trans,train_yD_trans],
                    validation_data=(padded_val_X, [val_yV_trans,val_yA_trans,val_yD_trans]),
                    batch_size=32, epochs=200, verbose=1, 
                    callbacks=[es, mc], shuffle=True
                    )
       
        saved_model = load_model('/content/gdrive/My Drive/TextAnalysis/best_'+WHICH_MODEL+'.h5',
                                 custom_objects={'pearson_r': module.pearson_r})
        globals()['train_inter%s' %WHICH_MODEL] = get_intermediate_output(padded_train_X, saved_model, "shared")        
        globals()['val_inter%s' %WHICH_MODEL] = get_intermediate_output(padded_val_X, saved_model, "shared")
        globals()['test_inter%s' %WHICH_MODEL] = get_intermediate_output(padded_test_X, saved_model, "shared")
        
        print('===== Finished {0} model the {1} time ====='.format(WHICH_MODEL, i))
        print('')
    # ensemble
    print('')
    print('===== Fitting ensemble model the {0} time ====='.format(i))
    merged_train = np.concatenate((train_intermultitask_biLSTM_reg,
                                   train_intermultitask_CNN_reg,
                                   train_intermultitask_biGRU_reg), axis=1)
    merged_val = np.concatenate((val_intermultitask_biLSTM_reg,
                                   val_intermultitask_CNN_reg,
                                   val_intermultitask_biGRU_reg), axis=1)
    merged_test = np.concatenate((test_intermultitask_biLSTM_reg,
                               test_intermultitask_CNN_reg,
                               test_intermultitask_biGRU_reg), axis=1)
        
    module = import_module("multitask_ensemble")
    ensemble_model = module.create_model(max_length=len(merged_train[0]))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
               patience=15)
    mc = ModelCheckpoint('/content/gdrive/My Drive/TextAnalysis/best_ensemble_model.h5', 
                 monitor='val_outputV_mae', mode='min', verbose=0, save_best_only=True)
    history = ensemble_model.fit(merged_train, [train_yV_trans,train_yA_trans,train_yD_trans],
                validation_data=(merged_val, [val_yV_trans,val_yA_trans,val_yD_trans]),
                batch_size=32, epochs=200, verbose=1, 
                callbacks=[es, mc], shuffle=True
                )
    saved_ensemble_model = load_model('/content/gdrive/My Drive/TextAnalysis/best_ensemble_model.h5',
                                      custom_objects={'pearson_r': module.pearson_r})

    y_predV, y_predA, y_predD = saved_ensemble_model.predict(merged_test)
    
    # inverse transform
    y_predV = scalerV.inverse_transform(y_predV)
    y_predV = np.round(y_predV, 3).flatten()
    
    mae_wo = mean_absolute_error(y_trueV, y_predV)
    manual_mae_wo.append(mae_wo)
    
    mse_wo = mean_squared_error(y_trueV, y_predV)
    manual_mse_wo.append(mse_wo)
    
    rmse_wo = mean_squared_error(y_trueV, y_predV, squared=False)
    manual_rmse_wo.append(rmse_wo)
    
    corr_wo = pearsonr(y_trueV, y_predV)[0]
    manual_corr_wo.append(corr_wo)
    
    var_name = 'best_ensemble_model'+str(i)
    #draw_regression(y_trueV, y_predV, var_name)
    print('mae: {0}'.format(manual_mae_wo))
    print('mse: {0}'.format(manual_mse_wo))
    print('rmse: {0}'.format(manual_rmse_wo))
    print('corr: {0}'.format(manual_corr_wo))
    print('===== Finished ensemble model the {0} time ====='.format(i))
    print('')
    
lst = [manual_mae_wo, manual_mse_wo, manual_rmse_wo, manual_corr_wo]
names = ["manual_mae_wo","manual_mse_wo",
         "manual_rmse_wo","manual_corr_wo"]
export_data = zip(*lst)

filename = '/output/ensemble.csv'
with open(filename, 'w', newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(names)
      wr.writerows(export_data)
myfile.close()








