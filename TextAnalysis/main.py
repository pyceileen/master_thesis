# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:45:15 2020

@author: Pei-yuChen
"""
import os 
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from util.read_dataset import load_data
from util.embedding import get_max_length, make_padded_doc, get_embedding_matrix

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv
from importlib import import_module


DATA_PATH = 'data\emobank.csv'
EMBED_PATH = 'util\glove.6B.300d.txt'


(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=False)


#%%

make_token = Tokenizer()
make_token.fit_on_texts(train_X)

max_length = get_max_length(train_X)
vocab_size = len(make_token.word_index) + 1
embedding_weights = get_embedding_matrix(EMBED_PATH, vocab_size, 300, make_token)

padded_train_X = make_padded_doc(train_X, max_length, make_token)
padded_val_X = make_padded_doc(val_X, max_length, make_token)
padded_test_X = make_padded_doc(test_X, max_length, make_token)


y_true = test_y      
#%%

experiments = ['biGRU', 'biLSTM',
               'CNN-biLSTM',
               'biGRU-attention','biLSTM-attention',
               ]
N_TRAIN = 10
for i in experiments:
    WHICH_MODEL = i

    module = import_module(WHICH_MODEL)

    manual_mae_wo = []
    manual_mse_wo = []
    manual_rmse_wo = []
    manual_corr_wo = []  
    
    for j in range(N_TRAIN):
        print('')
        print('===== Fitting {0} model the {1} time ====='.format(WHICH_MODEL, j+1))
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                           patience=15)
        mc = ModelCheckpoint('models\best_'+WHICH_MODEL+'.h5', 
                             monitor='val_mae', mode='min', verbose=1, save_best_only=True)
        
        model = module.create_model(vocab_size,max_length,embedding_weights)
        
        history = model.fit(padded_train_X,train_y,
                            validation_data=(padded_val_X,val_y),
                            epochs=200, verbose=1, batch_size=64,
                            callbacks=[es, mc], shuffle=True
                            )
        saved_model = load_model('models\best_'+WHICH_MODEL+'.h5', custom_objects={'Attention': module.Attention,
                                                                            'pearson_r': module.pearson_r,
                                                                            })
        
        
        y_pred = saved_model.predict(padded_test_X)
        y_pred = np.round(y_pred, 3).flatten()
        
        mae_wo = mean_absolute_error(y_true, y_pred)         
        manual_mae_wo.append(mae_wo)
        
        mse_wo = mean_squared_error(y_true, y_pred)
        manual_mse_wo.append(mse_wo)
        
        rmse_wo = mean_squared_error(y_true, y_pred, squared=False)
        manual_rmse_wo.append(rmse_wo)
        
        corr_wo = pearsonr(y_true, y_pred)[0]
        manual_corr_wo.append(corr_wo)
        
        print('mae: {0}'.format(manual_mae_wo))
        print('mse: {0}'.format(manual_mse_wo))
        print('rmse: {0}'.format(manual_rmse_wo))
        print('corr: {0}'.format(manual_corr_wo))
        print('===== Finished {0} model the {1} time.'.format(WHICH_MODEL, j+1))
        print('')
        

    lst = [manual_mae_wo, manual_mse_wo, manual_rmse_wo, manual_corr_wo]
    names = ["manual_mae_wo","manual_mse_wo",
             "manual_rmse_wo","manual_corr_wo"]
    export_data = zip(*lst)
    
    filename = 'output/{0}.csv'.format(WHICH_MODEL)
    with open(filename, 'w', newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerow(names)
          wr.writerows(export_data)
    myfile.close()
    print('End {0} model'.format(WHICH_MODEL))
##### end regression #####

