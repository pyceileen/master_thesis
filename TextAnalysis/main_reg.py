# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:45:15 2020

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


(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=False)


#%%
##### for regression #####
make_token = Tokenizer()
make_token.fit_on_texts(train_X)

max_length = get_max_length(train_X)
embedding_layer = make_embedding_layer(train_X, max_length, make_token, trainable=False)

padded_train_X = make_padded_doc(train_X, max_length, make_token)
padded_val_X = make_padded_doc(val_X, max_length, make_token)
padded_test_X = make_padded_doc(test_X, max_length, make_token)


y_true = test_y

#%%
experiments = ['biGRU_reg']
for i in experiments:
    WHICH_MODEL = i
    print('Start {0} model...'.format(WHICH_MODEL))

    module = import_module(WHICH_MODEL)

    manual_mae_wo = []
    manual_mse_wo = []
    manual_rmse_wo = []
    manual_corr_wo = []  
    
    for j in range(10):
        print('Fitting {0} model the {1} time...'.format(WHICH_MODEL, j))
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                           min_delta=0.001, patience=10)
        mc = ModelCheckpoint('best_'+WHICH_MODEL+'.h5', 
                             monitor='val_mae', mode='min', verbose=1, save_best_only=True)
        
        model = module.create_model(max_length,embedding_layer)
        history = model.fit(padded_train_X, train_y,
                            validation_data=(padded_val_X, val_y),
                            batch_size=32, epochs=200, verbose=1, 
                            callbacks=[es, mc], shuffle=True
                            )
        saved_model = load_model('best_'+WHICH_MODEL+'.h5', custom_objects={#'Attention': module.Attention,
                                                                        'pearson_r': module.pearson_r})
        
        #test_loss = saved_model.evaluate(padded_test_X, test_y, sample_weight = sample_weight_test)
        #model_evaluate.append(test_loss)
        
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
        
        var_name = 'best_'+WHICH_MODEL+'_'+str(j)
        draw_regression(y_true, y_pred, var_name)
        print('Finished fitting the model the {0} time.'.format(j))

    #reshape = [list(x) for x in zip(*model_evaluate)]
    #model_evaluate_loss, model_evaluate_mae, model_evaluate_corr = reshape
    lst = [manual_mae_wo, manual_mse_wo, manual_rmse_wo, manual_corr_wo]
    names = ["manual_mae_wo","manual_mse_wo",
             "manual_rmse_wo","manual_corr_wo"]
    export_data = zip(*lst)
    
    filename = "output\\{0}.csv".format(WHICH_MODEL)
    with open(filename, 'w', newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerow(names)
          wr.writerows(export_data)
    myfile.close()
    print('End {0} model'.format(WHICH_MODEL))
##### end regression #####