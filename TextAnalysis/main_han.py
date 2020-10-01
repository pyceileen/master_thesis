# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:14:55 2020

@author: Pei-yuChen
"""
import os 
PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

import numpy as np
import tensorflow as tf
#np.random.seed(1237)
#tf.random.set_seed(1237)

from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from util.read_dataset import load_data
from util.embedding import get_embedding_matrix
#from util.miscellaneous import draw_regression

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv
from importlib import import_module

DATA_PATH = '/data/emobank.csv'
EMBED_PATH = '/util/glove.6B.300d.txt'

(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data(
    DATA_PATH, ternary=False, sent_tok=True, multitask=False)



MAX_SENTS=max([len(train_X[i]) for i in range(len(train_X))])

from nltk.tokenize import word_tokenize
max_len = []
for i in range(len(train_X)):
    for j in range(len(train_X[i])):
        within_doc = []        
        sen_len = len(word_tokenize(train_X[i][j]))
        within_doc.append(sen_len)
        thismax = max(within_doc)
    max_len.append(thismax)
MAX_SENT_LENGTH=max(max_len)

join_text_train=[]
for i in range(len(train_X)):
    text = ' '.join(train_X[i])
    join_text_train.append(text)

    
make_token = Tokenizer()
make_token.fit_on_texts(join_text_train)

MAX_NB_WORDS = len(make_token.word_index) + 1

max_length = MAX_SENT_LENGTH
vocab_size = len(make_token.word_index) + 1
embedding_weights = get_embedding_matrix(EMBED_PATH, vocab_size, 300, make_token)




def process_raw_text(raw_text, max_sents=MAX_SENTS, vocab_size=MAX_NB_WORDS,
                     max_sent_len=MAX_SENT_LENGTH, tokenizer=make_token):
    join_text=[]
    for i in range(len(raw_text)):
        text = ' '.join(raw_text[i])
        join_text.append(text)
        
    '[# of reviews each batch, # of sentences, # of words in each sentence].'
    df = np.zeros((len(join_text), max_sents, max_sent_len), dtype='int32')

    for i, sentences in enumerate(raw_text):
        for j, sent in enumerate(sentences):
            if j < max_sents:
                wordTokens = text_to_word_sequence(sent)
                k = 0
                for _, word in enumerate(wordTokens):
                    try:                
                        if k < max_sent_len and tokenizer.word_index[word] < vocab_size:
                            df[i, j, k] = tokenizer.word_index[word]
                            k = k + 1
                    except KeyError:
                        df[i, j, k] = 0
                        k = k + 1
    return(df)
            
def batch_generator(X, y, batch_size):
    padded_X = process_raw_text(X)
    
    X_shape = padded_X[0].shape # input shape
    y_shape = y[0].shape # input shape
    
    X_size = (batch_size,) + X_shape
    y_size = (batch_size,) + y_shape
    
    batch_X = np.zeros(X_size)
    batch_y = np.zeros(y_size)
    while True:
        for i in range(batch_size):
            # choose random index in features
            index= np.random.choice(len(padded_X),1)
            batch_X[i] = padded_X[index]
            batch_y[i] = y[index]
    
        yield (batch_X, batch_y)
	
	
	
	
padded_train_X = process_raw_text(train_X)
padded_val_X = process_raw_text(val_X)
padded_test_X = process_raw_text(test_X)

y_true = test_y  	
	
	
	
experiments = [#'han_reg',
               #'regionalLSTM_reg',
               'regionalCNN-LSTM_reg',
              ]
for i in experiments:
    WHICH_MODEL = i
    print('')
    print('===== Start {0} model... ====='.format(WHICH_MODEL))

    module = import_module(WHICH_MODEL)
   
    manual_mae_wo = []
    manual_mse_wo = []
    manual_rmse_wo = []
    manual_corr_wo = []
    
    for j in range(8):
        print('')
        print('=== Fitting {0} model the {1} time... ==='.format(WHICH_MODEL, j))
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                           patience=15)
        mc = ModelCheckpoint('/content/gdrive/My Drive/TextAnalysis/best_'+WHICH_MODEL+'.h5', 
                             monitor='val_mae', mode='min', verbose=0, save_best_only=True)
        
        model = module.create_model(vocab_size,max_length,MAX_SENTS,embedding_weights)
        history = model.fit(padded_train_X, train_y,
                            validation_data = (padded_val_X, val_y),
                            batch_size=32,
                            epochs=200, verbose=1, 
                            callbacks=[es, mc], shuffle=True
                            )
        saved_model = load_model('/content/gdrive/My Drive/TextAnalysis/best_'+WHICH_MODEL+'.h5', 
                                 custom_objects={#'Attention': module.Attention,
                                                 'pearson_r': module.pearson_r})
       
        y_pred = saved_model.predict(process_raw_text(test_X))
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
        #draw_regression(y_true, y_pred, var_name)
        
        print('mae: {0}'.format(manual_mae_wo))
        print('mse: {0}'.format(manual_mse_wo))
        print('rmse: {0}'.format(manual_rmse_wo))
        print('corr: {0}'.format(manual_corr_wo))
        print('===== Finished {0} model the {1} time.'.format(WHICH_MODEL, j))
        print('')


    #reshape = [list(x) for x in zip(*model_evaluate)]
    #model_evaluate_loss, model_evaluate_mae, model_evaluate_corr = reshape
    lst = [manual_mae_wo,manual_mse_wo,manual_rmse_wo,manual_corr_wo]
    names = ["manual_mae_wo","manual_mse_wo",
             "manual_rmse_wo","manual_corr_wo"]
    export_data = zip(*lst)
    
    filename = 'output/{0}.csv'.format(WHICH_MODEL)
    with open(filename, 'w', newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerow(names)
          wr.writerows(export_data)
    myfile.close()
    print('')
    print('===== End {0} model ====='.format(WHICH_MODEL))

########## end han model ########## 

