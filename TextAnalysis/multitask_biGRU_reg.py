# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:01:21 2020

@author: Pei-yuChen
"""

from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return (r)

def create_model(vocab_size,max_length,embedding_matrix):
    l2_reg = regularizers.l2(1e-3)
    main_input  = Input(shape=(max_length,))  
    e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix],
                  input_length=max_length, trainable=False)(main_input)
    embedding = Dropout(0.25)(e)

    bi1 = Bidirectional(GRU(128, return_sequences=True,
                        kernel_regularizer=l2_reg))(embedding)
    drop1 = Dropout(0.25)(bi1)
    
    bi2 = Bidirectional(GRU(128,
                        kernel_regularizer=l2_reg), name='shared')(drop1)
    drop2 = Dropout(0.25)(bi2)
    
    denseV = Dense(32, activation='relu')(drop2)
    dropV2 = Dropout(0.25)(denseV)
    denseA = Dense(32, activation='relu')(drop2)
    dropA2 = Dropout(0.25)(denseA)
    denseD = Dense(32, activation='relu')(drop2)
    dropD2 = Dropout(0.25)(denseD)
    
    outV = Dense(1, activation='sigmoid', name='outputV')(dropV2)
    outA = Dense(1, activation='sigmoid',name='outputA')(dropA2)
    outD = Dense(1, activation='sigmoid',name='outputD')(dropD2)
    
    model = Model(inputs=main_input, outputs=[outV,outA,outD])
    model.compile(loss= 'mae',
                  optimizer=Adam(lr=1e-4),
                  metrics=['mae',pearson_r])

    return model   

