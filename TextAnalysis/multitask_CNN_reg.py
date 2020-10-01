# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:41:01 2020

@author: Pei-yuChen
"""

from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

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
    
    conv1 = Conv1D(filters=128, kernel_size=2, 
                   padding='valid', kernel_regularizer=l2_reg)(embedding)
    drop1 = Dropout(0.25)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(drop1)
    conv2 = Conv1D(filters=128, kernel_size=3, 
                   padding='valid', kernel_regularizer=l2_reg)(pool1)
    drop2 = Dropout(0.25)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    flat1 = Flatten()(pool2)
    
    conv3 = Conv1D(filters=128, kernel_size=3, 
                   padding='valid', kernel_regularizer=l2_reg)(embedding)
    drop3 = Dropout(0.25)(conv3)
    pool3 = MaxPooling1D(pool_size=2)(drop3)
    conv4 = Conv1D(filters=128, kernel_size=3, 
                   padding='valid', kernel_regularizer=l2_reg)(pool3)
    drop4 = Dropout(0.25)(conv4)
    pool4 = MaxPooling1D(pool_size=2)(drop4)
    flat2 = Flatten()(pool4)
    
    conv5 = Conv1D(filters=128, kernel_size=4, 
                   padding='valid', kernel_regularizer=l2_reg)(embedding)
    drop5 = Dropout(0.25)(conv5)
    pool5 = MaxPooling1D(pool_size=2)(drop5)
    conv6 = Conv1D(filters=128, kernel_size=3, 
                   padding='valid', kernel_regularizer=l2_reg)(pool5)
    drop6 = Dropout(0.25)(conv6)
    pool6 = MaxPooling1D(pool_size=2)(drop6)
    flat3 = Flatten()(pool6)
    
    merged = concatenate([flat1, flat2, flat3], name='shared')
    
    denseV = Dense(32, activation='relu')(merged)
    dropV2 = Dropout(0.25)(denseV)
    denseA = Dense(32, activation='relu')(merged)
    dropA2 = Dropout(0.25)(denseA)
    denseD = Dense(32, activation='relu')(merged)
    dropD2 = Dropout(0.25)(denseD)
    
    outV = Dense(1, activation='linear', name='outputV')(dropV2)
    outA = Dense(1, activation='linear',name='outputA')(dropA2)
    outD = Dense(1, activation='linear',name='outputD')(dropD2)
    
    model = Model(inputs=main_input, outputs=[outV,outA,outD])
    
    model.compile(loss= 'mae',
                    optimizer=Adam(),
                    metrics=["mae", pearson_r])
    return model








