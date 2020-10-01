# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:23:14 2020

@author: Pei-yuChen
"""

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Input, Embedding, TimeDistributed
from tensorflow.keras.layers import Layer, RepeatVector
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

    
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
    
    word_encoder = Bidirectional(GRU(128, 
                        return_sequences=True, kernel_regularizer=l2_reg,
                        ))(embedding)
    drop1 = Dropout(0.25)(word_encoder)

    bi2 = Bidirectional(GRU(64, 
                        return_sequences=False, kernel_regularizer=l2_reg,
                        ))(drop1)
    drop2 = Dropout(0.25)(bi2)
  
 
    dense1 = Dense(64, 
			activation='relu')(drop2)
    drop3 = Dropout(0.25)(dense1)     

    output = Dense(1, #kernel_initializer='he_normal',
                   activation='linear', name='output')(drop3)
    
    model = Model(inputs=main_input, outputs=output)
    
    model.compile(loss='mae',
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['mae', pearson_r])
    print(model.summary())
    return model
