# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:38:06 2020

@author: Pei-yuChen
"""

from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D
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
     
    conv1 = Conv1D(filters=128, kernel_size=3,
                    padding='valid', activation='relu', strides=1)(embedding)
    dropcnn = Dropout(0.25)(conv1)
    pool1 = MaxPooling1D(pool_size=2)(dropcnn)

    conv2 = Conv1D(filters=128, kernel_size=3,
                  padding='valid', activation='relu', strides=1)(pool1)
    dropcnn2 = Dropout(0.25)(conv2)
    pool2 = MaxPooling1D(pool_size=2)(dropcnn2)


    lstm1 = Bidirectional(LSTM(128,
                               kernel_regularizer=l2_reg,
                               return_sequences=True))(pool2)
    drop1 = Dropout(0.25)(lstm1)

    lstm2 = Bidirectional(LSTM(64,
                               kernel_regularizer=l2_reg,
                               return_sequences=False))(drop1)
    drop2 = Dropout(0.25)(lstm2)
    
    dense1 = Dense(64, activation='relu')(drop2)
    drop3 = Dropout(0.25)(dense1)     
    
    output = Dense(1,
                   activation='linear', name='output')(drop3)
    
    model = Model(inputs=main_input, outputs=output)
    model.compile(loss= 'mae',
                  optimizer=Adam(lr=1e-4),
                  metrics=['mae', pearson_r])
    print(model.summary())
    return model