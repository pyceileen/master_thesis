# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:15:48 2020

@author: Pei-yuChen
"""
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D
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



def create_model(max_length):    
    main_input  = Input(shape=(max_length,))
      
    dense1 = Dense(128, activation='relu')(main_input)
    drop1 = Dropout(0.25)(dense1)
    
    dense2 = Dense(100, activation='relu')(drop1)
    drop2 = Dropout(0.25)(dense2)
    
    dense3V = Dense(64, activation='relu')(drop2)
    drop3V = Dropout(0.25)(dense3V)
    dense3A = Dense(64, activation='relu')(drop2)
    drop3A = Dropout(0.25)(dense3A)
    dense3D = Dense(64, activation='relu')(drop2)
    drop3D = Dropout(0.25)(dense3D)

    dense4V = Dense(32, activation='relu')(drop3V)
    drop4V = Dropout(0.25)(dense4V)
    dense4A = Dense(32, activation='relu')(drop3A)
    drop4A = Dropout(0.25)(dense4A)
    dense4D = Dense(32, activation='relu')(drop3D)
    drop4D = Dropout(0.25)(dense4D)

    outV = Dense(1, activation='sigmoid', name='outputV')(drop4V)
    outA = Dense(1, activation='sigmoid',name='outputA')(drop4A)
    outD = Dense(1, activation='sigmoid',name='outputD')(drop4D)

    
    model = Model(inputs=main_input, outputs=[outV,outA,outD])
    model.compile(loss= 'mae',
                  optimizer=Adam(lr=1e-4),
                  metrics=['mae',pearson_r])

    return model   