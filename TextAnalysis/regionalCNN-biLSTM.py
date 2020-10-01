# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:04:01 2020

@author: Pei-yuChen
"""

from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
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


def create_model(vocab_size,max_length,MAX_SENTS,embedding_matrix):
    l2_reg = regularizers.l2(1e-3)
    main_input  = Input(shape=(max_length,))  
    e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], 
                  input_length=max_length, trainable=False)(main_input)
    embedding = Dropout(0.25)(e)

    
    word_encoder = Conv1D(filters=128, kernel_size=3,
                          padding='valid', activation='relu', strides=1)(embedding)
    dropcnn = Dropout(0.25)(word_encoder)
    l_maxpooling = MaxPooling1D(pool_size=2)(dropcnn)
    word_encoder2 = Conv1D(filters=128, kernel_size=3,
                          padding='valid', activation='relu', strides=1)(l_maxpooling)
    dropcnn2 = Dropout(0.25)(word_encoder2)
    l_maxpooling2 = MaxPooling1D(pool_size=2)(dropcnn2)

    l_cnn = Flatten()(l_maxpooling2)
    drop1 = Dropout(0.25)(l_cnn)
    sentEncoder = Model(main_input, drop1)
    
    
    texts_in = Input(shape=(MAX_SENTS, max_length), dtype='int32') 
    text_encoder = TimeDistributed(sentEncoder)(texts_in)
    l_lstm_sent = Bidirectional(LSTM(128, return_sequences = True,
                                     kernel_regularizer=l2_reg,))(text_encoder)
    drop1 = Dropout(0.25)(l_lstm_sent)

    lstm2 = Bidirectional(LSTM(64,
                                kernel_regularizer=l2_reg,))(drop1)
    drop2 = Dropout(0.25)(lstm2)

    dense1 = Dense(64, activation='relu')(drop2)
    drop3 = Dropout(0.25)(dense1)

    output = Dense(1, activation='linear', name='output')(drop3)    
    model = Model(inputs=texts_in, outputs=output)
    
    model.compile(loss= 'mae',
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['mae', pearson_r])

    print(model.summary())
    return model