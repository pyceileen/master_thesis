# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:36:31 2020

@author: Pei-yuChen
"""

from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, initializers

from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Bidirectional, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class Attention(Layer):
	def __init__(self, regularizer=None, **kwargs):
		super(Attention, self).__init__(**kwargs)
		self.regularizer = regularizer
		self.supports_masking = True

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.context = self.add_weight(name='context', 
									   shape=(input_shape[-1], 1),
									   initializer=initializers.RandomNormal(
									   		mean=0.0, stddev=0.05, seed=None),
									   regularizer=self.regularizer,
									   trainable=True)
		super(Attention, self).build(input_shape)

	def call(self, x, mask=None):
		attention_in = K.exp(K.squeeze(K.dot(x, self.context), axis=-1))
		attention = attention_in/K.expand_dims(K.sum(attention_in, axis=-1), -1)

		if mask is not None:
			# use only the inputs specified by the mask
			# import pdb; pdb.set_trace()
			attention = attention*K.cast(mask, 'float32')

		weighted_sum = K.batch_dot(K.permute_dimensions(x, [0, 2, 1]), attention)
		return weighted_sum

	def compute_output_shape(self, input_shape):
		print(input_shape)
		return (input_shape[0], input_shape[-1])
    
	def get_config(self):
		config = super(Attention, self).get_config()
		return config

    
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
                        return_sequences=True, kernel_regularizer=l2_reg,
                        ))(drop1)
    drop2 = Dropout(0.25)(bi2)
  
 
    dense1 = Dense(64, 
			activation='relu')(drop2)
    drop3 = Dropout(0.25)(dense1)     

    attention_weighted_word = Attention(regularizer=l2_reg)(drop3)

    output = Dense(1, #kernel_initializer='he_normal',
                   activation='linear', name='output')(attention_weighted_word)
    
    model = Model(inputs=main_input, outputs=output)
    
    model.compile(loss='mae',
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['mae', pearson_r])
    print(model.summary())
    return model
