# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 18:05:09 2020

@author: Pei-yuChen
"""

from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers, initializers
from tensorflow.keras import backend as K

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
    
    
    
    
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Reshape, Permute
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU, BatchNormalization



def create_model(input_shape, N_EMOTIONS):
    
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    l2_reg = regularizers.l2(1e-3)
    model = Sequential() 
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))
    
    model.add(Conv2D(filters=128, 
                     kernel_size=(3,3),
                     strides=(1,1),
                     padding='valid',
                     input_shape=input_shape))
    model.add(LeakyReLU()) 
    model.add(MaxPooling2D(pool_size=(2,4),
                           strides=(2,4),
                           padding='valid')) 
    
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), dilation_rate=2))
    model.add(LeakyReLU())
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), dilation_rate=2))
    model.add(LeakyReLU())
    model.add(BatchNormalization()) 

    model.add(Conv2D(256, (3, 3), dilation_rate=2))
    model.add(LeakyReLU())
    model.add(BatchNormalization())  

    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))
    model.add(LeakyReLU())

    model.add(Bidirectional(LSTM(128, 
                                 return_sequences=True,
                                 kernel_regularizer=l2_reg)))

    model.add(Attention(regularizer=l2_reg))
    
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Dense(N_EMOTIONS, #kernel_initializer='he_normal',
                   activation='softmax'))

    model.compile(loss= 'categorical_crossentropy',
                  optimizer=Adam(lr=1e-4),
                  metrics=['acc'])
    print(model.summary())
    return model    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    