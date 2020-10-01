# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:32:36 2020

@author: Eline
"""
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Bidirectional, GRU, TimeDistributed
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

def create_model(vocab_size,max_length,MAX_SENTS,embedding_matrix):
    l2_reg = regularizers.l2(1e-8)
    main_input  = Input(shape=(max_length,))  
    e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], 
                  input_length=max_length, trainable=False)(main_input)
    embedding = Dropout(0.25)(e)
    
    word_encoder = Bidirectional(GRU(50, 
                        return_sequences=True, kernel_regularizer=l2_reg
                        ))(embedding)
    drop1 = Dropout(0.25)(word_encoder)
    dense_transform_w = Dense(100, 
                              activation='relu', 
                              name='dense_transform_w', 
                              kernel_regularizer=l2_reg)(drop1)
    drop2 = Dropout(0.25)(dense_transform_w)
    attention_weighted_word = Model(
			main_input, Attention(name='word_attention', regularizer=l2_reg)(drop2))
    word_attention_model = attention_weighted_word
    #attention_weighted_word.summary()
    
    
    texts_in = Input(shape=(MAX_SENTS, max_length), dtype='int32') #max sentence 2
    attention_weighted_sentences = TimeDistributed(attention_weighted_word)(texts_in)
    sentence_encoder = Bidirectional(GRU(50, 
                                         return_sequences=True, kernel_regularizer=l2_reg,
                                         ))(attention_weighted_sentences)
    drop3 = Dropout(0.25)(sentence_encoder)
    dense_transform_s = Dense(100, 
                              activation='relu', 
                              name='dense_transform_s',
                              kernel_regularizer=l2_reg)(drop3)
    drop4 = Dropout(0.25)(dense_transform_s)
    attention_weighted_text = Attention(name='sentence_attention', regularizer=l2_reg)(drop4)
    
    output = Dense(1, 
                   activation='linear', name='output')(attention_weighted_text)
    
    model = Model(inputs=texts_in, outputs=output)
    
    model.compile(loss= 'mae',
                  optimizer=Adam(lr=1e-4),
                  metrics=['mae',pearson_r])

    return model