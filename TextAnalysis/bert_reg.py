# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:26:03 2020

@author: Pei-yuChen
"""

import tensorflow as tf
import tensorflow_hub as hub
import bert
from tensorflow.keras.models import  Model
from tqdm import tqdm
import numpy as np
from collections import namedtuple

print("TensorFlow Version:",tf.__version__)
print("Hub version: ",hub.__version__)

#%%
# BERT model is first initialized with the pre-trained parameters
# and all of the parameters are fine-tuned using labeled data from the downstream tasks.
bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)

#%%
def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))
 
def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))
 
def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def create_single_input(sentence,MAX_LEN):
  stokens = tokenizer.tokenize(sentence)
  stokens = stokens[:MAX_LEN]
  stokens = ["[CLS]"] + stokens + ["[SEP]"]
  ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
  masks = get_masks(stokens, MAX_SEQ_LEN)
  segments = get_segments(stokens, MAX_SEQ_LEN)
 
  return ids,masks,segments
 
def create_input_array(sentences):
  input_ids, input_masks, input_segments = [], [], []
  for sentence in tqdm(sentences,position=0, leave=True):
    ids,masks,segments=create_single_input(sentence,MAX_SEQ_LEN-2)
    input_ids.append(ids)
    input_masks.append(masks)
    input_segments.append(segments)
 
  return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

#%%
from util.read_dataset import load_data
from util.embedding import get_max_length
DATA_PATH = 'emobank.csv'
(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=False, multitask=False)

max_length = get_max_length(train_X)

MAX_SEQ_LEN=max_length
input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                       name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                   name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,
                                    name="segment_ids")

pooled_output, sequence_output = bert_layer([
    input_word_ids, input_mask, segment_ids])

FullTokenizer=bert.bert_tokenization.FullTokenizer
vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
tokenizer=FullTokenizer(vocab_file,do_lower_case)

dense1 = tf.keras.layers.Dense(256, activation='relu')(sequence_output)
drop1 = tf.keras.layers.Dropout(0.25)(dense1)
dense2 = tf.keras.layers.Dense(128, activation='relu')(drop1)
drop2 = tf.keras.layers.Dropout(0.25)(dense2)
out = tf.keras.layers.Dense(1, activation="linear", name="dense_output")(drop2)
 
model = tf.keras.models.Model(
      inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

from keras import backend as K
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

model.compile(loss='mae',
                  optimizer='adam',
                  metrics=['mae', pearson_r])
#%%

inputs_train=create_input_array(train_X)
inputs_val=create_input_array(val_X)
inputs_test=create_input_array(test_X)

history = model.fit(inputs_train, train_y,
          validation_data = (inputs_val, val_y),
          epochs=8,batch_size=32,
          shuffle=True)
#%%
y_pred = model.predict(inputs_test)
y_true = test_y

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y_true, y_pred)
mean_squared_error(y_true, y_pred)
mean_squared_error(y_true, y_pred, squared=False)
pearsonr(y_true, y_pred)[0]







