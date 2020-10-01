# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 13:53:07 2020

@author: Pei-yuChen
"""

#%% not tested yet
##### han vis #####
from util.preprocessing import text_preprocessing
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model

def encode_texts(texts):
	encoded_texts = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH))
	for i, text in enumerate(texts):
		encoded_text = np.array(pad_sequences(
			make_token.texts_to_sequences(text), 
			maxlen=MAX_SENT_LENGTH))[:MAX_SENTS]
		encoded_texts[i][-len(encoded_text):] = encoded_text
	return encoded_texts

def encode_input(x, log=False):
	x = np.array(x)
	if not x.shape:
		x = np.expand_dims(x, 0)
	texts = np.array([text_preprocessing(text) for text in x])
	return encode_texts(texts)

text = "Today is a great day! Because I got a job offer."
normalized_text = text_preprocessing(text)
encoded_text = encode_input(text)[0]


# get word activations
word_attention_model = model.get_layer('time_distributed_1').layer
hidden_word_encoding_out = Model(inputs=word_attention_model.input,
                                 outputs=word_attention_model.get_layer('dense_transform_w').output)
hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
word_context = word_attention_model.get_layer('word_attention').get_weights()[0]
u_wattention = encoded_text*np.exp(np.squeeze(np.dot(hidden_word_encodings, word_context)))

reverse_word_index = {value:key for key,value in make_token.word_index.items()}
    
# generate word, activation pairs
nopad_encoded_text = encoded_text[-len(normalized_text):]
nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
reconstructed_texts = [[reverse_word_index[int(i)] 
                        for i in sentence] for sentence in nopad_encoded_text]
nopad_wattention = u_wattention[-len(normalized_text):]
nopad_wattention = nopad_wattention/np.expand_dims(np.sum(nopad_wattention, -1), -1)
nopad_wattention = np.array([attention_seq[-len(sentence):] 
                             for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])

word_activation_maps = []
for i, text in enumerate(reconstructed_texts):
			word_activation_maps.append(list(zip(text, nopad_wattention[i])))

# get sentence activations
hidden_sentence_encoding_out = Model(inputs=model.input,
                                     outputs=model.get_layer('dense_transform_s').output)
hidden_sentence_encodings = np.squeeze(
    hidden_sentence_encoding_out.predict(np.expand_dims(encoded_text, 0)), 0)
sentence_context = model.get_layer('sentence_attention').get_weights()[0]
u_sattention = np.exp(np.squeeze(np.dot(hidden_sentence_encodings, sentence_context), -1))

nopad_sattention = u_sattention[-len(normalized_text):]

nopad_sattention = nopad_sattention/np.expand_dims(np.sum(nopad_sattention, -1), -1)

activation_map = list(zip(word_activation_maps, nopad_sattention))	

#抽句子
doc_word = []
for i in range(len(activation_map)):
    sent_word = []
    for j in range(len(activation_map[i][0])):
        word = activation_map[i][0][j][0]
        sent_word.append(word)
    doc_word.append(sent_word)

#抽重量
doc_weight = []
for i in range(len(activation_map)):
    word_weight = []
    for j in range(len(activation_map[i][0])):
        word = activation_map[i][0][j][1]
        word_weight.append(word)
    doc_weight.append(word_weight)

#抽句重
sent_weight = []
for i in range(len(activation_map[0])):
    word = activation_map[i][1]
    sent_weight.append(word)

x = np.array(doc_weight)
reshape_weight = []
for i in range(len(x)):
    a = np.array(x[i]).reshape(1, len(x[i]))
    reshape_weight.append(a)

w = np.array(doc_word)
reshape_word = []
for i in range(len(w)):
    a = np.array(w[i]).reshape(1, len(w[i]))
    reshape_word.append(a)

import seaborn as sns
import matplotlib.pyplot as plt
#plt.figure(figsize=(5, 0.1))


nrows = len(reshape_weight)
fig, axes = plt.subplots(nrows=nrows, figsize=(10, 1.5))    
for i in range(len(reshape_weight)):
    sns.heatmap(reshape_weight[i], ax=axes[i], cbar=False, 
            cmap="YlGnBu", annot=reshape_word[i], fmt = '',
            xticklabels=False, yticklabels=False)
fig.colorbar(axes[0].collections[0], ax=axes, location="right", aspect=20)
plt.show()

##### end han vis #####







#%% not tested yet

##### attention vis #####
from util.preprocessing import text_preprocessing
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from nltk.tokenize import word_tokenize

def encode_texts(texts):
	encoded_texts = np.zeros((len(texts), 1, max_length))
	for i, text in enumerate(texts):
		encoded_text = np.array(pad_sequences(
			make_token.texts_to_sequences(text), 
			maxlen=max_length))[:1]
		encoded_texts[i][-len(encoded_text):] = encoded_text
	return encoded_texts

def encode_input(x, log=False):
	texts = np.array([[text_preprocessing(x)]])
	return encode_texts(texts)

text = "After all the hard work, she finally achieved her goal"
normalized_text = text_preprocessing(text)
encoded_text = encode_input(text)[0]


# get word activations

hidden_word_encoding_out = Model(inputs=model.input,
                                 outputs=model.get_layer('dense_transform_w').output)
hidden_word_encodings = hidden_word_encoding_out.predict(encoded_text)
word_context = model.get_layer('word_attention').get_weights()[0]
u_wattention = encoded_text*np.exp(np.squeeze(np.dot(hidden_word_encodings, word_context)))

reverse_word_index = {value:key for key,value in make_token.word_index.items()}
    
# generate word, activation pairs
nopad_encoded_text = encoded_text[-len(normalized_text):]
nopad_encoded_text = [list(filter(lambda x: x > 0, sentence)) for sentence in nopad_encoded_text]
reconstructed_texts = [[reverse_word_index[int(i)] 
                        for i in sentence] for sentence in nopad_encoded_text]
nopad_wattention = u_wattention[-len(normalized_text):]
nopad_wattention = nopad_wattention/np.expand_dims(np.sum(nopad_wattention, -1), -1)
nopad_wattention = np.array([attention_seq[-len(sentence):] 
                             for attention_seq, sentence in zip(nopad_wattention, nopad_encoded_text)])

word_activation_maps = []
for i, text in enumerate(reconstructed_texts):
			word_activation_maps.append(list(zip(text, nopad_wattention[i])))

#抽句子
doc_word = []
for i in range(len(word_activation_maps[0])):
    word = word_activation_maps[0][i][0]
    doc_word.append(word)

#抽重量
word_weight = []
for i in range(len(word_activation_maps[0])):
    weight = word_activation_maps[0][i][1]
    word_weight.append(weight)

x = np.array(word_weight)
word_weight=x.reshape(1,len(x))

x = np.array(doc_word)
doc_word=x.reshape(1,len(x))


import seaborn as sns
import matplotlib.pyplot as plt
#plt.figure(figsize=(5, 0.1))


fig, ax = plt.subplots(figsize=(10, 1))    
sns.heatmap(word_weight, ax=ax, cbar=True, 
            cmap="YlGnBu", annot=doc_word, fmt = '',
            xticklabels=False, yticklabels=False,
            cbar_kws={"orientation": "horizontal"},
            cbar_ax=fig.add_axes([-1, 0.3, .5, 0.3]))
plt.show()

##### end attention vis #####

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

# Define two rows for subplots
# Use axes divider to put cbar on top
# plot heatmap without colorbar
fig, ax = plt.subplots(figsize=(10,1.5))    
ax = sns.heatmap(word_weight, ax=ax, cbar=False, 
            cmap="YlGnBu", annot=doc_word, fmt = '',
            xticklabels=False, yticklabels=False)
# split axes of heatmap to put colorbar
ax_divider = make_axes_locatable(ax)
# define size and padding of axes for colorbar
cax = ax_divider.append_axes('bottom', size = '70%', pad=0.1)
# make colorbar for heatmap. 
# Heatmap returns an axes obj but you need to get a mappable obj (get_children)
colorbar(ax.get_children()[0], cax = cax, orientation = 'horizontal')
# locate colorbar ticks
cax.xaxis.set_ticks_position('bottom')

plt.show()







