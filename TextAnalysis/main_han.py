# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:14:55 2020

@author: Pei-yuChen
"""
import os 
os.chdir(r"C:\Users\Pei-yuChen\Desktop\Programming\Model\TextAnalysis")

import numpy as np
import tensorflow as tf
np.random.seed(1237)
tf.random.set_seed(1237)

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from util.read_dataset import load_data
from util.embedding import make_embedding_layer
from util.miscellaneous import draw_regression

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import csv
from importlib import import_module

DATA_PATH = 'emobank.csv'
EMBED_PATH = 'util\glove.6B.300d.txt'

(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=True)

#%%
##### for han model #####
MAX_SENTS=max([len(train_X[i]) for i in range(len(train_X))])

from nltk.tokenize import word_tokenize
max_len = []
for i in range(len(train_X)):
    for j in range(len(train_X[i])):
        within_doc = []        
        sen_len = len(word_tokenize(train_X[i][j]))
        within_doc.append(sen_len)
        thismax = max(within_doc)
    max_len.append(thismax)
MAX_SENT_LENGTH=max(max_len)

join_text_train=[]
for i in range(len(train_X)):
    text = ' '.join(train_X[i])
    join_text_train.append(text)

    
make_token = Tokenizer()
make_token.fit_on_texts(join_text_train)

MAX_NB_WORDS = len(make_token.word_index) + 1

'''
reshape for HAN model
 [# of reviews each batch, # of sentences, # of words in each sentence].
'''
df_train = np.zeros((len(join_text_train), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
for i, sentences in enumerate(train_X):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and make_token.word_index[word] < MAX_NB_WORDS:
                    df_train[i, j, k] = make_token.word_index[word]
                    k = k + 1

join_text_val=[]
for i in range(len(val_X)):
    text = ' '.join(val_X[i])
    join_text_val.append(text)
df_val = np.zeros((len(join_text_val), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
for i, sentences in enumerate(val_X):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                try:                
                    if k < MAX_SENT_LENGTH and make_token.word_index[word] < MAX_NB_WORDS:
                        df_val[i, j, k] = make_token.word_index[word]
                        k = k + 1
                except KeyError:
                    df_val[i, j, k] = 0
                    k = k + 1

join_text_test=[]
for i in range(len(test_X)):
    text = ' '.join(test_X[i])
    join_text_test.append(text)
df_test = np.zeros((len(join_text_test), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
for i, sentences in enumerate(test_X):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                try:                
                    if k < MAX_SENT_LENGTH and make_token.word_index[word] < MAX_NB_WORDS:
                        df_test[i, j, k] = make_token.word_index[word]
                        k = k + 1
                except KeyError:
                    df_test[i, j, k] = 0
                    k = k + 1
                    
                    
max_length = MAX_SENT_LENGTH
embedding_layer = make_embedding_layer(train_X, max_length, make_token, trainable=False)

y_true = test_y

#%%
experiments = ['regionalLSTM_reg']
for i in experiments:
    WHICH_MODEL = i
    print('Start {0} model...'.format(WHICH_MODEL))

    module = import_module(WHICH_MODEL)
   
    manual_mae_wo = []
    manual_mse_wo = []
    manual_rmse_wo = []
    manual_corr_wo = []
    
    for j in range(10):
        print('Fitting {0} model the {1} time...'.format(WHICH_MODEL, j))
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                           min_delta=0.001, patience=10)
        mc = ModelCheckpoint('best_'+WHICH_MODEL+'.h5', 
                             monitor='val_mean_absolute_error', mode='min', verbose=1, save_best_only=True)
        
        model = module.create_model(max_length,MAX_SENTS,embedding_layer)
        history = model.fit(df_train, train_y,
                            validation_data=(df_val, val_y),
                            batch_size=32, epochs=200, verbose=1, 
                            callbacks=[es, mc], shuffle=True,
                            )
        saved_model = load_model('best_'+WHICH_MODEL+'.h5', custom_objects={#'Attention': module.Attention,
                                                                        'pearson_r': module.pearson_r})
        
        #test_loss = saved_model.evaluate(df_test, test_y, sample_weight = sample_weight_test)
        #model_evaluate.append(test_loss)
        
        y_pred = saved_model.predict(df_test)
        y_pred = np.round(y_pred, 3).flatten()
        
        mae_wo = mean_absolute_error(y_true, y_pred)         
        manual_mae_wo.append(mae_wo)
        
        mse_wo = mean_squared_error(y_true, y_pred)
        manual_mse_wo.append(mse_wo)
        
        rmse_wo = mean_squared_error(y_true, y_pred, squared=False)
        manual_rmse_wo.append(rmse_wo)
        
        corr_wo = pearsonr(y_true, y_pred)[0]
        manual_corr_wo.append(corr_wo)
        
        var_name = 'best_'+WHICH_MODEL+'_'+str(j)
        draw_regression(y_true, y_pred, var_name)
        
        print('Finished fitting the model the {0} time.'.format(j))

    #reshape = [list(x) for x in zip(*model_evaluate)]
    #model_evaluate_loss, model_evaluate_mae, model_evaluate_corr = reshape
    lst = [manual_mae_wo,manual_mse_wo,manual_rmse_wo,manual_corr_wo]
    names = ["manual_mae_wo","manual_mse_wo",
             "manual_rmse_wo","manual_corr_wo"]
    export_data = zip(*lst)
    
    filename = "output\\{0}.csv".format(WHICH_MODEL)
    with open(filename, 'w', newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerow(names)
          wr.writerows(export_data)
    myfile.close()
    print('End {0} model'.format(WHICH_MODEL))

########## end han model ########## 



#%%
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







#%%

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









