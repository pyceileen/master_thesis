# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:22:10 2020

@author: Pei-yuChen
"""

#%%
import os 
os.chdir(r"C:\Users\Pei-yuChen\Desktop\Programming\Model")

from util.read_dataset import load_data



DATA_PATH = 'reader.csv'

(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=False)

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.axes3d import Axes3D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#%%
# load the whole embedding dictionary into memory
embeddings_index = dict()
f = open('util\glove.6B.300d.txt', encoding='utf-8')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

#%%
make_token = Tokenizer()
make_token.fit_on_texts(train_X)

# define vocab size
vocab_size = len(make_token.word_index) + 1

# max length 
max_length = max([len(s.split()) for s in train_X])

# integer encode the documents
encoded_X_train = make_token.texts_to_sequences(train_X)

# pad to the same length
padded_X_train = pad_sequences(encoded_X_train, maxlen=max_length, padding='post')

#%%
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 300)) # because 300d
for word, i in make_token.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

#%% get average
x = []

for a in train_X:
    words = a.split(' ')
    vector = np.zeros((300))
    word_num = 0
    for b in words:
        if str(b).lower() in embeddings_index:
            vector += embeddings_index[str(b).lower()]
            word_num += 1
    if word_num > 0:
        vector = vector/word_num
    x.append(vector)    

x = np.asarray(x)

#%%        
embed_r = TSNE(n_components=2).fit_transform(x)     

#%%   
import matplotlib.pyplot as plt
plt.figure(1, figsize=(20, 30),)
plt.scatter(embed_r[:, 0], embed_r[:, 1],s=25, c=train_y, alpha=0.8, cmap='Purples')

plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
#%%
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

#%%
fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2],c=y_train['trinaryV'])
plt.show()














