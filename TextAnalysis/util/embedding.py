# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:42:04 2020

@author: Pei-yuChen
"""
import numpy as np
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

def get_vocab_size(doc, tokenizer):
    vocab_size = len(tokenizer.word_index) + 1
    return (vocab_size)

def get_max_length(doc):  
# =============================================================================
#     if isinstance(doc[0], list): # if it's a list
#         all_len = []
#         for i in range(len(doc)):
#             for j in range(len(doc[i])):
#                 within_doc = []
#                 num_words = len(word_tokenize(doc[i][j]))
#                 within_doc.append(num_words)
#                 thismax = max(within_doc)
#             all_len.append(thismax)
#         max_length = max(all_len)
#         return (max_length)
# =============================================================================

    max_length = max([len(s.split()) for s in doc])
    return (max_length)

def get_embedding_matrix(embedding_path, vocab_size, dim, tokenizer):
    # load the whole embedding dictionary into memory
    embeddings_index = dict()
    f = open(embedding_path, encoding='utf-8')
    
    print('Generating GloVe embedding...')
    for line in tqdm(f):
    	values = line.split()
    	word = values[0]
    	coefs = np.asarray(values[1:], dtype='float32')
    	embeddings_index[word] = coefs
    f.close()
    
    embedding_matrix = np.zeros((vocab_size, dim)) # because 100d
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Finish loading GloVe embedding')
    return (embedding_matrix)  

def make_padded_doc(doc, max_length, tokenizer):
    # integer encode the documents
    encoded_doc = tokenizer.texts_to_sequences(doc)
    
    # pad to the same length
    padded_doc = pad_sequences(encoded_doc, maxlen=max_length, padding='post')    
    return (padded_doc)


EMBED_PATH = 'util\glove.6B.300d.txt'
def make_embedding_layer(train_X, max_length, tokenizer, trainable=False):
    
    vocab_size = get_vocab_size(train_X, tokenizer)    
    embedding_matrix = get_embedding_matrix(EMBED_PATH, vocab_size, 300, tokenizer)        
    
    e = Embedding(input_dim=vocab_size, output_dim=300, weights=[embedding_matrix], 
                  input_length=max_length, trainable=trainable)
    
    return(e)




