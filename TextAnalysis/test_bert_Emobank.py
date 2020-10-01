# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:55:43 2020

@author: Pei-yuChen
"""

import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)


#%%
import torch
from transformers import BertForSequenceClassification

PRETRAINED_MODEL_NAME = "bert-base-uncased" 
NUM_LABELS = 1


model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)
model.load_state_dict(torch.load('models\checkpoint_train.pt',map_location=device))

#%%
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)        

def gettensors(text):
    text_a = text
        
    word_pieces = ["[CLS]"]
    tokens_a = tokenizer.tokenize(text_a)
    word_pieces += tokens_a + ["[SEP]"]
    len_a = len(word_pieces)


    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    tokens_tensor = torch.tensor(ids)
    segments_tensor = torch.tensor([1] * len_a, 
                                    dtype=torch.long)
                                    
    tokens_tensor = tokens_tensor.unsqueeze(0)
    segments_tensor = segments_tensor.unsqueeze(0)
    masks_tensor = torch.zeros(tokens_tensor.shape, 
                                dtype=torch.long)
    masks_tensor = masks_tensor.masked_fill(
        tokens_tensor != 0, 1)
       
    return (tokens_tensor, segments_tensor, masks_tensor)

#%% 
import pandas as pd

label_df = pd.read_csv('data\emobank.csv')
df_filtered = label_df.loc[label_df['split']=='test'].reset_index()


data = df_filtered['text']
label = df_filtered['V']

#%%
results = []
for i in range(len(data)):    
    text = data[i]
    
    if next(model.parameters()).is_cuda:
        tokens_tensors, \
            segments_tensors, \
                masks_tensors = [t.to("cuda:0") for t in gettensors(text) if t is not None]
    else:
        tokens_tensors, segments_tensors, masks_tensors = gettensors(text)
    
    outputs = model(input_ids=tokens_tensors, 
                    token_type_ids=segments_tensors, 
                    attention_mask=masks_tensors
                    )
    results.append(outputs[0].detach().numpy().flatten())

results = [l.tolist() for l in results]
results = [item for sublist in results for item in sublist]

#%%
def split(array, cutoff1, cutoff2):
    new_array = []
    for i in array:
        if i < cutoff1:
            i = 'neg'
        elif i > cutoff2:
            i = 'pos'
        else:
            i = 'neu'
        new_array.append(i)
        
    return (new_array)


results_cat = split(results, 2.3,3.7)
label_cat = split(label, 2.3,3.7)

from sklearn.metrics import accuracy_score
print(accuracy_score(label_cat, results_cat))














