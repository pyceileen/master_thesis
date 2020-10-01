# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 08:07:59 2020

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
model.load_state_dict(torch.load('bert_final.pt',map_location=device))

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

label_df = pd.read_csv('data\label_df_IEMOCAP_4emo.csv')

# filter out inconsistent
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
def binary(array):
    new_array = []
    for i in array:
        if i == 'hap':
            i = 'pos'
        elif i == 'neu':
            i = 'neu'
        else:
            i = 'neg'
        new_array.append(i)        
    return (new_array)

label_df['cat_V'] = split(label_df['V'], 2.2, 3.9)
label_df['cat_emotion'] = binary(label_df['emotion'])

df_filtered = label_df.loc[label_df['cat_V']==label_df['cat_emotion']]
df_filtered = df_filtered.reset_index()
#%%
test = df_filtered['transcriptions']

results = []
for i in range(len(test)):    
    text = test[i]
    
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
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

print('corr: ', pearsonr(label_df['V'], results)[0])
print('mae: ', mean_absolute_error(label_df['V'], results))


#%%

results_cat = split(results, 2.3, 3.5)
true_cat = df_filtered['cat_emotion']
#true_cat = split(true_cat, 2.5, 3)


from sklearn.metrics import accuracy_score, confusion_matrix
print('acc: ', accuracy_score(true_cat, results_cat))
confusion_matrix(true_cat, results_cat)











