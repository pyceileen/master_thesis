# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 15:28:30 2020

@author: Pei-yuChen
"""
import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import torch

#%%
'''
Preprocess dataset
'''
import pandas as pd
from util.read_dataset import load_data

DATA_PATH = 'data/emobank.csv'
(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=False)

df_train = pd.DataFrame(data=(train_X), columns=["text_a"])
df_train['label'] = train_y
df_val = pd.DataFrame(data=(val_X), columns=["text_a"])
df_val['label'] = val_y
df_test = pd.DataFrame(data=(test_X), columns=["text_a"])
df_test['label'] = test_y

df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
print("training data:", len(df_all))

df_all.to_csv("data/df_all_cat.tsv", sep="\t", index=False)

#%%

'''
convert data so that bert can use it

- tokens_tensor： including [CLS] and [SEP]
- segments_tensor： sentence boundary
- label_tensor
'''
from torch.utils.data import Dataset    
class EmotionDataset(Dataset):
    
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "val", "test"]  
        self.mode = mode
        
        self.df = pd.read_csv("data/df_all_cat.tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.tokenizer = tokenizer  
    
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, = self.df.loc[idx, ['text_a']].values
            label_tensor = None

        else:
            text_a, label = self.df.loc[idx, ['text_a','label']].values
            label_tensor = torch.tensor(label)
            
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)
        

        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        

        segments_tensor = torch.tensor([1] * len_a, 
                                        dtype=torch.long)
                                        
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

#%%

"""
DataLoader that returns mini batch
Return 4 tensors that BERT needs when training：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""
from torch.nn.utils.rnn import pad_sequence

def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # there's label
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

#%%

"""
Define a function that can return training results
"""
from scipy.stats import pearsonr
import numpy as np
def get_predictions(model, dataloader, compute_corr=False):
    predictions = None
    label_container = None
      
    with torch.no_grad():
        # loop through the dataset
        for data in dataloader:
            #  move tensors to GPU
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            
            #  3 tensors: tokens, segments, masks
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0] # == outputs
            labels = data[3] #  return tokens_tensors, segments_tensors, masks_tensors, label_ids
   
            # record the current results            
            if predictions is None:
                predictions = logits
            else:
                predictions = torch.cat((predictions, logits)) 
            
            if label_container is None:
                label_container = labels
            else:
                label_container = torch.cat((label_container, labels)) 
    if compute_corr:
        y_pred = predictions.cpu().numpy()
        y_pred = np.round(y_pred, 3).flatten()
        y_true = label_container.cpu().numpy()
        corr = pearsonr(y_true, y_pred)[0]
        return predictions, corr
    return predictions

#%%

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
# prepare material for training
PRETRAINED_MODEL_NAME = "bert-base-uncased"  # define the model used
BATCH_SIZE = 32

tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

trainset = EmotionDataset("train", tokenizer=tokenizer)
valset = EmotionDataset("val", tokenizer=tokenizer)

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch, shuffle=True)
valloader = DataLoader(valset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch, shuffle=True)

NUM_LABELS = 1
model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
model = model.to(device)

#%%

import time
from util.miscellaneous import format_time
from util.pytorchtools import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error


total_t0 = time.time()
NUM_EPOCHS = 4
N_TRAIN = 1
for i in range(N_TRAIN):

    print('')
    print('======== START TRAINING {} OUT OF {} TIMES ========'.format(i+1, N_TRAIN))

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True, checkpoint_name="bert_final_regression")
    for epoch in range(NUM_EPOCHS):

        print('======== Epoch {:} / {:} ========'.format(epoch + 1, NUM_EPOCHS))
        ###################
        # train the model #
        ###################
        print('Training...')
        
        t0 = time.time() # Measure how long the training epoch takes.   
        running_loss = 0.0        
        model.train() # training mode
    
        for step, data in enumerate(trainloader):  
            
            n_batches = len(trainloader)
            # Progress update every 25 batches. (a batch has 32 data)
            if step == n_batches or step % 25 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                
                # Report progress.
                print('    Batch {:>3} of {}.   Elapsed: {}.'.format(step, n_batches, elapsed))
    
            # unpack dataloader
            tokens_tensors, segments_tensors, \
            masks_tensors, labels = [t.to(device) for t in data]
    
            # gradient zero
            optimizer.zero_grad()
            
            # forward pass
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)
    
            loss = outputs[0]
            # backward
            loss.backward()
            optimizer.step()
    
            # current batch loss
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(trainloader)
        
        _, train_acc = get_predictions(model, trainloader, compute_corr=True)
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)        
        
        print("")       
        print('    [epoch %d] loss: %.3f (avg: %.3f), acc: %.3f' %
              (epoch + 1, running_loss, avg_train_loss, train_acc))
        print("    Training epcoh {} took: {:}".format(epoch + 1,training_time))
        print("")


        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_train_loss, model)
            
        if early_stopping.early_stop:
            print("Early stopping")
            break    
        print('')


print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))    
print('')






