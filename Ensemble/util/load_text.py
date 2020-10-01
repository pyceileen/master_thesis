# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:01:11 2020

@author: Pei-yuChen
"""
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


def load_BERT(model_type):
    
    model_list = ['regression', '3_categories']
    
    if model_type not in model_list:
        raise ValueError("Invalid model. Expected one of: %s" % model_list)    

    PRETRAINED_MODEL_NAME = "bert-base-uncased" 
    
    if model_type == 'regression':
        NUM_LABELS = 1
        MODEL = 'bert_final_regression.pt'
    else:
        NUM_LABELS = 3
        MODEL = 'bert_final_3cat.pt'        
    
    
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = model.to(device)
    model.load_state_dict(torch.load('models/' + MODEL,map_location=device))

    return(model)    
    

def gettensors(text):
    PRETRAINED_MODEL_NAME = "bert-base-uncased" 
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
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


def predict_text(model, text):
    if next(model.parameters()).is_cuda: # if gpu
        tokens_tensors, \
            segments_tensors, \
                masks_tensors = [t.to("cuda:0") for t in gettensors(text) if t is not None]

        outputs = model(input_ids=tokens_tensors, 
                token_type_ids=segments_tensors, 
                attention_mask=masks_tensors
                )
        text_emotion = outputs[0].cpu().detach().numpy().flatten()
        return (text_emotion)
        
    else: # if cpu
        tokens_tensors, segments_tensors, masks_tensors = gettensors(text)
    
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors
                        )
        text_emotion = outputs[0].cpu().detach().numpy().flatten()
        return (text_emotion)



















