# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:27:04 2020

@author: Pei-yuChen
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()
from util.preprocessing import text_preprocessing

def chunk_to_arrays(df, ternary=False, multitask=False):
    X = df['cleaned_text'].values
    split = df['split'].values
    if multitask:
        if ternary:
            yV = df['ternarized_V'].values
            yA = df['ternarized_A'].values
            yD = df['ternarized_D'].values
        else:
            yV = df['V'].values
            yA = df['A'].values
            yD = df['D'].values
        return X, yV, yA, yD, split

    else:     
        if ternary:
            y = df['ternarized_V'].values
        else:
            y = df['V'].values
        return X, y, split

def binirize_Y(df):
    binaryV = []
    binaryV = ['positive' if x >3 else 'negative' for x in df['V']]
    return binaryV

def ternarized_V(v):
    if v >= 3.12:
        return 2
    elif v < 2.84:
        return 0
    else:
        return 1

def load_data(data_path, ternary=False, sent_tok=False, multitask=False):
    print('\nLoading dataset...')
    
    df = pd.read_csv(data_path, usecols=['split', 'V', 'A', 'D', 'text'])
    df_clean = df.copy()
    
    df_clean['cleaned_text'] = df_clean['text'].progress_apply(
        lambda x: text_preprocessing(x, sent_tok=sent_tok))
    
    
    if sent_tok ==True:
        df_clean = df_clean[(df_clean['clean_text'].str[0] != '')]
    else:
        df_clean = df_clean[df_clean.cleaned_text != ""]  # delete blank rows
    
    if ternary:
        df_clean['ternarized_V'] = df_clean['V'].apply(lambda x: ternarized_V(x))
        df_clean['ternarized_A'] = df_clean['A'].apply(lambda x: ternarized_V(x))
        df_clean['ternarized_D'] = df_clean['D'].apply(lambda x: ternarized_V(x))

    
    if multitask==True:
        X, yV, yA, yD, split = chunk_to_arrays(df_clean, ternary=ternary, multitask=multitask)
        train_X = X[np.where(split == 'train')]
        train_yV = yV[np.where(split == 'train')]
        train_yA = yA[np.where(split == 'train')]
        train_yD = yD[np.where(split == 'train')]
        
        val_X = X[np.where(split == 'dev')]
        val_yV = yV[np.where(split == 'dev')]
        val_yA = yA[np.where(split == 'dev')]
        val_yD = yD[np.where(split == 'dev')]
        
        test_X = X[np.where(split == 'test')]
        test_yV = yV[np.where(split == 'test')]
        test_yA = yA[np.where(split == 'test')]
        test_yD = yD[np.where(split == 'test')]
        
        print('\nFinished loading dataset.')
        return (train_X, train_yV,train_yA,train_yD), (val_X, val_yV,val_yA,val_yD), (test_X, test_yV,test_yA,test_yD)
    
    else: 
        X, y, split = chunk_to_arrays(df_clean, ternary=ternary, multitask=multitask)
        train_X = X[np.where(split == 'train')]
        train_y = y[np.where(split == 'train')]
        
        val_X = X[np.where(split == 'dev')]
        val_y = y[np.where(split == 'dev')]
        
        test_X = X[np.where(split == 'test')]
        test_y = y[np.where(split == 'test')]
        
        print('\nFinished loading dataset.')
        return (train_X, train_y), (val_X, val_y), (test_X, test_y)



