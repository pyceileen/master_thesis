# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:24:33 2020

@author: Pei-yuChen
"""

import os 
os.chdir(r"C:\Users\Pei-yuChen\Desktop\Programming\Model\TextAnalysis")

import numpy as np
from util.read_dataset import load_data
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = 'emobank.csv'
(train_X, train_y), (val_X, val_y), (test_X, test_y) = load_data(
    data_path=DATA_PATH, ternary=False, sent_tok=False)
train_X = np.concatenate((train_X, val_X), axis=0)
train_y = np.concatenate((train_y, val_y), axis=0)


tfidf = TfidfVectorizer()
svclassifier = svm.SVR()
pipe = Pipeline([
        ('vect', tfidf),
        ('clf', svclassifier)
        ])
    
tuned_parameters= {
        'vect__min_df': [0,0.0001,0.001], # 0
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)], # ngram_range=(1, 2)
        'clf__kernel': ['rbf', 'linear', 'sigmoid'], # linear 
        'clf__gamma': (1e-2, 1e-3, 1e-4), # gamma=0.01
        'clf__C': [0.001,0.1,0.5], # C=0.5
        }

grid_search = GridSearchCV(estimator=pipe,
                           param_grid=tuned_parameters,
                           cv=10,
                           n_jobs=-1,
                           verbose=10
                           )
grid_search.fit(train_X, train_y)  
#%%
print("Best parameters set found on validation set:")
print(grid_search.best_params_)

y_pred = grid_search.predict(test_X)
y_pred = np.round(y_pred, 3)
y_true = test_y

mae_wo = mean_absolute_error(y_true, y_pred)         

mse_wo = mean_squared_error(y_true, y_pred)

rmse_wo = mean_squared_error(y_true, y_pred, squared=False)

corr_wo = pearsonr(y_true, y_pred)[0]














