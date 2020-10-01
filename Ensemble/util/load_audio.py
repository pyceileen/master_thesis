# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:41:37 2020

@author: Pei-yuChen
"""
# adopt from https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/
from tensorflow.keras.models import load_model
from importlib import import_module
import numpy as np
from sklearn.linear_model import LogisticRegression
import librosa

def audio2wave(audio_name):
    wave = []
    y, sr = librosa.load(audio_name, sr=16000, duration=7.5) 
    wave.append(y)
    return(wave)


def load_all_models(n_models, model_type):
    model_list = ['3_categories', '4_categories']
    
    if model_type not in model_list:
        raise ValueError("Invalid model. Expected one of: %s" % model_list) 
        
    all_models = list()
    module = import_module('ACRNN2D')
    
    for i in range(n_models):
        
        if model_type == '3_categories':
            filename = 'models\\audio_final_3cat_' + str(i + 1) + '.h5'
        else:
            filename = 'models\\audio_final_4cat_' + str(i + 1) + '.h5'
        
		  # load model from file
        individual_model = load_model(filename, 
                                custom_objects={'Attention':module.Attention})
		  # add to list of members
        all_models.append(individual_model)
        print('>>> %s Loaded!' % filename)
    return (all_models)    
    
    
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(all_models, inputX):
    stackX = None
    for individual_model in all_models:
		  # make prediction
        yhat = individual_model.predict(inputX, verbose=0)
		  # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.dstack((stackX, yhat))
	  # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
    return (stackX)
 
# fit a model based on the outputs from the ensemble members
def fit_logistic(all_models, inputX, inputy):
	 # create dataset using ensemble
    stackedX = stacked_dataset(all_models, inputX)
	 # fit standalone model
    logistic_model = LogisticRegression()
    logistic_model.fit(stackedX, inputy)
    return (logistic_model)
 

















