# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:09:33 2020

@author: Pei-yuChen
"""

import librosa
import numpy as np
from sklearn import preprocessing


def normalization(data):
    
    scaled_data = []
    for i in range(len(data)):
        scaled = preprocessing.scale(data[i], axis = 1)*255
        scaled_data.append(scaled)
    scaled_data = np.stack(scaled_data, axis=0)
    return(scaled_data)


def prepare_data_librosa(audios, 
                         features='logmel',
                         scaled=True):
    
    features_list = ['logmel', 'mfcc']
    
    if features not in features_list:
        raise ValueError("Invalid features. Expected one of: %s" % features_list)    

    
    ###########################################################################
    
    ###########
    #  LOGMEL #
    ###########
    if features=='logmel':
        
        ## main feature
        X1 = []
        for i in range(len(audios)):
            mel = librosa.feature.melspectrogram(y=audios[i], sr=16000, 
                                                 n_mels=40, n_fft=512, 
                                                 win_length=int(0.025*16000), 
                                                 hop_length=int(0.01*16000),
                                                 fmin=300, fmax=8000                                      
                                                 )
            logmel= librosa.power_to_db(mel, ref=np.max)
            X1.append(logmel)
        
        ## delta
        X2 = []
        for i in range(len(X1)):
            delta = librosa.feature.delta(X1[i], order=2)
            X2.append(delta)
        
        ## delta delta
        X3 = []
        for i in range(len(X2)):
            deltadelta = librosa.feature.delta(X2[i], order=2)
            X3.append(deltadelta)
       
        
        ##### standardization #####
        if scaled:
            feature_list=[X1,X2,X3]            
            for X in feature_list:
                for j in range(len(X)):                    
                    mean = np.mean(X[j])
                    std = np.std(X[j])
                    X[j] = (X[j] - mean)/std+1e-5
           
        ## zero padding
        max_length = 751
        #max([X1[i].shape[1] for i in range(len(X1))]) # find max length
        
        X_list=[X1,X2,X3]
        for X in X_list:            
            for i in range(len(X)):
                if X[i].shape[1] <max_length:
                    X[i] = librosa.util.fix_length(X[i], max_length)                
        X1 = np.stack(X1, axis=0)
        X2 = np.stack(X2, axis=0)    
        X3 = np.stack(X3, axis=0)    
        
        
        ## reshape
        N_DATA = X1.shape[0]
        N_COEF = X1.shape[1] # number of coefficients (40)
        N_FRAMES = X1.shape[2] # 126
        
        audio_X = []
        for i in range(len(X1)): # loop through each data entry
            X1_inx = X1[i]
            X2_inx = X2[i]
            X3_inx = X3[i]
            X1X2X3_inx = np.dstack((X1_inx, X2_inx, X3_inx))
            audio_X.append(X1X2X3_inx)
            
        N_FEATURES = 3    
        audio_X = np.concatenate(audio_X).ravel()
        audio_X = audio_X.reshape(N_DATA,N_COEF,N_FRAMES,N_FEATURES)         
        
        return(audio_X)


    ###########
    #   MFCC  #
    ###########    
    else: 

        ## main feature
        X1 = []
        for i in range(len(audios)):
            mfcc = librosa.feature.mfcc(y=audios[i], sr=16000, 
                                       n_mels=40, n_fft=512,n_mfcc=40, 
                                       win_length=int(0.025*16000), 
                                       hop_length=int(0.01*16000),
                                       fmin=300, fmax=8000                                      
                                       )
            X1.append(mfcc)
        
        ## delta
        X2 = []
        for i in range(len(X1)):
            delta = librosa.feature.delta(X1[i], order=2)
            X2.append(delta)
        
        ## delta delta
        X3 = []
        for i in range(len(X2)):
            deltadelta = librosa.feature.delta(X2[i], order=2)
            X3.append(deltadelta)
       
        
        ##### standardization #####
        if scaled:
            feature_list=[X1,X2,X3]            
            for X in feature_list:
                for j in range(len(X)):                    
                    mean = np.mean(X[j], axis=1).reshape(40,1)
                    std = np.std(X[j], axis=1).reshape(40,1)
                    X[j] = (X[j] - mean)/std+1e-5
           
        ## zero padding
        max_length = 751
        # max([X1[i].shape[1] for i in range(len(X1))]) # find max length
        
        X_list=[X1,X2,X3]
        for X in X_list:            
            for i in range(len(X)):
                if X[i].shape[1] <max_length:
                    X[i] = librosa.util.fix_length(X[i], max_length)                
        X1 = np.stack(X1, axis=0)
        X2 = np.stack(X2, axis=0)    
        X3 = np.stack(X3, axis=0)    
        
        ## reshape
        N_DATA = X1.shape[0]
        N_COEF = X1.shape[1] # number of coefficients (40)
        N_FRAMES = X1.shape[2] # 126
        
        audio_X = []
        for i in range(len(X1)): # loop through each data entry
            X1_inx = X1[i]
            X2_inx = X2[i]
            X3_inx = X3[i]
            X1X2X3_inx = np.dstack((X1_inx, X2_inx, X3_inx))
            audio_X.append(X1X2X3_inx)
            
        N_FEATURES = 3    
        audio_X = np.concatenate(audio_X).ravel()
        audio_X = audio_X.reshape(N_DATA,N_COEF,N_FRAMES,N_FEATURES)         
        
        return(audio_X)
    
        

            
def reshape_image(data, N_STEP):
    
    N_DATA = data.shape[0]
    N_COEF = data.shape[1] # number of coefficients (40)
    N_FRAMES = data.shape[2] # 126


    data_reshape = []
    for i in range(len(data)): # loop through each data entry
        # split into time_steps
        split_one_data = np.hsplit(data[i],N_STEP)
        
        # flatten and reshape: (40,126) -> (time_steps,40,126//time_steps)
        # 40 MFCC
        flatten = np.concatenate(split_one_data).ravel()
        reshape_one_data = flatten.reshape(N_STEP,N_COEF,N_FRAMES//N_STEP)
        data_reshape.append(reshape_one_data)    
    
    # flatten list of arrays and reshape    
    data_reshape = np.concatenate(data_reshape).ravel()
    data_reshape = data_reshape.reshape(N_DATA,N_STEP,N_COEF,N_FRAMES//N_STEP)
    
    return(data_reshape)
    
    















