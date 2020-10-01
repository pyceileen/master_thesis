# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:22:57 2020

@author: Pei-yuChen
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import confusion_matrix


def get_class_weight(y_array):
    weight = compute_class_weight('balanced',
                                        np.unique(y_array), y_array)
    classx = np.unique(y_array)
    zipped = zip(classx, weight)                            
    weights_list = dict(zipped)
    return(weights_list)

def get_sample_weight(y_array):
    count = np.unique(y_array, return_counts=True)
    weight = len(y_array) / (len(count[0]) * count[1])
    class_weight = dict(zip(np.unique(y_array),weight))
    sample_weight = compute_sample_weight(class_weight, y_array)
    return(sample_weight)


def draw_confusion_matrix(y_pred, y_true, title_name):
    
    label_list = ["ang", "hap", "neu", "sad"]
    cfm = confusion_matrix(y_true, y_pred, labels=label_list)
    df_cm = pd.DataFrame(cfm, index=label_list, columns=label_list)
    
    cfm_norm = df_cm.astype('float') / df_cm.sum(axis=1)[:, np.newaxis]
    
    labels = (100. * cfm_norm).round(2).astype(str) + '%'
        
    plt.figure(figsize=(10,7))
    sns.set(font_scale=1) # for label size
    sns.despine(offset=10, trim=True);
    
    ax = sns.heatmap(cfm_norm, annot=labels, annot_kws={"size": 12}, fmt='', 
                vmin=0, vmax=0.70, cmap="Purples", linewidths=.5, cbar=False) # font size
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.set_ticks([0, 0.30, 0.60])
    cbar.set_ticklabels(["0%", "30%", "60%"])
    
    plt.xlabel("Predicted", labelpad=10)
    plt.ylabel("True", labelpad=10)
    plt.title("audio model SVM")
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
