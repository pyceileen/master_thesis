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
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats


def get_class_weight(y_array):
    weights_list = compute_class_weight('balanced',
                                        np.unique(y_array), y_array)
    weights_list = dict(enumerate(weights_list))
    return(weights_list)

def get_sample_weight(y_array):
    count = np.unique(y_array, return_counts=True)
    weight = len(y_array) / (len(count[0]) * count[1])
    class_dict = dict(zip(np.unique(y_array),weight))
    sample_weight = list(map(class_dict.get, y_array))
    return(np.array(sample_weight))
    
def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)
def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)
def weighted_corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

def y2string(raw_y):
    y_string = []
    for i in raw_y:
        if i == 0:
            x = 'negative'
        elif i == 1:
            x = 'neutral'
        else:
            x = 'positive'
        y_string.append(x)
    return(y_string)

def make_classfication_report(y_pred, y_true):    
    y_pred_str = y2string(y_pred)
    y_true_str = y2string(y_true)
    
    return(print(classification_report(y_true_str, y_pred_str)))


def draw_confusion_matrix(y_pred, y_true, title_name):
    y_pred_str = y2string(y_pred)
    y_true_str = y2string(y_true)
    
    label_list = ["negative", "neutral", "positive"]
    cfm = confusion_matrix(y_true_str, y_pred_str, labels=label_list)
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
    plt.title(title_name)
    plt.show()
    
def draw_regression(y_true, y_pred, title):
    plt.figure()
    plt.axis('square')
    plt.xlim([1, 5.3])
    plt.ylim([1, 5.3])
    plt.yticks(np.arange(1, 5.3, 0.5))
    plt.xticks(np.arange(1, 5.3, 0.5))
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.scatter(y_true, y_pred, s=25, alpha=0.8)
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true,y_pred)
    line = slope*y_true+intercept
    plt.plot(y_true, line, 'r', label='r = {:.3f}'.format(r_value), color='lime')
    plt.legend(fontsize=9)
    plt.title(title)
    plt.savefig("output\\figures\\"+title)   
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
