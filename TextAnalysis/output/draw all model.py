# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:12:26 2020

@author: Pei-yuChen
"""

import os 
os.chdir(r"C:\Users\Pei-yuChen\Desktop\Programming\Model\TextAnalysis\output")


import pandas as pd
import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
from scipy import stats

mae = pd.read_csv('mae.csv')
corr = pd.read_csv('corr.csv')

mae_mean = mae.mean(axis = 0)
mae_error = stats.sem(mae)

corr_mean = corr.mean(axis = 0)
corr_error = stats.sem(corr)

name = ['SVR', 'BiGRU', 'BiLSTM', 'CNN\n-BiLSTM', 'Attention\n-BiGRU',
       'Attention\n-BiLSTM', 'HAN', 'Regional\nLSTM', 'Regional\n CNN-LSTM']


#%%
fig = plt.figure(figsize=(12,5))
bax = brokenaxes(ylims=((0, 0.005), (0.18, 0.225)), hspace=0.08)
bax.bar(name, mae_mean,
       yerr=mae_error,
       width=0.7,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=5)
bax.axhline(y=mae_mean[0], color='r', linestyle='--')
bax.axs[-1].set_xticks(range(0,9))
bax.axs[-1].set_xticklabels(name)
plt.title("Mean absolute error")

#%%
fig = plt.figure(figsize=(12,5))
bax = brokenaxes(ylims=((0, 0.01), (0.5, 0.625)), hspace=0.05)
bax.bar(name, corr_mean,
       yerr=corr_error,
       width=0.7,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=5)
bax.axhline(y=corr_mean[0], color='r', linestyle='--')
bax.axs[-1].set_xticks(range(0,9))
bax.axs[-1].set_xticklabels(name)
plt.title("Correlation coefficient")

