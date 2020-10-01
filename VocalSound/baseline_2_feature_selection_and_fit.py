# -*- coding: utf-8 -*-
"""
Created on Thu May 14 10:18:03 2020

@author: Pei-yuChen
"""
import os 
PATH = os.path.dirname(os.path.realpath(__file__)) # get script path
os.chdir(PATH)

import pandas as pd
import numpy as np
labels_df = pd.read_csv('data\\baesline_label_df_4emo.csv')
features_df = pd.read_csv('data\\baseline_features_all_4emo.csv')

labels_df_ensemble = pd.read_csv('data\\baesline_label_df_4emo_ensemble.csv')
features_df_ensemble = pd.read_csv('data\\baseline_features_all_4emo_ensemble.csv')

train_X = features_df
train_y = labels_df['emotion']

test_X = features_df_ensemble
test_y = labels_df_ensemble['emotion']
#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFpr, f_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from util.miscellaneous import get_class_weight



unique, counts = np.unique(train_y, return_counts=True)
print(dict(zip(unique, counts)))

weight_list = get_class_weight(train_y)   
print(weight_list)


standardizer = StandardScaler()
vt = VarianceThreshold()
regression_selector = SelectKBest(f_classif, k=300)
pca_95 = PCA(0.95)
svm_classifier = svm.SVC()

pipe = Pipeline([
        ('standardize', standardizer),
        ('removeVariance', vt),
        ('selectFeatures', regression_selector),
        ('pca', pca_95),
        ('clf', svm_classifier)
        ])

 #?;speaker independent
tuned_parameters= {
        'clf__kernel':['poly'] , #['rbf', 'linear', 'poly'], 
        'clf__gamma': [0.01], #[1e-2, 1e-3, 1e-4],
        'clf__C': [0.1], #[0.001,0.1, 0.5], 
        #'clf__class_weight': [weight_list]
        }

grid_search = GridSearchCV(estimator=pipe,
                           param_grid=tuned_parameters,
                           #cv=skf.split(train_X, train_y),
                           cv=10,
                           n_jobs=-1,
                           verbose=10,
                           return_train_score=True
                           )
grid_search.fit(train_X, train_y)  

#%% test
print("Best parameters set found on validation set:")
print(grid_search.best_params_)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score

y_pred = grid_search.predict(test_X)
y_true = test_y

print(accuracy_score(y_true,y_pred)) #0.6695906432748538
print(precision_score(y_true,y_pred, average='macro')) #0.5413568505721187
print(recall_score(y_true,y_pred, average='macro')) #0.485061795775698
print(f1_score(y_true,y_pred, average='macro')) #0.49516460167392123

print(classification_report(y_true, y_pred)) #

# =============================================================================
#               precision    recall  f1-score   support
# 
#          ang       0.78      0.44      0.56        41
#          hap       0.00      0.00      0.00        33
#          neu       0.64      0.87      0.74       174
#          sad       0.75      0.63      0.68        94
# 
#     accuracy                           0.67       342
#    macro avg       0.54      0.49      0.50       342
# weighted avg       0.62      0.67      0.63       342
# 
# =============================================================================
#%%
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


#%% speaker independent

#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from util.miscellaneous import get_class_weight

train_inds, test_inds = next(GroupShuffleSplit(
    test_size=1, n_splits=2).split(features_df, groups=labels_df['speaker']))

train_X = features_df.iloc[train_inds]
train_y_df = labels_df.iloc[train_inds]
train_y = train_y_df['emotion']

test_X = features_df.iloc[test_inds]
test_y_df = labels_df.iloc[test_inds]
test_y = test_y_df['emotion']

unique, counts = np.unique(train_y, return_counts=True)
print(dict(zip(unique, counts)))

weight_list = get_class_weight(train_y)   
print(weight_list)


standardizer = StandardScaler()
vt = VarianceThreshold()
regression_selector = SelectFpr(f_classif, alpha=0.01)
pca_95 = PCA(0.95)
svm_classifier = svm.SVC()

pipe = Pipeline([
        ('standardize', standardizer),
        ('removeVariance', vt),
        ('selectFeatures', regression_selector),
        ('pca', pca_95),
        ('clf', svm_classifier)
        ])

 #?;speaker independent
tuned_parameters= {
        'clf__kernel':['poly'] , #['rbf', 'linear', 'poly'], 
        'clf__gamma': [0.01], #[1e-2, 1e-3, 1e-4],
        'clf__C': [0.1], #[0.001,0.1, 0.5], 
        'clf__class_weight': [weight_list]
        }

logo   = LeaveOneGroupOut()
grid_search = GridSearchCV(estimator=pipe,
                           param_grid=tuned_parameters,
                           cv=logo.split(train_X, train_y, groups=train_y_df['speaker']),
                           #cv=10,
                           n_jobs=-1,
                           verbose=10
                           )
grid_search.fit(train_X, train_y)  


#%% test
print("Best parameters set found on validation set:")
print(grid_search.best_params_)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = grid_search.predict(test_X)
y_true = test_y

print(accuracy_score(y_true,y_pred)) #6923076923076923
print(balanced_accuracy_score(y_true,y_pred)) #4846477484815796
print(classification_report(y_true, y_pred)) #


# =============================================================================
#               precision    recall  f1-score   support
# 
#          ang       1.00      0.07      0.12        15
#          hap       0.50      0.17      0.26        29
#          neu       0.73      0.83      0.78       127
#          sad       0.63      0.86      0.73        37
# 
#     accuracy                           0.69       208
#    macro avg       0.71      0.48      0.47       208
# weighted avg       0.70      0.69      0.65       208
# 
# =============================================================================
