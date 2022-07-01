# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: Shuxy
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from gc import collect
from time import sleep
import os
import warnings
warnings.filterwarnings('ignore')


#read data
train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_2.csv')
test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_2.csv')
#over
train_original = train_df.copy()
test_original = test_df.copy()
  
lr_All = LogisticRegression(n_jobs=-1)
X_train = train_original.drop(['Label'], axis=1)
y_train = train_original['Label']
X_test = test_original.drop(['Label'], axis=1)

lr_All.fit(X_train, y_train)
test_original['predict_proba'] = lr_All.predict_proba(X_test)[:, 1]
    
Weight_importance = lr_All.coef_[0]

read_dir = []
clean_idx = 0
p_weight=pd.DataFrame(index=X_test.index.tolist(),columns=X_test.columns.tolist())
for iteration in [50]:
    list_dir = "/home/liukang/Doc/No_Com/Ma_old_Nor_Gra_01_001_0_005/Ma_old_Nor_Gra_01_001_0_005-2_50.csv"

    ki = pd.read_csv(list_dir)
    ki = ki.iloc[:, 0].tolist()
    #use to select features used for calculating patient similarity. Here, we use all features in X
    select_table = pd.read_csv('/home/liukang/Doc/No_Com/test_para/top-1921.csv')
    select_table['feature_name'] = X_train.columns.tolist()
    
    for index, val in zip([1], [1921]): 
        need_col = select_table.loc[select_table.iloc[:, index] == 1, 'feature_name'].tolist()

        for idx in range(len(test_df)):
            train_rank = train_df.copy()
            pre_data = test_df.loc[idx, :'CCS279']
            mat_copy = train_df.loc[:, :'CCS279'].copy()
            
            mat_copy -= pre_data
            mat_copy = mat_copy.astype('float64')
            mat_copy *= ki
            mat_copy = abs(mat_copy)
    
            train_rank['Distance'] = mat_copy.loc[:, need_col].sum(axis=1)
            train_rank.sort_values('Distance', inplace=True)
            train_rank.reset_index(drop=True, inplace=True)
                
            len_split = int(len(train_rank) / 10)
            
            train_data = train_rank.loc[:len_split, :'Label']
            weight_tr = train_rank.loc[:len_split, 'Distance'].tolist()
            weight_tr = [(weight_tr[0] + 0.01) / (val + 0.01) for val in weight_tr]
            
            X_train = train_data.drop('Label', axis=1)
            y_train = train_data['Label']
            X_test = test_df.loc[[idx], :'CCS279']
            
            #lr = LogisticRegression(n_jobs=-1)
            #lr.fit(X_train, y_train, sample_weight=weight_tr)
            #test_original.loc[idx, 'update_{}_ori_proba'.format(iteration)] = lr.predict_proba(X_test)[:, 1]
            
            #mat_result
            fit_train = X_train * Weight_importance
            fit_test = X_test * Weight_importance    
            
            lr = LogisticRegression(n_jobs=-1)
            lr.fit(fit_train, y_train, sample_weight=weight_tr)
            test_original.loc[idx, 'update_{}_mat_proba'.format(val)] = lr.predict_proba(fit_test)[:, 1]
            test_original.loc[idx, 'update_{}_mat_intercept'.format(val)] =lr.intercept_
            p_weight.iloc[idx,:]=lr.coef_[0]
            
            clean_idx += 1
            
            if clean_idx % 10 == 0:
                collect()
            
            else:
                continue
        p_weight_final=p_weight * Weight_importance
        #output prediction and intercept for each patient
        test_original.iloc[:, -4:].to_csv('/home/liukang/Doc/No_Com/test_para/Ma_old_Nor_Gra_01_001_0_005-2_50.csv'.format(val, iteration), index=False)
        #output coef for each patient
        p_weight_final.to_csv('/home/liukang/Doc/No_Com/test_para/weight_Ma_old_Nor_Gra_01_001_0_005-2_50.csv'.format(val, iteration), index=False)