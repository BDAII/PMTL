# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: Shuxy
"""
from sklearn.feature_selection import chi2, VarianceThreshold, SelectKBest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from gc import collect
from time import sleep
import os
import warnings
warnings.filterwarnings('ignore')

for p in [0.1]:
    #read data
    train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_5.csv')
    test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_5.csv')
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
    Wi = pd.DataFrame({'col_name':X_train.columns.tolist(), 'Feature_importance':lr_All.coef_[0]})
    
    read_dir = []
    clean_idx = 0
    
    ki = [abs(i) for i in list(Weight_importance)]
    
    for idx in range(len(test_df)):
        train_rank = train_df.copy()
        pre_data = test_df.loc[idx, :'CCS279']
        mat_copy = train_df.loc[:, :'CCS279'].copy()
        
        mat_copy -= pre_data
        mat_copy = mat_copy.astype('float64')
        mat_copy *= ki
        mat_copy = abs(mat_copy)
    
        train_rank['Distance'] = mat_copy.sum(axis=1)
        train_rank.sort_values('Distance', inplace=True)
        train_rank.reset_index(drop=True, inplace=True)
            
        len_split = int(len(train_rank) * p)
        
        train_data = train_rank.loc[:len_split, :'Label']
        #weight_tr = train_rank.loc[:len_split, 'Distance'].tolist()
        #weight_tr = [(weight_tr[0] + 0.01) / (val + 0.01) for val in weight_tr]
        
        X_train = train_data.drop('Label', axis=1)
        y_train = train_data['Label']
        X_test = test_df.loc[[idx], :'CCS279']
        s = SelectKBest(chi2, k=200)
        chi2_train = s.fit_transform(X_train, y_train)
        chi2_test = s.transform(X_test)
        
        col_idx = [int(i) for i in s.get_support()]
        col_idx = [i+j-1 for i,j in zip(range(1921), col_idx) if j > 0]
        need_wi = Wi.iloc[col_idx, 1].tolist()
        
        #ori_result
        lr = LogisticRegression(n_jobs=-1)
        lr.fit(chi2_train, y_train)#, sample_weight=weight_tr
        test_original.loc[idx, 'update_{}_ori_proba'.format(p)] = lr.predict_proba(chi2_test)[:, 1]
        
        #mat_result
        fit_train = chi2_train * need_wi
        fit_test = chi2_test * need_wi        
        
        lr = LogisticRegression(n_jobs=-1)
        lr.fit(fit_train, y_train)#, sample_weight=weight_tr
        test_original.loc[idx, 'update_{}_mat_proba'.format(p)] = lr.predict_proba(fit_test)[:, 1]
    
        
        clean_idx += 1
        
        if clean_idx % 10 == 0:
            collect()
        
        else:
            continue
    group_num=int(1/p)
    test_original.iloc[:, -4:].to_csv('/home/liukang/Doc/feature_selection/{}/knn_chi2_200_.csv'.format(group_num), index=False)