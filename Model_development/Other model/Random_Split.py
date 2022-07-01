#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:09:41 2019

@author: shuxinyu
"""
import numpy as np
import pandas as pd
import random

from sklearn.feature_selection import chi2, VarianceThreshold, SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

AUC_record=pd.DataFrame()
for round_num in range(10):
    result_concat=pd.DataFrame()
    for test_num in [1,2,3,4]:
        train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(test_num))
        test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(test_num))
        
        Lr = LogisticRegression(n_jobs=-1)
        X_train, y_train = train_df.loc[:, :'CCS279'], train_df['Label']
        X_test, y_test = test_df.loc[:, :'CCS279'], test_df['Label']
        
        Lr.fit(X_train, y_train)
        Wi = pd.DataFrame({'col_name':X_train.columns.tolist(), 'Feature_importance':Lr.coef_[0]})
        
        train_data=train_df
        train_data=train_data.sample(frac=1)
        train_data.reset_index(drop=True,inplace=True)
        test_data=test_df
        test_data=test_data.sample(frac=1)
        test_data.reset_index(drop=True,inplace=True)
        train_X,train_y=train_data.loc[:, :'CCS279'], train_data['Label']
        test_X,y_test=test_data.loc[:, :'CCS279'], test_data['Label']
        test_y=pd.DataFrame()
        test_y['Label']=y_test
        test_y_t=y_test
        
        #use 1/k samples for modeling
        for k in [5,10,20]:
            #for idx in range(k):
            kf_2 = StratifiedKFold(n_splits=k)
            kf_3 = StratifiedKFold(n_splits=k)
            k_time=1
            for train_idx, valid_idx in zip(kf_2.split(train_X, train_y), kf_3.split(test_X, test_y_t)):
                split_train = train_X.loc[train_idx[1], :'CCS279']
                split_y = train_y[train_idx[1]]
                
                lr = LogisticRegression(n_jobs=-1)
                lr.fit(split_train, split_y)
                
                test_y.loc[valid_idx[1], 'ori_{}'.format(k)] = lr.predict_proba(test_X.loc[valid_idx[1], :'CCS279'])[:, 1]
                
                lr = LogisticRegression(n_jobs=-1)
                lr.fit(split_train * list(Wi['Feature_importance']), split_y)
                
                test_y.loc[valid_idx[1], 'mat_{}'.format(k)] = lr.predict_proba(test_X.loc[valid_idx[1], :'CCS279'] * list(Wi['Feature_importance']))[:, 1]
                test_y.loc[valid_idx[1], 'group_{}'.format(k)] =k_time
                k_time=k_time+1
            #group_name='group_{}'.format(k)
            #mat_group_name='mat_group_{}'.format(k)
            #AUC_record.loc[0,group_name]=roc_auc_score(test_y['Label'],test_y['ori'])
            #AUC_record.loc[0,mat_group_name]=roc_auc_score(test_y['Label'],test_y['mat'])
            #test_y.loc[:, ['Label','ori', 'mat','group']].to_csv('/home/liukang/Doc/feature_selection/{}/Random_Kmeans_ori_chi2_200_.csv'.format(k), index=False)
        test_result=test_y
        result_concat=pd.concat([result_concat,test_result])
    result_concat.to_csv('/home/liukang/Doc/feature_selection/test/Random_Split_result_{}.csv'.format(round_num), index=False)
    for cluster_num in [5,10,20]:
        AUC_record.loc[round_num,'ori_{}'.format(cluster_num)]=roc_auc_score(result_concat['Label'],result_concat['ori_{}'.format(cluster_num)])
        AUC_record.loc[round_num,'mat_{}'.format(cluster_num)]=roc_auc_score(result_concat['Label'],result_concat['mat_{}'.format(cluster_num)])
AUC_record.to_csv('/home/liukang/Doc/feature_selection/test/Random_Split_AUC.csv', index=False)
       

 
'''
table = test_df.copy()
col_name = table.loc[:, 'Label':].columns.tolist()
col_name = col_name[2:]
label = table['Label']
for col in col_name:
    print('col_name:{}, AUC:{}'.format(col, roc_auc_score(label, table[col])))
#test_df.to_csv('/home/shuxinyu/C_T/FS_5.csv', index=False)
'''