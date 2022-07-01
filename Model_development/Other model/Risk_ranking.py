#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:09:41 2019

@author: shuxinyu
"""
import numpy as np
import pandas as pd

from sklearn.feature_selection import chi2, VarianceThreshold, SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

AUC_record=pd.DataFrame()
result_concat=pd.DataFrame()
result_concat=pd.DataFrame()
for test_num in [5]:
    train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(test_num))
    test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(test_num))
    
    Lr = LogisticRegression(n_jobs=-1)
    X_train, y_train = train_df.loc[:, :'CCS279'], train_df['Label']
    Lr.fit(X_train, y_train)
    train_df['ori_predict']=Lr.predict_proba(train_df.loc[:, :'CCS279'])[:,1]
    test_df['ori_predict']=Lr.predict_proba(test_df.loc[:,:'CCS279'])[:,1]
    train_df.sort_values('ori_predict',inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df.sort_values('ori_predict',inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    
    y_test=pd.DataFrame()
    y_test['Label']=test_df['Label']
    
    for k in [5,10,20]:
        first_train_sample=0
        first_test_sample=0
        for idx in range(1,k+1):
            last_train_sample=int((idx/k)*train_df.shape[0])
            last_test_sample=int((idx/k)*test_df.shape[0])
            X_train=train_df.loc[first_train_sample:last_train_sample,:'CCS279']
            y_train=train_df.loc[first_train_sample:last_train_sample,'Label']
            lr = LogisticRegression(n_jobs=-1)
            lr.fit(X_train,y_train)
            X_test=test_df.loc[first_test_sample:last_test_sample,:'CCS279']
            #y_test['label']=test_df.loc[first_test_sample:last_test_sample,'Label']
            #y_test.loc[first_test_sample:last_test_sample,'Label']=test_df.loc[first_test_sample:last_test_sample,'Label']
            y_test.loc[first_test_sample:last_test_sample,'predict_{}'.format(k)]=lr.predict_proba(X_test)[:,1]
            y_test.loc[first_test_sample:last_test_sample,'group_{}'.format(k)]=idx
            first_train_sample=last_train_sample+1
            first_test_sample=last_test_sample+1
    test_result=y_test
    result_concat=pd.concat([result_concat,test_result])
for group_num in [5,10,20]:
    AUC_record.loc[0,'AUC_{}'.format(group_num)]=roc_auc_score(result_concat['Label'],result_concat['predict_{}'.format(group_num)])
result_concat.to_csv('/home/liukang/Doc/feature_selection/test/risk_order_result.csv', index=False)
