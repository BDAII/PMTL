# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: liukang
"""
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

result_total=pd.DataFrame()

#target subgroup
disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

#read data
for data_num in range(1,5):
    #test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    #training data
    train_ori=pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
    
    #record=pd.DataFrame()
    result_record=pd.DataFrame()
    result_record['Label']=test_ori['Label']
    X_train = train_ori.drop(['Label'], axis=1)
    y_train=train_ori['Label']
    X_test=test_ori.drop(['Label'], axis=1)
    
    #learn global model
    lr_All = LogisticRegression(n_jobs=-1)
    lr_All.fit(X_train, y_train)
    result_record['predict_proba' ] = lr_All.predict_proba(X_test)[:, 1]
    
    #knowledge used for transfer
    Weight_importance = lr_All.coef_[0]
    
    result_record['DG_ori'] = 0
    result_record['DG_mat'] = 0
    
    for disease_num in range(disease_list.shape[0]):
        #find patients with a certain disease
        train_feature_true=train_ori.loc[:,disease_list.iloc[disease_num,0]]>0
        train_meaningful_sample=train_ori.loc[train_feature_true]
        X_train=train_meaningful_sample.drop(['Label'], axis=1)
        y_train=train_meaningful_sample['Label']
        
        test_feature_true=test_ori.loc[:,disease_list.iloc[disease_num,0]]>0
        test_meaningful_sample=test_ori.loc[test_feature_true]
        X_test=test_meaningful_sample.drop(['Label'], axis=1)
        
        #Disease subgroup modeling without transfer learning
        lr_DG_ori=LogisticRegression(n_jobs=-1,C=1)
        #best_C = 0.1
        lr_DG_ori.fit(X_train, y_train)
        result_record.loc[X_test.index.tolist(),'DG_ori']=lr_DG_ori.predict_proba(X_test)[:, 1]
        
        #Disease subgroup modeling with transfer learning
        lr_DG_mat=LogisticRegression(n_jobs=-1,C=1)
        #best_C=2
        #transfer learning 
        fit_train=X_train*Weight_importance
        fit_test=X_test*Weight_importance
        
        lr_DG_mat.fit(fit_train, y_train)
        result_record.loc[X_test.index.tolist(),'DG_mat']=lr_DG_mat.predict_proba(fit_test)[:, 1]
    result_total=pd.concat([result_total,result_record])
result_total.to_csv('/home/liukang/Doc/disease_top_20_analysis_No_com.csv'.format(data_num), index=False)
