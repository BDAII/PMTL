#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 18:53:07 2021

@author: liukang
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

subgroup_list = ['Drg76']
#['Drg2','Drg254','Drg255','Drg256','Drg257','Drg260']
subgroup_standard = [1]
final_standard = 1

y_select_test_record = []
predict_DG_ori_record = []
predict_DG_mat_record = []
predict_global_record = []
predict_person_record = []

for data_num in range(1,5):
    
    #test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    #training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
    
    personal_result = pd.read_csv('/home/liukang/Doc/No_Com/test_para/Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num))
    
    X_train = train_ori.drop(['Label'], axis=1)
    y_train = train_ori['Label']
    
    X_test = test_ori.drop(['Label'], axis=1)
    y_test = test_ori['Label']
    
    lr_all = LogisticRegression(n_jobs=-1)
    lr_all.fit(X_train,y_train)
    
    weight_importance = lr_all.coef_[0]
    
    train_final_true = np.zeros(train_ori.shape[0])
    test_final_true = np.zeros(test_ori.shape[0])
    
    for subgroup_num in range(len(subgroup_list)):
        
        train_subgroup_true = train_ori.loc[:,subgroup_list[subgroup_num]] == subgroup_standard[subgroup_num]
        train_final_true = train_final_true + train_subgroup_true.values
        
        test_subgroup_true = test_ori.loc[:,subgroup_list[subgroup_num]] == subgroup_standard[subgroup_num]
        test_final_true = test_final_true + test_subgroup_true.values
    
    train_final_select = train_final_true == final_standard
    test_final_select = test_final_true == final_standard
    
    select_personal_record = personal_result.loc[test_final_select]
    select_personal_result = select_personal_record['update_1921_mat_proba']
    predict_person_record.extend(select_personal_result)
    
    select_train = train_ori.loc[train_final_select]
    select_test = test_ori.loc[test_final_select]
    
    X_select_train = select_train.drop(['Label'], axis=1)
    y_select_train = select_train['Label']
    
    X_select_test = select_test.drop(['Label'], axis=1)
    y_select_test = select_test['Label']
    
    y_select_test_record.extend(y_select_test)
    
    lr_sub_ori = LogisticRegression(n_jobs=-1)
    lr_sub_ori.fit(X_select_train,y_select_train)
    
    predict_DG_ori = lr_sub_ori.predict_proba(X_select_test)[:, 1]
    predict_DG_ori_record.extend(predict_DG_ori)
    
    X_select_fit_train = X_select_train * weight_importance
    X_select_fit_test = X_select_test * weight_importance
    
    lr_sub_mat = LogisticRegression(n_jobs=-1)
    lr_sub_mat.fit(X_select_fit_train, y_select_train)
    
    predict_DG_mat = lr_sub_mat.predict_proba(X_select_fit_test)[:, 1]
    predict_DG_mat_record.extend(predict_DG_mat)
    
    predict_global = lr_all.predict_proba(X_select_test)[:, 1]
    predict_global_record.extend(predict_global)

AUC_DG_ori = roc_auc_score(y_select_test_record, predict_DG_ori_record)
AUC_DG_mat = roc_auc_score(y_select_test_record, predict_DG_mat_record)
AUC_global = roc_auc_score(y_select_test_record, predict_global_record)
AUC_person = roc_auc_score(y_select_test_record, predict_person_record)
    
    
        