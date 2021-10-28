#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:54:44 2021

@author: liukang
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')
feature_list=pd.read_csv('/home/liukang/Doc/Error_analysis/top20_auc_gain_feature.csv')

auc_record = pd.DataFrame()
test_data_total = pd.DataFrame()
general_result_record = pd.DataFrame()

for data_num in range(1,5):
    
    #train_data = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
    test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    person_coef = pd.read_csv('/home/liukang/Doc/No_Com/test_para/weight_Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num))
    
    test_data_total = pd.concat([test_data_total,test_data])
    
    y_test = test_data['Label']
    X_test = test_data.drop(['Label'],axis=1)
    
    general_score = X_test * person_coef
    general_score['Label'] = y_test
    general_result_record = pd.concat([general_result_record, general_score])

select_score = general_result_record.loc[:,feature_list['Feature']]
auc_global_selected_feature = roc_auc_score(general_result_record['Label'],select_score.sum(axis=1))
auc_record.loc[0,'global'] = auc_global_selected_feature

total_predict_result = pd.DataFrame()

for disease_num in range(20):
    
    disease = disease_list.loc[disease_num,'Drg']
    
    subgroup_true = test_data_total.loc[:,disease]==1
    
    subgroup_data = test_data_total.loc[subgroup_true]
    
    subgroup_select_score = select_score.loc[subgroup_true]
        
    auc_subgroup_selected_feature = roc_auc_score(subgroup_data['Label'], subgroup_select_score.sum(axis=1))
    auc_record.loc[0,disease] = auc_subgroup_selected_feature
    
    predict_result = pd.DataFrame()
    predict_result['Label'] = subgroup_data['Label']
    predict_result['predict_PMTL'] = subgroup_select_score.sum(axis=1)
    total_predict_result = pd.concat([total_predict_result,predict_result])
    
    