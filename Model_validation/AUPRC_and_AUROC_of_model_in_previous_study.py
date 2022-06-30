#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:27:11 2022

@author: liukang
"""

from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as performance_measure
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

result = pd.read_csv('/home/liukang/Doc/AUPRC/test_result_10_No_Com.csv')

disease_list=pd.read_csv('/home/liukang/Doc/Error_analysis/subgroup_in_previous_study.csv')

models = result.iloc[:,1:-1].columns.tolist()

target_model = 'update_1921_mat_proba'
round_num = 200

disease_score_record = pd.DataFrame()
disease_performance_record = pd.DataFrame()

test_total = pd.DataFrame()
train_data = {}
test_data = {}
Weight_importance_record = {}
for data_num in range(1,5):
    
    train_data[data_num] = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
    test_data[data_num] = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    test_total = pd.concat([test_total,test_data[data_num]])
    
    train_ori = train_data[data_num]
    X_train = train_ori.drop('Label',axis=1)
    y_train=train_ori['Label']
    lr_All = LogisticRegression(n_jobs=-1)
    lr_All.fit(X_train, y_train)
    Weight_importance = lr_All.coef_[0]
    Weight_importance_record[data_num] = Weight_importance

test_total.reset_index(drop=True, inplace=True)

#build_subgroup_model
subgroup_result_total = {}
for disease_num in range(disease_list.shape[0]):
    
    disease_name = disease_list.loc[disease_num,'Subgroup']
    subgroup_in_disease = disease_list.loc[disease_num,'Feature'].split(",")
    age_standard = disease_list.loc[disease_num,'Age']
    
    subgroup_result = pd.DataFrame()
    
    for data_num in range(1,5):
        
        subgroup_result_in_data = pd.DataFrame()
        select_train_data = train_data[data_num]
        select_test_data = test_data[data_num]
        
        #select_train_data
        subgroup_train_data = select_train_data.loc[:,subgroup_in_disease]
        subgroup_train_data_sum = subgroup_train_data.sum(axis=1)
        subgroup_train_data_true = subgroup_train_data_sum >= 1
        if age_standard != 0:
            
            age_train_true = select_train_data.loc[:,'Demo1'] == age_standard
            subgroup_train_data_true = subgroup_train_data_true & age_train_true
            
        meaningful_train_sample = select_train_data.loc[subgroup_train_data_true]
        X_subgroup_train = meaningful_train_sample.drop('Label',axis=1)
        y_subgroup_train = meaningful_train_sample['Label']
        
        #select_test_data
        subgroup_test_data = select_test_data.loc[:,subgroup_in_disease]
        subgroup_test_data_sum = subgroup_test_data.sum(axis=1)
        subgroup_test_data_true = subgroup_test_data_sum >= 1
        if age_standard != 0:
            
            age_test_true = select_test_data.loc[:,'Demo1'] == age_standard
            subgroup_test_data_true = subgroup_test_data_true & age_test_true
            
        meaningful_test_sample = select_test_data.loc[subgroup_test_data_true]
        X_subgroup_test = meaningful_test_sample.drop('Label',axis=1)
        y_subgroup_test = meaningful_test_sample['Label']
        
        subgroup_result_in_data['Label'] = y_subgroup_test
        
        #subgroup_model_without_transfer
        lr_DG_ori=LogisticRegression(n_jobs=-1)
        lr_DG_ori.fit(X_subgroup_train, y_subgroup_train)
        subgroup_result_in_data.loc[:,'DG_ori'] = lr_DG_ori.predict_proba(X_subgroup_test)[:, 1]
        
        #subgroup_model_with_transfer
        lr_DG_mat=LogisticRegression(n_jobs=-1)
        X_subgroup_fit_train = X_subgroup_train * Weight_importance_record[data_num]
        X_subgroup_fit_test = X_subgroup_test * Weight_importance_record[data_num]
        lr_DG_mat.fit(X_subgroup_fit_train, y_subgroup_train)
        subgroup_result_in_data.loc[:,'DG_mat'] = lr_DG_mat.predict_proba(X_subgroup_fit_test)[:, 1]
        
        subgroup_result = pd.concat([subgroup_result,subgroup_result_in_data])
        
    subgroup_result = subgroup_result.reset_index(drop=True)
    subgroup_result_total[disease_name] = subgroup_result 

#model_comparison        
for disease_num in range(disease_list.shape[0]):
    
    select_data = test_total.copy()
    disease_name = disease_list.loc[disease_num,'Subgroup']
    subgroup_in_disease = disease_list.loc[disease_num,'Feature'].split(",")
    age_standard = disease_list.loc[disease_num,'Age']
    
    subgroup_data = select_data.loc[:,subgroup_in_disease]
    subgroup_data_sum = subgroup_data.sum(axis=1)
    subgroup_data_true = subgroup_data_sum >= 1
    
    if age_standard != 0:
        
        age_train_true = select_data.loc[:,'Demo1'] == age_standard
        subgroup_data_true = subgroup_data_true & age_train_true
    
    meaningful_result = result.loc[subgroup_data_true]
    meaningful_result = meaningful_result.copy()
    meaningful_result = meaningful_result.reset_index(drop=True)
    
    subgroup_model_result = subgroup_result_total[disease_name]
    meaningful_result.loc[:,'DG_mat'] = subgroup_model_result['DG_mat'].values
    meaningful_result.loc[:,'DG_ori'] = subgroup_model_result['DG_ori'].values
    
    result_select = meaningful_result
    
    performance_result = pd.DataFrame(index=range(round_num),columns=models)
    performance_general = pd.DataFrame(index=models,columns = ['final_performance','mean','std','95%_upper','95%_lower'])
    for model_num in range(len(models)):
        
        final_subgroup_performance = performance_measure(result_select['Label'],result_select.loc[:,models[model_num]])
        disease_performance_record.loc['Drg{}'.format(disease_list.iloc[disease_num, 0]),models[model_num]] = final_subgroup_performance
        performance_general.loc[models[model_num],'final_performance'] = final_subgroup_performance
        
    for round_id in range(round_num):
        
        sample_result = result_select.sample(frac=1,replace=True)
        
        for model_num in range(len(models)):
            
            performance =performance_measure(sample_result['Label'],sample_result.loc[:,models[model_num]])
            performance_result.loc[round_id,models[model_num]] = performance
        
    performance_general.loc[:,'mean'] = performance_result.mean(axis=0)
    performance_general.loc[:,'std'] = performance_result.std(axis=0)
    
    performance_general['95%_upper'] = performance_general['final_performance'] + (performance_general['std'] * 1.96)
    performance_general['95%_lower'] = performance_general['final_performance'] - (performance_general['std'] * 1.96)
    performance_general.to_csv("/home/liukang/Doc/AUPRC/AUPRC_compare_previous_study_{}.csv".format(disease_name))
    
    for model_num in range(len(models)):
        
        cov_between_models = np.mean(performance_result.loc[:,models[model_num]].values * performance_result.loc[:,target_model].values) - (performance_general.loc[models[model_num],'mean'] * performance_general.loc[target_model,'mean'])
        p_score = (performance_general.loc[target_model,'final_performance'] - performance_general.loc[models[model_num],'final_performance']) / np.sqrt((performance_general.loc[target_model,'std'] ** 2)+(performance_general.loc[models[model_num],'std'] ** 2)  - (2 * cov_between_models))
        disease_score_record.loc['Drg{}'.format(disease_list.iloc[disease_num, 0]),models[model_num]] = p_score

one_size_p_record = pd.DataFrame(index=disease_score_record.index.tolist(),columns=disease_score_record.columns.tolist())
one_size_p_record.loc[:,:] = norm.sf(abs(disease_score_record))
two_size_p_record = 2 * one_size_p_record

two_size_p_record.to_csv("/home/liukang/Doc/AUPRC/AUPRC_compare_previous_study_p.csv")
disease_performance_record.to_csv("/home/liukang/Doc/AUPRC/AUPRC_compare_previous_study_ori.csv")

    
    
    
  
      