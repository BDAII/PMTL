#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:01:59 2022

@author: liukang
"""

from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as performance_measure
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


result = pd.read_csv('/home/liukang/Doc/AUPRC/test_result_10_No_Com.csv')

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_31_no_drg.csv')

models = result.iloc[:,1:-1].columns.tolist()

target_model = 'update_1921_mat_proba'
round_num = 200

disease_score_record = pd.DataFrame()
disease_performance_record = pd.DataFrame()


for disease_num in range(disease_list.shape[0]):
    
    performance_result = pd.DataFrame(index=range(round_num),columns=models)
    performance_general = pd.DataFrame(index=models,columns = ['final_performance','mean','std','95%_upper','95%_lower'])
    
    disease_true = result.loc[:,'Drg']== disease_list.iloc[disease_num, 0]
    result_select = result.loc[disease_true]
    
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
    
    for model_num in range(len(models)):
        
        cov_between_models = np.mean(performance_result.loc[:,models[model_num]].values * performance_result.loc[:,target_model].values) - (performance_general.loc[models[model_num],'mean'] * performance_general.loc[target_model,'mean'])
        p_score = (performance_general.loc[target_model,'final_performance'] - performance_general.loc[models[model_num],'final_performance']) / np.sqrt((performance_general.loc[target_model,'std'] ** 2)+(performance_general.loc[models[model_num],'std'] ** 2)  - (2 * cov_between_models))
        disease_score_record.loc['Drg{}'.format(disease_list.iloc[disease_num, 0]),models[model_num]] = p_score
        
one_size_p_record = pd.DataFrame(index=disease_score_record.index.tolist(),columns=disease_score_record.columns.tolist())
one_size_p_record.loc[:,:] = norm.sf(abs(disease_score_record))
two_size_p_record = 2 * one_size_p_record

two_size_p_record.to_csv("/home/liukang/Doc/AUPRC/AUPRC_compare_top31_p.csv")
disease_performance_record.to_csv("/home/liukang/Doc/AUPRC/AUPRC_compare_top31_ori.csv")
