#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:53:05 2022

@author: liukang
"""
from scipy.stats import wilcoxon
from scipy import stats
from scipy.stats import norm
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as performance_measure
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

result = pd.read_csv('/home/liukang/Doc/AUPRC/test_result_10_No_Com_without_top_subgroup_AKI50.csv')
models = result.iloc[:,1:-1].columns.tolist()
round_num = 200

performance_result = pd.DataFrame(index=range(round_num),columns=models)
performance_general = pd.DataFrame(index=models,columns = ['final_performance','mean','std','95%_upper','95%_lower'])

for model_num in range(len(models)):
    
    performance_general.loc[models[model_num],'final_performance'] = performance_measure(result['Label'],result.loc[:,models[model_num]])

for round_id in range(round_num):
    
    sample_result = result.sample(frac=1,replace=True)
    
    for model_num in range(len(models)):
        
        performance =performance_measure(sample_result['Label'],sample_result.loc[:,models[model_num]])
        performance_result.loc[round_id,models[model_num]] = performance
        
performance_general.loc[:,'mean'] = performance_result.mean(axis=0)
performance_general.loc[:,'std'] = performance_result.std(axis=0)

performance_general['95%_upper'] = performance_general['final_performance'] + (performance_general['std'] * 1.96)
performance_general['95%_lower'] = performance_general['final_performance'] - (performance_general['std'] * 1.96)

#z-test
normaltest_record = pd.DataFrame()
for j in range(performance_result.shape[1]):
    
    k2, p = stats.normaltest(performance_result.iloc[:,j])
    normaltest_record.loc[models[j],'p'] = p

z_score_record = pd.DataFrame()
for model_target in range(len(models)):
    
    target_model = models[model_target]
    
    for model_num in range(len(models)):
        
        cov_between_models = np.mean(performance_result.loc[:,models[model_num]].values * performance_result.loc[:,target_model].values) - (performance_general.loc[models[model_num],'mean'] * performance_general.loc[target_model,'mean'])
        z_score = (performance_general.loc[target_model,'final_performance'] - performance_general.loc[models[model_num],'final_performance']) / np.sqrt((performance_general.loc[target_model,'std'] ** 2)+(performance_general.loc[models[model_num],'std'] ** 2)  - (2 * cov_between_models))
        z_score_record.loc[target_model,models[model_num]] = z_score

one_size_p_record = pd.DataFrame(index=models,columns=models)
one_size_p_record.loc[:,:] = norm.sf(abs(z_score_record))
two_size_p_record = 2 * one_size_p_record


two_size_p_record.to_csv("/home/liukang/Doc/AUPRC/AUPRC_compare_without_AKI50_p.csv")
performance_general.to_csv("/home/liukang/Doc/AUPRC/AUPRC_compare_without_AKI50_ori.csv")
