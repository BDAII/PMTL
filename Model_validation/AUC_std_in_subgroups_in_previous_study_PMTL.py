#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:40:55 2021

@author: liukang
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

disease_list=pd.read_csv('/home/liukang/Doc/Error_analysis/subgroup_in_previous_study.csv')
result = pd.read_csv('/home/liukang/Doc/AUPRC/test_result_10_No_Com.csv')
round_num = 1000

disease_final_result = pd.DataFrame()

test_total = pd.DataFrame()
for i in range(1,5):
    
    test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(i))
    test_total = pd.concat([test_total,test_data])

test_total.reset_index(drop=True, inplace=True)

for disease_num in range(disease_list.shape[0]):
    
    disease_name = disease_list.loc[disease_num,'Subgroup']
    subgroup_in_disease = disease_list.loc[disease_num,'Feature'].split(",")
    age_standard = disease_list.loc[disease_num,'Age']
    
    test_feature_true = np.zeros(test_total.shape[0])
    
    for subgroup in subgroup_in_disease:
        
        subgroup_test_true = test_total.loc[:,subgroup] >= 1
        test_feature_true = test_feature_true + subgroup_test_true.values
    
    test_feature_true = test_feature_true >= 1
        
    if age_standard != 0:
        
        age_test_true = test_total.loc[:,'Demo1'] == age_standard
        test_feature_true = test_feature_true & age_test_true
        
    disease_test_result = result.loc[test_feature_true]
    disease_AUC = roc_auc_score(disease_test_result['Label'],disease_test_result['update_1921_mat_proba'])
    
    sampling_AUC_result = pd.DataFrame()
    for i in range(round_num):
        
        sampling_disease_test_result = disease_test_result.sample(frac=1,replace=True)
        sampling_AUC_result.loc[i,'AUC'] = roc_auc_score(sampling_disease_test_result['Label'],sampling_disease_test_result['update_1921_mat_proba'])
    
    disease_AUC_std = np.std(sampling_AUC_result.loc[:,'AUC'].values)
    
    disease_final_result.loc[disease_num,'Subgroup'] = disease_name
    disease_final_result.loc[disease_num,'AUC'] = disease_AUC
    disease_final_result.loc[disease_num,'AUC_std'] = disease_AUC_std
    
disease_final_result.to_csv('/home/liukang/Doc/AUPRC/AUC_std_in_previous_study.csv')

'''   
#['predict_proba','update_1921_mat_proba','update_1921_ori_proba','DG_ori','DG_mat']
models = ['predict_proba','update_1921_mat_proba','update_1921_ori_proba','DG_ori','DG_mat']
round_num = 1000
auprc_result = pd.DataFrame(index=range(round_num),columns=models)
auprc_general = pd.DataFrame(index=models,columns = ['AUPRC','mean','std','95%_upper','95%_lower'])

for k in range(1,6):
    
    auprc_general.iloc[k-1,0] = average_precision_score(result['Label'],result.iloc[:,k])

for i in range(round_num):
    
    sample_result = result.sample(frac=1,replace=True)
    
    for j in range(1,6):
        
        auprc_result.iloc[i,j-1] = average_precision_score(sample_result['Label'],sample_result.iloc[:,j])

auprc_general['std'] = auprc_result.std(axis=0).values
auprc_general['mean'] = auprc_result.mean(axis=0).values
auprc_general['95%_upper'] = auprc_general['AUPRC'] + (auprc_general['std'] * 1.96)
auprc_general['95%_lower'] = auprc_general['AUPRC'] - (auprc_general['std'] * 1.96)

cov_gvpb = np.mean(auprc_result.loc[:,'predict_proba'].values * auprc_result.loc[:,'update_1921_mat_proba'].values) - (auprc_general.loc['predict_proba','mean'] * auprc_general.loc['update_1921_mat_proba','mean'])
cov_pvpb = np.mean(auprc_result.loc[:,'update_1921_ori_proba'].values * auprc_result.loc[:,'update_1921_mat_proba'].values) - (auprc_general.loc['update_1921_ori_proba','mean'] * auprc_general.loc['update_1921_mat_proba','mean'])
cov_svsb = np.mean(auprc_result.loc[:,'DG_ori'].values * auprc_result.loc[:,'DG_mat'].values) - (auprc_general.loc['DG_ori','mean'] * auprc_general.loc['DG_mat','mean'])
cov_svpb = np.mean(auprc_result.loc[:,'DG_ori'].values * auprc_result.loc[:,'update_1921_mat_proba'].values) - (auprc_general.loc['DG_ori','mean'] * auprc_general.loc['update_1921_mat_proba','mean'])
cov_sbvpb = np.mean(auprc_result.loc[:,'DG_mat'].values * auprc_result.loc[:,'update_1921_mat_proba'].values) - (auprc_general.loc['DG_mat','mean'] * auprc_general.loc['update_1921_mat_proba','mean'])
cov_gvs = np.mean(auprc_result.loc[:,'predict_proba'].values * auprc_result.loc[:,'DG_ori'].values) - (auprc_general.loc['predict_proba','mean'] * auprc_general.loc['DG_ori','mean'])

t_score_1 = (auprc_general.loc['update_1921_mat_proba','AUPRC'] - auprc_general.loc['predict_proba','AUPRC']) / np.sqrt((auprc_general.loc['update_1921_mat_proba','std'] ** 2)+(auprc_general.loc['predict_proba','std'] ** 2)  - (2 * cov_gvpb))
t_score_2 = (auprc_general.loc['update_1921_mat_proba','AUPRC'] - auprc_general.loc['update_1921_ori_proba','AUPRC']) / np.sqrt((auprc_general.loc['update_1921_mat_proba','std'] ** 2)+(auprc_general.loc['update_1921_ori_proba','std'] ** 2) - (2 * cov_pvpb))
t_score_3 = (auprc_general.loc['DG_mat','AUPRC'] - auprc_general.loc['DG_ori','AUPRC']) / np.sqrt((auprc_general.loc['DG_mat','std'] ** 2)+(auprc_general.loc['DG_ori','std'] ** 2) - (2 * cov_svsb))
t_score_4 = (auprc_general.loc['update_1921_mat_proba','AUPRC'] - auprc_general.loc['DG_ori','AUPRC']) / np.sqrt((auprc_general.loc['update_1921_mat_proba','std'] ** 2)+(auprc_general.loc['DG_ori','std'] ** 2) - (2 * cov_svpb))
t_score_5 = (auprc_general.loc['update_1921_mat_proba','AUPRC'] - auprc_general.loc['DG_mat','AUPRC']) / np.sqrt((auprc_general.loc['update_1921_mat_proba','std'] ** 2)+(auprc_general.loc['DG_mat','std'] ** 2) - (2 * cov_sbvpb))
t_score_6 = (auprc_general.loc['DG_ori','AUPRC'] - auprc_general.loc['predict_proba','AUPRC']) / np.sqrt((auprc_general.loc['DG_ori','std'] ** 2)+(auprc_general.loc['predict_proba','std'] ** 2)  - (2 * cov_gvs))
'''