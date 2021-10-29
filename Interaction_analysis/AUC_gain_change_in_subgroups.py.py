#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:24:26 2020

@author: liukang
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression 

#disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')
#ori_test_data = pd.read_csv('/home/liukang/Doc/valid_df/train_5.csv')
#feature_list = ori_test_data.drop(['Label'],axis=1)
#feature_name=np.append(feature_list.columns.tolist(),'constant')
#auc_record = pd.DataFrame(index=['ori','fs'],columns=disease_list['Drg'])


def PMTL_analysis(subgroup):
    
    #Features' AUC gain in PMTL
    test_total = pd.DataFrame()
    person_coef_total = pd.DataFrame()
    person_result_total = pd.DataFrame()

    
    for data_num in range(1,5): 
        
        test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
        person_coef = pd.read_csv('/home/liukang/Doc/No_Com/test_para/weight_Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num))
        person_result = pd.read_csv('/home/liukang/Doc/No_Com/test_para/Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num))
        
        test_total = pd.concat([test_total,test_df])
        person_coef_total = pd.concat([person_coef_total,person_coef])
        person_result_total = pd.concat([person_result_total,person_result])
        
    X_test_total = test_total.drop('Label',axis=1)
    X_test_total['constant'] = 1
    person_coef_total['constant'] = person_result_total.loc[:,'update_1921_mat_intercept']
    
    feature_name = X_test_total.columns.tolist()
    auc_gain = pd.DataFrame(columns=['auc_gain'],index=feature_name)
    
    per_ori_record = pd.DataFrame()
    
    subgroup_true = test_total.loc[:,subgroup] == 1
    
    select_X_test = X_test_total.loc[subgroup_true]
    select_coef = person_coef_total.loc[subgroup_true]
    #ori_score_total = X_test_total * person_coef_total
    score_total = select_X_test * select_coef
    
    mean_select_coef = select_coef.mean(axis=0).values
    mean_coef_record = pd.DataFrame()
    mean_coef_record['feature'] = feature_name
    mean_coef_record['coef'] = mean_select_coef
    mean_coef_record.set_index('feature',inplace=True)
    male_coef = select_coef.loc[:,'Demo3_1'].values
    female_coef = select_coef.loc[:,'Demo3_2'].values
    gender_coef = male_coef - female_coef
    mean_coef_record.loc[['Demo3_1','Demo3_2'],'coef'] = np.mean(gender_coef)
    
    
    score_in_ori_per_model = score_total
        
    Label_data = test_total.loc[subgroup_true]
    
    true_sample = Label_data['Label'] == 1
    false_sample = Label_data['Label'] == 0
    
    score_in_ori_per_true = score_in_ori_per_model.loc[true_sample]
    score_in_ori_per_false = score_in_ori_per_model.loc[false_sample]
    
    mean_score_in_ori_per_true = score_in_ori_per_true.mean()
    mean_score_in_ori_per_false = score_in_ori_per_false.mean()
    
    score_diff_ori_per = mean_score_in_ori_per_true-mean_score_in_ori_per_false
    
    per_ori_record['feature'] = feature_name
    per_ori_record['score_diff'] = score_diff_ori_per.values
    positive_feature = per_ori_record.loc[:,'score_diff'] >= 0
    select_feature = per_ori_record.loc[positive_feature]
    sample_after_fs = score_in_ori_per_model.loc[:,select_feature['feature']]
    
    final_per_score = score_in_ori_per_model.sum(axis=1)
    auc_ori = roc_auc_score(Label_data['Label'],final_per_score)
    auc_fs = roc_auc_score(Label_data['Label'],sample_after_fs.sum(axis=1))
    
    gender_score_true = mean_score_in_ori_per_true['Demo3_1'] + mean_score_in_ori_per_true['Demo3_2']
    gender_score_false = mean_score_in_ori_per_false['Demo3_1'] + mean_score_in_ori_per_false['Demo3_2']
    gender_score_diff = gender_score_true - gender_score_false
    per_ori_record.set_index('feature',inplace=True)
    per_ori_record.loc[['Demo3_1','Demo3_2'],'score_diff'] = gender_score_diff
    per_ori_record.to_csv('/home/liukang/Doc/Error_analysis/PMTL_score_diff_gender_change_{}.csv'.format(subgroup))
    
    
    for feature_num in range(X_test_total.shape[1]):
        
        if feature_name[feature_num] =='Demo3_2' or feature_name[feature_num] == 'Demo3_1':
            
            score_select_female = score_total.loc[:,'Demo3_1']
            score_select_male = score_total.loc[:,'Demo3_2']
            score_new = final_per_score - score_select_female -score_select_male
            
        else:
            
            score_select = score_total.loc[:,feature_name[feature_num]]
            score_new = final_per_score - score_select
            
        auc_new = roc_auc_score(Label_data['Label'],score_new)
        auc_gap = auc_ori-auc_new
        auc_gain.loc[feature_name[feature_num],'auc_gain'] = auc_gap
        
    auc_gain.to_csv('/home/liukang/Doc/Error_analysis/PMTL_feature_auc_gain_gender_change_{}.csv'.format(subgroup))
    #auc_record.loc['ori',0] = auc_ori
    #auc_record.loc['fs',0] = auc_fs
    return mean_coef_record, per_ori_record, auc_gain


def PM_analysis(subgroup):
    
    #Features' AUC gain in PM
    test_total = pd.DataFrame()
    person_coef_total = pd.DataFrame()
    person_result_total = pd.DataFrame()
    
    
    for data_num in range(1,5): 
        
        test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
        #person_coef = pd.read_csv('/home/liukang/Doc/No_Com/test_para/weight_Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num))
        #person_result = pd.read_csv('/home/liukang/Doc/No_Com/test_para/Ma_old_Nor_Gra_01_001_0_005-{}_50.csv'.format(data_num))
        
        test_total = pd.concat([test_total,test_df])
        #person_coef_total = pd.concat([person_coef_total,person_coef])
        #person_result_total = pd.concat([person_result_total,person_result])
        
    X_test_total = test_total.drop('Label',axis=1)
    X_test_total['constant'] = 1
    #person_coef_total['intercept'] = person_result_total.loc[:,'update_1921_mat_intercept']
    
    feature_name = X_test_total.columns.tolist()
    auc_gain = pd.DataFrame(columns=['auc_gain'],index=feature_name)
    
    per_ori_record = pd.DataFrame()
    
    subgroup_true = test_total.loc[:,subgroup] == 1
    
    #ori_score_total = X_test_total * person_coef_total
    score_total = pd.read_csv('/home/liukang/Doc/Error_analysis/person_ori_score_{}.csv'.format(subgroup))
    
    score_in_ori_per_model = score_total
        
    Label_data = test_total.loc[subgroup_true]
    Label_data.reset_index(drop=True, inplace=True)
    
    select_coef = pd.read_csv('/home/liukang/Doc/Error_analysis/person_ori_coef_record_{}.csv'.format(subgroup))
    mean_select_coef = select_coef.mean(axis=0).values
    mean_coef_record = pd.DataFrame()
    mean_coef_record['feature'] = feature_name
    mean_coef_record['coef'] = mean_select_coef
    mean_coef_record.set_index('feature',inplace=True)
    male_coef = select_coef.loc[:,'Demo3_1'].values
    female_coef = select_coef.loc[:,'Demo3_2'].values
    gender_coef = male_coef - female_coef
    mean_coef_record.loc[['Demo3_1','Demo3_2'],'coef'] = np.mean(gender_coef)
    
    true_sample = Label_data['Label'] == 1
    false_sample = Label_data['Label'] == 0
    
    score_in_ori_per_true = score_in_ori_per_model.loc[true_sample]
    score_in_ori_per_false = score_in_ori_per_model.loc[false_sample]
    
    mean_score_in_ori_per_true = score_in_ori_per_true.mean()
    mean_score_in_ori_per_false = score_in_ori_per_false.mean()
    
    score_diff_ori_per = mean_score_in_ori_per_true-mean_score_in_ori_per_false
    
    per_ori_record['feature'] = feature_name
    per_ori_record['score_diff'] = score_diff_ori_per.values
    positive_feature = per_ori_record.loc[:,'score_diff'] >= 0
    select_feature = per_ori_record.loc[positive_feature]
    sample_after_fs = score_in_ori_per_model.loc[:,select_feature['feature']]
    
    final_per_score = score_in_ori_per_model.sum(axis=1)
    auc_ori = roc_auc_score(Label_data['Label'],final_per_score)
    auc_fs = roc_auc_score(Label_data['Label'],sample_after_fs.sum(axis=1))
    
    gender_score_true = mean_score_in_ori_per_true['Demo3_1'] + mean_score_in_ori_per_true['Demo3_2']
    gender_score_false = mean_score_in_ori_per_false['Demo3_1'] + mean_score_in_ori_per_false['Demo3_2']
    gender_score_diff = gender_score_true - gender_score_false
    per_ori_record.set_index('feature',inplace=True)
    per_ori_record.loc[['Demo3_1','Demo3_2'],'score_diff'] = gender_score_diff
    #per_ori_record.to_csv('/home/liukang/Doc/Error_analysis/PM_score_diff_gender_change_{}.csv'.format(subgroup))
    
    
    for feature_num in range(X_test_total.shape[1]):
        
        if feature_name[feature_num] =='Demo3_2' or feature_name[feature_num] == 'Demo3_1':
            
            score_select_female = score_total.loc[:,'Demo3_1']
            score_select_male = score_total.loc[:,'Demo3_2']
            score_new = final_per_score - score_select_female -score_select_male
            
        else:
            
            score_select = score_total.loc[:,feature_name[feature_num]]
            score_new = final_per_score - score_select
            
        auc_new = roc_auc_score(Label_data['Label'],score_new)
        auc_gap = auc_ori-auc_new
        auc_gain.loc[feature_name[feature_num],'auc_gain'] = auc_gap
        
    #auc_gain.to_csv('/home/liukang/Doc/Error_analysis/PM_feature_auc_gain_gender_change_{}.csv'.format(subgroup))
    return mean_coef_record, per_ori_record, auc_gain



def global_analysis(subgroup):
    
    #Features' AUC gain in global model
    global_score_record = pd.DataFrame()
    global_y_record = pd.DataFrame()
    X_test_total = pd.DataFrame()
    for data_num in range(1,5):
        
        train_data = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
        test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
        
        y_train = train_data['Label']
        X_train = train_data.drop(['Label'],axis=1)
        
        y_test = test_data['Label']
        X_test = test_data.drop(['Label'],axis=1)
        X_test['constant'] = 1
        X_test_total = pd.concat([X_test_total,X_test])
        
        lr = LogisticRegression(n_jobs=-1)
        lr.fit(X_train,y_train)
        
        coef = lr.coef_[0]
        intercept = lr.intercept_
        final_coef = np.append(coef,intercept)
        
        score_test = X_test * final_coef
        
        global_score_record = pd.concat([global_score_record,score_test])
        global_y_record = pd.concat([global_y_record,test_data])
        
        
    feature_name = X_test.columns.tolist()
    subgroup_true = global_y_record.loc[:,subgroup] ==1
    
    global_score_record = global_score_record.loc[subgroup_true]
    global_y_record = global_y_record.loc[subgroup_true]
    X_test_total = X_test_total.loc[subgroup_true]
    
    subgroup_feature_mean = X_test_total.mean(axis=0).values
    feature_avg_change = subgroup_feature_mean / global_feature_mean.values
    tune_global_score_diff_with_avg_test = global_in_general_score_diff.loc[:,'score_diff'].values * feature_avg_change
    
    tune_global_score_diff_with_avg = pd.DataFrame(index=feature_name)
    tune_mean_score_in_true = pd.DataFrame(index=feature_name)
    tune_mean_score_in_false = pd.DataFrame(index=feature_name)
    
    tune_mean_score_in_true['score'] = global_mean_score_in_true.loc[:,'score'].values * feature_avg_change
    tune_mean_score_in_false['score'] = global_mean_score_in_false.loc[:,'score'].values * feature_avg_change
    tune_global_score_diff_with_avg['score_diff'] = tune_mean_score_in_true.loc[:,'score'].values - tune_mean_score_in_false.loc[:,'score'].values
    
    tune_gender_score_true = tune_mean_score_in_true.loc['Demo3_1','score'] + tune_mean_score_in_true.loc['Demo3_2','score']
    tune_gender_score_false = tune_mean_score_in_false.loc['Demo3_1','score'] + tune_mean_score_in_false.loc['Demo3_2','score']
    tune_gender_score_diff = tune_gender_score_true - tune_gender_score_false
    tune_global_score_diff_with_avg.loc[['Demo3_1','Demo3_2'],'score_diff'] = tune_gender_score_diff
    
    true_global_sample = global_y_record.loc[:,'Label'] == 1
    false_global_sample = global_y_record.loc[:,'Label'] == 0
    
    score_in_true = global_score_record.loc[true_global_sample]
    score_in_false = global_score_record.loc[false_global_sample]
    
    mean_score_in_true = score_in_true.mean()
    mean_score_in_false = score_in_false.mean()
    
    score_diff_global = mean_score_in_true-mean_score_in_false
    
    global_record = pd.DataFrame()
    global_record['feature'] = feature_name
    global_record['score_diff'] = score_diff_global.values
    positive_global_feature = global_record.loc[:,'score_diff'] > 0
    select_global_feature = global_record.loc[positive_global_feature]
    global_sample_after_fs = global_score_record.loc[:,select_global_feature['feature']]
    
    
    auc_global = roc_auc_score(global_y_record['Label'],global_score_record.sum(axis=1))
    auc_global_fs = roc_auc_score(global_y_record['Label'],global_sample_after_fs.sum(axis=1))
    
    gender_score_true = mean_score_in_true['Demo3_1'] + mean_score_in_true['Demo3_2']
    gender_score_false = mean_score_in_false['Demo3_1'] + mean_score_in_false['Demo3_2']
    gender_score_diff = gender_score_true - gender_score_false
    global_record.set_index('feature',inplace=True)
    global_record.loc[['Demo3_1','Demo3_2'],'score_diff'] = gender_score_diff
    #global_record.to_csv('/home/liukang/Doc/Error_analysis/global_score_diff_gender_change_{}.csv'.format(subgroup))
    
    
    final_global_score = global_score_record.sum(axis=1)
    auc_global = roc_auc_score(global_y_record['Label'],final_global_score)
    
    auc_gain = pd.DataFrame(columns=['auc_gain'],index=feature_name)
    
    for feature_num in range(X_test_total.shape[1]):
        
        if feature_name[feature_num] == 'Demo3_1' or feature_name[feature_num] == 'Demo3_2':
            
            score_select_female = global_score_record.loc[:,'Demo3_1']
            score_select_male = global_score_record.loc[:,'Demo3_2']
            score_new = final_global_score - score_select_male - score_select_female
            
        else:
            
            score_select = global_score_record.loc[:,feature_name[feature_num]]
            score_new = final_global_score - score_select
            
        auc_new = roc_auc_score(global_y_record['Label'],score_new)
        auc_gap = auc_global-auc_new
        auc_gain.loc[feature_name[feature_num],'auc_gain'] = auc_gap
        
    #auc_gain.to_csv('/home/liukang/Doc/Error_analysis/global_feature_auc_gain_gender_change_{}.csv'.format(subgroup))
    
    #combine_record.to_csv('/home/liukang/Doc/Error_analysis/score_diff_with_transfer/score_diff_with_transfer_order_all_6_{}.csv'.format(disease_list.iloc[disease_num,0]))
    return tune_global_score_diff_with_avg, global_record, auc_gain, tune_global_score_diff_with_avg_test


def subgroup_analysis(subgroup):
    
    #Features' AUC gain in subgroup model
    global_score_record = pd.DataFrame()
    global_y_record = pd.DataFrame()
    X_test_total = pd.DataFrame()
    coef_record = pd.DataFrame()
    
    for data_num in range(1,5):
        
        train_data = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
        test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
        
        subgroup_true_train = train_data.loc[:,subgroup] ==1
        subgroup_true_test = test_data.loc[:,subgroup] ==1
        
        train_data = train_data.loc[subgroup_true_train]
        test_data = test_data.loc[subgroup_true_test]
        
        
        y_train = train_data['Label']
        X_train = train_data.drop(['Label'],axis=1)
        
        y_test = test_data['Label']
        X_test = test_data.drop(['Label'],axis=1)
        X_test['constant'] = 1
        X_test_total = pd.concat([X_test_total,X_test])
        
        lr = LogisticRegression(n_jobs=-1)
        lr.fit(X_train,y_train)
        
        coef = lr.coef_[0]
        intercept = lr.intercept_
        final_coef = np.append(coef,intercept)
        
        coef_record['coef{}'.format(data_num)] = final_coef
        
        score_test = X_test * final_coef
        
        global_score_record = pd.concat([global_score_record,score_test])
        global_y_record = pd.concat([global_y_record,test_data])
        
        
    
    feature_name = X_test.columns.tolist()
    coef_record['feature'] = feature_name
    coef_record.set_index('feature',inplace=True)
    
    mean_coef_record = pd.DataFrame()
    mean_coef_record['feature'] = feature_name
    mean_coef_record['coef'] = coef_record.mean(axis=1).values
    mean_coef_record.set_index('feature',inplace=True)
    male_coef = coef_record.loc['Demo3_1',:].values
    female_coef = coef_record.loc['Demo3_2',:].values
    gender_coef = male_coef - female_coef
    mean_coef_record.loc[['Demo3_1','Demo3_2'],'coef'] = np.mean(gender_coef)
    
    
    true_global_sample = global_y_record.loc[:,'Label'] == 1
    false_global_sample = global_y_record.loc[:,'Label'] == 0
    
    score_in_true = global_score_record.loc[true_global_sample]
    score_in_false = global_score_record.loc[false_global_sample]
    
    mean_score_in_true = score_in_true.mean()
    mean_score_in_false = score_in_false.mean()
    
    score_diff_global = mean_score_in_true-mean_score_in_false
    
    global_record = pd.DataFrame()
    global_record['feature'] = feature_name
    global_record['score_diff'] = score_diff_global.values
    positive_global_feature = global_record.loc[:,'score_diff'] > 0
    select_global_feature = global_record.loc[positive_global_feature]
    global_sample_after_fs = global_score_record.loc[:,select_global_feature['feature']]
    
    
    auc_global = roc_auc_score(global_y_record['Label'],global_score_record.sum(axis=1))
    auc_global_fs = roc_auc_score(global_y_record['Label'],global_sample_after_fs.sum(axis=1))
    
    gender_score_true = mean_score_in_true['Demo3_1'] + mean_score_in_true['Demo3_2']
    gender_score_false = mean_score_in_false['Demo3_1'] + mean_score_in_false['Demo3_2']
    gender_score_diff = gender_score_true - gender_score_false
    global_record.set_index('feature',inplace=True)
    global_record.loc[['Demo3_1','Demo3_2'],'score_diff'] = gender_score_diff
    #global_record.to_csv('/home/liukang/Doc/Error_analysis/subgroup_score_diff_gender_change_{}.csv'.format(subgroup))
    
    
    final_global_score = global_score_record.sum(axis=1)
    auc_global = roc_auc_score(global_y_record['Label'],final_global_score)
    
    auc_gain = pd.DataFrame(columns=['auc_gain'],index=feature_name)
    
    for feature_num in range(X_test_total.shape[1]):
        
        if feature_name[feature_num] == 'Demo3_1' or feature_name[feature_num] == 'Demo3_2':
            
            score_select_female = global_score_record.loc[:,'Demo3_1']
            score_select_male = global_score_record.loc[:,'Demo3_2']
            score_new = final_global_score - score_select_male - score_select_female
            
        else:
            
            score_select = global_score_record.loc[:,feature_name[feature_num]]
            score_new = final_global_score - score_select
            
        auc_new = roc_auc_score(global_y_record['Label'],score_new)
        auc_gap = auc_global-auc_new
        auc_gain.loc[feature_name[feature_num],'auc_gain'] = auc_gap
    
    #auc_gain.to_csv('/home/liukang/Doc/Error_analysis/subgroup_feature_auc_gain_gender_change_{}.csv'.format(subgroup))
    return mean_coef_record, global_record, auc_gain


def subgroup_analysis_best_C(subgroup):
    
    #select best C
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_5.csv')
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_5.csv')
    
    train_feature_true=train_ori.loc[:,subgroup]>0
    train_meaningful_sample=train_ori.loc[train_feature_true]
    X_train=train_meaningful_sample.drop(['Label'], axis=1)
    y_train=train_meaningful_sample['Label']
    
    test_feature_true=test_ori.loc[:,disease_list.iloc[disease_num,0]]>0
    test_meaningful_sample=test_ori.loc[test_feature_true]
    X_test=test_meaningful_sample.drop(['Label'], axis=1)
    y_test=test_meaningful_sample['Label']
    
    AUROC_and_C_record = pd.DataFrame()
    
    for round_num in range(100):
        
        C_now = 1-0.01*round_num
        
        lr_DG_ori=LogisticRegression(n_jobs=-1,C=C_now)
        lr_DG_ori.fit(X_train, y_train)
        
        AUROC_and_C_record.loc[round_num,'C'] = C_now
        AUROC_and_C_record.loc[round_num,'AUC']=roc_auc_score(y_test, lr_DG_ori.predict_proba(X_test)[:, 1])
        
    max_AUROC = np.amax(AUROC_and_C_record.loc[:,'AUC'])
    select_model = AUROC_and_C_record.loc[:,'AUC'] == max_AUROC
    select_C = np.mean(AUROC_and_C_record.loc[select_model,'C'].values)
    
    #Features' AUC gain in subgroup model
    global_score_record = pd.DataFrame()
    global_y_record = pd.DataFrame()
    X_test_total = pd.DataFrame()
    coef_record = pd.DataFrame()
    
    for data_num in range(1,5):
        
        train_data = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
        test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
        
        subgroup_true_train = train_data.loc[:,subgroup] ==1
        subgroup_true_test = test_data.loc[:,subgroup] ==1
        
        train_data = train_data.loc[subgroup_true_train]
        test_data = test_data.loc[subgroup_true_test]
        
        
        y_train = train_data['Label']
        X_train = train_data.drop(['Label'],axis=1)
        
        y_test = test_data['Label']
        X_test = test_data.drop(['Label'],axis=1)
        X_test['constant'] = 1
        X_test_total = pd.concat([X_test_total,X_test])
        
        lr = LogisticRegression(n_jobs=-1,C=select_C)
        lr.fit(X_train,y_train)
        
        coef = lr.coef_[0]
        intercept = lr.intercept_
        final_coef = np.append(coef,intercept)
        
        coef_record['coef{}'.format(data_num)] = final_coef
        
        score_test = X_test * final_coef
        
        global_score_record = pd.concat([global_score_record,score_test])
        global_y_record = pd.concat([global_y_record,test_data])
        
        
    
    feature_name = X_test.columns.tolist()
    coef_record['feature'] = feature_name
    coef_record.set_index('feature',inplace=True)
    
    mean_coef_record = pd.DataFrame()
    mean_coef_record['feature'] = feature_name
    mean_coef_record['coef'] = coef_record.mean(axis=1).values
    mean_coef_record.set_index('feature',inplace=True)
    male_coef = coef_record.loc['Demo3_1',:].values
    female_coef = coef_record.loc['Demo3_2',:].values
    gender_coef = male_coef - female_coef
    mean_coef_record.loc[['Demo3_1','Demo3_2'],'coef'] = np.mean(gender_coef)
    
    
    true_global_sample = global_y_record.loc[:,'Label'] == 1
    false_global_sample = global_y_record.loc[:,'Label'] == 0
    
    score_in_true = global_score_record.loc[true_global_sample]
    score_in_false = global_score_record.loc[false_global_sample]
    
    mean_score_in_true = score_in_true.mean()
    mean_score_in_false = score_in_false.mean()
    
    score_diff_global = mean_score_in_true-mean_score_in_false
    
    global_record = pd.DataFrame()
    global_record['feature'] = feature_name
    global_record['score_diff'] = score_diff_global.values
    positive_global_feature = global_record.loc[:,'score_diff'] > 0
    select_global_feature = global_record.loc[positive_global_feature]
    global_sample_after_fs = global_score_record.loc[:,select_global_feature['feature']]
    
    
    auc_global = roc_auc_score(global_y_record['Label'],global_score_record.sum(axis=1))
    auc_global_fs = roc_auc_score(global_y_record['Label'],global_sample_after_fs.sum(axis=1))
    
    gender_score_true = mean_score_in_true['Demo3_1'] + mean_score_in_true['Demo3_2']
    gender_score_false = mean_score_in_false['Demo3_1'] + mean_score_in_false['Demo3_2']
    gender_score_diff = gender_score_true - gender_score_false
    global_record.set_index('feature',inplace=True)
    global_record.loc[['Demo3_1','Demo3_2'],'score_diff'] = gender_score_diff
    #global_record.to_csv('/home/liukang/Doc/Error_analysis/subgroup_score_diff_gender_change_{}.csv'.format(subgroup))
    
    
    final_global_score = global_score_record.sum(axis=1)
    auc_global = roc_auc_score(global_y_record['Label'],final_global_score)
    
    auc_gain = pd.DataFrame(columns=['auc_gain'],index=feature_name)
    
    for feature_num in range(X_test_total.shape[1]):
        
        if feature_name[feature_num] == 'Demo3_1' or feature_name[feature_num] == 'Demo3_2':
            
            score_select_female = global_score_record.loc[:,'Demo3_1']
            score_select_male = global_score_record.loc[:,'Demo3_2']
            score_new = final_global_score - score_select_male - score_select_female
            
        else:
            
            score_select = global_score_record.loc[:,feature_name[feature_num]]
            score_new = final_global_score - score_select
            
        auc_new = roc_auc_score(global_y_record['Label'],score_new)
        auc_gap = auc_global-auc_new
        auc_gain.loc[feature_name[feature_num],'auc_gain'] = auc_gap
    
    #auc_gain.to_csv('/home/liukang/Doc/Error_analysis/subgroup_feature_auc_gain_gender_change_{}.csv'.format(subgroup))
    return mean_coef_record, global_record, auc_gain


def TLsubgroup_analysis(subgroup):
    
    #Features' AUC gain in subgroup&TL model
    global_score_record = pd.DataFrame()
    global_y_record = pd.DataFrame()
    X_test_total = pd.DataFrame()
    coef_record = pd.DataFrame()
    
    for data_num in range(1,5):
        
        train_data = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
        test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
        
        X_train_global = train_data.drop('Label',axis=1)
        y_train_global = train_data['Label']
        
        X_test_global = test_data.drop('Label',axis=1)
        y_test_global = test_data['Label']
        
        lr_global = LogisticRegression(n_jobs=-1)
        lr_global.fit(X_train_global,y_train_global)
        
        global_coef = lr_global.coef_[0]
        transfer_middle_weight = np.append(global_coef,1)
        
        subgroup_true_train = train_data.loc[:,subgroup] ==1
        subgroup_true_test = test_data.loc[:,subgroup] ==1
        
        train_data = train_data.loc[subgroup_true_train]
        test_data = test_data.loc[subgroup_true_test]
        
        y_train = train_data['Label']
        X_train = train_data.drop(['Label'],axis=1)
        
        X_train = X_train * global_coef
        
        y_test = test_data['Label']
        X_test = test_data.drop(['Label'],axis=1)
        X_test['constant'] = 1
        X_test_total = pd.concat([X_test_total,X_test])
        
        lr = LogisticRegression(n_jobs=-1)
        lr.fit(X_train,y_train)
        
        coef = lr.coef_[0]
        intercept = lr.intercept_
        final_coef = np.append(coef,intercept)
        
        coef_record['coef{}'.format(data_num)] = transfer_middle_weight * final_coef
        
        score_test = X_test * transfer_middle_weight * final_coef
        
        global_score_record = pd.concat([global_score_record,score_test])
        global_y_record = pd.concat([global_y_record,test_data])
        
        
    feature_name = X_test.columns.tolist()
    coef_record['feature'] = feature_name
    coef_record.set_index('feature',inplace=True)
    
    mean_coef_record = pd.DataFrame()
    mean_coef_record['feature'] = feature_name
    mean_coef_record['coef'] = coef_record.mean(axis=1).values
    mean_coef_record.set_index('feature',inplace=True)
    male_coef = coef_record.loc['Demo3_1',:].values
    female_coef = coef_record.loc['Demo3_2',:].values
    gender_coef = male_coef - female_coef
    mean_coef_record.loc[['Demo3_1','Demo3_2'],'coef'] = np.mean(gender_coef)
    
    true_global_sample = global_y_record.loc[:,'Label'] == 1
    false_global_sample = global_y_record.loc[:,'Label'] == 0
    
    score_in_true = global_score_record.loc[true_global_sample]
    score_in_false = global_score_record.loc[false_global_sample]
    
    mean_score_in_true = score_in_true.mean()
    mean_score_in_false = score_in_false.mean()
    
    score_diff_global = mean_score_in_true-mean_score_in_false
    
    global_record = pd.DataFrame()
    global_record['feature'] = feature_name
    global_record['score_diff'] = score_diff_global.values
    positive_global_feature = global_record.loc[:,'score_diff'] > 0
    select_global_feature = global_record.loc[positive_global_feature]
    global_sample_after_fs = global_score_record.loc[:,select_global_feature['feature']]
    
    
    auc_global = roc_auc_score(global_y_record['Label'],global_score_record.sum(axis=1))
    auc_global_fs = roc_auc_score(global_y_record['Label'],global_sample_after_fs.sum(axis=1))
    
    gender_score_true = mean_score_in_true['Demo3_1'] + mean_score_in_true['Demo3_2']
    gender_score_false = mean_score_in_false['Demo3_1'] + mean_score_in_false['Demo3_2']
    gender_score_diff = gender_score_true - gender_score_false
    global_record.set_index('feature',inplace=True)
    global_record.loc[['Demo3_1','Demo3_2'],'score_diff'] = gender_score_diff
    #global_record.to_csv('/home/liukang/Doc/Error_analysis/TLsubgroup_score_diff_gender_change_{}.csv'.format(subgroup))
    
    
    final_global_score = global_score_record.sum(axis=1)
    auc_global = roc_auc_score(global_y_record['Label'],final_global_score)
    
    auc_gain = pd.DataFrame(columns=['auc_gain'],index=feature_name)
    
    for feature_num in range(X_test_total.shape[1]):
        
        if feature_name[feature_num] == 'Demo3_1' or feature_name[feature_num] == 'Demo3_2':
            
            score_select_female = global_score_record.loc[:,'Demo3_1']
            score_select_male = global_score_record.loc[:,'Demo3_2']
            score_new = final_global_score - score_select_male - score_select_female
            
        else:
            
            score_select = global_score_record.loc[:,feature_name[feature_num]]
            score_new = final_global_score - score_select
            
        auc_new = roc_auc_score(global_y_record['Label'],score_new)
        auc_gap = auc_global-auc_new
        auc_gain.loc[feature_name[feature_num],'auc_gain'] = auc_gap
        
    #auc_gain.to_csv('/home/liukang/Doc/Error_analysis/TLsubgroup_feature_auc_gain_gender_change_{}.csv'.format(subgroup))
    return mean_coef_record, global_record, auc_gain


#MAIN PROCESS
disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')

X_train_global = pd.DataFrame()
X_test_global = pd.DataFrame()

for data_num in range(1,5):
    
    train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
    X_train_global = pd.concat([X_train_global,train_df])
    
    test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    test_df.drop('Label', axis=1, inplace=True)
    test_df['constant'] = 1
    X_test_global = pd.concat([X_test_global,test_df])

y_train_all = train_df['Label']
X_train_all = train_df.drop(['Label'],axis=1)

lr_ALL = LogisticRegression(n_jobs=-1)
lr_ALL.fit(X_train_all,y_train_all)

coef = lr_ALL.coef_[0]
intercept = lr_ALL.intercept_
final_coef_ALL = np.append(coef,intercept)
#final_coef_exp_ALL = np.exp(final_coef_ALL)

feature_name = X_test_global.columns.tolist()
global_feature_mean = X_test_global.mean(axis=0)
feature_mean_zero = global_feature_mean == 0
global_feature_mean[feature_mean_zero] = 0.00001

global_in_general_score_diff = pd.read_csv("/home/liukang/Doc/Error_analysis/global_score_diff_gender_change.csv")
global_mean_score_in_true = pd.read_csv('/home/liukang/Doc/Error_analysis/global_score_in_true.csv',index_col=0)
global_mean_score_in_false = pd.read_csv('/home/liukang/Doc/Error_analysis/global_score_in_false.csv',index_col=0)

#feature_free_risk_left_side = 1/ (global_feature_mean * (final_coef_exp_ALL - 1) + 1)
#feature_free_risk_right_upper = final_coef_ALL * global_feature_mean * (global_feature_mean - 1) * (1-final_coef_exp_ALL)
#feature_free_risk_right_bottom = global_in_general_score_diff.loc[:,'score_diff'].values * ((global_feature_mean * (final_coef_exp_ALL - 1) +1)**2)

#feature_free_risk = feature_free_risk_left_side - (feature_free_risk_right_upper / feature_free_risk_right_bottom)

global_total_score_diff_tune_avg = pd.DataFrame(index=feature_name)
global_total_score_diff = pd.DataFrame(index=feature_name)
global_total_AUC_gain = pd.DataFrame(index=feature_name)
PMTL_total_coef = pd.DataFrame(index=feature_name)
PMTL_total_score_diff = pd.DataFrame(index=feature_name)
PMTL_total_AUC_gain = pd.DataFrame(index=feature_name)
#PM_total_coef = pd.DataFrame(index=feature_name)
#PM_total_score_diff = pd.DataFrame(index=feature_name)
#PM_total_AUC_gain = pd.DataFrame(index=feature_name)
#subgroup_total_coef = pd.DataFrame(index=feature_name)
#subgroup_total_score_diff = pd.DataFrame(index=feature_name)
#subgroup_total_AUC_gain = pd.DataFrame(index=feature_name)
subgroup_best_total_coef = pd.DataFrame(index=feature_name)
subgroup_best_total_score_diff = pd.DataFrame(index=feature_name)
subgroup_best_total_AUC_gain = pd.DataFrame(index=feature_name)
#TLsubgroup_total_coef = pd.DataFrame(index=feature_name)
#TLsubgroup_total_score_diff = pd.DataFrame(index=feature_name)
#TLsubgroup_total_AUC_gain = pd.DataFrame(index=feature_name)

for disease_num in range(20):
    
    disease = disease_list.loc[disease_num,'Drg']
    
    PMTL_coef, PMTL_score_diff,  PMTL_AUC_gain = PMTL_analysis(disease)
    PMTL_total_coef[disease] = PMTL_coef['coef']
    PMTL_total_score_diff[disease] = PMTL_score_diff['score_diff']
    PMTL_total_AUC_gain[disease] = PMTL_AUC_gain['auc_gain']
    
    #PM_coef, PM_score_diff,  PM_AUC_gain = PM_analysis(disease)
    #PM_total_coef[disease] = PM_coef['coef']
    #PM_total_score_diff[disease] = PM_score_diff['score_diff']
    #PM_total_AUC_gain[disease] = PM_AUC_gain['auc_gain']
    
    #subgroup_coef, subgroup_score_diff,  subgroup_AUC_gain = subgroup_analysis(disease)
    #subgroup_total_coef[disease] = subgroup_coef['coef']
    #subgroup_total_score_diff[disease] = subgroup_score_diff['score_diff']
    #subgroup_total_AUC_gain[disease] = subgroup_AUC_gain['auc_gain']
    
    #subgroup_best_coef, subgroup_best_score_diff,  subgroup_best_AUC_gain = subgroup_analysis_best_C(disease)
    #subgroup_best_total_coef[disease] = subgroup_best_coef['coef']
    #subgroup_best_total_score_diff[disease] = subgroup_best_score_diff['score_diff']
    #subgroup_best_total_AUC_gain[disease] = subgroup_best_AUC_gain['auc_gain']
    
    #TLsubgroup_coef, TLsubgroup_score_diff,  TLsubgroup_AUC_gain = TLsubgroup_analysis(disease)
    #TLsubgroup_total_coef[disease] = TLsubgroup_coef['coef']
    #TLsubgroup_total_score_diff[disease] = TLsubgroup_score_diff['score_diff']
    #TLsubgroup_total_AUC_gain[disease] = TLsubgroup_AUC_gain['auc_gain']
    
    global_score_diff_tune_avg, global_score_diff, global_AUC_gain, test_result = global_analysis(disease)
    global_total_score_diff_tune_avg[disease] = global_score_diff_tune_avg['score_diff']
    global_total_score_diff[disease] = global_score_diff['score_diff']
    global_total_AUC_gain[disease] = global_AUC_gain['auc_gain']
    
PMTL_total_coef.to_csv('/home/liukang/Doc/Error_analysis/PMTL_coef_in_top20_subgroup.csv')
PMTL_total_score_diff.to_csv('/home/liukang/Doc/Error_analysis/PMTL_score_diff_in_top20_subgroup.csv')
PMTL_total_AUC_gain.to_csv('/home/liukang/Doc/Error_analysis/PMTL_auc_gain_in_top20_subgroup.csv')
#PM_total_coef.to_csv('/home/liukang/Doc/Error_analysis/PM_coef_in_top20_subgroup.csv')
#PM_total_score_diff.to_csv('/home/liukang/Doc/Error_analysis/PM_score_diff_in_top20_subgroup.csv')
#PM_total_AUC_gain.to_csv('/home/liukang/Doc/Error_analysis/PM_auc_gain_in_top20_subgroup.csv')
#subgroup_total_coef.to_csv('/home/liukang/Doc/Error_analysis/subgroup_coef_in_top20_subgroup.csv')
#subgroup_total_score_diff.to_csv('/home/liukang/Doc/Error_analysis/subgroup_score_diff_in_top20_subgroup.csv')
#subgroup_total_AUC_gain.to_csv('/home/liukang/Doc/Error_analysis/subgroup_auc_gain_in_top20_subgroup.csv')
#subgroup_best_total_coef.to_csv('/home/liukang/Doc/Error_analysis/subgroup_best_C_coef_in_top20_subgroup.csv')
#subgroup_best_total_score_diff.to_csv('/home/liukang/Doc/Error_analysis/subgroup_best_C_score_diff_in_top20_subgroup.csv')
#subgroup_best_total_AUC_gain.to_csv('/home/liukang/Doc/Error_analysis/subgroup_best_C_auc_gain_in_top20_subgroup.csv')
#TLsubgroup_total_coef.to_csv('/home/liukang/Doc/Error_analysis/TLsubgroup_coef_in_top20_subgroup.csv')
#TLsubgroup_total_score_diff.to_csv('/home/liukang/Doc/Error_analysis/TLsubgroup_score_diff_in_top20_subgroup.csv')
#TLsubgroup_total_AUC_gain.to_csv('/home/liukang/Doc/Error_analysis/TLsubgroup_auc_gain_in_top20_subgroup.csv')
#global_total_score_diff_tune_avg.to_csv('/home/liukang/Doc/Error_analysis/global_total_score_diff_tune_avg_in_top20_subgroup.csv')
#global_total_score_diff.to_csv('/home/liukang/Doc/Error_analysis/global_score_diff_in_top20_subgroup.csv')
#global_total_AUC_gain.to_csv('/home/liukang/Doc/Error_analysis/global_auc_gain_in_top20_subgroup.csv')


