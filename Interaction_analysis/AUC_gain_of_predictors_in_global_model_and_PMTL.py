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

#calculate predictor AUC gain in PMTL
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
X_test_total['intercept'] = 1
person_coef_total['intercept'] = person_result_total.loc[:,'update_1921_mat_intercept']

feature_name = X_test_total.columns.tolist()
auc_gain = pd.DataFrame(columns=['auc_gain'],index=feature_name)

score_total = X_test_total * person_coef_total

per_ori_record = pd.DataFrame()

score_in_ori_per_model = score_total
    
Label_data = test_total
 
true_sample = Label_data['Label'] == 1
false_sample = Label_data['Label'] == 0

score_in_ori_per_true = score_in_ori_per_model.loc[true_sample]
score_in_ori_per_false = score_in_ori_per_model.loc[false_sample]

mean_score_in_ori_per_true = score_in_ori_per_true.mean()
mean_score_in_ori_per_false = score_in_ori_per_false.mean()

score_diff_ori_per = mean_score_in_ori_per_true-mean_score_in_ori_per_false

per_ori_record['ori_per_feature'] = feature_name
per_ori_record['score_diff_ori_per'] = score_diff_ori_per.values
positive_feature = per_ori_record.loc[:,'score_diff_ori_per'] >= 0
select_feature = per_ori_record.loc[positive_feature]
sample_after_fs = score_in_ori_per_model.loc[:,select_feature['ori_per_feature']]

final_per_score = score_in_ori_per_model.sum(axis=1)
auc_ori = roc_auc_score(Label_data['Label'],final_per_score)
auc_fs = roc_auc_score(Label_data['Label'],sample_after_fs.sum(axis=1))

gender_score_true = mean_score_in_ori_per_true['Demo3_1'] + mean_score_in_ori_per_true['Demo3_2']
gender_score_false = mean_score_in_ori_per_false['Demo3_1'] + mean_score_in_ori_per_false['Demo3_2']
gender_score_diff = gender_score_true - gender_score_false
per_ori_record.set_index('ori_per_feature',inplace=True)
per_ori_record.loc[['Demo3_1','Demo3_2'],'score_diff_ori_per'] = gender_score_diff
per_ori_record.to_csv('/home/liukang/Doc/Error_analysis/PMTL_score_diff_gender_change.csv')



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

auc_gain.to_csv('/home/liukang/Doc/Error_analysis/PMTL_feature_auc_gain.csv')
 

    
#auc_record.loc['ori',0] = auc_ori
#auc_record.loc['fs',0] = auc_fs

#calculate predictor AUC gain in global model
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

true_global_sample = global_y_record.loc[:,'Label'] == 1
false_global_sample = global_y_record.loc[:,'Label'] == 0

score_in_true = global_score_record.loc[true_global_sample]
score_in_false = global_score_record.loc[false_global_sample]

mean_score_in_true = score_in_true.mean()
mean_score_in_false = score_in_false.mean()

mean_score_in_true_csv = pd.DataFrame(index=feature_name)
mean_score_in_false_csv = pd.DataFrame(index=feature_name)
mean_score_in_true_csv['score'] = mean_score_in_true.values
mean_score_in_false_csv['score'] = mean_score_in_false.values
mean_score_in_true_csv.to_csv('/home/liukang/Doc/Error_analysis/global_score_in_true.csv')
mean_score_in_false_csv.to_csv('/home/liukang/Doc/Error_analysis/global_score_in_false.csv')


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
global_record.to_csv('/home/liukang/Doc/Error_analysis/global_score_diff_gender_change.csv')



final_global_score = global_score_record.sum(axis=1)
auc_global = roc_auc_score(global_y_record['Label'],final_global_score)

feature_name = X_test_total.columns.tolist()
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
    
auc_gain.to_csv('/home/liukang/Doc/Error_analysis/global_feature_auc_gain.csv')

    #combine_record.to_csv('/home/liukang/Doc/Error_analysis/score_diff_with_transfer/score_diff_with_transfer_order_all_6_{}.csv'.format(disease_list.iloc[disease_num,0]))
