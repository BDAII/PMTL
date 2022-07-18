# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: Shuxy
"""

import numpy as np
import pandas as pd
import threading
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from gc import collect
from time import sleep
import os
import warnings
warnings.filterwarnings('ignore')

k=0.1
m_sample_ki=0.01
iteration = 50
val = 1921
threading_num = 30
last_X_feature = 'CCS279'
#read data
train_df = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/train-data-no-empty.csv',dtype=np.float32)
test_df = pd.read_csv('/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/data/df_data/pre_48h/test-data-no-empty.csv',dtype=np.float32)
#read similarity measure
list_dir = '/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/kang_test/result/Ma_old_Nor_Gra_mp_ver2_01_001_0_005-1_{}.csv'.format(iteration)
#over

global_lock = threading.Lock()

#feature_happened = train_df.loc[:,:'MED_9995'] > 0
#feature_happened_count = feature_happened.sum(axis=0)
#feature_sum = train_df.loc[:,:'MED_9995'].sum(axis=0)
#feature_average_if = feature_sum / feature_happened_count
#train_df.loc[:,:'MED_9995'] = train_df.loc[:,:'MED_9995'] / feature_average_if
#test_df.loc[:,:'MED_9995'] = test_df.loc[:,:'MED_9995'] / feature_average_if
#del feature_happened

#train_original = train_df.copy()
test_original = test_df.copy()

lr_All = LogisticRegression(n_jobs=-1,solver='liblinear')
global_X_train = train_df.drop(['Label'], axis=1)
global_y_train = train_df['Label']
X_test = test_df.drop(['Label'], axis=1)

lr_All.fit(global_X_train, global_y_train)
test_original['predict_proba'] = lr_All.predict_proba(X_test)[:, 1]
    
Weight_importance = lr_All.coef_[0]

#read_dir = []
#clean_idx = 0
p_weight=pd.DataFrame(index=X_test.index.tolist(),columns=X_test.columns.tolist())
X_columns_name = global_X_train.columns.tolist()

del global_y_train
del X_test

def personalized_modeling(pre_data,I_idx,X_test):
    
    #mat_copy = train_rank_X - pre_data              
    #mat_copy *= ki
    
    #mat_copy = abs(mat_copy)
    
    similar_rank = pd.DataFrame()
    
    similar_rank['data_id'] = train_df.index.tolist()
    similar_rank['Distance'] = (abs((global_X_train - pre_data) * ki)).sum(axis=1)
    
    #similar_rank['Distance'] = mat_copy.sum(axis=1)
    #mat_copy = []
    
    similar_rank.sort_values('Distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    select_id = similar_rank.loc[:len_split,'data_id'].values
    
    #print(np.sum(train_rank['Demo1']))
    
    train_data = train_df.iloc[select_id, :]
    X_train = train_data.loc[:, :last_X_feature]      
    fit_train = X_train * Weight_importance
    y_train = train_data['Label']
    
    #print('shape_{}'.format(train_data.shape[0]))
    #print(np.mean(similar_rank.iloc[:len_split, 1].values))
    #print(similar_rank.iloc[0, 1])
    
    fit_test = X_test * Weight_importance
    
    sample_ki = similar_rank.loc[:len_split,'Distance'].tolist()
    sample_ki = [(sample_ki[0] + m_sample_ki) / (dist + m_sample_ki) for dist in sample_ki]
    
    
    lr = LogisticRegression(n_jobs=-1,solver='liblinear')
    lr.fit(fit_train, y_train, sample_ki)
    
    global_lock.acquire()
    
    test_original.loc[I_idx, 'update_{}_mat_proba'.format(val)] = lr.predict_proba(fit_test)[:, 1]
    test_original.loc[I_idx, 'update_{}_mat_intercept'.format(val)] =lr.intercept_
    p_weight.iloc[I_idx,:]=lr.coef_[0]
    
    global_lock.release()





ki = pd.read_csv(list_dir)
ki = ki.iloc[:, 0].tolist()
#select_table = pd.read_csv("/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/kang_test/top-3937.csv")
#select_table['feature_name'] = X_columns_name
#need_col = select_table.loc[select_table.iloc[:, 1] == 1, 'feature_name'].tolist()

#I_idx_now = 0

len_split = int(len(train_df) * k)

threading_round = int(test_df.shape[0] / threading_num)

for threading_round_idx in range(threading_round):
    
    threadList = []
    
    for threading_num_idx in range(threading_num):
        
        s_idx = (threading_round_idx * threading_num) + threading_num_idx
        
        pre_data_select = test_df.loc[s_idx, :last_X_feature]
        #true_select = select_data.loc[s_idx, 'Label']
        X_test_select = test_df.loc[[s_idx], :last_X_feature]
        
        thread = threading.Thread(target=personalized_modeling, args=(pre_data_select,s_idx,X_test_select))
        thread.start()
        threadList.append(thread)
        #I_idx_now += 1
        
    for join_num in range(threading_num):
        
        threadList[join_num].join()
    
    collect()

finish_sample = s_idx + 1

if test_df.shape[0] != finish_sample:
    
    threadList = []
    
    for last_idx in range(finish_sample, test_df.shape[0]):
        
        pre_data_select = test_df.loc[last_idx, :last_X_feature]
        X_test_select = test_df.loc[[last_idx], :last_X_feature]
        thread = threading.Thread(target=personalized_modeling, args=(pre_data_select,last_idx,X_test_select))
        thread.start()
        threadList.append(thread)
    i=0
    for join_num in range(finish_sample, test_df.shape[0]):
        
        threadList[i].join()
        i = i+1
        
        
       
p_weight_final=p_weight * Weight_importance
#output prediction and intercept for each patients
test_original.iloc[:, -4:].to_csv('/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/kang_test/result/test_result/Ma_old_Nor_Gra_01_001_0_005-1_50.csv'.format(val, iteration), index=False)
#output coef for each patients
p_weight_final.to_csv('/panfs/pfs.local/work/liu/xzhang_sta/yuanborong/kang_test/result/test_result/weight_Ma_old_Nor_Gra_01_001_0_005-1_50.csv'.format(val, iteration), index=False)