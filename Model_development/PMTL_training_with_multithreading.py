# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: Shuxy
"""
#import multiprocessing
import threading
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.linear_model import LogisticRegression, LinearRegression
from gc import collect
import sys
import warnings
warnings.filterwarnings('ignore')

fold_id = sys.argv[1]
fold_id = int(fold_id)
print('Begin {}'.format(fold_id))

#read data, check dtpye
data_path = '/home/yuanborong/new_data/pre_24h/'
saving_path = '/home/yuanborong/code/Personalized/Ma_old_Nor_Gra_01_001_0_005/result/'
train_df =df = pd.read_csv(data_path + 'train_24h_{}.csv'.format(fold_id),dtype=np.int8) #data for model training
test_df = pd.read_csv(data_path + 'test_24h_{}.csv'.format(fold_id),dtype=np.int8) #data for model testing
#delete ID
train_df.drop(['ID'] , inplace=True , axis=1)
test_df.drop(['ID'] , inplace=True , axis=1)
#over

#learning_rate = 0.1
l_rate = 0.00001
Iteration = 50
Number_of_Iteration = 1000
threading_round = 40
threading_num = 25
k = 0.1 #select 10% of all sample as similar sample
#regularization parameters
regularization_c=0.05
m_sample_ki=0.01
global_lock = threading.Lock()
#train_original save ori_auc
#train_original = train_df.copy()
#test_original = test_df.copy()

#learn global model use for transfer learning
lr_All = LogisticRegression(n_jobs=-1,solver='liblinear')
X_train = train_df.drop(['Label'], axis=1) #Label recorded AKI(1) or not(0)
y_train = train_df['Label']
X_test = test_df.drop(['Label'], axis=1)

lr_All.fit(X_train, y_train)
#test_original['predict_proba'] = lr_All.predict_proba(X_test)[:, 1]

X_columns_name = X_train.columns
del X_train
del y_train
del X_test

print('train data size {}'.format(train_df.shape) )
print('test data size {}'.format(test_df.shape))

Weight_importance = lr_All.coef_[0]
clean_idx = 0

#initialization of similarity measure
ki = [abs(i) for i in Weight_importance]
ki = [i / sum(ki) for i in ki]

def learn_similarity_measure(pre_data,true,I_idx,X_test):
    
    #mat_copy = train_rank_X - pre_data              
    #mat_copy *= ki
    
    #mat_copy = abs(mat_copy)
    
    #similarity sample matching
    similar_rank = pd.DataFrame()
    
    similar_rank['data_id'] = train_rank.index.tolist()
    similar_rank['Distance'] = (abs((train_rank_X - pre_data) * ki)).sum(axis=1)
    
    #similar_rank['Distance'] = mat_copy.sum(axis=1)
    #mat_copy = []
    
    similar_rank.sort_values('Distance', inplace=True)
    similar_rank.reset_index(drop=True, inplace=True)
    select_id = similar_rank.iloc[:len_split, 0].values
    
    #print(np.sum(train_rank['Demo1']))
    
    train_data = train_rank.iloc[select_id, :]
    X_train = train_data.loc[:, :'CCS279'] #CCS279 is the last feature in X in our data      
    fit_train = X_train * Weight_importance
    y_train = train_data['Label']
    
    #print(train_data.shape[0])
    #print(np.mean(similar_rank.iloc[:len_split, 1].values))
    #print(similar_rank.iloc[0, 1])
    
    #transfer learning
    fit_test = X_test * Weight_importance
    
    sample_ki = similar_rank.iloc[:len_split, 1].tolist()
    sample_ki = [(sample_ki[0] + m_sample_ki) / (val + m_sample_ki) for val in sample_ki]
    
    #personalized modeling
    lr = LogisticRegression(n_jobs=-1,solver='liblinear')
    lr.fit(fit_train, y_train, sample_ki)
    
    proba = lr.predict_proba(fit_test)[:, 1]
    
    #record error of prediction and difference between target and similarity
    X_train = X_train - pre_data
    X_train = abs(X_train)
    mean_r = np.mean(X_train)
    y = abs(true - proba)
    
    global Iteration_data
    global y_Iteration
    
    global_lock.acquire()
    
    Iteration_data.loc[I_idx, :] = mean_r
    y_Iteration[I_idx] = y
    
    global_lock.release()
    
    
    #print(np.sum(mean_r))
    #print(y)


#evaluate personalized model with current similarity measure
for k_idx in range(1, Iteration+1):
    
    Iteration_data = pd.DataFrame(index=range(Number_of_Iteration), columns=X_columns_name)
    y_Iteration = pd.Series(index=range(Number_of_Iteration))
    I_idx_now = 0
    
    for threading_round_idx in range(threading_round):
        
        last_idx = list(range(len(train_df)))
        shuffle(last_idx)
        last_data = train_df
        last_data = last_data.loc[last_idx, :]
        last_data.reset_index(drop=True, inplace=True)
        #last_data['predict_proba'] = lr_All.predict_proba(last_data.drop('Label', axis=1))[:, 1]
        
        select_data = last_data.loc[:threading_num - 1, :]
        select_data.reset_index(drop=True, inplace=True)
        
        train_rank = last_data.loc[threading_num:, :].copy()
        train_rank.reset_index(drop=True, inplace=True)
        train_rank_X = train_rank.drop('Label', axis=1)
        len_split = int(len(train_rank) * k)
        train_rank_dtype = train_rank.dtypes
        #ues_core = int(multiprocessing.cpu_count())
        #pool = multiprocessing.Pool(processes=ues_core)
        
        threadList = []
        
        for threading_num_idx in range(threading_num):
            
            s_idx = threading_num_idx
            
            pre_data_select = select_data.loc[s_idx, :'CCS279']
            true_select = select_data.loc[s_idx, 'Label']
            X_test_select = select_data.loc[[s_idx], :'CCS279']
            
            thread = threading.Thread(target=learn_similarity_measure, args=(pre_data_select,true_select,I_idx_now,X_test_select))
            thread.start()
            threadList.append(thread)
            I_idx_now += 1
        
        for join_num in range(threading_num):
            
            threadList[join_num].join()
            
            #pool.apply_async(learn_similarity_measure, args=(train_rank_data,pre_data_select,true_select,I_idx_now,X_test_select,))
        
        #clean_idx += 1
        #if clean_idx % 20 == 0:
        collect()
        
        #else:
            #continue
                
    #pool.close()
    #pool.join()
    
    #update similarity measure
    new_similar = Iteration_data * ki
    y_pred = new_similar.sum(axis=1)
    
    new_ki = []
    risk_gap = [real - pred for real, pred in zip(list(y_Iteration), list(y_pred))]
    for idx, value in enumerate(ki):
        features_x = list(Iteration_data.iloc[:, idx])
        plus_list = [a * b for a, b in zip(risk_gap, features_x)]
        new_value = value + l_rate * (sum(plus_list)-regularization_c*value)
        new_ki.append(new_value)
        
    new_ki = list(map(lambda x:x if x > 0 else 0, new_ki))
    ki = new_ki.copy()
    
    if k_idx % 10 == 0:
        table = pd.DataFrame({'Ma_update_{}'.format(k_idx):ki})
        #output similarity measure
        table.to_csv(saving_path + 'Ma_old_Nor_Gra_mp_ver3_01_001_0_005-{}_{}.csv'.format(fold_id , k_idx), index=False)
                        
    else:
        continue