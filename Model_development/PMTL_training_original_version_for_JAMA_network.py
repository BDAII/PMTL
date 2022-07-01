# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: Shuxy
"""

import numpy as np
import pandas as pd
from random import shuffle
from sklearn.linear_model import LogisticRegression, LinearRegression
from gc import collect

import warnings
warnings.filterwarnings('ignore')


#read data
train_df =df = pd.read_csv('/home/liukang/Doc/valid_df/train_1.csv')
test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_1.csv')
#over

#select 10% of all sample as similar sample
for k in [0.1]:
        #train_original save ori_auc
        train_original = train_df.copy()
        test_original = test_df.copy()
        
        #learn global model use for transfer learning  
        lr_All = LogisticRegression(n_jobs=-1)
        X_train = train_original.drop(['Label'], axis=1)
        y_train = train_original['Label']
        X_test = test_original.drop(['Label'], axis=1)
        
        lr_All.fit(X_train, y_train)
        test_original['predict_proba'] = lr_All.predict_proba(X_test)[:, 1]
            
        Weight_importance = lr_All.coef_[0]
        clean_idx = 0
        
        Number_of_Iteration = 1000
        Iteration = 100
        #learning_rate = 0.1
        
        for l_rate in [0.00001]:
            #initialization of similarity measure
            ki = [abs(i) for i in Weight_importance]
            ki = [i / sum(ki) for i in ki]
            #for update weight_importance
            regularization_c=0.05
            #regularization parameters
            m_sample_ki=0.01
            #min weight of sample for logisticregression
            for k_idx in range(1, Iteration+1):
                #evaluate personalized model with current similarity measure
                last_idx = list(range(len(train_df)))
                shuffle(last_idx)
                last_data = train_df
                last_data = last_data.loc[last_idx, :]
                last_data.reset_index(drop=True, inplace=True)
                #last_data['predict_proba'] = lr_All.predict_proba(last_data.drop('Label', axis=1))[:, 1]
                
                Iteration_data = pd.DataFrame(index=range(Number_of_Iteration), columns=X_train.columns)
                y_Iteration = pd.Series(index=range(Number_of_Iteration))
                I_idx = 0
                
                select_data = last_data.loc[:Number_of_Iteration - 1, :]
                select_data.reset_index(drop=True, inplace=True)
                
                for s_idx in range(len(select_data)):
                    #similarity sample matching
                    train_rank = last_data.loc[Number_of_Iteration - 1:, :].copy()
                    pre_data = select_data.loc[s_idx, :'CCS279'] #CCS279 is the last feature in X in our data
                    mat_copy = train_rank.drop('Label', axis=1) #Label recorded AKI(1) or not(0)
                    
                    mat_copy -= pre_data                
                    mat_copy *= ki
                    
                    mat_copy = abs(mat_copy)
                    
                    train_rank['Distance'] = mat_copy.sum(axis=1)
                    train_rank.sort_values('Distance', inplace=True)
                    train_rank.reset_index(drop=True, inplace=True)
                    
                    len_split = int(len(train_rank) * k)
                    
                    train_data = train_rank.iloc[:len_split, :-1]
                    X_train = train_data.loc[:, :'CCS279']      
                    fit_train = X_train * Weight_importance
                    y_train = train_data['Label']
                    X_test = select_data.loc[[s_idx], :'CCS279']
                    
                    #transfer learning
                    fit_test = X_test * Weight_importance
                    
                    true = select_data.loc[s_idx, 'Label']
                    
                    sample_ki = train_rank.iloc[:len_split, -1].tolist()
                    sample_ki = [(sample_ki[0] + m_sample_ki) / (val + m_sample_ki) for val in sample_ki]
                    
                    #personalized modeling
                    lr = LogisticRegression(n_jobs=-1)
                    lr.fit(fit_train, y_train, sample_ki)
                    
                    proba = lr.predict_proba(fit_test)[:, 1]
                    
                    #record error of prediction and difference between target and similarity
                    X_train -= pre_data
                    X_train = abs(X_train)
                    mean_r = np.mean(X_train)
                    y = abs(true - proba)
                   
                    Iteration_data.loc[I_idx, :] = mean_r
                    y_Iteration[I_idx] = y
                    
                    I_idx += 1
                    
                    clean_idx += 1
                    if clean_idx % 10 == 0:
                        collect()
                    
                    else:
                        continue
                
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
                    table.to_csv('/home/liukang/Doc/No_Com/Ma_old_Nor_Gra_01_001_0_005/Ma_old_Nor_Gra_01_001_0_005-1_{}.csv'.format(k_idx), index=False)
                        
                else:
                    continue