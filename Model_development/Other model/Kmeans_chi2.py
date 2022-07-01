#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:09:41 2019

@author: shuxinyu
"""
import numpy as np
import pandas as pd

from sklearn.feature_selection import chi2, VarianceThreshold, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

AUC_record=pd.DataFrame()
for round_num in range(5):
    result_concat=pd.DataFrame()
    for test_num in [5]:
        train_df = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(test_num))
        test_df = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(test_num))
        
        Lr = LogisticRegression(n_jobs=-1)
        X_train, y_train = train_df.loc[:, :'CCS279'], train_df['Label']
        X_test, y_test = test_df.loc[:, :'CCS279'], test_df['Label']
        
        Lr.fit(X_train, y_train)
        Wi = pd.DataFrame({'col_name':X_train.columns.tolist(), 'Feature_importance':Lr.coef_[0]})
        
        train_X=X_train
        train_y=y_train
        test_X=X_test
        test_y=pd.DataFrame(y_test)
        for k in [5,10,20]:
            cluster_Kmeans = KMeans(n_clusters=k, n_jobs=-1)
            train_X['cluster_label'] = cluster_Kmeans.fit_predict(train_X * Wi['Feature_importance'].tolist())
            test_X['cluster_label'] = cluster_Kmeans.predict(test_X * Wi['Feature_importance'].tolist())
            
            for label in range(k):
                Label_data = train_X.loc[train_X['cluster_label'] == label, :'CCS279']
                test_data = test_X.loc[test_X['cluster_label'] == label, :'CCS279']
                #----chi2_200-----
                X_train, y_tr = Label_data.loc[:, :'CCS279'], train_y[Label_data.index.tolist()]
                X_test = test_data.loc[:, :'CCS279']
                #chi2_data = Label_data.copy()
                #chi2_train = chi2_data.loc[:, :'CCS279']
                #chi2_y = chi2_data['Label']
                s = SelectKBest(chi2, k=200)
                chi2_train = s.fit_transform(X_train, y_tr)
                chi2_test = test_X.loc[test_X['cluster_label'] == label, :'CCS279']
                test_idx = chi2_test.index.tolist()
                chi2_test = s.transform(chi2_test)
                
                col_idx = [int(i) for i in s.get_support()]
                col_idx = [i+j-1 for i,j in zip(range(1921), col_idx) if j > 0]
                need_wi = Wi.iloc[col_idx, 1].tolist()
                
                lr_ori = LogisticRegression(n_jobs=-1)
                lr_ori.fit(chi2_train, y_tr)
                test_y.loc[test_idx, 'ori_Kmeans_{}'.format(k)] = lr_ori.predict_proba(chi2_test)[:, 1]
                
                lr_ori = LogisticRegression(n_jobs=-1)
                lr_ori.fit(chi2_train * need_wi, y_tr)
                test_y.loc[test_idx, 'mat_Kmeans_{}'.format(k)] = lr_ori.predict_proba(chi2_test * need_wi)[:, 1]
            #group_name='group_{}'.format(k)
            #mat_group_name='mat_group_{}'.format(k)
            #AUC_record.loc[0,group_name]=roc_auc_score(test_y['Label'],test_y['ori_chi2_200'])
            #AUC_record.loc[0,mat_group_name]=roc_auc_score(test_y['Label'],test_y['mat_chi2_200'])
            #test_y.loc[:, ['Label', 'ori_chi2_200', 'mat_chi2_200']].to_csv('/home/liukang/Doc/feature_selection/{}/FS_Kmeans_ori_chi2_200_.csv'.format(k), index=False)
            test_X.drop('cluster_label',axis=1,inplace=True)
            train_X.drop('cluster_label',axis=1,inplace=True)
        test_result=test_y
        result_concat=pd.concat([result_concat,test_result])
    for cluster_num in [5,10,20]:
        AUC_record.loc[round_num,'ori_Kmeans_{}'.format(cluster_num)]=roc_auc_score(result_concat['Label'],result_concat['ori_Kmeans_{}'.format(cluster_num)])
        AUC_record.loc[round_num,'mat_Kmeans_{}'.format(cluster_num)]=roc_auc_score(result_concat['Label'],result_concat['mat_Kmeans_{}'.format(cluster_num)])
AUC_record.to_csv('/home/liukang/Doc/feature_selection/test/Kmeans_chi_AUC.csv', index=False)