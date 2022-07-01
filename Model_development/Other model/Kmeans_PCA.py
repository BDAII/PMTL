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
                
                X_train, y_train = Label_data.loc[:, :'CCS279'], train_y[Label_data.index.tolist()]
                X_test = test_data.loc[:, :'CCS279']
                
                #----pca----
                pca_ori_train = X_train.copy()
                pca_y = y_train.copy()
                pca_ori_test = X_test.copy()
                test_idx = test_data.index.tolist()
                
                pca_mat_train = X_train.copy()
                pca_mat_test = X_test.copy()
                pca_mat_train *= list(Wi['Feature_importance'])
                pca_mat_test *= list(Wi['Feature_importance'])
                
                pca_ori = PCA(n_components=0.95)
                pca_mat = PCA(n_components=0.95)
                
                pca_ori_train = pca_ori.fit_transform(pca_ori_train)
                pca_mat_train = pca_mat.fit_transform(pca_mat_train)
                
                pca_ori_test = pca_ori.transform(pca_ori_test)
                pca_mat_test = pca_mat.transform(pca_mat_test)
                
                lr_ori = LogisticRegression(n_jobs=-1)
                lr_mat = LogisticRegression(n_jobs=-1)
                
                lr_ori.fit(pca_ori_train, pca_y)
                test_y.loc[test_idx, 'ori_Kmeans_{}'.format(k)] = lr_ori.predict_proba(pca_ori_test)[:, 1]
                
                lr_mat.fit(pca_mat_train, pca_y)
                test_y.loc[test_idx, 'mat_Kmeans_{}'.format(k)] = lr_mat.predict_proba(pca_mat_test)[:, 1]
            #group_name='group_{}'.format(k)
            #mat_group_name='mat_group_{}'.format(k)
            #AUC_record.loc[0,group_name]=roc_auc_score(test_y['Label'],test_y['ori_Kmeans_PCA'])
            #AUC_record.loc[0,mat_group_name]=roc_auc_score(test_y['Label'],test_y['mat_Kmeans_PCA'])
            #test_y.loc[:, ['Label','ori_Kmeans_PCA', 'mat_Kmeans_PCA']].to_csv('/home/liukang/Doc/feature_selection/{}/PCA_Kmeans_ori_chi2_200_.csv'.format(k), index=False)
            test_X.drop('cluster_label',axis=1,inplace=True)
            train_X.drop('cluster_label',axis=1,inplace=True)
        test_result=test_y
        result_concat=pd.concat([result_concat,test_result])
    for cluster_num in [5,10,20]:
        AUC_record.loc[round_num,'ori_Kmeans_{}'.format(cluster_num)]=roc_auc_score(result_concat['Label'],result_concat['ori_Kmeans_{}'.format(cluster_num)])
        AUC_record.loc[round_num,'mat_Kmeans_{}'.format(cluster_num)]=roc_auc_score(result_concat['Label'],result_concat['mat_Kmeans_{}'.format(cluster_num)])
AUC_record.to_csv('/home/liukang/Doc/feature_selection/test/Kmeans_PCA_AUC.csv', index=False)
            