#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 17:38:13 2019

@author: shuxinyu
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

for k in [5, 10, 20]:
    train_data = pd.read_csv('/home/liukang/Doc/valid_df/train_5.csv')
    test_data = pd.read_csv('/home/liukang/Doc/valid_df/test_5.csv')
    
    X_train, y_train = train_data.drop('Label', axis=1), train_data['Label']
    X_test = test_data.drop('Label', axis=1)
    
    lr = LogisticRegression(n_jobs=-1)
    lr.fit(X_train, y_train)
    coef = lr.coef_[0]
    coef = [abs(i) for i in coef]
    
    cluster = KMeans(n_clusters=k, n_jobs=-1)
    cluster.fit(X_train * coef)
    
    train_data['group'] = cluster.labels_
    test_data['group'] = cluster.predict(X_test * coef)
    
    for val in range(k):
        train_X, train_y = train_data.loc[train_data['group'] == val, :'CCS279'], train_data.loc[train_data['group'] == val, 'Label']
        test_X = test_data.loc[test_data['group'] == val, :'CCS279']
        
        lr = LogisticRegression(n_jobs=-1)
        lr.fit(train_X, train_y)
        
        test_data.loc[test_X.index.tolist(), 'cluster_predict'] = lr.predict_proba(test_X)[:, 1]
        
    test_data.loc[:, ['Label', 'cluster_predict', 'group']].to_csv('/home/liukang/Doc/feature_selection/{}/Kmeans_ori.csv'.format(k), index=False)