# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: liukang
"""
from pymare import meta_regression
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold, SelectKBest
warnings.filterwarnings('ignore')

#result_total=pd.DataFrame()

#target subgroup
feature_list=pd.read_csv('/home/liukang/Doc/Meta_regression/age_and_lab5+10.csv')

beta_record = pd.DataFrame()
var_record = pd.DataFrame()
target_record = pd.DataFrame()

for data_num in range(1,6):
    
    beta_data = pd.read_csv("/home/liukang/Doc/Meta_regression/Beta_all_{}.csv".format(data_num))
    beta_record = pd.concat([beta_record,beta_data])
    
    var_data = pd.read_csv("/home/liukang/Doc/Meta_regression/Var_all_{}.csv".format(data_num))
    var_record = pd.concat([var_record,var_data])
    
    target_data = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    target_record = pd.concat([target_record,target_data])
    
target_record.drop('Label',axis=1,inplace=True)

for feature_num in range(1):
    #disease_list.shpae[0]
    #read_data
    feature_true = target_record.loc[:,feature_list.iloc[feature_num,0]]>4
    target_select = target_record.loc[feature_true]
    
    beta_select = beta_record.loc[feature_true]
    feature_beta = beta_select['{}'.format(feature_list.iloc[feature_num,0])]
    
    var_select = var_record.loc[feature_true]
    feature_var= var_select['{}'.format(feature_list.iloc[feature_num,0])]
    
    target_select.reset_index(drop=True,inplace=True)
    feature_beta.reset_index(drop=True,inplace=True)
    feature_var.reset_index(drop=True,inplace=True)
    
    #sub_top_feature=pd.read_csv('/home/liukang/Doc/Meta_regression/sub_top_20_{}.csv'.format(disease_list.iloc[disease_num,0]))
    
    #no_med
    target_select.loc[:,'Med0':'Med1270']=0
    #X_similar.loc[:,'Med0':'Med1270']=0
    
    #no_target_feature
    #X_target.loc[:,'Drg0':'Drg314']=0
    #X_similar.loc[:,'Drg0':'Drg314']=0
    
    #select_feature
    #feature_or=np.exp(Beta_and_Var['Beta'])
    #feature_or_var=np.exp(Beta_and_Var['Beta']) * np.exp(Beta_and_Var['Beta']) * Beta_and_Var['Var']
    
    Feature_importance=pd.DataFrame(index=target_select.columns.tolist(),columns=['weight','p_value'])
    for feature_id in range(target_select.shape[1]):
        patient_count=np.sum(target_select.iloc[:,feature_id])
        if patient_count<10:
            continue
        meta_feature=meta_regression(y=feature_beta,v=feature_var,X=target_select.iloc[:,feature_id])
        #meta_feature=meta_regression(y=Beta_and_Var['Beta'],v=Beta_and_Var['Var'],X=X_target.iloc[:,feature_id])
        result_feature=meta_feature.to_df()
        Feature_importance.iloc[feature_id,0]=result_feature.loc[1,'estimate']
        Feature_importance.iloc[feature_id,1]=result_feature.loc[1,'p-value']
    
    feature_sin=Feature_importance.loc[:,'p_value']<0.01
    meaningful_feature=Feature_importance.loc[feature_sin]
    meaningful_feature['abs_weight']=meaningful_feature['weight'].abs()
    #top_feature=meaningful_feature.sort_values('p_value',ascending=True)
    #select_feature=top_feature.iloc[:,:]
    
    X_tar_patient = target_select.loc[:,meaningful_feature.index.tolist()]
    #X_tar_patient = X_target.loc[:,sub_top_feature.iloc[:10,0]]
    '''
    X_tar_patient.drop(['Med1078'],axis=1,inplace=True)
    
    X_tar_patient_corr=X_tar_patient.corr(method='pearson')
    
    pca = PCA(n_components=0.8)
    pca_train = pca.fit_transform(X_tar_patient)
    com=pca.components_
    com_e=pca.explained_variance_ratio_
    com_s=pca.singular_values_
    
    com_df=pd.DataFrame(index=range(com.shape[0]),columns=X_tar_patient.columns.tolist())
    com_df.loc[:,:]=com
    X_tar_patient_mean=X_tar_patient.mean(axis=0)
    com_mat_df=com_df * X_tar_patient_mean
    '''
    
    #Meta_regression
    result_tar=meta_regression(y=feature_beta,v=feature_var,X=X_tar_patient)
    #result_tar=meta_regression(y=Beta_and_Var['Beta'],v=Beta_and_Var['Var'],X=X_tar_patient)
    #result_sim=meta_regression(y=Beta_and_Var['Beta'],v=Beta_and_Var['Var'],X=X_sim_patient)
    result_tar_df=result_tar.to_df()
    result_tar_df['abs_z-score'] = np.abs(result_tar_df['z-score'])
    #result_sim_df=result_sim.to_df()
    #result_tar_df.to_csv("/home/liukang/Doc/Meta_regression/Meta_regression+MetaFSall_tar_no_med_result_{}.csv".format(feature_list.iloc[feature_num,0]))
    
    '''
    vp_record=pd.DataFrame(index=X_tar_patient.columns.tolist(),columns=['weight','se','p-value'],)
    for feature_num in range(X_tar_patient.shape[1]):
        #Meta_regression
        result_tar=meta_regression(y=Beta_and_Var['Beta'],v=Beta_and_Var['Var'],X=X_tar_patient.loc[:,X_tar_patient.columns.tolist()[feature_num]])
        #result_tar=meta_regression(y=Beta_and_Var['Beta'],v=Beta_and_Var['Var'],X=X_tar_patient.loc[:,X_tar_patient.columns.tolist()[feature_num]])
        #result_sim=meta_regression(y=Beta_and_Var['Beta'],v=Beta_and_Var['Var'],X=X_sim_patient)
        result_tar_single_df=result_tar.to_df()
        #result_sim_df=result_sim.to_df()
        vp_record.iloc[feature_num,0]=result_tar_single_df.loc[1,'estimate']
        vp_record.iloc[feature_num,1]=result_tar_single_df.loc[1,'se']
        vp_record.iloc[feature_num,2]=result_tar_single_df.loc[1,'p-value']
    #result_sim_df.to_csv("/home/liukang/Doc/Meta_regression/Meta_regression+MetaFS_sim_result_NoMed_{}.csv".format(disease_list.iloc[disease_num,0]))
    '''
