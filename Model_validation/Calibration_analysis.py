# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:07:48 2018

@author: liukang
"""
from decimal import Decimal
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#auc_drg=pd.DataFrame()
#auc_total=pd.DataFrame()
brier_score_record = pd.DataFrame()
brier_score_se_record = pd.DataFrame()
brier_score_compare_z_record = pd.DataFrame()
disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20_no_drg.csv')
#drg_list=pd.read_csv('/home/liukang/Doc/drg_list.csv')
result=pd.read_csv("/home/liukang/Doc/calibration/test_result_10_No_Com.csv")
#result=pd.read_csv("/home/liukang/Doc/calibration/test_result_10_No_Com_without_top_subgroup_AKI50.csv")
#result['drg']=drg_list.iloc[:,0]
group_num = 10
round_num = 1000

#compute calibration in global patients
plt.figure(figsize=(12, 9.5))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#ax2 = plt.subplot2grid((3, 1), (2, 0))

#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

result_list = result.columns.tolist()[1:4]

for result_type in result_list:
    
    observation, prediction = calibration_curve(result['Label'], result[result_type], n_bins=group_num, strategy='quantile')
    
    brier_score = np.mean(np.square(np.array(observation - prediction)))
    brier_score_record.loc['General',result_type] = brier_score
    brier_score_short = Decimal(brier_score).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")
    
    ax1.plot(prediction, observation, "s-", label="%s = %s" % (result_type, brier_score_short))
    
    #ax2.hist(result[result_type], range=(0, 1), bins=10, label=result_type, histtype="step", lw=2)
    
ax1.set_ylabel("Fraction of positives",fontsize=20)
ax1.set_xlabel("Mean predicted value",fontsize=20)
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right",fontsize=15)
#ax1.set_title('General patients',fontsize=20)

#ax2.set_xlabel("Mean predicted value",fontsize=20)
#ax2.set_ylabel("Count",fontsize=20)
#ax2.legend(loc="upper center", ncol=2, fontsize=15)
plt.rc('font',family='Times New Roman') 

plt.tight_layout()
plt.savefig("/home/liukang/Doc/calibration/result/calibration_quantile_global_low_risk.png")
#plt.show()

boostrap_brier = pd.DataFrame()
for i in range(round_num):
    
    sample_result = result.sample(frac=1,replace=True)
    
    for result_type in result_list:
        
         observation, prediction = calibration_curve(sample_result['Label'], sample_result[result_type], n_bins=group_num, strategy='quantile')
         brier_score = np.mean(np.square(np.array(observation - prediction)))
         boostrap_brier.loc[i,result_type] = brier_score

for result_type in result_list:
    
    brier_score_se_record.loc['General',result_type] = np.std(boostrap_brier[result_type])

for i in range(len(result_list)):
    
    first_result = boostrap_brier.loc[:,result_list[i]].values
    
    for j in range(i+1,len(result_list)):
        
        second_result = boostrap_brier.loc[:,result_list[j]].values
        
        cov = np.mean(first_result * second_result) - (np.mean(first_result) * np.mean(second_result))
        brier_score_compare_z_record.loc['General','{}_V_{}'.format(result_list[i],result_list[j])] = abs(brier_score_record.loc['General',result_list[i]]-brier_score_record.loc['General',result_list[j]]) / np.sqrt((brier_score_se_record.loc['General',result_list[i]]**2)+(brier_score_se_record.loc['General',result_list[j]]**2)-(2*cov))

    
    



#compute calibration in each subgroup
#result_list = result.columns.tolist()[1:6]
result_list = ['GM','PMTL','SM']
total_drg_result = pd.DataFrame()

for disease_num in range(disease_list.shape[0]):
    
    plt.figure(figsize=(12, 9.5))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    
    #ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    #plt.xticks(fontsize=15)
    #plt.yticks(fontsize=15)
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    result_true=result.loc[:,'Drg']==disease_list.iloc[disease_num,0]
    meaningful_result=result.loc[result_true]
    total_drg_result = pd.concat([total_drg_result,meaningful_result])
    
    for result_type in result_list:
        
        observation, prediction = calibration_curve(meaningful_result['Label'], meaningful_result[result_type], n_bins=5, strategy='quantile')
        
        brier_score = np.mean(np.square(np.array(observation - prediction)))
        brier_score_record.loc[disease_list.iloc[disease_num,1],result_type] = brier_score
        brier_score_short = Decimal(brier_score).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")
        
        ax1.plot(prediction, observation, "s-", label="%s = %s" % (result_type, brier_score_short))
        #ax2.hist(meaningful_result[result_type], range=(0, 1), bins=10, label=result_type, histtype="step", lw=2)
        
    #ax1.set_ylabel("Fraction of positives",fontsize=20)
    #ax1.set_xlabel("Mean predicted value",fontsize=20)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right",fontsize=15)
    ax1.set_title('{}'.format(disease_list.iloc[disease_num,1]),fontsize=20)
    
    #ax2.set_xlabel("Mean predicted value",fontsize=20)
    #ax2.set_ylabel("Count",fontsize=20)
    #ax2.legend(loc="upper center", ncol=2,fontsize=15)
    plt.rc('font',family='Times New Roman') 
    
    plt.tight_layout()
    plt.savefig("/home/liukang/Doc/calibration/result/calibration_quantile_Drg{}.png".format(disease_list.iloc[disease_num,0]))
    #plt.show()
    
    
    boostrap_brier = pd.DataFrame()
    for i in range(round_num):
        
        sample_result = meaningful_result.sample(frac=1,replace=True)
        
        for result_type in result_list:
            
            observation, prediction = calibration_curve(sample_result['Label'], sample_result[result_type], n_bins=5, strategy='quantile')
            brier_score = np.mean(np.square(np.array(observation - prediction)))
            boostrap_brier.loc[i,result_type] = brier_score
    
    for result_type in result_list:
        
        brier_score_se_record.loc[disease_list.iloc[disease_num,1],result_type] = np.std(boostrap_brier[result_type])
    
    for i in range(len(result_list)):
        
        first_result = boostrap_brier.loc[:,result_list[i]].values
        
        for j in range(i+1,len(result_list)):
            
            second_result = boostrap_brier.loc[:,result_list[j]].values
            
            cov = np.mean(first_result * second_result) - (np.mean(first_result) * np.mean(second_result))
            brier_score_compare_z_record.loc[disease_list.iloc[disease_num,1],'{}_V_{}'.format(result_list[i],result_list[j])] = abs(brier_score_record.loc[disease_list.iloc[disease_num,1],result_list[i]]-brier_score_record.loc[disease_list.iloc[disease_num,1],result_list[j]]) / np.sqrt((brier_score_se_record.loc[disease_list.iloc[disease_num,1],result_list[i]]**2)+(brier_score_se_record.loc[disease_list.iloc[disease_num,1],result_list[j]]**2)-(2*cov))



#compute calibration in all high-risk patients
plt.figure(figsize=(12, 9.5))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

#ax2 = plt.subplot2grid((3, 1), (2, 0))

#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

for result_type in result_list:
    
    observation, prediction = calibration_curve(total_drg_result['Label'], total_drg_result[result_type], n_bins=group_num, strategy='quantile')
    
    brier_score = np.mean(np.square(np.array(observation - prediction)))
    brier_score_record.loc['All Top-20',result_type] = brier_score
    brier_score_short = Decimal(brier_score).quantize(Decimal("0.0001"), rounding = "ROUND_HALF_UP")
    
    ax1.plot(prediction, observation, "s-", label="%s = %s" % (result_type, brier_score_short))
    #ax2.hist(total_drg_result[result_type], range=(0, 1), bins=10, label=result_type, histtype="step", lw=2)
    
ax1.set_ylabel("Fraction of positives",fontsize=20)
ax1.set_xlabel("Mean predicted value",fontsize=20)
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right",fontsize=15)
#ax1.set_title('Top-20 high risk admissions',fontsize=20)

#ax2.set_xlabel("Mean predicted value",fontsize=20)
#ax2.set_ylabel("Count",fontsize=20)
#ax2.legend(loc="upper center", ncol=2,fontsize=15)
plt.rc('font',family='Times New Roman') 

plt.tight_layout()
plt.savefig("/home/liukang/Doc/calibration/result/calibration_quantile_Top20.png")
#plt.show()

boostrap_brier = pd.DataFrame()
for i in range(round_num):
    
    sample_result = total_drg_result.sample(frac=1,replace=True)
    
    for result_type in result_list:
        
         observation, prediction = calibration_curve(sample_result['Label'], sample_result[result_type], n_bins=group_num, strategy='quantile')
         brier_score = np.mean(np.square(np.array(observation - prediction)))
         boostrap_brier.loc[i,result_type] = brier_score

for result_type in result_list:
    
    brier_score_se_record.loc['All Top-20',result_type] = np.std(boostrap_brier[result_type])

for i in range(len(result_list)):
    
    first_result = boostrap_brier.loc[:,result_list[i]].values
    
    for j in range(i+1,len(result_list)):
        
        second_result = boostrap_brier.loc[:,result_list[j]].values
        
        cov = np.mean(first_result * second_result) - (np.mean(first_result) * np.mean(second_result))
        brier_score_compare_z_record.loc['All Top-20','{}_V_{}'.format(result_list[i],result_list[j])] = abs(brier_score_record.loc['All Top-20',result_list[i]]-brier_score_record.loc['All Top-20',result_list[j]]) / np.sqrt((brier_score_se_record.loc['All Top-20',result_list[i]]**2)+(brier_score_se_record.loc['All Top-20',result_list[j]]**2)-(2*cov))


#output
brier_score_record.to_csv("/home/liukang/Doc/calibration/result/brier_score_all.csv")
brier_score_se_record.to_csv("/home/liukang/Doc/calibration/result/brier_se_all.csv")
brier_score_compare_z_record.to_csv("/home/liukang/Doc/calibration/result/brier_z_all.csv")

