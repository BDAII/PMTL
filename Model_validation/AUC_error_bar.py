#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:11:59 2019

@author: liukang
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time


xx=np.array([1,2,3,4])
GPM_yy=np.array([0.755,0.758,0.74,0.718])
GPM_err=np.array([0.0065,0.0065,0.007,0.0075])

GPMTL_yy=np.array([0.76,0.779,0.778,0.775])
GPMTL_err=np.array([0.007,0.0066,0.0066,0.0066])

random_lr_yy=np.array([0.755,0.713,0.684,0.655])
random_lr_err=np.array([0.0065,0.0075,0.0075,0.0075])

random_lr_tf_yy=np.array([0.76,0.75,0.742,0.728])
random_lr_tf_err=np.array([0.007,0.007,0.007,0.0075])

fig=plt.figure(figsize=(15,8.034))
axes=fig.add_axes([0.1,0.1,0.8,0.8])
axes.errorbar(xx,GPM_yy,yerr=GPM_err,fmt='o',ecolor='orange',color='orange',elinewidth=3,capsize=6)
axes.errorbar(xx,GPMTL_yy,yerr=GPMTL_err,fmt='o',ecolor='orange',color='orange',elinewidth=3,capsize=6)
axes.errorbar(xx,random_lr_yy,yerr=random_lr_err,fmt='^',ecolor='deepskyblue',color='deepskyblue',elinewidth=3,capsize=6)
axes.errorbar(xx,random_lr_tf_yy,yerr=random_lr_tf_err,fmt='^',ecolor='deepskyblue',color='deepskyblue',elinewidth=3,capsize=6)
axes.plot(xx,GPM_yy,'orange',linestyle='--',marker='o',label='PM',linewidth=3,markersize=7)
axes.plot(xx,GPMTL_yy,'orange',marker='o',label='PMTL',linewidth=3,markersize=7)
axes.plot(xx,random_lr_yy,'deepskyblue',marker='^',linestyle='--',label='GM',linewidth=3,markersize=7)
axes.plot(xx,random_lr_tf_yy,'deepskyblue',marker='^',label='GMTL',linewidth=3,markersize=7)
axes.set_xlabel('Sample size',fontsize=20)
axes.set_ylabel('AUROC (95% CI)',fontsize=20)
plt.xticks(fontsize=18)
axes.set_xticks([1,2,3,4])
axes.set_xticklabels(['100%','20%','10%','5%'])
plt.yticks(fontsize=18)
plt.rc('font',family='Times New Roman') 
plt.legend(loc='lower left',fontsize=20)
plt.savefig("/home/liukang/Doc/error_2.png")
plt.show()
