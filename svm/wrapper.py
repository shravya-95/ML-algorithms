# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 06:27:40 2020

@author: Shravya Gade
"""

import numpy as np
from sklearn.datasets import load_boston
from my_cross_val import my_cross_val


########################LOAD BOSTON###################################
boston=load_boston()
X,R=boston.data,boston.target

r50=np.percentile(R,50)
r75=np.percentile(R,75)

Y50=np.empty((0,1))
for r in R:
    if r>=r50:
        Y50=np.append(Y50,1)
    else:
        Y50=np.append(Y50,-1)
        
Y75=np.empty((0,1))
for r in R:
    if r>=r75:
        Y75=np.append(Y75,1)
    else:
        Y75=np.append(Y75,-1)

##########################CALLING CROSS VAL#############################

methods=['MySVM2','LogisticRegression']

data=['Boston50','Boston75']

ind=1
   
for d in data:
    if d=="Boston50":

        X1=X
        Y=Y50
    if d=="Boston75":
       
        X1=X
        Y=Y75
    for i in methods:
        if i == "MySVM2":
            m=[40,200,0]
            for mi in m:
                p="MySVM2 with m = "+str(mi)
                print(p+" "+d)
                my_cross_val(i,X1,Y,5,mi)
        if i =="LogisticRegression":
            p="LohisticRegression with "
            print(p+d)
            m=0
            my_cross_val(i,X1,Y,5,m)
            