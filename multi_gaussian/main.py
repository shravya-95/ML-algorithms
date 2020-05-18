# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 01:54:37 2020

@author: Shravya Gade
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
from my_cross_val import my_cross_val

#########################LOAD DIGITS###############################
digits=load_digits()
Xdigits,Ydigits=digits.data,digits.target

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
        Y50=np.append(Y50,0)
        
Y75=np.empty((0,1))
for r in R:
    if r>=r75:
        Y75=np.append(Y75,1)
    else:
        Y75=np.append(Y75,0)
##########################CALLING CROSS VAL#############################

methods=["MultiGaussClassify","MultiGaussClassifyD","LogisticRegression"]

data=['Boston50','Boston75','Digits']
ind=1
for i in methods:
    if i=="MultiGaussClassify":
        p="MultiGaussClassify with full covariance matrix on "
    if i=="MultiGaussClassifyD":
        p="MultiGaussClassify with diagonal covariance matrix on "
    if i=="LogisticRegression":
        p="LogisticRegression with "
    for d in data:
        print(p+d)
        if d=="Boston50":
            X1=X
            Y=Y50
        if d=="Boston75":
            X1=X
            Y=Y75
        if d=="Digits":
            X1=Xdigits
            Y=Ydigits
        my_cross_val(i,X1,Y,5)
            
            

        
        
    
