# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 01:48:22 2020

@author: Shravya Gade
"""

import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from MyLogisticReg2 import MyLogisticReg2

from statistics import mean 
from statistics import stdev 



# -*- coding: utf-8 -*-




def my_cross_val(method,X,Y,k):
    
    index=[i for i in range(X.shape[0])]
    
    
    random.shuffle(index)
        
    size=int(X.shape[0]/k)
    
    start=0
    stop=size
    error=[]
    #Logistric Regression method
    if method =='LogisticRegression':
        for i in range(0,k):
            e=0
            train=index[0:start]+index[stop:]
            test=index[start:stop]
            Xtrain=np.array([X[i] for i in train])
            Ytrain=np.array([Y[i] for i in train])
            myLR = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000).fit(Xtrain,Ytrain)
            Xtest=np.array([X[i] for i in test])
            Ytest=np.array([Y[i] for i in test])
            Ypred = myLR.predict(Xtest)
            
            for yp,yt in zip(Ypred,Ytest):
                if yp!=yt:
                    e+=1
            err=e/np.shape(Ypred)[0]
            error.append(err)
            print("Fold ",i+1,":",err)
            start+=size
            stop+=size
        print("Mean: ",mean(error),"\n Standard Deviation: ", stdev(error))
    #SVC method
    
    elif method=='MyLogisticReg2':
        d=np.shape(X)[1]
        myLR=MyLogisticReg2(d)
        for i in range(0,k):
            e=0
            train=index[0:start]+index[stop:]
            test=index[start:stop]
            Xtrain=np.array([X[i] for i in train])
            Ytrain=np.array([Y[i] for i in train])
            myLR.fit(Xtrain,Ytrain)
            Xtest=np.array([X[i] for i in test])
            Ytest=np.array([Y[i] for i in test])
            Ypred=myLR.predict(Xtest)
            for yp,yt in zip(Ypred,Ytest):
                if yp!=yt:
                    e+=1
            err=e/np.shape(Ypred)[0]
            error.append(err)
            print("Fold ",i+1,":",err)
            start+=size
            stop+=size
        print("Mean: ",mean(error),"\n Standard Deviation: ", stdev(error))

    
        
        