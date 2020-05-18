# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:00:54 2020

@author: Shravya Gade
"""

import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from statistics import mean 
from statistics import stdev 


def my_cross_val(method,X,Y,k):
    
    index=[i for i in range(X.shape[0])]
    
    
    random.shuffle(index)
        
    size=int(X.shape[0]/k)
    
    start=0
    stop=size
    error=[]
    #Logistric Regression method
    if method in ["LR","lr","Lr"]:
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
    elif method in ["SVC","svc","Svc"]:
        mySVC=SVC(gamma='scale',C=10)
        for i in range(0,k):
            e=0
            train=index[0:start]+index[stop:]
            test=index[start:stop]
            Xtrain=np.array([X[i] for i in train])
            Ytrain=np.array([Y[i] for i in train])
            mySVC.fit(Xtrain,Ytrain)
            Xtest=np.array([X[i] for i in test])
            Ytest=np.array([Y[i] for i in test])
            Ypred=mySVC.predict(Xtest)
            for yp,yt in zip(Ypred,Ytest):
                if yp!=yt:
                    e+=1
            err=e/np.shape(Ypred)[0]
            error.append(err)
            print("Fold ",i+1,":",err)
            start+=size
            stop+=size
        print("Mean: ",mean(error),"\n Standard Deviation: ", stdev(error))
        
    #Linear SVC method    
    elif method in ["LSVC","lsvc","Lsvc"]:
        myLSVC=LinearSVC(max_iter=2000)
        for i in range(0,k):
            e=0
            train=index[0:start]+index[stop:]
            test=index[start:stop]
            Xtrain=np.array([X[i] for i in train])
            Ytrain=np.array([Y[i] for i in train])
            myLSVC.fit(Xtrain,Ytrain)
            Xtest=np.array([X[i] for i in test])
            Ytest=np.array([Y[i] for i in test])
            Ypred=myLSVC.predict(Xtest)
            for yp,yt in zip(Ypred,Ytest):
                if yp!=yt:
                    e+=1
            err=e/np.shape(Ypred)[0]
            error.append(err)
            print("Fold ",i+1,":",err)
            start+=size
            stop+=size
        print("Mean: ",mean(error),"\n Standard Deviation: ", stdev(error))
    
        
        