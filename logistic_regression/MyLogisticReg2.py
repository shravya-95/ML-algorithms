# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 05:45:40 2020

@author: Shravya Gade
"""

import numpy as np
from math import exp

class MyLogisticReg2:
    def __init__(self,d):
        self.d=d
#        self.w0=np.random.uniform(low=-0.001,high=0.001,size=(1,))
#        self.w=np.random.uniform(low=-0.001,high=0.001,size=(d,))
        self.w0=0
        self.w=np.array(np.zeros(d,))
    def coef_predict(self,x):
        yp=self.w0
        for i in range(len(x)):
            yp+=self.w[i]*x[i]
            #print (1.0 + exp(-yp))
#            print (yp)
        return 1.0/(1.0 + exp(-yp))
    def fit(self,X,Y):
        lr=0.000005
        tmax=2000
        preverr=0
        sumerr=0
        for e in range(tmax):
            
            for i in range(len(X)):
                
                yp=self.coef_predict(X[i])
                err=Y[i] - yp
#                print (yp)
                sumerr+=err**2
                self.w0=self.w0+lr*err*yp*(1.0-yp)
                for j in range(self.d):
                    self.w[j]=self.w[j]+lr*err*yp*(1.0-yp)*X[i][j]
            #print ("Error in "+ str(e)+" round is "+str(sumerr))
            
            if preverr-sumerr>0 and preverr-sumerr<0.001:
                break
            preverr=sumerr
            sumerr=0
        
    def predict(self,X):
        yp=np.array(np.zeros(len(X),))
        for i in range(len(X)):
            yp[i]=round(self.coef_predict(X[i]))
        return yp

        
        