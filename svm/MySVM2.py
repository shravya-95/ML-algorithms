# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 06:13:53 2020

@author: Shravya Gade
"""

import numpy as np



class MySVM2:
    def __init__(self,d,m):
        self.d=d
        self.m=m


        self.w=np.random.uniform(low=-0.01,high=0.01,size=(d+1,))

    def coef_predict(self,x,y):
        g=0
        if y*np.matmul(self.w.T,x)<1:
            g=-y*x
        
        return g
    def fit(self,X,Y):
        lr=0.0001
        tmax=100000
        
        prev_w=self.w
        curr_cost=0
        ndl=0

        new_col=np.ones((len(X),1))
        X=np.hstack((new_col,X))
        for e in range(tmax):
            if self.m!=0:
                mini_batch=np.random.choice(len(X),self.m)
            else:
                mini_batch=range(len(X))
            for i in mini_batch:
                
                loss=self.coef_predict(X[i],Y[i])
#                print (loss)
                ndl=ndl+loss
            grad=lr*((ndl/len(X)))
            self.w=self.w-lr*grad
#            curr_cost= (self.compute_cost(self.w, X, Y))

#            if lr*grad<0.00001:
#                print ("Ended at epoch"+str(e))
#         
            prev_cost=curr_cost
            curr_cost=0
        
    def predict(self,X):
        yp=np.array(np.zeros(len(X),))
        for i in range(len(X)):
            yp[i]=(np.matmul(self.w[1:].T, X[i]) +self.w[0])
#            print (yp[i])
            yp[i]=np.sign(yp[i])
        return yp

        

