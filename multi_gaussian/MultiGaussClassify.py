# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:14:33 2020

@author: Shravya Gade
"""
import numpy as np
import math
import random

class MultiGaussClassify:
    def __init__(self,k,d):
        self.probC=np.full((k),1/k)
        self.mui=np.full((k,d),0)
        c=np.identity((d))
        self.covi=np.tile(c,(k,1)).reshape(k,d,d)
    def fit(self,X,Y,diag=False):

        def cov(ind,mu,diag):
            n=len(ind)
            d=len(X[0])
            var=np.empty([n,d])
            for i in range(n):
                var[i]=X[ind[i]]-mu
            varT=var.T
            cov=np.matmul(varT,var)/n
            if diag==True:
                cov=cov*np.identity(d)
            try:
                math.log(np.linalg.det(cov))
            except ValueError:
                e=np.identity(d)*0.05
                cov=cov+e
                
            return cov
            
    
    
        def mu(ind):
            n=len(ind)
            d=len(X[0])
            avg=np.empty(d)
            
            for j in range(d):
                mui=0
                for i in ind:
                    mui+=X[i,j]
                avg[j]=mui/n
                #print (avg)
            return avg

        pci={}
        for i,v in enumerate(Y):
            try:
                pci[v].append(i)
            except KeyError:
                pci[v]=[i]
        self.target=list(pci.keys())
        
            
        cn=len(self.target)
        #self.mui=np.empty([cn,len(X[0])])
        
        for i in range(cn):
            self.mui[i]=mu(pci[self.target[i]])
        
        #c=cov(pci[1],mui[1])
        #np.shape(c)
        #covi=np.empty([cn,64,64])
        for i in range(cn):
            self.covi[i]=cov(pci[self.target[i]],self.mui[i],diag)
        #self.probC=np.empty(cn)
        for i in range(cn):
            self.probC[i]=len(pci[self.target[i]])/np.shape(X)[0]
    def predict(self,X):
        yPred=np.empty([np.shape(X)[0]])
        k=len(self.target)
        j=0
        for x in X:
        
            gi=[]
            for i in range(k):
                a=-(math.log(np.linalg.det(self.covi[i])))/2
                b=(np.matmul(np.matmul(np.array([x-self.mui[i]]),np.linalg.inv(self.covi[i])),(np.array([(x-self.mui[i])]).T)))
                b=-(b[0][0])/2
                
                c=math.log(self.probC[i])
                g=a+b+c
                gi.append(g)
            m=max(gi)
            #ind=gi.index(m)
            
            ind=[]
            for z,v in enumerate(gi):
                if v==m:
                    ind.append(z)
            if len(ind)>1:
                random.shuffle(ind)
            
            yPred[j]=self.target[ind[0]]
            j+=1
        return yPred
                
            
            
        
        
        
        