# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 23:01:59 2020

@author: Shravya Gade
"""
import numpy as np

def quad_proj(X):
    featureLength=int(X.shape[1])
    X2sq=np.empty((0,featureLength*2))
    for x in X:
        sq=np.append(x,x*x)
        X2sq=np.append(X2sq,[sq],axis=0)
        
    X2prod=np.empty((0,(int((featureLength*(featureLength-1))/2))))
    xprod=np.empty((0,(int((featureLength*(featureLength-1))/2))))
    for x in X:
        for j,xj in enumerate(x):
            for jp in range(j+1,featureLength):
                xprod=np.append(xprod,xj*x[jp])
        X2prod=np.append(X2prod,[xprod],axis=0)
        xprod=np.empty((0,(int((featureLength*(featureLength-1))/2))))
    
    X2=np.append(X2sq,X2prod,axis=1)
    return X2