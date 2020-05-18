# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 22:58:28 2020

@author: Shravya Gade
"""

import numpy as np

def rand_proj(X,d):


    mu , sigma = 0,1
    G=np.random.normal(mu, sigma, 2048).reshape(64,d)
    X1=np.matmul(X,G)
    
    return X1

