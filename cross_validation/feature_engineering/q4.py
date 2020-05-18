# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:19:49 2020

@author: Shravya Gade
"""
import numpy as np
from sklearn.datasets import load_digits
from rand_proj import rand_proj
from quad_proj import quad_proj
from my_cross_val import my_cross_val

#########################LOAD DIGITS###############################
digits=load_digits()
X,Y=digits.data,digits.target
X1=rand_proj(X,32)
X2=quad_proj(X)

print("  1. LinearSVC with ~X1  \n")
my_cross_val("LSVC",X1,Y,10)
print("  2. LinearSVC with ~X2  \n")
my_cross_val("LSVC",X2,Y,10)
print("  3. SVC with ~X1  \n")
my_cross_val("SVC",X1,Y,10)

print("  4. SVC with ~X2  \n")
my_cross_val("SVC",X2,Y,10)
print("  5. Logistic Regression with ~X1  \n")
my_cross_val("LR",X1,Y,10)

print("  1. Logistic Regression with ~X2  \n")
my_cross_val("LR",X2,Y,10)
