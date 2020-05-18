# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:00:54 2020

@author: Shravya Gade
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

from my_train_test import my_train_test
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
        
print("  1. LinearSVC with Boston50  \n")
my_train_test("LSVC",X,Y50,0.75,10)
print(" 2. LinearSVC with Boston75  \n")
my_train_test("LSVC",X,Y75,0.75,10)
print(" 3. LinearSVC with Digits \n")
my_train_test("LSVC",Xdigits,Ydigits,0.75,10)
print(" 4. SVC with Boston50 \n")
my_train_test("SVC",X,Y50,0.75,10)
print("5. SVC with Boston75 \n")
my_train_test("SVC",X,Y75,0.75,10)

print("6. SVC with Digits")
my_train_test("SVC",Xdigits,Ydigits,0.75,10)

print("7. Logistic Regression with Boston50 \n")
my_train_test("LR",X,Y50,0.75,10)
print("8. Logistic Regression with Boston75 \n")
my_train_test("LR",X,Y75,0.75,10)

print("9. Logistic Regression with Digits \n")
my_train_test("LR",Xdigits,Ydigits,0.75,10)


