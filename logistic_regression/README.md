Author: Krishna Shravya Gade </br>
Date: 04-14-2020 </br>

## Instructions to run:
1) Open terminal and make sure you have python 3.6 or above installed in your machine </br>
2) Go to the path in the downloaded folder where you see "q3.py" </br>
2) Run command "python wrapper.py" </br>

## Description:

We will develop code for 2-class logistic regression with one set of parameters (w, w0) where w ∈ Rd, w0 ∈ R. Assuming the two classes are {0, 1}, and the data x ∈ Rd, the posterior probability of class C1 is given by: </br>
P(1|x) = exp(wTx +w0) 1 + exp(wTx +w0) </br>
and P(0|x) = 1 − P(1|x). f We will develop code for MyLogisticReg2 with corresponding MyLogisticReg2.fit(X,y) and MyLogisticReg2.predict(X) functions. Parameters for the model can be initialized following suggestions in the textbook. In the fit function, the parameters will be estimated using gradient descent as described in the textbook and in class. We will compare the performance of MyLogisticReg2 with LogisticRegression2 on two datasets: Boston50 and Boston75. Using my cross val with 5-fold cross-validation, report the error rates in each fold as well as the mean and standard deviation of error rates across all folds for the two methods: MyLogisticReg2 and LogisticRegression, applied to the two 2-class classification datasets: Boston50 and Boston75 </br>