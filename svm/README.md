Author: Krishna Shravya Gade </br>
Date: 04-27-2020 </br>

## Instructions to run:
1) Open terminal and make sure you have python 3.6 or above installed in your machine </br>
2) Go to the path in the downloaded folder where you see "wrapper.py" </br>
2) Run command "python wrapper.py"

## Description
We will develop code for 2-class SVMs with parameters (w, w0) where w ∈ Rd, w0 ∈ R. Assume a given dataset {(xt, yt), t = 1, . . . ,N}, where xt ∈ Rd and yt ∈ {−1, 1}. Recall from our discussion in class that training SVMs involves minimizing the following objective function: </br>
f(w, w0) =
1 n
n i=1
max{0, 1 − yt(wTxt +w0)} + w2 .
λ 2
(3) </br>
We will use λ = 5 in this case. </br>


We will develop code for MySVM2 with corresponding MySVM2.fit(X,y) and MySVM2.predict(X) functions. Parameters for the model can be initialized following what you had done for MyLogisticReg2. In the fit function, the parameters will be estimated using mini-batch stochastic gradient descent with different mini-batch sizes m ≤ n. In particular, you will
(4)modify your MyLogisticReg2 code by using gradients for the SVM objective in (3) instead of the logistic regression objective in (4). Further, you will have to add the mini-batch stochastic gradient descent (SGD) functionality which, for a pre-specified mini-batch size m, picks m unique points at random to do the gradient descent in each iteration. We will run experiments with different values of m.
We will compare the performance of MySVM2 for different values of mini-batch size m with LogisticRegression1 on two datasets: Boston50 and Boston75. Recall that Boston has 506 data points, and a 5-fold cross-validation leaves n ≈ 400 points for training in each fold.2 For mini-batch SGD, we will consider three different values of m </br>

## Code details:

Using my cross val with 5-fold cross-validation, report the error rates in each fold as well as the mean and standard deviation of error rates across all folds for the four methods: MySVM2 with m = 40,m = 200, and m = n, and LogisticRegression, applied to the two 2-class classification datasets: Boston50 and Boston75. </br>
