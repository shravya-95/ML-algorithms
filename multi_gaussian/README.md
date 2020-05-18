Author: Krishna Shravya Gade </br>
Date: 03.06.2020 </br>

## How to run:
1.Make sure latest sci-kit version is installed using pip3 </br>
2.Place all the 3 folders in the current working directory of python 3 </br>
3.Run the main file using the below commands: </br>
	python3 main.py

## Description

We will develop two parametric classifiers by modeling each class’s conditional distribution p(x|Ci) as multivariate Gaussians with (a) full covariance matrix Σi and (b) diagonal covariance matrix Σi. In particular, using the training data, we will compute the maximum likelihood estimate of the class prior probabilities p(Ci) and the class conditional probabilities p(x|Ci) based on the maximum likelihood estimates of the mean ˆµi and the (full/diagonal) covariance ˆΣi for each class Ci. The classification will be done based on the following discriminant function: </br>
gi(x) = log p(Ci) + log p(x|Ci) </br>
We will develop code for a class MultiGaussClassify with two key functions: MultiGaussClassify.fit(self,X,y,diag) and MultiGaussClassify.predict(self,X). For fit(self,X,y,diag), the inputs (X, y) are respectively the feature matrix and class labels, and diag is boolean (TRUE or FALSE) which indicates whether the estimated class covariance matrices should be a full matrix (diag=FALSE) or a diagonal matrix (diag=TRUE). For predict(X), the input X is the feature matrix corresponding to the test set and the output should be the predicted labels for each point in the test set. For the class, the init (self,k,d) function can initialize the parameters for each class to be uniform prior, zero mean, and identity covariance, i.e., p(Ci) = 1/k, µi = 0 and Σi = I, i = 1, . . . , k. Here, the number of classes k and the dimensionality d of features is passed as an argument to the constructor of MultiGaussClassify.