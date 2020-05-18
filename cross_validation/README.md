Author: Krishna Shravya Gade </br>
Date: 02.17.2020 </br>

## How to run:
1. Make sure latest sci-kit version is installed using pip3 </br>
2. Place all the 3 folders in the current working directory of python 3 </br>
3. Run the main file using the below commands: </br>
	Q3 i) python3 q3i.py </br>
	Q3 ii) python3 q3ii.py </br>
	Q4 iii) python3 q4.py </br>

We will be considering three datasets (derived from two available datasets) for these assignments: </br>
(a) Boston: The Boston housing dataset comes pre-packaged with scikit-learn. The dataset
has 506 points, 13 features, and 1 target (response) variable. You can find more information
about the dataset here:
https://github.com/rupakc/UCI-Data-Analysis/tree/master/Boston Housing Dataset/Boston Housing
While the original dataset is for a regression problem, we will create two classification datasets
for the homework. Note that you only need to work with the response r to create these
classification datasets. </br>
i. Boston50: Let τ50 be the median (50th percentile) over all r (response) values. Create
a 2-class classification problem such that y = 1 if r ≥ τ50 and y = 0 if r < τ50. </br>
ii. Boston75: Let τ75 be the 75th percentile over all r (response) values. Create a 2-class
classification problem such that y = 1 if r ≥ τ75 and y = 0 if r < τ75. </br>
(b) Digits: The Digits dataset comes prepackaged with scikit-learn. The dataset has 1797
points, 64 features, and 10 classes corresponding to ten numbers 0, 1, . . . , 9. The dataset was
(likely) created from the following dataset:
http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits
The 2-class classification datasets from Boston50, Boston75, and the 10-class classification dataset
from Digits will be used in the following two problems.
