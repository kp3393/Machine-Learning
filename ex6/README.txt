(1) FOLDER FORMAT.
Problem is defined in ex6.pdf
There are three subfolders in this main folder.
a) Figures --> Consists of all the graphs and figures derived from the dataset. 
               The numbering is same as that in the pdf of the problem statement.
b) Input data --> Consisting of input dataset which comes along with the problem. 
                  Move it to the 'Python code' folder to test it.
c) Python code --> folder consisting of python code. Here there are two codes. 
                   c.1) supportVM.py --> JUST consists of all the functions.
                   c.2) part1.py --> solution to dataset ex6data1.mat to ex6data3.mat

(2) PROBLEMS FACED/ THINGS TO REMEMBER.
1. USING SVM FOR TRAINING DATASET
sklearn was used for training data set. It is a two step process to train a SVM using sklearn.
STEP 1: create an object using svm.SVC(). 
        Here mention your regularisation parameter C, kernel i.e. linear or Gaussian ('rbf' in this case)
        and gamma (sigma for Gaussian in this case)
STEP 2: use fit() method of the class to train it on the desired dataset

2. PLOTTING DECISION BOUNDARY
This is explained in detail along with supporting code in supportVM.py's visualizeBoundary() method.