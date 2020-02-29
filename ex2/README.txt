(1) FOLDER FORMAT.
Problem is defined in ex2.pdf
There are three subfolders in this main folder.
a) Figures --> Consists of all the graphs and figures derived from the dataset. 
               The numbering is same as that in the pdf of the problem statement.
b) Input data --> Consisting of input dataset which comes along with the problem. 
                  Move it to the 'Python code' folder to test it.
c) Python code --> folder consisting of python code. Here there are three codes. 
                   c.1) regression.py --> JUST consists of all the functions.
                   c.2) data1.py --> solution to ex1data1.txt 
                   c.3) data2.py --> solution to ex1data2.txt

(2) PROBLEMS FACED/ THINGS TO REMEMBER.
A) function mapFeature.
--> WHAT DOES IT DO AND WHY? 
This function was used to map features to quadratic feature set. 
To fit our data better we need to create more feature from each data point.
Here we map the features into all polynomial terms of x1 and x2 up to the sixth power.
--> THINGS TO REMEMBER
a) It maintains the same number of ROWS .i.e. the sample size is same.
b) Adds columns of features.
c) First column is of ones so no need to add column of ones separately for the theta0.
d) If the degree is 6 we will have 28 (7+6+5+4+3+2+1) columns/ features.
e) Columns --> 1 x1 x2 x1^2 x1x2 x2^2 x1^3 .... x1x2^5 x2^6

B) contour plot in decision boundary.
--> Contour plot will always need three coordinates (u,v,z) in our case.
u and v are 1D arrays (microchip1 and microchip2).
z will be a function of these two. Traditionally we would have gone for meshgrid.
In this case z is a square matrix. Each element in z = (theta)'X. Where theta is 
the optimised array we obtained from optimisation and X is the mapped feature array.
So basically each point in u and v which represent out features are converted into
polynomials returning 1x28 elements. This 1x28 when multiplied with 
optimised theta shaped 28x1 we get a scalar value for a cell at i,j in square matrix z.