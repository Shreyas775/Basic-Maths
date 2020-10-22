#!/usr/bin/python
# -*- coding: latin-1 -*-
				#Linear Algebra main subject for machine Learning

#A vector is a tuple of one or more values called scalars.Vectors are built from components, which are ordinary numbers.Vectors are often represented using a lowercase character such as v; for example:   v = (v 1 , v 2 , v 3 )


#Vector is a one dimensional array. vector row = (1    vector column = (1,2,3). 2 dimension means vectors vector. Each row is one vector.
						  #2
						  #3)

print "--------------------------------Vector--------------------------------------"

import numpy as np
import math
from scipy.linalg import lu
v = np.array([1,2,3])
print "vector = ",v,"\n" 




#------------------------------------------------------------------------------------------------------------------------------------------
print "--------------------------------Vector Addition--------------------------------------"
#Vector Addition
#Two vectors of equal length can be added together to create a new third vector.c = a + b  c = (a 1 + b 1 , a 2 + b 2 , a 3 + b 3 )
#addition with vector and scalar in which each vector will be added by scalar
a = np.array([1, 2, 3])
print(a)
# define scalar
b = 2
print(b)
# broadcast
c = a + b
print c,"\n" 

'''Scalar and Two-Dimensional Array
the result of the addition with the value 2 added to each value in the array.
[[1 2 3]
[1 2 3]]
2
[[3 4 5]
[3 4 5]]'''

'''One-Dimensional and Two-Dimensional Arrays
[[1 2 3]
[1 2 3]]
[1 2 3]
[[2 4 6]
[2 4 6]]'''

'''Limitations of Broadcasting
A = array([
[1, 2, 3],
[1, 2, 3]])
print(A.shape)
# define one-dimensional array
b = array([1, 2])
print(b.shape)
# attempt broadcast
C = A + b
print(C)
(2, 3)
(2,)
ValueError: operands could not be broadcast together with shapes (2,3) (2,)'''

num1 = np.array([1,2,3])
num2 = np.array([1,2,3])
c = num1+num2
print "vector addition = ",c,"\n" 




#-----------------------------------------------------------------------------------------------------------------------------------------
print "--------------------------Vector Substraction------------------------------"
#Vector Subtraction
#scalar sub
a = np.array([1,2])
b = 2
c = a-b
print "Scalar subtraction",c

#scalar and two dimension
a = np.array([[1,2],[2,3]])
b = 2
c = a-b
print "scalar and two dimension",c

#one dimen and two dimen
a = np.array([[1,2],[1,2]])
b = np.array([1,2])
c = a-b
print "one dimen and two dimen",c,"\n" 
#limitation broadcast it will happen if the shape was not same a row shape or column shape must be same




#-----------------------------------------------------------------------------------------------------------------------------------------
print "------------------------Vector Multiplication-------------------------------"
#Vector Multiplication
#scalar multiply
a = np.array([1,2])
b = 2
c = a*b
print "Scalar multiplication",c

#scalar and two dimension
a = np.array([[1,2],[2,3]])
b = np.array([2])
c = a*b
print "scalar and two dimension",c

#one dimen and two dimen
a = np.array([[1,2],[1,2]])
b = np.array([1,2])
c = a*b
print "one dimen and two dimen",c,"\n" 
#limitation broadcast it will happen if the shape was not same a row shape or column shapemust be same




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Vector Division--------------------------------"
#Vector Division
#scalar division
a = np.array([1,2])
b = 0
c = a/b
print "Scalar division",c

#scalar and two dimension
a = np.array([[1.0,2],[2,3]])
b = np.array([2])
c = a/b
print "scalar and two dimension",c

#one dimen and two dimen
a = np.array([[1,2],[1,2]])
b = np.array([1,2])
c = a/b
print "one dimen and two dimen",c,"\n" 
#limitation broadcast it will happen if the shape was not same a row shape or column shapemust be same




#-----------------------------------------------------------------------------------------------------------------------------------------
print "---------------------------Vector Dot product------------------------------"
#Vector Dot Product
'''We can calculate the sum of the multiplied elements of two vectors of the same length to give a scalar. c = (a1×b1 + a2×b2 + a3×b3)
dimension should be same scalar multiply cant apply if we apply it will behave like simple scalar multiplication'''
a = np.array([1,2,3])
b = np.array([1,2,3])
d = 2
c = a.dot(b)
print "Vector Dot product = ",c,"\n"






#-----------------------------------------------------------------------------------------------------------------------------------------
print "------------------------------Vector Norm-----------------------------------"
#Vector Norm
#Calculating the size or length of a vector is often required either directly or as part of a broader vector or vector-matrix operation. The length of the vector is referred to as the vector norm or the vector’s magnitude.

'''Vector L 1 Norm
L 1 (v) = ||v|| 1
The L 1 norm is calculated as the sum of the absolute vector values, where the absolute value
of a scalar uses the notation |a 1 |. ||v|| 1 = |a 1 | + |a 2 | + |a 3 | if we put two dimension it is giving the maximum among that'''
a = np.array([[-1,-1,3]])
L1 = np.linalg.norm(a,1)
print "L1 norm = ",L1,"\n"


#Vector L 2 Norm
'''The L 2 norm calculates the distance of the vector coordinate from the origin of the vector
space.The L 2 norm is calculated as
the square root of the sum of the squared vector values.
||v|| 2 = SQRT(a1^2 + a2^2 + a3^2)'''
a = np.array([1, -2, 3])
L2 = np.linalg.norm(a)
print "L2 norm = ",L2,"\n"


#Vector Max Norm
'''The length of a vector can be calculated using the maximum norm, also called max norm. The notation for max norm is ||v|| inf , where inf is a subscript.
			L inf (v) = ||v|| inf
The max norm is calculated as returning the maximum value of the vector, hence the name. ||v|| inf = max a 1 , a 2 , a 3 it will convert to absolute and find the max. if two dimension means it will add each value in vector and give the maximum among that answer like the below'''
a = np.array([[1, -2, 3],[1,2,4]])
Maxnorm = np.linalg.norm(a, float("inf"))
print "Max norm = ",Maxnorm,"\n\n\n"








#-----------------------------------------------------------------------------------------------------------------------------------------
print "--------------------------------Matrix--------------------------------------"
a = np.array([[1,2,3],[1,2,3]])
print "Matrix = ",a,"\n"




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix Addition----------------------------"
#Matrix Addition
'''Two matrices with the same dimensions can be added together to create a new third matrix.
C = A + B
C[0, 0] = A[0, 0] + B[0, 0]
C[1, 0] = A[1, 0] + B[1, 0]
C[2, 0] = A[2, 0] + B[2, 0]
C[0, 1] = A[0, 1] + B[0, 1]
C[1, 1] = A[1, 1] + B[1, 1]
C[2, 1] = A[2, 1] + B[2, 1]'''
a = np.array([[1,2,3]])
b = np.array([[1,2,3],[1,2,3]])
c = a+b
print "Matrix addition = ",c,"\n"




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix Substraction----------------------------"
#Matrix Substraction
'''Two matrices with the same dimensions can be substracted together to create a new third matrix.
C = A - B
C[0, 0] = A[0, 0] - B[0, 0]
C[1, 0] = A[1, 0] - B[1, 0]
C[2, 0] = A[2, 0] - B[2, 0]
C[0, 1] = A[0, 1] - B[0, 1]
C[1, 1] = A[1, 1] - B[1, 1]
C[2, 1] = A[2, 1] - B[2, 1]'''
a = np.array([[2,12,13]])
b = np.array([[1,2,3],[1,2,3]])
c = a-b
print "Matrix Substraction = ",c,"\n"




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix Multiplication----------------------------"
#Matrix Multiplication
'''Two matrices with the same dimensions can be Multiplication together to create a new third matrix.
C = A * B
C[0, 0] = A[0, 0] * B[0, 0]
C[1, 0] = A[1, 0] * B[1, 0]
C[2, 0] = A[2, 0] * B[2, 0]
C[0, 1] = A[0, 1] * B[0, 1]
C[1, 1] = A[1, 1] * B[1, 1]
C[2, 1] = A[2, 1] * B[2, 1]'''
a = np.array([[2,12,13]])
b = np.array([[1,2,3],[1,2,3]])
c = a*b
print "Matrix Multiplication = ",c,"\n"




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix Division----------------------------"
#Matrix Division
'''Two matrices with the same dimensions can be divide together to create a new third matrix.
C = A / B
C[0, 0] = A[0, 0] / B[0, 0]
C[1, 0] = A[1, 0] / B[1, 0]
C[2, 0] = A[2, 0] / B[2, 0]
C[0, 1] = A[0, 1] / B[0, 1]
C[1, 1] = A[1, 1] / B[1, 1]
C[2, 1] = A[2, 1] / B[2, 1]'''
a = np.array([[2,12,13]])
b = np.array([[1,2,3.0],[1,2,3]])
c = a/b
print "Matrix Division = ",c,"\n"




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix -Matrix multiplication----------------------------"
#Matrix-Matrix Multiplication
'''Matrix multiplication, also called the matrix dot product is more complicated than the previous operations and involves a rule as not all matrices can be multiplied together. C = A · B 
The rule for matrix multiplication is as follows:
 The number of columns (n) in the first matrix (A) must equal the number of rows (m) in the second matrix (B). C(m, k) = A(m, n) · B(n, k)
C[0, 0] = A[0, 0] × B[0, 0] + A[0, 1] × B[1, 0]
C[1, 0] = A[1, 0] × B[0, 0] + A[1, 1] × B[1, 0]
C[2, 0] = A[2, 0] × B[0, 0] + A[2, 1] × B[1, 0]
C[0, 1] = A[0, 0] × B[0, 1] + A[0, 1] × B[1, 1]
C[1, 1] = A[1, 0] × B[0, 1] + A[1, 1] × B[1, 1]
C[2, 1] = A[2, 0] × B[0, 1] + A[2, 1] × B[1, 1]'''
a = np.array([[1,2,3],[1,2,3],[1,2,3]])
b = np.array([[1,2],[1,2],[1,2]])
c = a.dot(b)
print "Matrix matrix multiplication = ",c,"\n"




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix-vector multiplication----------------------------"
#Matrix-Vector Multiplication
'''A matrix and a vector can be multiplied together as long as the rule of matrix multiplication is observed. c = A · v
c[0] = A[0, 0] × v[0] + A[0, 1] × v[1]
c[1] = A[1, 0] × v[0] + A[1, 1] × v[1]
c[2] = A[2, 0] × v[0] + A[2, 1] × v[1]'''
a =np.array([[1,2],[1,2]])
b = np.array([1,2])
c = a.dot(b)
print "Matrix vector multiplication = ",c,"\n"




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix-Scalar multiplication----------------------------"
#Matrix-Scalar Multiplication
'''A matrix can be multiplied by a scalar. C = A · b
C[0, 0] = A[0, 0] × b
C[1, 0] = A[1, 0] × b
C[2, 0] = A[2, 0] × b
C[0, 1] = A[0, 1] × b
C[1, 1] = A[1, 1] × b
C[2, 1] = A[2, 1] × b'''
a = np.array([[1,2],[1,2]])
b = 2
c = a*b
print "Matrix scalar multiplication = ",c,"\n"




#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Types of matrix----------------------------"
#lower triangular = this will put all zero except that triangle index 
a = np.array([[1,2,3],[1,2,3],[3,2,3]])
lower = np.tril(a)
upper = np.triu(a)
print "Lower triangular = ",lower,"\n"
print "Upper triangular = ",upper,"\n"

#Diagonal matrix
a = np.array([[1,2,3],[2,3,4],[1,2,-3]])
d = np.diag(a)
diag = np.diag(d)
print "Diagonal matrix = ",diag,"\n"

#Identity matrix
a = np.identity(100)
print "Identity matrix = ",a,"\n"

#Orthogonal Matrix
'''Two vectors are orthogonal when their dot product equals zero. The length of each vector is 1 then the vectors are called orthonormal v · w =0 or v · w T = 0 
A matrix is orthogonal if its transpose is equal to its inverse. Q T = Q −1
Another equivalence for an orthogonal matrix is if the dot product of the matrix and itself equals the identity matrix. Q · Q T = I'''
Q = np.array([
[1, 0],
[0, -1]])
# inverse equivalence
V = np.linalg.inv(Q)
print("Inverse matrix = ",V)
# identity equivalence
I = Q.dot(Q.T)
print("Orthogonal matrix = ",I)








#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix operation----------------------------"
#Transpose matrix
#A defined matrix can be transposed, which creates a new matrix with the number of columns and rows flipped. This is denoted by the superscript T next to the matrix A T . C = A T
a = np.array([[111,2,3],[11,2,3],[11,2,3]])
print "Transpose matrix = ",a.T,"\n"

#Inverse Matrix
#Matrix inversion is a process that finds another matrix that when multiplied with the matrix, results in an identity matrix.AB = BA = In B = A −1
print "Inverse matrix = ",np.linalg.inv(a),"\n"

#trace 
#A trace of a square matrix is the sum of the values on the main diagonal of the matrix (top-left to bottom-right). tr(A) = A[0, 0] + A[1, 1] + A[2, 2]
print "Trace = ",np.trace(a),"\n"

#Determinant
#The determinant of a square matrix is a scalar representation of the volume of the matrix. the determinant of a matrix A tells you the volume of a box with sides given by rows of A. The determinant is a zero when the matrix has no inverse.
print "Determinant = ",np.linalg.det(a),"\n"

#Rank 
#The rank of a matrix is the estimate of the number of linearly independent rows or columns in a matrix. The rank of a matrix M is often denoted as the function rank(). rank(A)
print "Rank = ",np.linalg.matrix_rank(a),"\n"






#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Matrix Decomposition----------------------------"
a = np.array([[111,13,14],[1000,1,1],[150,111,1]])
p,l,u = lu(a)
print "lu decompostion = ",p,"\n"
values,vector = np.linalg.eig(a)
print "Eigen values = ",values,"\n"
print "Eigen vectors = ",vector,"\n" 







#-----------------------------------------------------------------------------------------------------------------------------------------
print "-----------------------------Basic Operation----------------------------"
#Mean
a = np.array([
[1,2,3,4,5,6],
[1,2,3,4,5,7]])
mean = np.mean(a)
print "mean = ",mean
# column means
col_mean = np.mean(a, axis=0)
print "Column mean = ",col_mean
# row means
row_mean = np.mean(a, axis=1)
print "Row mean = ",row_mean,"\n"

#Variance
result = np.var(a, ddof=1)
print "Variance = ",result
# column variances
col_var = np.var(a, ddof=1, axis=0)
print "Column variance = ",col_var
# row variances
row_var = np.var(a, ddof=1, axis=1)
print "Row variance = ",row_var,"\n"

#Standard deviation
# column standard deviations
col_std = np.std(a, ddof=1, axis=0)
print "column std_dev = ",col_std
# row standard deviations
row_std = np.std(a, ddof=1, axis=1)
print "Row std_dev = ",row_std,"\n"

#Covariance
# define first vector
x = np.array([1,2,3,4,5,6,7,8,9])
print(x)
# define second covariance
y = np.array([9,8,7,6,5,4,3,2,1])
print(y)
# calculate covariance
Sigma = np.cov(x,y)[0,1]
print "Covariance = ",Sigma,"\n"

#Correlation
# calculate correlation
corr = np.corrcoef(x,y)[0,1]
print "Correlation = ",corr,"\n"

#covariance matrix
X = np.array([
[1.0, 1.0],
[2.0, 1.0]])
#[1.0, 1.0],
#[1.0, 5.0]])
Sigma = np.cov(X.T)
row_var = np.cov(X.T[0],X.T[1])[0,1]
print(Sigma,"variance = ",row_var)
row_var = np.var(X, ddof=1, axis=1)
print(Sigma,"variance = ",row_var),"\n"


a = np.array([[1,1],[0,1],[-1,1]])
u,si,v = np.linalg.svd(a)
print si

