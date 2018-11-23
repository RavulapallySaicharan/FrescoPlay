# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 02:56:50 2018

@author: ravul
"""

import numpy as np

# Matrix A
A = np.array([[1,2],[3,4]])
print('Shape of the A :',A.shape)

# Matrix B
B =  np.array([[2,3,4],[5,6,7]])
print('Shape of the B: ',B.shape)

# Dot product (dot product is the matrix multiplication), Inner dimension should be matched for the two matrices M = A . B
Dotprod = np.dot(A,B)
print('Answer of the matrix multiplication :',Dotprod)
print('Shape of dot product :',Dotprod.shape)

# Element wise matrix multiplication (both should be of same dimension) M = A * B
Matmul = np.multiply(A,A)
print('\nAnswer of the element wise matrix multiplication :',Matmul)
print('Shape of the element wise multiplication :', Matmul.shape)

# Broadcasting (The smaller arrays are broadcasted across the larger array so that they have compatible shapes)
## Multiplication example of the broadcasting
A = np.array([[11,32,43],[34,25,35],[34,67,89]])
B = np.array([10,20,30])
print('\nA shape :',A.shape,
      '\nB shape :',B.shape)
C = A * B
print('\nResult :',C,
      '\nShape :',C.shape)

## Addition example of the broadcasting
A = np.array([[11,32,43],[34,25,35],[34,67,89]])
B = 10
print('\nA shape :',A.shape)
C = A + B
print('\nResult :',C,
      '\nShape :',C.shape)

# Ndarray (np.ndarray is class and np.array is the method to create ndarray)
a = np.ndarray([1,2,3,2,3,4,3,4])
print(type(a))
b = np.array([1,2,3,2,3,4,3,4])
print(type(b))

areshape1 = b.reshape(2,4)
print('\nAfter reshape to 2x4 :\n',areshape1)
''' areshape2 = b.reshape() ## not possible should print error 
print('\nAfter reshape to 3x3 :\n',areshape2) '''



