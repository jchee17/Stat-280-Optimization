#-*- coding: utf-8 -*-
# Jerry Chee
# Optimization hw4 problem 2

from __future__ import unicode_literals
from __future__ import division
from numpy import linalg as LA
import numpy as np
from scipy import linalg

# general functions
def forward_err(X_star, A_inv):
    return linalg.norm(X_star - A_inv, 'fro')

def backward_err(X_star, A):
    n = A.shape[0]
    I = np.identity(n) 
    return linalg.norm(I - A.dot(X_star), 'fro') 

def gen_matrix(n, dtype):
    """ generates various matrices, ensures invertible"""
    # int entires
    if dtype == "int":
        M = np.matrix(np.random.randint(-1000000,1000000, size=(n,n)))
        while LA.det(M) == 0:
            M = np.matrix(np.random.randint(-1000000,1000000, size=(n,n)))
    # diagonal matrix with rational entries
    # create rational entry matrix by element-wise divide two int matrices
    elif dtype == "rat_diag":
        M1 = np.random.randint(-1000000,1000000, size=(10,10))
        M2 = np.random.randint(-1000000,1000000, size=(10,10))
        M  = np.matrix(M1/M2) # array, element wise division
        M  = np.diag(np.diag(M))
        while LA.det(M) == 0:
            M1 = np.random.randint(-1000000,1000000, size=(10,10))
            M2 = np.random.randint(-1000000,1000000, size=(10,10))
            M  = np.matrix(M1/M2)
            M  = np.diag(np.diag(M))
    # real entries
    elif dtype == "real":
        M = np.matrix(np.random.rand(n,n))
        while linalg.det(M) == 0:
             = np.matrix(np.random.rand(n,n))
    
    return M

def newton_inverse(A, epsilon):
    """ A matrix, alpha/epislon scalar """
    # matrix size
    n = A.shape[0]
    I = np.identity(n)
    
    # we set alpha to be 1/4 ||A||_{2}^{2}
    alpha = 1.0 / ((n*n) * np.square(linalg.norm(A, 'fro')))

    # starting values
    X = alpha * A.transpose()
    E = I - A.dot(X)
    
    # main loop
    i = 0
    while linalg.norm(E, 'fro') > epsilon:
       # print(X)
        X = X.dot(2*I - A.dot(X))
        E = I - A.dot(X)
        i += 1

    return X

# ==========================================
# i)
num_trials = 10
epsilon = 0.0001
# A (2x2), random int
for i in range(num_trials):
    A = gen_matrix(2, "int")
    X_star = newton_inverse(A,epsilon)
    A_inv = LA.inv(A)
    print(forward_err(X_star, A_inv))

# A (10x10) diagonal, rational
for i in range(num_trials):
    A = gen_matrix(10, "rat_diag")
    X_star = newton_inverse(A,epsilon)
    A_inv = LA.inv(A)
    print(forward_err(X_star, A_inv))
    
# ii)
for n in [10, np.power(10,2), np.power(10,3)]: #, np.power(10,4)]:
    print(n)
    for i in range(num_trials):
        A = gen_matrix(n, "real")
        X_star = newton_inverse(A, epsilon)
        #A_inv = linalg.inv(A)
        print(backward_err(X_star, A), backward_err(A_inv, A))
    print('')
