#-*- coding: utf-8 -*-   
# Jerry Chee
# Optimization hw4 problem 4

from __future__ import unicode_literals
from __future__ import division
import timeit
from numpy import linalg as LA
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def obj_fn(A, b, c, x):
    """ objective function defined in problem 4"""
    return np.dot(c.transpose(),x) - np.sum(np.log(b - A.dot(x)))

def grad_helper(A, b, c, x, k):
    """ helper function for grad_fn"""
    tmp = 0.0
    for i in range(b.shape[0]):
        tmp += (A[i,k] / (b[i] - np.dot(A[i], x)))
    return c[k] + tmp
    
def grad_old(A, b, c, x):
    n = x.shape[0]
    grad = np.zeros((n,1))
    for k in range(n):
        grad[k] = grad_helper(A,b,c,x,k)
    return grad

def grad_fn(A, b, c, x):
    """ returns gradient of obj_fn"""
    return c + np.sum(A / (b - A.dot(x)), axis=0).reshape((100,1))
    
def hess_helper(A, b, x, k, t):
    first = A[:,k] * A[:,t]
    second = np.reciprocal(np.square(b - A.dot(x)))
    return np.dot(first, second)

def hess_fn(A, b, x):
    """ returns hessian of obj_fn"""
    n = x.shape[0]
    hess = np.zeros((n ,n))
    for k in range(n):
        for t in range(n):
            hess[k,t] = hess_helper(A,b,x,k,t)
    return hess

def generate_conditions(distribution):
    """ ensures that A,b,c such that x in domain of function"""
    const = 1
    if distribution == "uniform":
        A = const * np.random.rand(500,100)
        x = const * np.random.rand(100,1)
        b = A.dot(x) + 1
        c = const * np.random.rand(100,1)
    elif distribution == "stdnormal":
        A = const * np.random.randn(500,100)
        x = const * np.random.randn(100,1)
        b = A.dot(x) + 1
        c = const * np.random.randn(100,1)
    return (A,b,c,x)

def back_line_search(obj_fn, A, b, c, x, alpha_0, p_k, c_1, p):
    """ performs backtracking line search """
    alpha = alpha_0
    x_new = x + alpha*p_k
    compare_domain =  b - A.dot(x_new)
    # first reduce alpha until new x_k+1 in domain of objective function 
    while np.any(compare_domain <= 0): # update alpha
        alpha = p * alpha
        # update x_new, compare_domain
        x_new = x + alpha*p_k
        compare_domain =  b - A.dot(x_new)

    compare_armijo = obj_fn(A,b,c,x) + c_1 * alpha * np.dot(grad_fn(A,b,c,x).transpose(),p_k)
    # now reduce alpha until armijo condition met     
    while np.all(obj_fn(A,b,c,x_new) > compare_armijo): 
        # update alpha, compare_armijo
        alpha = p * alpha
        compare_armijo = obj_fn(A,b,c,x) + c_1 * alpha * np.dot(grad_fn(A,b,c,x).transpose(),p_k)
        # update x_new
        x_new = x + alpha*p_k
    
    return alpha

def steepest_descent(obj_fn, grad_fn, A, b, c, x_0, 
        c_1, p, alpha_0, epsilon):
    """ performs steepest descent with backtracking line search"""
    # setup
    start = timeit.default_timer()
    errors = []    
    # 1st search dir and step size
    p_new = np.negative(grad_fn(A,b,c,x_0))
    alpha_new = back_line_search(obj_fn, A,b, c, x_0, alpha_0, p_new, c_1, p)
    # 1st step
    x_new = x_0 + alpha_new * p_new

    iterations = 0
    while linalg.norm(p_new, 'fro') > epsilon:
        iterations += 1
        # error log
        errors.append(obj_fn(A,b,c,x_new)[0][0])
        
        #print(linalg.norm(p_new, 'fro'))
        # search dir and step size
        p_new = - grad_fn(A,b,c,x_new)
        alpha_new = back_line_search(obj_fn, A, b, c, x_new, alpha_0, p_new, c_1,p)
        # step
        x_new = x_new + alpha_new * p_new

    print(timeit.default_timer() - start)
    return (iterations, errors, x_new)

def newton(obj_fn, grad_fn, hess_fn, A, b, c, x_0, 
        c_1, p, alpha_0, epsilon_sd, epsilon_n):
    """ performs newton methon with backtracking line search"""
    # timing
    start = timeit.default_timer()
    # first perform steepest descent
    (iterations, errors, x_new) = steepest_descent(obj_fn, grad_fn, 
            A, b, c, x_0, c_1, p, alpha_0, epsilon_sd)
    
    # initial search dir and step size
    grad = grad_fn(A,b,c,x_new)
    hess = hess_fn(A,b,x_new)
    p_new = -linalg.inv(hess).dot(grad)
    alpha_new = back_line_search(obj_fn, A,b, c, x_new, alpha_0, p_new, c_1, p)
    lambda_new = (grad.transpose().dot(hess)).dot(grad)
    # 1st step
    x_new = x_new + alpha_new * p_new

    while (lambda_new/2) > epsilon_n:
        iterations += 1
        # error log
        errors.append(obj_fn(A,b,c,x_new)[0][0])
        #print(lambda_new/2)
        # search dir and step size
        grad = grad_fn(A,b,c,x_new)
        hess = hess_fn(A,b,x_new)
        p_new = -linalg.inv(hess).dot(grad)
        alpha_new = back_line_search(obj_fn, A,b, c, x_new, 
                alpha_0, p_new, c_1, p)
        lambda_new = (grad.transpose().dot(hess)).dot(grad)
        # step
        x_new = x_new + alpha_new * p_new
    
    print(timeit.default_timer() - start)
    return (iterations, errors, x_new)
        
# =============================================================================
# error plotting for c_1, p, epsilon. Hold each constant, change the other.

# parameters
ls_c_1 = [0.01, 0.25, 0.90]
const_c_1 = 0.25
ls_p = [0.05, 0.50, 0.95]
const_p = 0.50
ls_epsilon = [0.001, np.power(10.0,-5), np.power(10.0,-8)]
const_epsilon = 0.001
alpha_0 = 1

# generate A,b,c,x
(A,b,c,x) = generate_conditions("stdnormal")

# first vary c_1, hold else constant
dict_errors_sd = {}
dict_errors_n  = {}
p = const_p
epsilon = const_epsilon
for c_1 in ls_c_1:
    (iterations, errors, x_star) = steepest_descent(obj_fn, grad_fn, 
            A, b, c, x, c_1, p, alpha_0, epsilon)
    dict_errors_sd[c_1] = np.log(errors - obj_fn(A,b,c,x_star)[0][0])

    (iterations, errors, x_star) = newton(obj_fn, grad_fn, hess_fn,
            A, b, c, x, c_1, p, alpha_0, 0.01, epsilon)
    dict_errors_n[c_1] = np.log(errors - obj_fn(A,b,c,x_star)[0][0])

# plot steepest descent, varying c_1
l_01 = range(len(dict_errors_sd[0.01]))
l_25 = range(len(dict_errors_sd[0.25]))
l_90 = range(len(dict_errors_sd[0.90]))
plt.plot(l_01, dict_errors_sd[0.01], label = "c_1 = 0.01")
plt.plot(l_25, dict_errors_sd[0.25],  label = "c_1 = 0.25")
plt.plot(l_90, dict_errors_sd[0.90],  label = "c_1 = 0.90")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("log error")
plt.title("Steepest Descent log error varying c_1, p=0.50, epsilon=10^-3")
#plt.savefig("steepest_desc_c1.png")
plt.show()

# plot newton, varying c_1
l_01 = range(len(dict_errors_n[0.01]))
l_25 = range(len(dict_errors_n[0.25]))
l_90 = range(len(dict_errors_n[0.90]))
plt.plot(l_01, dict_errors_n[0.01], label = "c_1 = 0.01")
plt.plot(l_25, dict_errors_n[0.25],  label = "c_1 = 0.25")
plt.plot(l_90, dict_errors_n[0.90],  label = "c_1 = 0.90")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("log error")
plt.title("Newton Method log error varying c_1, p=0.50, epsilon=10^-3")
#plt.savefig("newton_method_c1.png")
plt.show()

# ===================================
# second vary p, hold else constant
dict_errors_sd = {}
dict_errors_n  = {}
c_1 = const_c_1
epsilon = const_epsilon
for p in ls_p:
    (iterations, errors, x_star) = steepest_descent(obj_fn, grad_fn, 
            A, b, c, x, c_1, p, alpha_0, epsilon)
    dict_errors_sd[p] = np.log(errors - obj_fn(A,b,c,x_star)[0][0])

    (iterations, errors, x_star) = newton(obj_fn, grad_fn, hess_fn,
            A, b, c, x, c_1, p, alpha_0, 0.01, epsilon)
    dict_errors_n[p] = np.log(errors - obj_fn(A,b,c,x_star)[0][0])

# plot steepest descent, varying p
l_05 = range(len(dict_errors_sd[0.05]))
l_50 = range(len(dict_errors_sd[0.50]))
l_95 = range(len(dict_errors_sd[0.95]))
plt.plot(l_05, dict_errors_sd[0.05], label = "p = 0.05")
plt.plot(l_50, dict_errors_sd[0.50],  label = "p = 0.50")
plt.plot(l_95, dict_errors_sd[0.95],  label = "p = 0.95")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("log error")
plt.title("Steepest Descent log error varying p, c_1=0.25, epsilon=10^-3")
#plt.savefig("steepest_desc_p.png")
plt.show()

# now plot newton method, varying p
l_05 = range(len(dict_errors_n[0.05]))
l_50 = range(len(dict_errors_n[0.50]))
l_95 = range(len(dict_errors_n[0.95]))
plt.plot(l_05, dict_errors_n[0.05], label = "p = 0.05")
plt.plot(l_50, dict_errors_n[0.50],  label = "p = 0.50")
plt.plot(l_95, dict_errors_n[0.95],  label = "p = 0.95")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("log error")
plt.title("Newton Method log error varying p, c_1=0.25, epsilon=10^-3")
#plt.savefig("newton_method_p.png")
plt.show()

# ===================================
# third vary epsilon, hold else constant
dict_errors_sd = {}
dict_errors_n  = {}
c_1 = const_c_1
p = const_p
P3 = np.power(10.0, -3)
P5 = np.power(10.0, -5)
P8 = np.power(10.0, -8)

for epsilon in ls_epsilon:
    # steepest descent has trouble converging for 10^-8
    if epsilon == P3:
        (iterations, errors, x_star) = steepest_descent(obj_fn, grad_fn, 
                A, b, c, x, c_1, p, alpha_0, epsilon)
        dict_errors_sd[epsilon] = np.log(errors - obj_fn(A,b,c,x_star)[0][0])
    
    (iterations, errors, x_star) = newton(obj_fn, grad_fn, hess_fn,
            A, b, c, x, c_1, p, alpha_0, 0.0001, epsilon)
    dict_errors_n[epsilon] = np.log(errors - obj_fn(A,b,c,x_star)[0][0])

# plot steepest descent, varying p
l_3 = range(len(dict_errors_sd[P3]))
#l_5 = range(len(dict_errors_sd[P5]))
#l_8 = range(len(dict_errors_sd[P8]))
plt.plot(l_3, dict_errors_sd[P3], label = "epsilon = 10^-3")
#plt.plot(l_5, dict_errors_sd[P5],  label = "epsilon = 10^-5")
#plt.plot(l_8, dict_errors_sd[P8],  label = "epsilon = 10^-8")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("log error")
plt.title("Steepest Descent log error varying epsilon, c_1=0.25, p=0.50")
#plt.savefig("steepest_desc_epsilon.png")
plt.show()

# plot newton, varying p
l_3 = range(len(dict_errors_n[P3]))
l_5 = range(len(dict_errors_n[P5]))
l_8 = range(len(dict_errors_n[P8]))
plt.plot(l_3, dict_errors_n[P3], label = "epsilon = 10^-3")
plt.plot(l_5, dict_errors_n[P5],  label = "epsilon = 10^-5")
plt.plot(l_8, dict_errors_n[P8],  label = "epsilon = 10^-8")
plt.legend()
plt.xlabel("iterations")
plt.ylabel("log error")
plt.title("Newton Method log error varying epsilon, c_1=0.25, p=0.50")
#plt.savefig("newton_method_epsilon.png")
plt.show()

