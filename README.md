# Stat-280-Optimization
Course code from HW.

hw4\_p2.py implements Newton's method for computing the matrix inverse. 

hw4\_p4.py implements steepest descent and Newton's method, both with 
backtracking line search, to minimize a function of the form:

f(x\_1, ... x\_100) = \sum\_{j=1}^{100} c\_{j}x\_{j} - 
\sum\_{i=1}{500} log ( b\_{i} - \sum\_{j=1}{100} a\_{ij} x\_{j} )
