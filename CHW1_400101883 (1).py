#!/usr/bin/env python
# coding: utf-8

# <h1 align="center">Mathematical Methods In Engineering - 25872</h1>
# <h4 align="center">Dr. Amiri</h4>
# <h4 align="center">Sharif University of Technology, Fall 2023</h4>
# <h4 align="center">Python Assignment 1</h4>
# <h4 align="center">feel free to ask your questions via telegram,
# 
# questions 1,4 : @BeNameBalasari and questions 2,3,5 : @maahmoradi

# You should write your code in the <font color='green'>Code Cell</font> and then run the <font color='green'>Evaluation Cell</font> to check the output of your code.<br>
# <font color='red'>**Please do not edit the existing codes.**</font>

# ## 1. Introduction to matrices
# In this question, we want to get familiar with performing simple matrix operations and obtaining special features of matrices in Python <br>
# #### 1-1-  Vector P-Norm
#  Let $p\geq 1$  be  a real   number . The  p-norm ( also called 
# $\ell ^{p}$-norm) of  vector 
# ${\displaystyle \mathbf {x} =(x_{1},\ldots ,x_{n})}$  is : 
# ${\displaystyle \|\mathbf {x} \|_{p}:=\left(\sum _{i=1}^{n}\left|x_{i}\right|^{p}\right)^{1/p}}$ and
# ${\displaystyle \|\mathbf {x} \|_{\infty}:=max(|x_1| , |x_2| , \cdots ,|x_n|)}$ 
# > Write a function that takes a vector and p as input and gives p-norm as output then use it to calculate the norm-3 of the  following vector ( if the p is np.inf the function should give the infinity norm of the matrix ) :
# $$
# x = \left(\begin{array}{cc} 
# -3.0\\ 1.0 \\2.0
# \end{array}\right)
# $$
# > then use np.linalg.norm to check your answer 

# In[8]:


# import required packages
import numpy as np


# In[9]:


# Code cell
def pnorm(x, p):
   return np.sum(np.abs(x)**p)**(1/p) 
       
example_x=[-3 , 1 , 2]
p=3
print('the norm-3 of the following vector: ',pnorm(example_x,p))

##chck my answer:
p_norm=np.linalg.norm(example_x,p)
print('using library :' , p_norm)


# #### 1-2- Matrix Norm
#  The 
#  operator  norm of matrix $ \mathbf{A}  $ is : 
# ${\displaystyle \|\mathbf {A} \|:=\max\limits_{x\neq 0}\left(\frac{||Ax||}{||x||}\right)}$
# 
# and  The Frobenius  norm  is defined  so 
# that   for  every  square  matrix $ \mathbf{A} $ : ${\displaystyle \|\mathbf {A} \|_F:=\left(\sum_{i,j=1}^{n}(|a_{ij}|^2)\right)^\frac{1}{2}}$
# 
# > Write a function that takes a Matrix  as input and gives frobenius norm as output then use it to calculate the norm of the  following Matrix :
# $$
# A = \begin{bmatrix}
#     7 & 5 & 1\\
#     1 & 7 & 1\\
#     5 & 5 & 7
# \end{bmatrix}
# $$
# > then use np.linalg.norm to check your answer and use it to calculate the operator norm of A 

# In[10]:


# Code cell
def fnorm(A):
    norm_value = 0
    for row in A:
        for element in row:
            norm_value += element ** 2
    norm_value = norm_value ** 0.5
    
    return norm_value
example_matrix=np.array([[7, 1, 5],
                        [5, 7, 5],
                        [1, 1, 7]])
print('Frobenius norm of the matrix: ', fnorm(example_matrix))

##to check my code:
n_v=np.linalg.norm(example_matrix)
print('using library :' ,n_v)


# #### 1-3- Matrix Determinant And Matrix inverse 
# Laplace expansion expresses the determinant of a matrix 
# A recursively in terms of determinants of smaller matrices, known as its minors. The minor 
# , $M_{i,j}$ is defined to be the determinant of the 
# ${\displaystyle (n-1)\times (n-1)}$ matrix that results from 
# A by removing the 
# i-th row and the 
# j-th column. The expression 
# ${\displaystyle (-1)^{i+j}M_{i,j}}$  is known as a cofactor. For every 
# i, one has the equality :
# $$
# det(\mathbf{A})=\sum_{j=1}^n\left((-1)^{i+j}a_{ij}M_{ij}\right)
# $$
# > Write a function that takes a Matrix  as input and gives Determinant as output then use it to calculate the Determinant of the  following Matrix :
# $$
# A = \begin{bmatrix}
#     7 & 5 & 1\\
#     1 & 7 & 1\\
#     5 & 5 & 7
# \end{bmatrix}
# $$
# > then use np.linalg.det to check your answer 

# In[11]:


# Code cell
def det(A):
    
    if len(A) == len(A[0]):
        order = len(A)

        # Base case: if the matrix is 1x1, return the only element
        if order == 1:
            return A[0][0]

        # Base case: if the matrix is 2x2, return the determinant using the formula ad - bc
        elif order == 2:
            return A[0][0] * A[1][1] - A[0][1] * A[1][0]

        else:
            determinant = 0
            for i in range(order):
                # Calculate the cofactor
                cofactor = (-1) ** i * A[0][i] * det(
                    [row[:i] + row[i + 1:] for row in A[1:]]
                )
                determinant += cofactor

            return determinant
    else:
        return "Error: Input matrix is not a square matrix."

example_matrix=[[7, 1, 5],
                [5, 7, 5],
                [1, 1, 7]]
print('Determinate of matrix A is: ', det(example_matrix))

##to check my code:
  
test_det=np.linalg.det(example_matrix)
    
print('using library :',test_det)


# The inverse of a Matrix is defined as : 
# $$
# \mathbf{A}^{-1}=\frac{adj(A)}{det(A)}
# $$
# The adjugate of $\mathbf{A} \ adj(\mathbf{A})$ is the transpose of $\mathbf{C}$, that is, the n × n matrix whose (i, j) entry is the (j, i) cofactor of A,
# $$
# adj(\mathbf{A})=\mathbf{C}^T=\left((-1)^{i+j}M_{ji}\right)_{1 \leq i,j \leq n}
# $$
# > Write a function that takes a Matrix  as input and gives inverse of Matrix as output then use it to calculate the inverse of the  following Matrix :
# $$
# A = \begin{bmatrix}
#     7 & 5 & 1\\
#     1 & 7 & 1\\
#     5 & 5 & 7
# \end{bmatrix}
# $$
# > then use np.linalg.inv to check your answer 

# In[12]:


# Code cell
def inv(A):

    # Check if the matrix is square
    if len(A) != len(A[0]):
        print("Input matrix must be square for inverse calculation")

    n = len(A)
    
    # Augment the matrix with the identity matrix
    augmented_matrix = [row + [int(i == j) for j in range(n)] for i, row in enumerate(A)]

    # Apply Gauss-Jordan elimination
    for col in range(n):
        # Find the pivot (non-zero element) in the current column
        pivot_row = next(i for i in range(col, n) if augmented_matrix[i][col] != 0)
        
        # Swap rows to move the pivot to the current row
        augmented_matrix[col], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[col]

        # Scale the current row to make the pivot 1
        pivot_value = augmented_matrix[col][col]
        augmented_matrix[col] = [element / pivot_value for element in augmented_matrix[col]]

        # Eliminate other rows
        for row in range(n):
            if row != col:
                factor = augmented_matrix[row][col]
                augmented_matrix[row] = [elem - factor * augmented_matrix[col][i] for i, elem in enumerate(augmented_matrix[row])]

    # Extract the inverse matrix from the augmented matrix
    inverse_matrix = [row[n:] for row in augmented_matrix]

    return inverse_matrix

    
example_matrix=[[7, 1, 5],
                         [5, 7, 5],
                         [1, 1, 7]]

print('Inverse of the matrix is:' )
for row in inv(example_matrix):
    print(row)
test=np.linalg.inv(example_matrix)
print('using library :\n',test)


# ## 2. Gauss-Jordan elimination
# #### 2-1-  Implementation
# The Gauss-Jordan Elimination method is an algorithm to solve a linear system of equations. This method solves the system by representing it as an augmented matrix, reducing it using row operations, and expressing the system in reduced row-echelon form to find the values of the variables. \
# The function gauss_jordan_elimination takes two arguments: the matrix A representing the coefficients of the equations, and the vector b representing the constants on the right-hand side of the equations. It sets flag to 1 and returns the solution vector x if exists, else sets flag to 0 in case of 'No Solution' or 'Infinite Solutions' and prints the corresponding case.
# 
# Here's an example of a system of equations $\textbf{Ax = b}$
# 
# the inputs of the function :
# 
# $$
# A = \begin{bmatrix}
#     2  & 1  & 5 \\
#     4  & 4  & -4 \\
#     1 & 3 & 1 
# \end{bmatrix}
# $$
# 
# $$
# b = \begin{bmatrix}
#     8  \\
#     4  \\
#     5 
# \end{bmatrix}
# $$
# 
# the corresponding output :
# 
# $$
# x = \begin{bmatrix}
#     1  \\
#     1  \\
#     1 
# \end{bmatrix}
# $$

# In[13]:


# import required packages
import numpy as np
import time


# In[14]:


# Code cell
def gauss_jordan_elimination(A, b):
    # Combine matrix A and vector b
    augmented_matrix = np.column_stack((A.astype(float), b.astype(float)))

    # Perform Gauss-Jordan elimination
    rows, cols = augmented_matrix.shape
    for i in range(rows):
        # Make the diagonal element 1
        pivot = augmented_matrix[i, i]
        augmented_matrix[i, :] /= pivot

        # Eliminate other rows
        for k in range(rows):
            if k != i:
                factor = augmented_matrix[k, i]
                augmented_matrix[k, :] -= factor * augmented_matrix[i, :]

    # Check for no solution or infinite solutions
    for i in range(rows):
        if np.all(augmented_matrix[i, :-1] == 0) and augmented_matrix[i, -1] != 0:
            print("No Solution")
            return None, 0

    # Check for infinite solutions
    if np.linalg.matrix_rank(A) < np.linalg.matrix_rank(augmented_matrix[:, :-1]):
        print("Infinite Solutions")
        return None, 0

    # Extract the solution vector from the augmented matrix
    solution = augmented_matrix[:, -1]
    print("Solution vector x:", solution)
    return solution, 1


# In[15]:


# Evaluation Cell
n = 10
A = np.zeros((10,10))
b = np.random.randint(100, size=(10))
while np.linalg.matrix_rank(A) != n: 
  A = np.random.randint(0,100,(n, n))
s = time.time()
elapsed = time.time() - s
x, flag = gauss_jordan_elimination(A,b)
assert flag == 1, "flag's not set correctly"
assert np.linalg.norm(A @ x - b) < 1e-7, "Ax = b is not satisfied"
print(f'status: successful, time elapsed: {np.round(elapsed, 5)} seconds')
b = np.random.randint(100, size=(10))
A = np.random.randint(0, 100, (n, n))
U, S, V = np.linalg.svd(A)
r = 9  
S[r:] = 0 
A = U.dot(np.diag(S)).dot(V)
b = np.random.randint(100, size=(10))
s = time.time()
elapsed = time.time() - s
x, flag = gauss_jordan_elimination(A,b)
assert flag == 0, "flag's not set correctly"
print(f'status: successful, time elapsed: {np.round(elapsed, 5)} seconds')


# ## 3. Statistics on random matrices
# 
# On average, a random matrix is invertible. But what if the random matrix has entries that are either 0 or 1 with equal probability? What is the probability that a 5 by 5 matrix whose entries are all zeros or ones is singular? And what is the average number of pivot columns? That is what you will find out in this exercise.
# 
# Create random integer matrices whose entries are either 0 or 1. To find the number of pivot columns of a matrix A, complete the function getrank below.
# 

# In[6]:


# Code cell
import random
import numpy as np

import random

import random

def generate_random_square_matrix(size):
    matrix = []
    for _ in range(size):
        row = [random.choice([0, 1]) for _ in range(size)]
        matrix.append(row)
    return matrix

def get_rank(matrix):
    if not matrix:
        return 0  # Empty matrix

    size = len(matrix)
    rank = 0

    for i in range(size):
        # Find the first non-zero entry in the current column
        column_values = [row[i] for row in matrix[rank:]]
        nonzero_indices = [rank + index for index, value in enumerate(column_values) if value != 0]

        if nonzero_indices:
            nonzero_row = min(nonzero_indices)
            
            # Swap the current row with the first non-zero entry row
            matrix[rank], matrix[nonzero_row] = matrix[nonzero_row], matrix[rank]

            # Make the current column all zeros below the pivot
            for j in range(rank + 1, size):
                factor = matrix[j][i] // matrix[rank][i]
                matrix[j][:] = [a - factor * b for a, b in zip(matrix[j], matrix[rank])]

            rank += 1

    return rank

# Example usage:
size = 5
random_square_matrix = generate_random_square_matrix(size)
print("Random Square Matrix:")
for row in random_square_matrix:
    print(row)

rank = get_rank(random_square_matrix)
print("Number of Pivot Columns (Rank):", rank)


# In[20]:


# Code cell

num = 100000;  # Number of random trials 
n = 5  # Size of matrix
number_of_pivots = 0
pobs=0
while num>=0 : 
    A=generate_random_square_matrix(n)
    rank=get_rank(A)
    if rank!=5:
        pobs+=1
    number_of_pivots+=rank
    num-=1
num = 100000
avg_number_of_pivots=number_of_pivots/num
print('avg number of pivots:',avg_number_of_pivots)
print('probability of being singular:',pobs/num)

# (avg number of pivots, probability of being singular)
# in the format [a.b,c.d] where you rounded the answer to one decimal place.

# write your code here


# We've looked at random matrices whose entries are 0 or 1. But how does size effect the rank and the probability of being singular? Repeat the previous exercise for 10 by 10 matrices with entries that are 0 or 1.
# 
# Explore and plot the probability of singularity and the average number of pivots for different sizes of random matrices. Can you determine how this probability depends on the size of the matrix? You might try to find the probability of a 10 by 10 random matrix with entries 0 or 1 has rank 10, 9, 8, 7 etc. You can check your probabilities against the expected value to see if your probabilities match your observations. You might try to connect these probabilities with determinant formulas to see if you can predict the numbers from other formulas and prove a relationship. Happy explorations!
# 
#  guess what happens as $n \rightarrow \infty $
# , but only run up to 80

# In[22]:


# Code cell

nvalues = [5,10,20,40,80]
num = 500
for n in nvalues :
    number_of_pivots = 0
    pobs=0
    while num>=0 : 
        A=generate_random_square_matrix(n)
        rank=get_rank(A)
        if rank!=5:
            pobs+=1
        number_of_pivots+=rank
        num-=1
    num = 500
    avg_number_of_pivots=number_of_pivots/num
    print(avg_number_of_pivots)


# ## 4. Application of Cholesky decomposition
# #### 4-1-  Cholesky decomposition
# In linear algebra, LU decomposition factors a matrix ($\textbf{A} :n \times n$) as the product of a lower triangular matrix ($\textbf{L} :n \times n$) and an upper triangular matrix ($\textbf{U} :n \times n$). The product sometimes includes a permutation matrix ($\textbf{P} :n \times n$) as well.
# $$ \textbf{PA} = \textbf{LU} $$
# We know that the elements on the main diagonal of the $\textbf{U}$ are the pivots. So $\textbf{U}$ can be decomposed into a diagonal matrix ($\textbf{D} :n \times n$) with elements whose pivots are on the main diagonal and a normalized $\textbf{U}$ matrix.
# $$ \textbf{PA} = \textbf{LDU}  $$
# if the A is symmetric then we have the Cholesky decomposition :
# $$
# \mathbf{A}^T=\mathbf{A} \longrightarrow \mathbf{A} =LD^\frac{1}{2}D^\frac{1}{2} L^T= (LD^\frac{1}{2})(LD^\frac{1}{2})^T=L'{L'} ^T
# $$
# There are various methods for calculating the Cholesky decomposition one of them is $\bold{Cholesky–Banachiewicz}$ algorithm :
# $$
# A=LL^T=\begin{bmatrix}
#     L_{11}& 0 & 0\\
#     L_{21} & L_{22} & 0\\
#     L_{31}  & L_{32}  & L_{33} 
# \end{bmatrix}\begin{bmatrix}
#     L_{11}  & L_{21}  & L_{31} \\
#     0  & L_{22}  & L_{32} \\
#     0 & 0 & L_{33} 
# \end{bmatrix}=\begin{bmatrix}
#     L_{11}^2  &   & (symmetric) \\
#     L_{21}L_{11}  & L_{21}^2+L_{22}^2  &  \\
#     L_{31}L_{11}& L_{31}L_{21}+L_{32}L_{22}& L_{33}^2+L_{32}^2+L_{31}^2 
# \end{bmatrix}
# $$
# 
# and therefore the following formulas for the entries of L:
# 
# $$
# L_{jj}=\sqrt{A_{jj}-\sum_{k=1}^{j-1}L_{jk}^2} \\
# \\
# L_{i,j}=\frac{\left(A_{ij}-\sum_{k=1}^{j-1}L_{jk}L_{ik}\right)}{L_{jj}} \  \ for \ \ i>j
# $$
# 
# > Use this algorithm to write a function that takes a matrix and gives its cholesky decomposition. also print the output for the following matrix :
# $$
# \mathbf{C}=
# \begin{bmatrix}
#     1 & 0.7 \\
#     0.7 & 1
# \end{bmatrix}
# $$ 
# 
# 
# > then use np.linalg.cholesky to check your answer 

# In[ ]:


# Code cell
def Cholesky(A):
    
    # write your code here
    
    pass


# #### 4-2-  Using Cholesky to generate correlated random numbers
# 
# The co-variance Matrix of any random vector Y
#  is given as $\mathbf{E}(YY^T)$
# , where Y
#  is a random column vector of size n×1
# . Now take a random vector, X
# , consisting of uncorrelated random variables with each random variable, $X_i$
# , having zero mean and unit variance 1
# . Since $X_i$
# 's are uncorrelated random variables with zero mean and unit variance, we have $\mathbf{E}(X_i X_j^T)=δ_{ij}$
# . Hence,
# $$
# \mathbf{E}(XX^T)=I
# $$
# To generate a random vector with a given covariance matrix $\mathbf{C}$
# , look at the Cholesky decomposition of $C$
#  i.e. $\mathbf{C}=LL^T$
#  
# Now look at the random vector $Z=LX$ :
# $$
# \mathbf{E}(ZZ^T)=\mathbf{E}\left((LX)(LX)^T\right)=L\mathbf{E}(XX^T)L^T =LL^T=\mathbf{C}
# $$
# Hence, the random vector $\mathbf{Z}$
#  has the desired co-variance matrix, $\mathbf{C}$
# 
#  >Make the covarience matirx of $C$
#   $$
# \mathbf{C}=
# \begin{bmatrix}
#     1 & 0.7 \\
#     0.7 & 1
# \end{bmatrix}
# $$ 
# 
#  
#  >Then we need another matrix with the desired standard deviation in the diagonal  $\Tau$
#  $$
# \mathbf{\Tau}=
# \begin{bmatrix}
#     1 & 0 \\
#     0 & 2
# \end{bmatrix}
# $$ 
# >Then find the cholesky decomposition of $C$
# 

# In[ ]:


# Code cell


# >Now  generate values for 2 independent random variables and put them in  2*1000 matrix $X$ (1000 samples)
# 
# you can use np.random.normal ( generates iid random variables each time )

# In[ ]:


# Code cell


# >then calculate $\mathbf{Z}=\Tau L X$ and then plot Z[1] in base of Z[0] and X[1] in base of X[0] 

# In[ ]:


# Code cell


# >now  check the correlation in generated samples with using np.correlate

# In[ ]:


# Code cell


# ## 5. Graphs (bonus)
# 
# I suggest running this question's code cells in google colab to install the $\textbf{PyGSP}$ package by  simply running the following code cell

# In[ ]:


get_ipython().system('pip install pygsp')


# 
# <!-- ![]( graph.png) -->
# <div style="text-align:center">
#     <img src="graph.png" alt="Image" />
# </div> 
# 
# 
# For the directed graph above the adjacency matrix is : 
# $
# A = \begin{bmatrix}
#     0  & 1  & 1 & 0\\
#     -1  & 0  & 1 & 1 \\
#     -1 & -1 & 0 & 1 \\
#     0  & -1  & -1 & 0
# \end{bmatrix}
# $ 
# 
# If we change the nodes numbering, the properties of the graph don't change yet the adjacency matrix changes. 
# 

# In[ ]:


# import required packages
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting
import cv2


# 
# #### 5-1-  Introduction to graphs 
# >construct the permutation matrix $\textbf{P}$ by modifying the Identity matrix. then use it to number the graph's nodes in reverse order

# In[ ]:


# Code cell

A_modified = ...


# In[ ]:


# Evaluation Cell
assert A_modified == np.matrix('0 -1 -1 0; 1 0 -1 -1; 1 1 0 -1; 0 1 1 0'), "wrong!"


# #### 5-2-  Graph image processing  
# 
# Inpainting is a classical signal processing problem where we wish to fill in the missing values in a
# partially observed signal. This is here done in the context of image processing for inferring missing pixel values in
# an image. The signal in the image is considered to be the image matrix flattened while the image is modeled as a 2D grid graph.
# Inpainting for an image can be formulated as below:
# 
# $$(M + \alpha L)x = y$$
# 
# where y is a partially observed graph signal (with missing values being 0), and M is a diagonal matrix that satisfies:
#  $$ M(i, i)=   \left\{
# \begin{array}{ll}
#       1, & if & y(i) & is & observed, \\
#       0, & if & y(i) & is & not & observed, \\
# \end{array} 
# \right.  $$
# 
# The Equation tries to find an x that nearly matches the observed values in y, and at the same time
# being smooth on the graph (the image here). The regularisation parameter α controls the trade-off between the data fidelity term and the
# smoothness prior. The solution can therefore be considered as an inpainted version of the partially observed signal.

# >Use the cv2 package to load a grayscale version of the cameraman image with a relatively low resolution, 64 by 64, then display the image using matplotlib.pyplot
# 
# - do not forget to convert RGB to gray to reduce the 3 color channels to one

# In[ ]:


# Code cell


# >Now flatten the image matrix to get the signal\
# >Then, construct the diagonal matrix M which has its diagonal 0 except for $p = 50%$ of its elements that are randomly set to 1\
# >At last, apply the observasion matrix M to the signal and construct y 

# In[ ]:


# Code cell


# In[ ]:


# do not edit this cell 
G = graphs.Grid2d(64,64)
L = G.L


# >To get x, apply the $\textbf{Cholesky decomposition}$ to ( $M+\alpha L$ ) and then solve the stated equation applying np.linalg.inv() to the two terms multiplied by x ( note that L is given by the code cell above )
# - check the soloution with "np.linalg.solve()"

# In[ ]:


# Code cell


# >For values of alpha 0.001 , 0.1 , 10 and p values 50 , 75 display the original, the damaged and the Inpainted image in subplots

# In[ ]:


# Code cell

