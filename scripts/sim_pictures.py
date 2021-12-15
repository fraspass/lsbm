#! /usr/bin/env python3
import numpy as np
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

np.random.seed(1771)
n = 1000
K = 2
### Means
m = {}
m[0] = np.array([0.75,0.25])
m[1] = np.array([0.25,0.75])
X = np.zeros((n,2))
for i in range(n):
    X[i] = m[z[i]]

### Allocations
z = np.array([0]*(n//2) + [1]*(n//2))

### Stochastic blockmodel
A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i,n):
        A[i,j] = np.random.binomial(n=1,p=np.inner(m[z[i]],m[z[j]]),size=1)
        A[j,i] = A[i,j]

Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:2]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))

from scipy.linalg import orthogonal_procrustes as proc
X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

plt.scatter(X_tilde[:,0], X_tilde[:,1], c=np.array(['#009E73','#0072B2'])[z], edgecolor='black',linewidth=0.3)
plt.scatter([m[0][0], m[1][0]], [m[0][1], m[1][1]], edgecolor='black',linewidth=0.3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("sbm_sim.pdf",bbox_inches='tight')
plt.show()

### Degree-corrected blockmodel
for i in range(n):
    X[i] = np.random.uniform() * m[z[i]]

A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i,n):
        A[i,j] = np.random.binomial(n=1,p=np.inner(X[i],X[j]),size=1)
        A[j,i] = A[i,j]

Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:2]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))

X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

plt.scatter(X_tilde[:,0], X_tilde[:,1], c=np.array(['#009E73','#0072B2'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,0])
uu2 = np.argsort(X[z==1,0])
plt.plot(X[z==0,0][uu1], X[z==0,1][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,0][uu2], X[z==1,1][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("dcsbm_sim.pdf",bbox_inches='tight')
plt.show()

### Quadratic
np.random.seed(1771)
gamma = [-2, 2]
for i in range(n):
    X[i,0] = np.random.uniform() / 2
    X[i,1] = (gamma[z[i]] * X[i,0] * (X[i,0] - 1) + X[i,0]) / 2

A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i,n):
        A[i,j] = np.random.binomial(n=1,p=np.inner(X[i],X[j]),size=1)
        A[j,i] = A[i,j]

Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:2]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))

X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

plt.scatter(X_tilde[:,0], X_tilde[:,1], c=np.array(['#009E73','#0072B2'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,0])
uu2 = np.argsort(X[z==1,0])
plt.plot(X[z==0,0][uu1], X[z==0,1][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,0][uu2], X[z==1,1][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("quadratic_sim.pdf",bbox_inches='tight')
plt.show()