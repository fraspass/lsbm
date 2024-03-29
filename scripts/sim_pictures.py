#! /usr/bin/env python3
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

np.random.seed(1771)
n = 1000
K = 2

### Allocations
z = np.array([0]*(n//2) + [1]*(n//2))
np.save('../data/block_sim/z.npy', z)

### Means
m = {}
m[0] = np.array([0.75,0.25])
m[1] = np.array([0.25,0.75])
X = np.zeros((n,2))
for i in range(n):
    X[i] = m[z[i]]

### Stochastic blockmodel
A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i,n):
        A[i,j] = np.random.binomial(n=1,p=np.inner(m[z[i]],m[z[j]]),size=1)
        A[j,i] = A[i,j]

np.save('../data/block_sim/A_sbm.npy', A)

## Embedding
Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:2]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))

## Align to true embedding
from scipy.linalg import orthogonal_procrustes as proc
X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

## Save
np.save('../data/block_sim/X_sbm.npy', X_hat)

## Plot
plt.scatter(X_tilde[:,0], X_tilde[:,1], c=np.array(['#13A0C6','#EB8F18'])[z], edgecolor='black',linewidth=0.3)
plt.scatter([m[0][0], m[1][0]], [m[0][1], m[1][1]], edgecolor='black',linewidth=0.3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("../pictures/sbm_sim.pdf",bbox_inches='tight')
plt.clf()

### Degree-corrected blockmodel
for i in range(n):
    X[i] = np.random.beta(a=1,b=1) * m[z[i]]

## Adjacency matrix
A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i,n):
        A[i,j] = np.random.binomial(n=1,p=np.inner(X[i],X[j]),size=1)
        A[j,i] = A[i,j]

np.save('../data/block_sim/A_dcsbm.npy', A)

## Embedding
Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:2]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))
X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

## Save
np.save('../data/block_sim/X_dcsbm.npy', X_hat)

## Plot
plt.scatter(X_tilde[:,0], X_tilde[:,1], c=np.array(['#13A0C6','#EB8F18'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,0])
uu2 = np.argsort(X[z==1,0])
plt.plot(X[z==0,0][uu1], X[z==0,1][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,0][uu2], X[z==1,1][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("../pictures/dcsbm_sim.pdf",bbox_inches='tight')
plt.clf()

### Quadratic
np.random.seed(1771)
gamma = [-1, -4]
for i in range(n):
    X[i,0] = np.random.beta(a=2,b=1) / 2
    X[i,1] = gamma[z[i]] * (X[i,0] ** 2) + X[i,0]
    ## X[i,1] = (gamma[z[i]] * X[i,0] * (X[i,0] - 1) + X[i,0]) / 2

## Adjacency matrix
A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i,n):
        A[i,j] = np.random.binomial(n=1,p=np.inner(X[i],X[j]),size=1)
        A[j,i] = A[i,j]

np.save('../data/block_sim/A_quad.npy', A)

## Embedding
Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:2]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))
X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

## Save
np.save('../data/block_sim/X_quad.npy', X_hat)

## Plot
plt.scatter(X_tilde[:,0], X_tilde[:,1], c=np.array(['#13A0C6','#EB8F18'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,0])
uu2 = np.argsort(X[z==1,0])
plt.plot(X[z==0,0][uu1], X[z==0,1][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,0][uu2], X[z==1,1][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("../pictures/quadratic_sim.pdf",bbox_inches='tight')
plt.clf()