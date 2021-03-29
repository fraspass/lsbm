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
z = np.repeat(np.arange(K),n//K)
theta = np.random.beta(a=1,b=1,size=n)
X = np.zeros((n,3))
X[:500,:] = np.array([(1-theta[:500])**2, theta[:500] ** 2, 2*theta[:500]*(1-theta[:500])]).T
X[500:1000,:] = np.array([theta[500:1000] ** 2, 2*theta[500:1000]*(1-theta[500:1000]), (1-theta[500:1000]) ** 2]).T
EA = np.dot(X,X.T)

A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i,n):
        A[i,j] = np.random.binomial(n=1,p=EA[i,j],size=1)
        A[j,i] = A[i,j]

Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:3]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))

fW = {}
fW[0] = lambda x: np.array([x])
fW[1] = lambda x: np.array([x ** 3, x ** 2, x, 1])
fW[2] = lambda x: np.array([x ** 3, x ** 2, x, 1])

from scipy.linalg import orthogonal_procrustes as proc
X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

plt.scatter(X_tilde[:,0], X_tilde[:,1], c=np.array(['#009E73','#0072B2'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,0])
uu2 = np.argsort(X[z==1,0])
plt.plot(X[z==0,0][uu1], X[z==0,1][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,0][uu2], X[z==1,1][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("x12_sim.pdf",bbox_inches='tight')
plt.show()

plt.scatter(X_tilde[:,1], X_tilde[:,2], c=np.array(['#009E73','#0072B2'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,1])
uu2 = np.argsort(X[z==1,2])
plt.plot(X[z==0,1][uu1], X[z==0,2][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,1][uu2], X[z==1,2][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_2$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_3$$')
plt.savefig("x23_sim.pdf",bbox_inches='tight')
plt.show()

## Truncated power splines
knots = {}
nknots = 3
mmin = np.min(X_tilde[:,0])
mmax = np.max(X_tilde[:,0])
knots =  np.linspace(start=mmin,stop=mmax,num=nknots+2)[1:-1]

def relu(x):
    return x * (x > 0)

fW[0] = lambda x: np.array([x])
for j in [1,2]:
    fW[j] = lambda x: np.array([1, x, x ** 2, x ** 3] + [relu(x - knot) ** 3 for knot in knots])

import lsbm
m = lsbm.lsbm_gibbs(X=X_tilde[:,:3], K=2, W_function=fW, fixed_function={})
np.random.seed(1771)
m.initialise(z=np.random.choice(2,size=m.n), theta=m.X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=1/m.n, mu_theta=m.X[:,0].mean(), sigma_theta=10, b_0=0.01)
q = m.mcmc(samples=10000, burn=1000, sigma_prop=0.5, thinning=1)

## Posterior similarity matrix
psm = np.zeros((m.n,m.n))
for i in range(q[1].shape[2]):
    psm += np.equal.outer(q[1][:,0,i],q[1][:,0,i])

psm /= q[1].shape[2]

## Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
cluster_model = AgglomerativeClustering(n_clusters=m.K, affinity='precomputed', linkage='average') 
clust = cluster_model.fit_predict(1-psm) + 1

## ARI
ari(z, clust)

### Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mm = lsbm.lsbm_gibbs(X=X_tilde[:,:3], K=2, W_function=fW, fixed_function={})
mm.initialise(z=clust-1, theta=m.X[:,0], Lambda_0=1/m.n, mu_theta=m.X[:,0].mean(), sigma_theta=10, b_0=0.01)
xx = np.linspace(np.min(X_tilde[:,0]),np.max(X_tilde[:,0]), 1000)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_tilde[:,0], X_tilde[:,1], X_tilde[:,2], c=np.array(['#009E73','#0072B2'])[clust-1], s=1, alpha=0.75)
ax.scatter(xx, np.inner(fW[1](xx).T, mm.mu[1][0]), np.inner(fW[2](xx).T, mm.mu[2][0]), s=1,c='#009E73')
ax.scatter(xx, np.inner(fW[1](xx).T, mm.mu[1][1]), np.inner(fW[2](xx).T, mm.mu[2][1]), s=1,c='#0072B2')
ax.scatter(X[:,0],X[:,1],X[:,2],s=1,c='black')
ax.view_init(elev=25, azim=45)
plt.savefig("x123_sim_splines.pdf",bbox_inches='tight')
plt.show()
