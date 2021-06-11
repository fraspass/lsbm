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

def relu(x):
    return x * (x > 0)

## X = np.loadtxt('Data/icll2.csv', delimiter=',')[:,:5]
X = np.load('Data/X_icl2.npy')[:,:5]
lab = np.loadtxt('Data/labs2.csv', dtype=int)

import lsbm
fW = {}
fW[0] = lambda x: np.array([x])
for j in range(5):
    fW[j] = lambda x: np.array([x ** 2, x])

## Truncated power splines
knots = {}
nknots = 3
mmin = np.min(X,axis=0)[0]
mmax = np.max(X,axis=0)[0]
knots =  np.linspace(start=mmin,stop=mmax,num=nknots+2)[1:-1]

fW[0] = lambda x: np.array([x])
for j in range(1,5):
    fW[j] = lambda x: np.array([x, x ** 2, x ** 3] + [relu(x - knot) ** 3 for knot in knots])

## This works when variance and theta are NOT resampled
m = lsbm.lsbm_gibbs(X=X[:,:5], K=4, W_function=fW)
np.random.seed(11711)
m.initialise(z=KMeans(n_clusters=m.K).fit_predict(m.X[:,:5]), theta=X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=1/m.n, mu_theta=X[:,0].mean(), sigma_theta=10, b_0=0.01)
q = m.mcmc(samples=10000, burn=1000, sigma_prop=0.5, thinning=1)

ari(lab, m.z)

psm = np.zeros((m.n,m.n))
for i in range(q[1].shape[2]):
    psm += np.equal.outer(q[1][:,0,i],q[1][:,0,i])

psm /= q[1].shape[2]

from sklearn.cluster import AgglomerativeClustering
cluster_model = AgglomerativeClustering(n_clusters=m.K, affinity='precomputed', linkage='average') 
clust = cluster_model.fit_predict(1-psm) + 1
ari(clust, lab)

import pandas as pd
pd.crosstab(clust, lab)

import matplotlib.pyplot as plt
plt.plot(q[0][:,0].T)
plt.show()

mm = lsbm.lsbm_gibbs(X=X[:,:5], K=5, W_function=fW, fixed_function={})
mm.initialise(z=lab, theta=X[:,0], 
        Lambda_0=1/mm.n, mu_theta=X[:,0].mean(), sigma_theta=10, b_0=0.01)

fig, ax = plt.subplots()
cdict = ['#1E88E5','#FFC107','#D81B60','#004D40']
mms = ['o', 'v', 'd', 's']
group = ['Mathematics','Medicine','Chemistry','Civil Engineering']
xx = np.linspace(np.min(X[:,0]),np.max(X[:,0]))
for g in [2,3,0,1]:
    ix = np.where(lab == g)
    ax.scatter(X[:,0][ix], X[:,1][ix], c = cdict[g], label = group[g], marker=mms[g],edgecolor='black',linewidth=0.3)
    ax.plot(xx, np.inner(fW[1](xx).T, mm.mu[1][g]), c = cdict[g])

ax.legend()
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("x12.pdf",bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
for g in [2,3,0,1]:
    ix = np.where(lab == g)
    ax.scatter(X[:,0][ix], X[:,2][ix], c = cdict[g], label = group[g], marker=mms[g],edgecolor='black',linewidth=0.3)
    ax.plot(xx, np.inner(fW[2](xx).T, mm.mu[2][g]), c = cdict[g])

plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_3$$')
plt.savefig("x13.pdf",bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
for g in [2,3,0,1]:
    ix = np.where(lab == g)
    ax.scatter(X[:,0][ix], X[:,3][ix], c = cdict[g], label = group[g], marker=mms[g],edgecolor='black',linewidth=0.3)
    ax.plot(xx, np.inner(fW[3](xx).T, mm.mu[3][g]), c = cdict[g])

plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_4$$')
plt.savefig("x14.pdf",bbox_inches='tight')
plt.show()

fig, ax = plt.subplots()
for g in [2,3,0,1]:
    ix = np.where(lab == g)
    ax.scatter(X[:,0][ix], X[:,4][ix], c = cdict[g], label = group[g], marker=mms[g],edgecolor='black',linewidth=0.3)
    ax.plot(xx, np.inner(fW[4](xx).T, mm.mu[4][g]), c = cdict[g])

plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_5$$')
plt.savefig("x15.pdf",bbox_inches='tight')
plt.show()


plt.scatter(X[:,0],X[:,2],c=m.z)
plt.show()

plt.scatter(X[:,0],X[:,3],c=m.z)
plt.show()

plt.scatter(X[:,0],X[:,4],c=m.z)
plt.show()

map_est = m.map(z=clust, theta=m.X[:,0])
ci = m.bootstrapped_ci(z=clust, theta=m.X[:,0], boot_samples=500)

curve = {}
for j in range(m.d):
    if not (j == 0 and m.first_linear):
        curve[j] = {}
        for k in range(m.K):
            curve[j][k] = np.zeros((200,500))
            for i in range(200):
                x = np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200)
                curve[j][k][i] = np.array([np.dot(ci[j][k][u],m.fW[j](x[i])) for u in range(500)])

import matplotlib.pyplot as plt
cols = np.array(['red','blue','green','orange'])
plt.scatter(m.X[:,0],m.X[:,1],c=cols[clust], s=20, alpha=0.5)
for k in range(m.K):
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), np.percentile(curve[1][k], q=97.5, axis=1), '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), np.percentile(curve[1][k], q=2.5, axis=1), '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), [np.dot(map_est[0][1][k],m.fW[1](x)) for x in np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200)], c=cols[k])

plt.show()

cols = np.array(['red','blue','green','orange'])
plt.scatter(m.X[:,0],m.X[:,1],c=cols[clust], s=20, alpha=0.5)
for k in range(m.K):
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250), map_est[1][k,1][:,0], '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250), map_est[1][k,1][:,1], '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250), map_est[0][k,1], c=cols[k])

plt.show()

plt.scatter(m.X[:,0],m.X[:,2],c=cols[clust], s=20, alpha=0.5)
for k in range(m.K):
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), np.percentile(curve[2][k], q=97.5, axis=1), '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), np.percentile(curve[2][k], q=2.5, axis=1), '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), [np.dot(map_est[0][2][k],m.fW[2](x)) for x in np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200)], c=cols[k])

plt.show()

plt.scatter(m.X[:,0],m.X[:,3],c=cols[clust], s=20, alpha=0.5)
for k in range(m.K):
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), np.percentile(curve[3][k], q=97.5, axis=1), '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), np.percentile(curve[3][k], q=2.5, axis=1), '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), [np.dot(map_est[0][3][k],m.fW[3](x)) for x in np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200)], c=cols[k])

plt.show()

plt.scatter(m.X[:,0],m.X[:,4],c=cols[clust], s=20, alpha=0.5)
for k in range(m.K):
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), np.percentile(curve[4][k], q=97.5, axis=1), '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), np.percentile(curve[4][k], q=2.5, axis=1), '--', c=cols[k])
    plt.plot(np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200), [np.dot(map_est[0][4][k],m.fW[4](x)) for x in np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),200)], c=cols[k])

plt.show()