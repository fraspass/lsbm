#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
from utilities import *

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Load data
X = np.load('Data/X_icl2.npy')[:,:5]
n = X.shape[0]
lab = np.loadtxt('Data/labs2.csv', dtype=int)

## Import lsbm_gp
import lsbm_gp

#####################
## Kernel function ##
#####################

## Basis functions
fW = {}
fW[0] = lambda x: np.transpose(np.array([x]))
for j in range(1,5):
    fW[j] = lambda x: np.transpose(np.array([x ** 2, x]))

## Weighting matrices
Delta = {}
Delta[0] = n**2 / np.array([[0.88593648]])
for j in range(1,5):
    Delta[j] = n**2 * np.linalg.inv(np.array([[ 11.26059931, -13.76523829],[-13.76523829,  17.7129081 ]]))

## Kernels
csi = {}
for k in range(4):
    csi[k,0] = lambda theta,theta_prime: Delta[0] * np.matmul(fW[0](theta).reshape(-1,1),fW[0](theta_prime).reshape(1,-1)) 
    for j in range(1,5):
        csi[k,j] = lambda theta,theta_prime: np.matmul(np.matmul(fW[j](theta),Delta[j]),np.transpose(fW[j](theta_prime)))

### Setup model and MCMC
M = 100; B = 10
m = lsbm_gp.lsbm_gp_gibbs(X=X[:,:5], K=4, csi=csi)
np.random.seed(11711)
m.initialise(z=np.random.choice(m.K,size=m.n), theta=X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            mu_theta=X[:,0].mean(), sigma_theta=10, b_0=0.001, first_linear=True)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)

## Estimate clustering
clust = estimate_communities(q=q[1],m=m)
## Evaluate adjusted Rand index
a = ari(clust, lab)
np.save('ICL/ari.npy',a)
clust = relabel_matching(lab, clust)

## Calculate MAP curves
map_est = m.map(z=clust, theta=m.X[:,0], range_values=np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250))

## Calculate confusion matrix
import pandas as pd
pd.crosstab(clust, lab)

## Plots
fig, ax = plt.subplots()
cdict = ['#1E88E5','#FFC107','#D81B60','#004D40']
mms = ['o', 'v', 'd', 's']
group = ['Mathematics','Medicine','Chemistry','Civil Engineering']
xx = np.linspace(np.min(X[:,0]),np.max(X[:,0]),250)
for g in [2,3,0,1]:
    ix = np.where(lab == g)
    ix2 = np.where(clust == g)
    ## ax.fill_between(xx, map_est[1][g,1][:,0], map_est[1][g,1][:,1], facecolor=lighten_color(cdict[g]), interpolate=True, alpha=0.5)
    ax.plot(xx, map_est[0][g,1], c = cdict[g], alpha=0.5)
    ax.scatter(X[:,0][ix], X[:,1][ix], c = cdict[g], label = group[g], marker=mms[g], edgecolor='black', linewidth=0.3)

ax.legend()
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')

for g in [2,3,0,1]:
    ix = np.where(lab == g)
    ax.plot(xx, map_est[0][g,1], c = cdict[g])
    ax.scatter(X[:,0][ix], X[:,1][ix], c = cdict[g], label = group[g], marker=mms[g], edgecolor='black', linewidth=0.3)

plt.savefig("x12.pdf",bbox_inches='tight')
plt.show()