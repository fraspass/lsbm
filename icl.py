#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
import lsbm
from utilities import *

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Number of samples
M = 10000
## Burnin
B = 1000

## Load data
X = np.load('Data/X_icl2.npy')[:,:5]
lab = np.loadtxt('Data/labs2.csv', dtype=int)

## Define latent functions
fW = {}
for k in range(4):
    fW[k,0] = lambda x: np.array([x])
    for j in range(1,5):
        fW[k,j] = lambda x: np.array([x ** 2, x])

## Set up the model and posterior sampler
m = lsbm.lsbm_gibbs(X=X[:,:5], K=4, W_function=fW)
np.random.seed(11711)
m.initialise(z=KMeans(n_clusters=m.K).fit_predict(m.X[:,:5]), theta=X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=(1/m.n)**2, mu_theta=X[:,0].mean(), sigma_theta=10, b_0=0.001)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)
np.save('ICL/out_theta.npy',q[0])
np.save('ICL/out_z.npy',q[1])

## Estimate clustering
clust = estimate_communities(q=q[1],m=m)
## Evaluate adjusted Rand index
a = ari(clust, lab)
np.save('ICL/ari.npy',a)
clust = relabel_matching(lab, clust)

### Plots
xx = np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250)
## Calculate MAP for the curves based on the estimated clustering
v = m.map(z=clust, theta=m.X[:,0], range_values=xx)
## Parameters of plots
cdict = ['#1E88E5','#FFC107','#D81B60','#004D40']
mms = ['o', 'v', 'd', 's']
group = ['Mathematics','Medicine','Chemistry','Civil Engineering']
### Scatterplots
for j in range(1,m.d):
    for g in [2,3,0,1]:
        ix = np.where(lab == g)
        ix2 = np.where(clust == g)
        plt.scatter(m.X[:,0][ix], m.X[:,j][ix], c=cdict[g], label=group[g], marker=mms[g], edgecolor='black', linewidth=0.3)
        plt.plot(v[0][g,0], v[0][g,j], c=cdict[g])
    plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
    plt.ylabel('$$\\hat{\\mathbf{X}}_'+str(j+1)+'$$')
    plt.legend()
    plt.savefig('ICL/icl_1'+str(j+1)+'.pdf',bbox_inches='tight')
    plt.show(block=False); plt.clf(); plt.cla(); plt.close()

## Repeat with truncated power splines
knots = {}
nknots = 3
mmin = np.min(X,axis=0)[0]
mmax = np.max(X,axis=0)[0]
knots =  np.linspace(start=mmin,stop=mmax,num=nknots+2)[1:-1]
for k in range(m.K):
    fW[k,0] = lambda x: np.array([x])
    for j in range(1,m.d):
        fW[k,j] = lambda x: np.array([x, x ** 2] + [relu(x - knot) ** 2 for knot in knots])

## Set up the model and posterior sampler
m = lsbm.lsbm_gibbs(X=X[:,:5], K=4, W_function=fW)
np.random.seed(11711)
m.initialise(z=KMeans(n_clusters=m.K).fit_predict(m.X[:,:5]), theta=X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=(1/m.n)**2, mu_theta=X[:,0].mean(), sigma_theta=10, b_0=0.001)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)
np.save('ICL/out_theta_splines.npy',q[0])
np.save('ICL/out_z_splines.npy',q[1])

## Estimate clustering
clust = estimate_communities(q=q[1],m=m)
clust = relabel_matching(lab, clust)
## Evaluate adjusted Rand index
a = ari(clust, lab)
np.save('ICL/ari_splines.npy',a)

### Plots
xx = np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250)
## Calculate MAP for the curves based on the estimated clustering
v = m.map(z=clust, theta=m.X[:,0], range_values=xx)
## Parameters of plots
cdict = ['#1E88E5','#FFC107','#D81B60','#004D40']
mms = ['o', 'v', 'd', 's']
group = ['Mathematics','Medicine','Chemistry','Civil Engineering']
### Scatterplots
for j in range(1,m.d):
    for g in [2,3,0,1]:
        ix = np.where(lab == g)
        ix2 = np.where(clust == g)
        plt.scatter(m.X[:,0][ix], m.X[:,j][ix], c=cdict[g], label=group[g], marker=mms[g], edgecolor='black', linewidth=0.3)
        plt.plot(v[0][g,0], v[0][g,j], c=cdict[g])
    plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
    plt.ylabel('$$\\hat{\\mathbf{X}}_'+str(j+1)+'$$')
    plt.legend()
    plt.savefig('ICL/icl_splines_1'+str(j+1)+'.pdf',bbox_inches='tight')
    plt.show(block=False); plt.clf(); plt.cla(); plt.close()