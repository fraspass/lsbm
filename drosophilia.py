#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Simulate 2-dimensional embedding of the Drosophilia connectome
n = [100,21,29,63]
fn = lambda x: np.array([-−2.5*(x**2)−2.25*x, 1.5*(x**2) + x, 0.2*(x**2) + 0.6*x, −1.750*(x**2)−1.375*x, 0.5500*(x**2)+0.6750*x]) 
means = {}
means[1] = np.array([-.4,-.25,-.25,-.25,0,.25])
means[2] = np.array([-.1,.075,.125,-.375,.075,.25])
means[3] = np.array([-.075,-.05,-0.125,0,0,0])
covs = {}
covs[1] = 0.025 * (np.ones((6,6)) * .7 + np.diag(np.ones(6) * .3))
covs[2] = 0.005 * (-np.ones((6,6)) * .8 + np.diag(np.ones(6) * 1.8))
covs[3] = 0.005 * (np.ones((6,6)) * .5 + np.diag(np.ones(6) * .5))

## Embedding and community allocations
X = np.zeros((np.sum(n),6))
z = np.zeros(np.sum(n))
np.random.seed(117)
for i in range(np.sum(n)):
    clust = int(np.searchsorted(a=np.cumsum(n), v=i))
    z[i] = clust
    if clust == 0:
        X[i,0] = np.random.uniform(-1.1,0,size=1) 
        X[i,1:] = np.random.normal(fn(X[i,0]),scale=np.sqrt(0.05),size=1)
    elif clust == 1:
        X[i] = np.random.multivariate_normal(mean=means[1], cov=covs[1], size=1) 
    elif clust == 2:
        X[i] = np.random.multivariate_normal(mean=means[2], cov=covs[2], size=1) 
    else:
        X[i] = np.random.multivariate_normal(mean=means[3], cov=covs[3], size=1) 

## Scatterplots
fig, axs = plt.subplots(3, 2)
axs[0, 0].scatter(X[:,0],X[:,1],c=z)
axs[0, 1].scatter(X[:,0],X[:,2],c=z)
axs[1, 0].scatter(X[:,0],X[:,3],c=z)
axs[1, 1].scatter(X[:,0],X[:,4],c=z)
axs[2, 0].scatter(X[:,0],X[:,5],c=z)
plt.show()

## Define latent basis functions
K = 4
d = 6
fW = {}
for j in range(d):
    if j == 0:
        fW[0,0] = lambda x: np.array([x])
    else:
        fW[0,j] = lambda x: np.array([x ** 2, x])
    for k in range(1,K):
        fW[k,j] = lambda x: np.array([1])

## Set up the model & posterior sampler
m = lsbm.lsbm_gibbs(X=X, K=4, W_function=fW)
np.random.seed(11711)
m.initialise(z=z, theta=X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=(1/m.n)**2, mu_theta=X[:,0].mean(), sigma_theta=10, b_0=0.001)
q = m.mcmc(samples=10000, burn=1000, sigma_prop=0.05, thinning=1)

## Posterior similarity matrix
psm = np.zeros((m.n,m.n))
for i in range(q[1].shape[2]):
    psm += np.equal.outer(q[1][:,0,i],q[1][:,0,i])

psm /= q[1].shape[2]

## Clustering based on posterior similarity matrix
from sklearn.cluster import AgglomerativeClustering
cluster_model = AgglomerativeClustering(n_clusters=m.K, affinity='precomputed', linkage='average') 
clust = cluster_model.fit_predict(1-psm) + 1
ari(clust, z)

## Scatterplots
fig, axs = plt.subplots(3, 2)
axs[0, 0].scatter(X[:,0],X[:,1],c=clust)
axs[0, 1].scatter(X[:,0],X[:,2],c=clust)
axs[1, 0].scatter(X[:,0],X[:,3],c=clust)
axs[1, 1].scatter(X[:,0],X[:,4],c=clust)
axs[2, 0].scatter(X[:,0],X[:,5],c=clust)
plt.show()