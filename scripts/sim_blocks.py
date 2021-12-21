#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
import lsbm

## Number of samples and burnin
M = 10000
B = 1000

## Simulated latent structure blockmodels
z = np.load('../data/block_sim/z.npy')
X_sbm = np.load('../data/block_sim/X_sbm.npy')
X_dcsbm = np.load('../data/block_sim/X_dcsbm.npy')
X_quad = np.load('../data/block_sim/X_quad.npy')

## Stochastic blockmodel latent functions
fW = {}
for k in range(2):
    fW[k,0] = lambda x: np.array([1])
    fW[k,1] = lambda x: np.array([1])

## Setup model and run MCMC -- SBM
m = lsbm.lsbm_gibbs(X=X_sbm, K=2, W_function=fW, fixed_function={})
np.random.seed(11711)
m.initialise(z=KMeans(n_clusters=2).fit_predict(m.X), theta=(np.abs(m.X[:,0])+np.random.normal(size=m.n,scale=0.000001)),
                            Lambda_0=(1/m.n**2), mu_theta=np.sqrt(np.abs(m.X[:,0])).mean(), sigma_theta=1, b_0=0.0001)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.001, thinning=1)
np.save('../Block_Sim/out_theta_sbm.npy', q[0])
np.save('../Block_Sim/out_z_sbm.npy', q[1])

## Estimate clustering
clust = lsbm.estimate_communities(q=q[1],m=m)
## Evaluate adjusted Rand index
a = ari(clust, z)
np.save('../Block_Sim/ari_sbm.npy', a)
np.save('../Block_Sim/clust_sbm.npy', clust)
np.save('../Block_Sim/z_sbm.npy', z)

## Degree-corrected stochastic blockmodel latent functions
fW = {}
for k in range(2):
    fW[k,0] = lambda x: np.array([x])
    fW[k,1] = lambda x: np.array([x])

## Setup model and run MCMC -- DCSBM
m = lsbm.lsbm_gibbs(X=X_dcsbm, K=2, W_function=fW, fixed_function={})
np.random.seed(11711)
m.initialise(z=KMeans(n_clusters=2).fit_predict(m.X), theta=(np.abs(m.X[:,0])+np.random.normal(size=m.n,scale=0.000001)),
                            Lambda_0=(1/m.n**2), mu_theta=np.sqrt(np.abs(m.X[:,0])).mean(), sigma_theta=1, b_0=0.0001, first_linear=True)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.001, thinning=1)
np.save('../Block_Sim/out_theta_dcsbm.npy', q[0])
np.save('../Block_Sim/out_z_dcsbm.npy', q[1])

## Estimate clustering
clust = lsbm.estimate_communities(q=q[1],m=m)
## Evaluate adjusted Rand index
a = ari(clust, z)
np.save('../Block_Sim/ari_dcsbm.npy', a)
np.save('../Block_Sim/clust_dcsbm.npy', clust)
np.save('../Block_Sim/z_dcsbm.npy', z)

## Quadratic stochastic blockmodel latent functions
fW = {}
for k in range(2):
    fW[k,0] = lambda x: np.array([x])
    fW[k,1] = lambda x: np.array([x ** 2, x])

## Setup model and run MCMC -- Quadratic model
m = lsbm.lsbm_gibbs(X=X_quad, K=2, W_function=fW, fixed_function={})
np.random.seed(11711)
m.initialise(z=KMeans(n_clusters=2).fit_predict(m.X), theta=(np.abs(m.X[:,0])+np.random.normal(size=m.n,scale=0.000001)), 
                            Lambda_0=(1/m.n**2), mu_theta=np.sqrt(np.abs(m.X[:,0])).mean(), sigma_theta=0.1, b_0=0.0001, first_linear=True)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.001, thinning=1)
np.save('../Block_Sim/out_theta_quad.npy', q[0])
np.save('../Block_Sim/out_z_quad.npy', q[1])

## Estimate clustering
clust = lsbm.estimate_communities(q=q[1],m=m)
## Evaluate adjusted Rand index
a = ari(clust, z)
np.save('../Block_Sim/ari_quad.npy', a)
np.save('../Block_Sim/clust_quad.npy', clust)
np.save('../Block_Sim/z_quad.npy', z)