#! /usr/bin/env python3
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import KMeans
import lsbm

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Number of samples and burnin
M = 10000
B = 1000

## Simulate matrix of edge probabilities
np.random.seed(1771)
n = 1000
K = 2
z = np.repeat(np.arange(K),n//K)
theta = np.random.beta(a=1,b=1,size=n)
X = np.zeros((n,3))
X[:500,:] = np.array([(1-theta[:500])**2, theta[:500] ** 2, 2*theta[:500]*(1-theta[:500])]).T
X[500:1000,:] = np.array([theta[500:1000] ** 2, 2*theta[500:1000]*(1-theta[500:1000]), (1-theta[500:1000]) ** 2]).T
EA = np.dot(X,X.T)
np.fill_diagonal(a=EA, val=np.zeros(n))
## Adjacency matrix
A = np.random.binomial(n=1,p=EA)
## Spectral embedding
Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:3]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))

## Quadratic latent functions
fW = {}
for k in range(2):
    fW[k,0] = lambda x: np.array([x ** 2, x, 1])
    fW[k,1] = lambda x: np.array([x ** 2, x, 1])
    fW[k,2] = lambda x: np.array([x ** 2, x, 1])

## Procrustes alignment for visualisation
from scipy.linalg import orthogonal_procrustes as proc
X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

## Plot of marginals
plt.scatter(X_tilde[:,0], X_tilde[:,1], c=np.array(['#009E73','#0072B2'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,0])
uu2 = np.argsort(X[z==1,0])
plt.plot(X[z==0,0][uu1], X[z==0,1][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,0][uu2], X[z==1,1][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("../Sim/x12_sim.pdf",bbox_inches='tight')
plt.show(block=False); plt.clf(); plt.cla(); plt.close()

plt.scatter(X_tilde[:,0], X_tilde[:,2], c=np.array(['#009E73','#0072B2'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,0])
uu2 = np.argsort(X[z==1,0])
plt.plot(X[z==0,0][uu1], X[z==0,2][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,0][uu2], X[z==1,2][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_3$$')
plt.savefig("../Sim/x13_sim.pdf",bbox_inches='tight')
plt.show(block=False); plt.clf(); plt.cla(); plt.close()

plt.scatter(X_tilde[:,1], X_tilde[:,2], c=np.array(['#009E73','#0072B2'])[z], edgecolor='black',linewidth=0.3)
uu1 = np.argsort(X[z==0,1])
uu2 = np.argsort(X[z==1,2])
plt.plot(X[z==0,1][uu1], X[z==0,2][uu1], '-', linewidth=3, c='black')
plt.plot(X[z==1,1][uu2], X[z==1,2][uu2], '-', linewidth=3, c='black')
plt.xlabel('$$\\hat{\\mathbf{X}}_2$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_3$$')
plt.savefig("../Sim/x23_sim.pdf",bbox_inches='tight')
plt.show(block=False); plt.clf(); plt.cla(); plt.close()

## Setup model and run MCMC
m = lsbm.lsbm_gibbs(X=X_tilde[:,:3], K=2, W_function=fW, fixed_function={})
np.random.seed(1771)
m.initialise(z=np.random.choice(2,size=m.n), theta=np.sqrt(np.abs(m.X[:,0])), 
                            Lambda_0=(1/m.n**2), mu_theta=np.sqrt(np.abs(m.X[:,0])).mean(), sigma_theta=10, b_0=0.001, first_linear=False)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)
np.save('../Sim/out_theta.npy',q[0])
np.save('../Sim/out_z.npy',q[1])

## Estimate clustering
clust = lsbm.estimate_communities(q=q[1],m=m)
clust = lsbm.relabel_matching(z, clust)
## Evaluate adjusted Rand index
a = ari(clust, z)
np.save('../Sim/ari.npy',a)

### Plot
from mpl_toolkits.mplot3d import Axes3D
xx = np.linspace(np.min(np.sqrt(np.abs(m.X[:,0]))),np.max(np.sqrt(np.abs(m.X[:,0]))),1000)
## Calculate MAP for the curves based on the estimated clustering
v = m.map(z=clust, theta=q[0][:,0].mean(axis=1), range_values=xx)
## Plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_tilde[:,0], X_tilde[:,1], X_tilde[:,2], c=np.array(['#009E73','#0072B2'])[z], s=1, alpha=0.75)
ax.scatter(v[0][0,0], v[0][0,1], v[0][0,2], s=1, c='#009E73')
ax.scatter(v[0][1,0], v[0][1,1], v[0][1,2], s=1, c='#0072B2')
ax.scatter(X[:,0],X[:,1],X[:,2],s=1,c='black',alpha=0.25)
ax.view_init(elev=25, azim=45)
plt.savefig("../Sim/x123_sim.pdf",bbox_inches='tight')
plt.show(block=False); plt.clf(); plt.cla(); plt.close()

## Cubic latent functions
fW = {}
for k in range(2):
    fW[k,0] = lambda x: np.array([x])
    fW[k,1] = lambda x: np.array([x ** 3, x ** 2, x, 1])
    fW[k,2] = lambda x: np.array([x ** 3, x ** 2, x, 1])

## Setup model and run MCMC
m = lsbm.lsbm_gibbs(X=X_tilde[:,:3], K=2, W_function=fW, fixed_function={})
np.random.seed(11711)
m.initialise(z=np.random.choice(2,size=m.n), theta=np.sqrt(np.abs(m.X[:,0])), 
                            Lambda_0=(1/m.n**2), mu_theta=np.sqrt(np.abs(m.X[:,0])).mean(), sigma_theta=10, b_0=0.001)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)
np.save('../Sim/out_cubic_theta.npy',q[0])
np.save('../Sim/out_cubic_z.npy',q[1])

## Estimate clustering
clust = lsbm.estimate_communities(q=q[1],m=m)
clust = lsbm.relabel_matching(z, clust)
## Evaluate adjusted Rand index
a = ari(clust, z)
np.save('../Sim/ari_cubic.npy',a)

### Plot
xx = np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),1000)
## Calculate MAP for the curves based on the estimated clustering
v = m.map(z=clust, theta=m.X[:,0], range_values=xx)
## Plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_tilde[:,0], X_tilde[:,1], X_tilde[:,2], c=np.array(['#009E73','#0072B2'])[z], s=1, alpha=0.75)
ax.scatter(v[0][0,0], v[0][0,1], v[0][0,2], s=1, c='#009E73')
ax.scatter(v[0][1,0], v[0][1,1], v[0][1,2], s=1, c='#0072B2')
ax.scatter(X[:,0],X[:,1],X[:,2],s=1,c='black',alpha=0.25)
ax.view_init(elev=25, azim=45)
plt.savefig("../Sim/x123_sim_cubic.pdf",bbox_inches='tight')
plt.show(block=False); plt.clf(); plt.cla(); plt.close()
