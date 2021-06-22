#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
import lsbm
from sklearn.preprocessing import LabelEncoder as labeler

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Import labels
lab = np.loadtxt('Data/drosophila_labels.csv', dtype=str)
lab = labeler().fit(lab).transform(lab)
## Import embeddings
X = np.loadtxt('Data/drosophila_dase.csv', delimiter=',')

def relu(x):
    return x * (x > 0)

## Truncated power splines
knots = {}
nknots = 3
mmin = np.min(X,axis=0)[0]
mmax = np.max(X,axis=0)[0]
knots =  np.linspace(start=mmin,stop=mmax,num=nknots+2)[1:-1]

fW = {}
for k in range(4):
    fW[k,0] = lambda x: np.array([x])
    for j in range(1,6):
        if k == 0:
            fW[k,j] = lambda x: np.array([x, x ** 2, x ** 3] + [relu(knot - x) ** 3 for knot in knots])
        elif k == 1:
            if j == 5:
                fW[k,j] = lambda x: np.array([1,x])
            else:
                fW[k,j] = lambda x: np.array([x])
        elif k == 3:
            fW[k,j] = lambda x: np.array([x])
        else:
            if j <= 2:
                fW[k,j] = lambda x: np.array([x])
            else:
                fW[k,j] = lambda x: np.array([1,x])

## Set up the model & posterior sampler
m = lsbm.lsbm_gibbs(X=X, K=4, W_function=fW)
## For running this code on real data, only the 6-dimensional Drosophilia connectome embeddings are needed.
np.random.seed(111)
## Initialisation - For simplicity, here it is initialised from the truth
## When running this code on real data, this could be initialised from k-means
## IMPORTANT: Label 0 MUST correspond to 'quadratic' community (at least some elements of it)! 
from sklearn.cluster import KMeans
z_init = KMeans(n_clusters=6, random_state=0).fit_predict(X) + 1
## Appropriate relabelling for matching functions
z_init[np.where(z_init == 6)[0]] = 0
z_init[np.where(z_init == 3)[0]] = 0
z_init[np.where(z_init == 4)[0]] = 0
z_init[np.where(z_init == 5)[0]] = 3
z_init[np.where(z_init == 2)[0]] = -1
z_init[np.where(z_init == 1)[0]] = 2
z_init[np.where(z_init == -1)[0]] = 1
z_init[np.where(z_init == 3)[0]] = -1
z_init[np.where(z_init == 1)[0]] = 3
z_init[np.where(z_init == -1)[0]] = 1

## Initialise model
np.random.seed(111)
m.initialise(z=np.copy(z_init), theta=X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=(1/m.n)**2, mu_theta=X[:,0].mean(), sigma_theta=10, b_0=0.001, first_linear=True)
## Run the sampler
np.random.seed(111)
q = m.mcmc(samples=1000, burn=10, sigma_prop=0.05, thinning=1)

## Scaled posterior similarity matrix
psm = np.zeros((m.n,m.n))
for i in range(q[1].shape[2]):
    psm += np.equal.outer(q[1][:,0,i],q[1][:,0,i])

## Posterior similarity matrix (estimate)
psm /= q[1].shape[2]

## Clustering based on posterior similarity matrix (hierarchical clustering)
from sklearn.cluster import AgglomerativeClustering
cluster_model = AgglomerativeClustering(n_clusters=m.K, affinity='precomputed', linkage='average') 
clust = cluster_model.fit_predict(1-psm)
## Evaluate adjusted Rand index
ari(clust, lab)

## Majority rule
from collections import Counter
cc = np.apply_along_axis(func1d=lambda x: int(Counter(x).most_common(1)[0][0]), axis=1, arr=q[1][:,0]) 
ari(cc, lab)

## Plot
xx = np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250)
v = m.map(z=cc, theta=X[:,0], range_values=xx)

cdict = ['#1E88E5','#FFC107','#D81B60','#23C14B']
mms = ['o', 'v', 's', 'd']
group = ['Kenyon Cells','Input Neurons','Output Neurons','Projection Neurons']
### Scatterplots
fig, ax = plt.subplots(1, 5)
for j in range(1,6):
    for g in range(4):
        ix = np.where(lab == g)
        ix2 = np.where(cc == g)
        pos_min = np.searchsorted(xx,np.min(X[:,0][ix2]))
        pos_max = np.searchsorted(xx,np.max(X[:,0][ix2])) + 1
        ax[j-1].scatter(X[:,0][ix], X[:,j][ix], c = cdict[g], label = group[g], marker=mms[g], edgecolor='black', linewidth=0.3)
        if g == 0:
            ax[j-1].plot(xx[pos_min:pos_max], v[0][g,j][pos_min:pos_max], c = 'black')
        else:
            ax[j-1].plot(v[0][g,0][pos_min:pos_max], v[0][g,j][pos_min:pos_max], c = 'black')
        ax[j-1].set_xlabel('$$\\hat{\\mathbf{X}}_1$$')
        ax[j-1].set_ylabel('$$\\hat{\\mathbf{X}}_'+str(j)+'$$')
        if j == 1:
            ax[j-1].legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=4)

## 

plt.savefig("x12.pdf",bbox_inches='tight')
plt.show()

### Define latent basis functions
# Cluster 0: Linearity in first component + quadratics
# Clusters 1-3: GMM 
K = 4
d = 6
fW = {}
for j in range(d):
    if j == 0:
        fW[0,0] = lambda x: np.array([1])
    else:
        fW[0,j] = lambda x: np.array([x ** 2, x])
    for k in range(1,K):
        fW[k,j] = lambda x: np.array([1])

## Set up the model & posterior sampler
m = lsbm.lsbm_gibbs(X=X, K=4, W_function=fW)
## For running this code on real data, only the 6-dimensional Drosophilia connectome embeddings are needed.
np.random.seed(111)
## Initialisation - For simplicity, here it is initialised from the truth
## When running this code on real data, this could be initialised from k-means
## IMPORTANT: Label 0 MUST correspond to 'quadratic' community (at least some elements of it)! 

## Initialise model
np.random.seed(111)
m.initialise(z=np.copy(z_init), theta=X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=(1/m.n)**2, mu_theta=X[:,0].mean(), sigma_theta=10, b_0=0.001, first_linear=[True,False,False,False])
## Run the sampler
np.random.seed(111)
q = m.mcmc(samples=1000, burn=10, sigma_prop=0.05, thinning=1)

## Majority rule
from collections import Counter
cc = np.apply_along_axis(func1d=lambda x: int(Counter(x).most_common(1)[0][0]), axis=1, arr=q[1][:,0]) 
ari(cc, lab)

## Plot
xx = np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250)
v = m.map(z=cc, theta=X[:,0], range_values=xx)

cdict = ['#1E88E5','#FFC107','#D81B60','#23C14B']
mms = ['o', 'v', 's', 'd']
group = ['Kenyon Cells','Input Neurons','Output Neurons','Projection Neurons']
### Scatterplots
fig, ax = plt.subplots(1, 5)
for j in range(1,6):
    for g in range(4):
        ix = np.where(lab == g)
        ix2 = np.where(cc == g)
        pos_min = np.searchsorted(xx,np.min(X[:,0][ix2]))
        pos_max = np.searchsorted(xx,np.max(X[:,0][ix2])) + 1
        ax[j-1].scatter(X[:,0][ix], X[:,j][ix], c = cdict[g], label = group[g], marker=mms[g], edgecolor='black', linewidth=0.3)
        if g == 0:
            ax[j-1].plot(xx[pos_min:pos_max], v[0][g,j][pos_min:pos_max], c = 'black')
        else:
            ax[j-1].scatter(v[0][g,0][pos_min:pos_max], v[0][g,j][pos_min:pos_max], c = 'black')
        ax[j-1].set_xlabel('$$\\hat{\\mathbf{X}}_1$$')
        ax[j-1].set_ylabel('$$\\hat{\\mathbf{X}}_'+str(j)+'$$')
        if j == 1:
            ax[j-1].legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=4)
