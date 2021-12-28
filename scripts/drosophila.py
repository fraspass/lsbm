#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
import lsbm
from sklearn.preprocessing import LabelEncoder as labeler

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Number of samples
M = 10000
## Burnin
B = 1000

## Import labels
lab = np.loadtxt('../data/drosophila_labels.csv', dtype=str)
lab = labeler().fit(lab).transform(lab)
## Import embeddings
X = np.loadtxt('../data/drosophila_dase.csv', delimiter=',')

## Truncated power splines
knots = {}
nknots = 3
mmin = np.min(X,axis=0)[0]
mmax = np.max(X,axis=0)[0]
knots =  np.linspace(start=mmin,stop=mmax,num=nknots+2)[1:-1]

## Define the functions
fW = {}
for k in range(4):
    fW[k,0] = lambda x: np.array([x])
    for j in range(1,6):
        if k == 0:
            fW[k,j] = lambda x: np.array([x, x ** 2, x ** 3]) # + [relu(knot - x) ** 3 for knot in knots])
        elif k == 1:
            fW[k,j] = lambda x: np.array([1,x])
        elif k == 3:
            fW[k,j] = lambda x: np.array([x])
        else:
            if j <= 2:
                fW[k,j] = lambda x: np.array([x])
            else:
                fW[k,j] = lambda x: np.array([1,x])

## Set up the model and posterior sampler
m = lsbm.lsbm_gibbs(X=X, K=4, W_function=fW)
np.random.seed(171)
## IMPORTANT: the initial values must be labelled carefully to match the corresponding curves  
from sklearn.cluster import KMeans
z_init = KMeans(n_clusters=6, random_state=0).fit_predict(X) + 1
## Initial labelling based on k-means output (following Priebe et al, 2017)
z_init[np.where(z_init == 6)[0]] = 0
z_init[np.where(z_init == 3)[0]] = 0
z_init[np.where(z_init == 4)[0]] = 0
z_init[np.where(z_init == 5)[0]] = 3

## Relabel k-means output using marginal likelihoods --> Match communities with 'best fitting' curve
z_optim, perm_optim = lsbm.marginal_likelihood_relabeler(z_init=z_init, m=m)
ari(z_optim, lab)

## Initialise model
np.random.seed(111)
m.initialise(z=np.copy(z_optim), theta=m.X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=(1/m.n)**2, mu_theta=m.X[:,0].mean(), sigma_theta=10, b_0=0.001, first_linear=True)
## Run the sampler
np.random.seed(111)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)
np.save('../Drosophila/out_theta_full.npy',q[0])
np.save('../Drosophila/out_z_full.npy',q[1])

## Estimate clustering
clust = lsbm.estimate_communities(q=q[1],m=m)
clust = lsbm.relabel_matching(lab, clust)
## Evaluate adjusted Rand index
a1 = ari(clust, lab)

## Majority rule (label switching appears to be avoided since the functions are different)
cc = lsbm.estimate_majority(q[1]) 
cc = lsbm.relabel_matching(lab, cc)
a2 = ari(cc, lab)
np.save('../Drosophila/ari_full.npy',np.array([a1,a2]))

### Plots
xx = np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250)
## Calculate MAP for the curves based on the estimated clustering
v = m.map(z=lab, theta=m.X[:,0], range_values=xx)

## Plots
cdict = ['#1E88E5','#FFC107','#D81B60','#23C14B']
mms = ['o', 'v', 's', 'd']
group = ['Kenyon Cells','Input Neurons','Output Neurons','Projection Neurons']
### Scatterplots
for j in range(1,m.d):
    for g in range(m.K):
        ix = np.where(lab == g)
        ix2 = np.where(cc == g)
        pos_min = np.searchsorted(xx,np.min(X[:,0][ix2]))
        pos_max = np.searchsorted(xx,np.max(X[:,0][ix2])) + 1
        plt.scatter(X[:,0][ix], X[:,j][ix], c=cdict[g], label = group[g], marker=mms[g], edgecolor='black', linewidth=0.3)
        plt.plot(v[0][g,0][pos_min:pos_max], v[0][g,j][pos_min:pos_max], c=cdict[g])
    plt.xlabel('$$\\hat{\\mathbf{Y}}_1$$')
    plt.ylabel('$$\\hat{\\mathbf{Y}}_'+str(j+1)+'$$')
    plt.legend()
    plt.savefig('../Drosophila/ddroso_1'+str(j+1)+'.pdf',bbox_inches='tight')
    plt.show(block=False); plt.clf(); plt.cla(); plt.close()

### Define latent basis functions (Priebe et al., 2017 & Athreya et al., 2018)
### - Cluster 0: Linearity in first component & quadratics
### - Clusters 1-3: GMM 
K = 4
d = 6
fW = {}
for j in range(d):
    if j == 0:
        fW[0,0] = lambda x: np.array([1])
    else:
        fW[0,j] = lambda x: np.array([x, x**2])
    for k in range(1,K):
        fW[k,j] = lambda x: np.array([1])

## Set up the model and posterior sampler
m = lsbm.lsbm_gibbs(X=X, K=4, W_function=fW, first_linear=[True,False,False,False])
np.random.seed(111)
## As before, relabel z_optim using marginal likelihoods (with the updated fW)
z_optim, perm_optim = lsbm.marginal_likelihood_relabeler(z_init=z_init, m=m, first_linear=[True,False,False,False])

## Initialise model
np.random.seed(111)
m.initialise(z=z_optim, theta=m.X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                            Lambda_0=(1/m.n)**2, mu_theta=m.X[:,0].mean(), sigma_theta=10, b_0=0.001)
## Run the sampler
np.random.seed(111)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)
np.save('../Drosophila/out_theta_priebe.npy',q[0])
np.save('../Drosophila/out_z_priebe.npy',q[1])

## Majority rule
cc = lsbm.estimate_majority(q[1]) 
cc = lsbm.relabel_matching(lab, cc)
a = ari(cc, lab)
np.save('../Drosophila/ari_priebe.npy',np.array([a]))

## Plot
xx = np.linspace(np.min(m.X[:,0]),np.max(m.X[:,0]),250)
v = m.map(z=cc, theta=X[:,0], range_values=xx)

## Plots
cdict = ['#1E88E5','#FFC107','#D81B60','#23C14B']
mms = ['o', 'v', 's', 'd']
group = ['Kenyon Cells','Input Neurons','Output Neurons','Projection Neurons']
### Scatterplots
for j in range(1,6):
    for g in range(4):
        ix = np.where(lab == g)
        ix2 = np.where(cc == g)
        pos_min = np.searchsorted(xx,np.min(X[:,0][ix2]))
        pos_max = np.searchsorted(xx,np.max(X[:,0][ix2])) + 1
        plt.scatter(X[:,0][ix], X[:,j][ix], c = cdict[g], label = group[g], marker=mms[g], edgecolor='black', linewidth=0.3)
        plt.plot(v[0][g,0][pos_min:pos_max], v[0][g,j][pos_min:pos_max], c = 'black')
    for g in range(1,4):
        if g == 1:
            plt.scatter(v[0][g,0][pos_min], v[0][g,j][pos_min], c = 'black', marker='X', label='Community centers')
        else:
            plt.scatter(v[0][g,0][pos_min], v[0][g,j][pos_min], c = 'black', marker='X')
    plt.xlabel('$$\\hat{\\mathbf{Y}}_1$$')
    plt.ylabel('$$\\hat{\\mathbf{Y}}_'+str(j+1)+'$$')
    plt.legend()
    plt.savefig('../Drosophila/droso_priebe_1'+str(j+1)+'.pdf',bbox_inches='tight')
    plt.show(block=False); plt.clf(); plt.cla(); plt.close()