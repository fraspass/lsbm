#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import KMeans
import lsbm
from utilities import *

import matplotlib.pyplot as plt
#from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Times']})
#rc('text', usetex=True)

## Number of samples 
M = 10000
B = 1000

## Contruct the adjacency matrix
A = np.zeros((65,65),int)
for line in np.loadtxt('Data/potter.csv',delimiter=',',dtype=str):
    ## Example line '0,45,-', where '-' denotes enmity ('+' for friendship) 
    if line[2] == '-':
        A[int(line[0]),int(line[1])] = 1
        A[int(line[1]),int(line[0])] = 1

## Remove nodes without links
one_link = np.where(np.logical_or(A.sum(axis=0) != 0, A.sum(axis=0) != 0))[0]
A = A[one_link][:,one_link]

## Names
names = np.loadtxt('Data/characters.csv', delimiter=',', dtype=str, skiprows=1)[:,1][one_link]

## Spectral decomposition and ASE
Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:2]
X = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))

## Basis functions

fW = {}
for j in range(2):
    for k in range(2):
        fW[k,j] = lambda x: np.array([x])

## Construct the MCMC sampler object
np.random.seed(117)
m = lsbm.lsbm_gibbs(X=X[:,:2], K=2, W_function=fW)
## Initialisation
m.initialise(z=KMeans(n_clusters=m.K).fit_predict(m.X), theta=(np.abs(m.X[:,0])+np.random.normal(size=m.n,scale=0.000001)), 
                            Lambda_0=1/m.n, g_prior=False, b_0=0.01)
## Sampler
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)
np.save('Harry/out_theta.npy',q[0])
np.save('Harry/out_z.npy',q[1])

## Estimate clustering
clust = estimate_communities(q=q[1],m=m)

## MAP curves
uu = m.map(clust,X[:,0],np.linspace(np.min(X[:,0]),np.max(X[:,0]),500))
## Figure 6(a)
fig, ax = plt.subplots()
xx = np.linspace(np.min(X[:,0]),np.max(X[:,0]),500)
ax.scatter(X[:,0][clust==0], X[:,1][clust==0],c='#FFC107',edgecolor='black',linewidth=0.3)
ax.scatter(X[:,0][clust==1], X[:,1][clust==1],c='#004D40',edgecolor='black',linewidth=0.3)
ax.plot(xx, uu[0][0,1], c='#FFC107')
ax.plot(xx, uu[0][1,1], c='#004D40')
#plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
#plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
ax.text(X[np.where(names=='Sirius Black')[0],0][0],X[np.where(names=='Sirius Black')[0],1][0],'Sirius Black')
ax.text(X[np.where(names=='Albus Dumbledore')[0],0][0],X[np.where(names=='Albus Dumbledore')[0],1][0],'Albus Dumbledore')
ax.text(X[np.where(names=='Dudley Dursley')[0],0][0],X[np.where(names=='Dudley Dursley')[0],1][0],'Dudley Dursley',va='top')
ax.text(X[np.where(names=='Hermione Granger')[0],0][0],X[np.where(names=='Hermione Granger')[0],1][0],'Hermione Granger')
ax.text(X[np.where(names=='Rubeus Hagrid')[0],0][0],X[np.where(names=='Rubeus Hagrid')[0],1][0],'Rubeus Hagrid',va='top')
ax.text(X[np.where(names=='Bellatrix Lestrange')[0],0][0],X[np.where(names=='Bellatrix Lestrange')[0],1][0],'Bellatrix Lestrange',ha='right')
ax.text(X[np.where(names=='Remus Lupin')[0],0][0],X[np.where(names=='Remus Lupin')[0],1][0],'Remus Lupin')
ax.text(X[np.where(names=='Draco Malfoy')[0],0][0],X[np.where(names=='Draco Malfoy')[0],1][0],'Draco Malfoy',ha='right')
ax.text(X[np.where(names=='Minerva McGonagall')[0],0][0],X[np.where(names=='Minerva McGonagall')[0],1][0],'Minerva McGonagall',va='top',ha='right')
ax.text(X[np.where(names=='Harry Potter')[0],0][0],X[np.where(names=='Harry Potter')[0],1][0],'Harry Potter')
ax.text(X[np.where(names=='Lord Voldemort')[0],0][0],X[np.where(names=='Lord Voldemort')[0],1][0],'Lord Voldemort',ha='right')
ax.text(X[np.where(names=='Severus Snape')[0],0][0],X[np.where(names=='Severus Snape')[0],1][0],'Severus Snape')
ax.text(X[np.where(names=='Dolores Janes Umbridge')[0],0][0],X[np.where(names=='Dolores Janes Umbridge')[0],1][0],'Dolores Janes Umbridge',va='top')
ax.text(X[np.where(names=='Fred Weasley')[0],0][0],X[np.where(names=='Fred Weasley')[0],1][0],'Fred Weasley',va='top')
ax.text(X[np.where(names=='Ron Weasley')[0],0][0],X[np.where(names=='Ron Weasley')[0],1][0],'Ron Weasley',va='top')
ax.text(X[np.where(names=='Dobby')[0],0][0],X[np.where(names=='Dobby')[0],1][0],'Dobby')
ax.text(X[np.where(names=='Aragog')[0],0][0],X[np.where(names=='Aragog')[0],1][0],'Aragog',va='top',ha='right')
ax.text(X[np.where(names=='Quirinus Quirrell')[0],0][0],X[np.where(names=='Quirinus Quirrell')[0],1][0],'Quirinus Quirrell')
ax.text(X[np.where(names=='Argus Filch')[0],0][0],X[np.where(names=='Argus Filch')[0],1][0],'Argus Filch')
ax.text(X[np.where(names=='Peter Pettigrew')[0],0][0],X[np.where(names=='Peter Pettigrew')[0],1][0],'Peter Pettigrew')
plt.savefig("Harry/x12_harry.pdf",bbox_inches='tight')
plt.show(block=False); plt.clf(); plt.cla(); plt.close()

## Setup for truncated power basis
## Re-define the basis functions for the example with truncated power basis splines
knots = {}
nknots = 3
mmin = np.min(X,axis=0)[0]
mmax = np.max(X,axis=0)[0]
knots =  np.linspace(start=mmin,stop=mmax,num=nknots+2)[1:-1]
for k in range(2):
    fW[k,0] = lambda x: np.array([x])
    fW[k,1] = lambda x: np.array([x, x ** 2, x ** 3] + [relu(x - knot) ** 3 for knot in knots])

## Sampler
np.random.seed(11711)
m = lsbm.lsbm_gibbs(X=X[:,:2], K=2, W_function=fW)
m.initialise(z=KMeans(n_clusters=m.K).fit_predict(m.X), theta=(np.abs(m.X[:,0])+np.random.normal(size=m.n,scale=0.001)), 
                            Lambda_0=1/m.n, g_prior=False, b_0=0.01)
q = m.mcmc(samples=M, burn=B, sigma_prop=0.01, thinning=1)
np.save('Harry/out_theta_splines.npy',q[0])
np.save('Harry/out_z_splines.npy',q[1])
## Estimate clustering
clust = estimate_communities(q=q[1],m=m)

## Figure 6(b)
uu = m.map(clust,X[:,0],np.linspace(np.min(X[:,0]),np.max(X[:,0]),500))
xx = np.linspace(np.min(X[:,0]),np.max(X[:,0]),500)
fig, ax = plt.subplots()
ax.scatter(X[:,0][clust==0], X[:,1][clust==0],c='#FFC107',edgecolor='black',linewidth=0.3)
ax.scatter(X[:,0][clust==1], X[:,1][clust==1],c='#004D40',edgecolor='black',linewidth=0.3)
ax.plot(xx, uu[0][0,1], c='#FFC107')
ax.plot(xx, uu[0][1,1], c='#004D40')
plt.savefig("Harry/x12_harry_splines.pdf",bbox_inches='tight')
plt.show(block=False); plt.clf(); plt.cla(); plt.close()

