#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering
from sknetwork.hierarchy import LouvainHierarchy, cut_straight
import lsbm

## Simulated latent structure blockmodels
z = np.load('../data/block_sim/z.npy')
X_sbm = np.load('../data/block_sim/X_sbm.npy'); A_sbm = np.load('../data/block_sim/A_sbm.npy')
X_dcsbm = np.load('../data/block_sim/X_dcsbm.npy'); A_dcsbm = np.load('../data/block_sim/A_dcsbm.npy')
X_dcsbm[X_dcsbm[:,0]<1e-10,0] = 0
X_quad = np.load('../data/block_sim/X_quad.npy'); A_quad = np.load('../data/block_sim/A_quad.npy')
X_quad[X_quad[:,0]<1e-10,0] = 0

## SBM
np.random.seed(171)
z_sbm = GMM(n_components=2,n_init=10).fit_predict(X_sbm)
print('GMM\t SBM\t', ari(z_sbm, z))
z_sbm = GMM(n_components=2,n_init=10).fit_predict(lsbm.row_normalise(X_sbm))
print('norm-GMM\t SBM\t', ari(z_sbm, z))
z_sbm = GMM(n_components=2,n_init=10).fit_predict(lsbm.theta_transform(X_sbm))
print('sphere-GMM\t SBM\t', ari(z_sbm, z))
hlouvain = LouvainHierarchy(); dendrogram_louvain = hlouvain.fit_transform(A_sbm)
z_sbm = cut_straight(dendrogram_louvain, n_clusters=2)
print('HLouvain\t SBM', ari(z_sbm, z))
z_sbm = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit_predict(X_sbm)
print('HClust\t SBM', ari(z_sbm,z))

## DCSBM
np.random.seed(171)
z_dcsbm = GMM(n_components=2,n_init=10).fit_predict(X_dcsbm)
print('GMM\t DCSBM\t',ari(z_dcsbm, z))
z_dcsbm = GMM(n_components=2,n_init=10).fit_predict(lsbm.row_normalise(X_dcsbm))
print('norm-GMM\t DCSBM\t',ari(z_dcsbm, z))
z_dcsbm = GMM(n_components=2,n_init=10).fit_predict(lsbm.theta_transform(X_dcsbm))
print('sphere-GMM\t DCSBM\t',ari(z_dcsbm, z))
hlouvain = LouvainHierarchy(); dendrogram_louvain = hlouvain.fit_transform(A_dcsbm)
z_dcsbm = cut_straight(dendrogram_louvain, n_clusters=2)
print('HLouvain\t DCSBM', ari(z_dcsbm, z))
z_dcsbm = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit_predict(X_dcsbm)
print('HClust\t DCSBM', ari(z_dcsbm,z))

## Quadratic model
np.random.seed(171)
z_quad = GMM(n_components=2,n_init=10).fit_predict(X_quad)
print('GMM\t Quad\t',ari(z_quad, z))
z_quad = GMM(n_components=2,n_init=10).fit_predict(lsbm.row_normalise(X_quad))
print('norm-GMM\t Quad\t',ari(z_quad, z))
z_quad = GMM(n_components=2,n_init=10).fit_predict(lsbm.theta_transform(X_quad))
print('sphere-GMM\t Quad\t',ari(z_quad, z))
hlouvain = LouvainHierarchy(); dendrogram_louvain = hlouvain.fit_transform(A_quad)
z_quad = cut_straight(dendrogram_louvain, n_clusters=2)
print('HLouvain\t Quad', ari(z_quad, z))
z_quad = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit_predict(X_quad)
print('HClust\t Quad', ari(z_quad,z))