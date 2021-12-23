#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture as GMM 
from sklearn.cluster import AgglomerativeClustering
from sknetwork.hierarchy import LouvainHierarchy, cut_straight
import lsbm

## Simulated HW LSBM
z = np.load('../data/z_hw.npy')
A = np.load('../data/A_hw.npy')
X = np.load('../data/X_hw.npy')

## 
np.random.seed(171)
z_est = GMM(n_components=2,n_init=10).fit_predict(X)
print('GMM\t HW\t',ari(z_est, z))
z_est = GMM(n_components=2,n_init=10).fit_predict(lsbm.row_normalise(X))
print('norm-GMM\t HW\t',ari(z_est, z))
z_est = GMM(n_components=2,n_init=10).fit_predict(lsbm.theta_transform(X))
print('sphere-GMM\t HW\t',ari(z_est, z))
hlouvain = LouvainHierarchy(); dendrogram_louvain = hlouvain.fit_transform(A)
z_est = cut_straight(dendrogram_louvain, n_clusters=2)
print('HLouvain\t HW', ari(z_est, z))
z_est = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete').fit_predict(X)
print('HClust\t HW', ari(z_est,z))