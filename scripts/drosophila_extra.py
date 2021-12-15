#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import LabelEncoder as labeler
np.random.seed(171)

## Import labels
lab = np.loadtxt('Data/drosophila_labels.csv', dtype=str)
lab = labeler().fit(lab).transform(lab)
## Import embeddings
X = np.loadtxt('Data/drosophila_dase.csv', delimiter=',')
## Import adjacency matrix
A = np.loadtxt('Data/drosophila_A.csv',delimiter=',',skiprows=1,dtype=int)

from sklearn.mixture import GaussianMixture as GMM
z1 = np.mean([ari(lab, GMM(n_components=4).fit_predict(X)) for _ in range(500)])
z2 = np.mean([ari(lab, GMM(n_components=4).fit_predict(X / np.linalg.norm(X,axis=1).reshape(-1,1))) for _ in range(500)])
z3 = np.mean([ari(lab, GMM(n_components=6).fit_predict(X)) for _ in range(500)])

from sklearn.cluster import AgglomerativeClustering
z3 = np.mean([ari(lab, AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete').fit_predict(X)) for _ in range(500)])

from sknetwork.hierarchy import LouvainHierarchy, Paris, cut_straight
biparis = Paris()
bihlouvain = LouvainHierarchy()
dendrogram_paris = biparis.fit_transform(A)
dendrogram_louvain = bihlouvain.fit_transform(A)
z_paris = cut_straight(dendrogram_paris, n_clusters=K)
z_hlouvain = cut_straight(dendrogram_louvain, n_clusters=K)
print('Paris:\t',ari(z_paris, lab))
print('HLouvain:\t',ari(z_hlouvain, lab))

from sknetwork.clustering import Louvain
bilouvain = Louvain()
z_louvain = bilouvain.fit_transform(A)
print('Louvain:\t',ari(z_louvain, lab))