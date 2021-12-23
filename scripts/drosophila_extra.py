#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.preprocessing import LabelEncoder as labeler
import lsbm

## Import labels
lab = np.loadtxt('../data/drosophila_labels.csv', dtype=str)
lab = labeler().fit(lab).transform(lab)
## Import embeddings
X = np.loadtxt('../data/drosophila_dase.csv', delimiter=',')
## Import adjacency matrix
A = np.loadtxt('../data/drosophila_A.csv',delimiter=',',skiprows=1,dtype=int)

from sklearn.mixture import GaussianMixture as GMM
np.random.seed(1771)
print('GMM\t Drosophila\t', np.mean([ari(lab, GMM(n_components=4).fit_predict(X)) for _ in range(1000)]))
Xs = lsbm.row_normalise(X)
np.random.seed(1771)
print('norm-GMM\t Drosophila\t', np.mean([ari(lab, GMM(n_components=4).fit_predict(Xs)) for _ in range(1000)]))
Xs = lsbm.theta_transform(X)
np.random.seed(1771)
print('sphere-GMM\t Drosophila\t', np.mean([ari(lab, GMM(n_components=4).fit_predict(Xs)) for _ in range(1000)]))

from sklearn.cluster import AgglomerativeClustering
print('HClust\t Drosophila\t', ari(lab, AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete').fit_predict(X)))

from sknetwork.hierarchy import LouvainHierarchy, Paris, cut_straight
biparis = Paris()
bihlouvain = LouvainHierarchy()
dendrogram_paris = biparis.fit_transform(A)
dendrogram_louvain = bihlouvain.fit_transform(A)
z_paris = cut_straight(dendrogram_paris, n_clusters=4)
z_hlouvain = cut_straight(dendrogram_louvain, n_clusters=4)
print('Paris\t Drosophila\t',ari(z_paris, lab))
print('HLouvain\t Drosophila\t',ari(z_hlouvain, lab))

from sknetwork.clustering import Louvain
bilouvain = Louvain()
z_louvain = bilouvain.fit_transform(A)
print('Louvain\t Drosophila\t',ari(z_louvain, lab))