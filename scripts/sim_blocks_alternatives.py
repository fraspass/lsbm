#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture as GMM 
import lsbm

## Simulated latent structure blockmodels
z = np.load('../data/block_sim/z_sbm.npy')
X_sbm = np.load('../data/block_sim/X_sbm.npy')
X_dcsbm = np.load('../data/block_sim/X_dcsbm.npy')
X_dcsbm[X_dcsbm[:,0]<1e-10,0] = 0
X_quad = np.load('../data/block_sim/X_quad.npy')
X_quad[X_quad[:,0]<1e-10,0] = 0

## SBM
np.random.seed(171)
z_sbm = GMM(n_components=2,n_init=10).fit_predict(X_sbm)
print('GMM\t SBM\t',ari(z_sbm, z))
z_sbm = GMM(n_components=2,n_init=10).fit_predict(lsbm.row_normalise(X_sbm))
print('norm-GMM\t SBM\t',ari(z_sbm, z))
z_sbm = GMM(n_components=2,n_init=10).fit_predict(lsbm.theta_transform(X_sbm))
print('sphere-GMM\t SBM\t',ari(z_sbm, z))

## DCSBM
np.random.seed(171)
z_dcsbm = GMM(n_components=2,n_init=10).fit_predict(X_dcsbm)
print('GMM\t DCSBM\t',ari(z_dcsbm, z))
z_dcsbm = GMM(n_components=2,n_init=10).fit_predict(lsbm.row_normalise(X_dcsbm))
print('norm-GMM\t DCSBM\t',ari(z_dcsbm, z))
z_dcsbm = GMM(n_components=2,n_init=10).fit_predict(lsbm.theta_transform(X_dcsbm))
print('sphere-GMM\t DCSBM\t',ari(z_dcsbm, z))

## Quadratic model
np.random.seed(171)
z_quad = GMM(n_components=2,n_init=10).fit_predict(X_quad)
print('GMM\t Quad\t',ari(z_quad, z))
z_quad = GMM(n_components=2,n_init=10).fit_predict(lsbm.row_normalise(X_quad))
print('norm-GMM\t Quad\t',ari(z_quad, z))
z_quad = GMM(n_components=2,n_init=10).fit_predict(lsbm.theta_transform(X_quad))
print('sphere-GMM\t Quad\t',ari(z_quad, z))