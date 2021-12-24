#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
import lsbm
import pgpem

# Set random seed for reproducibility
np.random.seed(171)

# Load data
z = np.load('../data/block_sim/z.npy')
X_sbm = np.load('../data/block_sim/X_sbm.npy'); A_sbm = np.load('../data/block_sim/A_sbm.npy')
X_dcsbm = np.load('../data/block_sim/X_dcsbm.npy'); A_dcsbm = np.load('../data/block_sim/A_dcsbm.npy')
X_dcsbm[X_dcsbm[:,0]<1e-10,0] = 0
X_quad = np.load('../data/block_sim/X_quad.npy'); A_quad = np.load('../data/block_sim/A_quad.npy')
X_quad[X_quad[:,0]<1e-10,0] = 0

# Set up the model and posterior sampler
m = pgpem.PGPEM(x=X_sbm, K=2, standardise=True, sig=1)
m.initialise_t(random=False)
m.fit_em(max_iter=20)
print('SBM: \t',ari(z, np.argmax(m.t, axis=1)))

m = pgpem.PGPEM(x=lsbm.row_normalise(X_dcsbm), K=2, standardise=True, sig=1)
m.initialise_t(random=False, row_normalise=False)
m.fit_em(max_iter=20)
print('DCSBM: \t',ari(z, np.argmax(m.t, axis=1)))

m = pgpem.PGPEM(x=lsbm.row_normalise(X_quad), K=2, standardise=True, sig=1)
m.initialise_t(random=False, row_normalise=False)
m.fit_em(max_iter=20)
print('Quadratic: \t',ari(z, np.argmax(m.t, axis=1)))