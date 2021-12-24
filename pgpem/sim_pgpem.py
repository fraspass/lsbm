#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
import lsbm
import pgpem

# Set random seed for reproducibility
np.random.seed(171)

# Load data
z = np.load('../data/z_hw.npy')
X = np.load('../data/X_hw.npy')

# Set up the model and posterior sampler
m = pgpem.PGPEM(x=lsbm.row_normalise(X), K=2, standardise=False, sig=1)
m.initialise_t(random=False)
m.fit_em(max_iter=25)
print('HW: \t', ari(z, np.argmax(m.t, axis=1)))