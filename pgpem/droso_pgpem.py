#! /usr/bin/env python3
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
import pgpem

# Set random seed for reproducibility
np.random.seed(171)

# Load data
X = np.loadtxt('../data/drosophila_dase.csv', delimiter=',')
lab = np.loadtxt('../data/droso_labels_numbers.csv', dtype=int).astype(int)

# Set up the model and posterior sampler
m = pgpem.PGPEM(x=X, K=4, standardise=False, sig=1.90)
m.initialise_t(random=False, row_normalise=True)
m.fit_em(max_iter=50)
print(ari(lab, np.argmax(m.t, axis=1)))