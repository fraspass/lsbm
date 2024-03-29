#! /usr/bin/env python3

#######################
## Utility functions ##
#######################

## Estimate communities using hierarchical clustering on the posterior similarity matrix
def estimate_communities(q,m):
    import numpy as np
    ## Scaled posterior similarity matrix
    psm = np.zeros((m.n,m.n))
    for i in range(q.shape[2]):
        psm += np.equal.outer(q[:,0,i],q[:,0,i])
    ## Posterior similarity matrix (estimate)
    psm /= q.shape[2]
    ## Clustering based on posterior similarity matrix (hierarchical clustering)
    from sklearn.cluster import AgglomerativeClustering
    cluster_model = AgglomerativeClustering(n_clusters=m.K, affinity='precomputed', linkage='average') 
    clust = cluster_model.fit_predict(1-psm)
    return clust

## Estimate ARI
def estimate_ari(q,z):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score as ari
    ## Scaled posterior similarity matrix
    aris = np.zeros((q.shape[1],q.shape[2]))
    for k in range(q.shape[1]):
        for i in range(q.shape[2]):
            aris[k,i] = ari(z, q[:,k,i])
    return aris

## Estimate communities using majority rule from output of MCMC
def estimate_majority(q):
    import numpy as np
    from collections import Counter
    cc = np.apply_along_axis(func1d=lambda x: int(Counter(x).most_common(1)[0][0]), axis=1, arr=q[:,0])
    return cc

## Relabel the initial allocations using the marginal likelihood
def marginal_likelihood_relabeler(z_init, m, seed=111):
    import numpy as np
    ## Copy the initial labels
    z_relabelled = np.copy(z_init)
    ## Initialise the quantities of interest (initialisation and best permutation)
    max_init = np.copy(z_init)
    max_perm = np.arange(m.K)
    ## Set seed
    np.random.seed(seed)
    ## Initialise the model using z_init
    m.initialise(z=z_init, theta=m.X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                                Lambda_0=(1/m.n)**2, mu_theta=m.X[:,0].mean(), sigma_theta=10, b_0=0.001)
    ## Calculate the marginal loglikelihood
    max_mar_loglik = m.marginal_loglikelihood()
    ## Calculate the possible permutations of the labels
    import itertools
    perms = list(itertools.permutations(list(range(m.K))))
    ## Iterate over the permutations of the labels
    for perm in perms[1:]:
        ## Relabel z_init
        z_relabelled = np.array(perm)[z_init]
        ## Initialise
        np.random.seed(seed)
        m.initialise(z=z_relabelled, theta=m.X[:,0]+np.random.normal(size=m.n,scale=0.01), 
                                Lambda_0=(1/m.n)**2, mu_theta=m.X[:,0].mean(), sigma_theta=10, b_0=0.001)
        ## Marginal likelihood
        ml = m.marginal_loglikelihood()
        ## Update only if the marginal likelihood is larger than previous maximum
        if ml > max_mar_loglik:
            max_mar_loglik = ml
            max_perm = perm
            max_init = np.copy(z_relabelled)
    ## Return output
    return max_init, max_perm

## Rectified linear unit
def relu(x):
    return x * (x > 0)

## Lighten colours (for confidence bands)
def lighten_color(color, amount=0.5):
    ## Lightens the given color by multiplying (1-luminosity) by the given amount.
    ## Input can be matplotlib color string, hex string, or RGB tuple.
    ## Credits to https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

## Relabel two vectors such that the number of mismatches is minimised
def relabel_matching(v1, v2):
    import numpy as np
    import pandas as pd
    K1 = int(np.max(v1)); K2 = int(np.max(v2))
    K = np.max([K1,K2]) + 1
    ## Copy the initial labels
    v2_relabelled = np.copy(v2)
    ## Initialise the quantities of interest (initialisation and best permutation)
    v2_best = np.copy(v2)
    max_score_diagonal = np.sum(np.diag(pd.crosstab(v1,v2)))
    ## Calculate the possible permutations of the labels
    import itertools
    perms = list(itertools.permutations(list(range(K))))
    for perm in perms:
        v2_relabelled = np.array(perm)[v2]
        score_diagonal = np.sum(np.diag(pd.crosstab(v1,v2_relabelled)))
        if score_diagonal > max_score_diagonal:
            max_score_diagonal = score_diagonal
            v2_best = np.copy(v2_relabelled)
    return v2_best

## Takes a vector and returns its spherical coordinates
def cart_to_sphere(x):
    import numpy as np
    x[x == 0] = -np.finfo(float).eps
    ## theta_1
    q = np.arccos(x[1] / np.linalg.norm(x[:2]))
    sphere_coord = [q] if x[0] >= 0 else [2*np.pi - q]
    ## Loop for theta_2, ..., theta_m-1
    for j in range(2,len(x)):
        sphere_coord += [2 * np.arccos(x[j] / np.linalg.norm(x[:(j+1)]))]
    ## Return the result in a numpy array
    return np.array(sphere_coord)

## Takes a matrix and returns the spherical coordinates obtained along the given axis
def theta_transform(X,axis=1):
    import numpy as np
    ## Apply the function theta_transform along the axis
    return np.apply_along_axis(func1d=cart_to_sphere, axis=axis, arr=X)

## Row-normalisation
def row_normalise(X):
    import numpy as np
    X_sum = np.linalg.norm(X, axis=1)
    X_sum[X_sum == 0] = np.finfo(float).eps
    return np.divide(X, X_sum.reshape(-1,1))