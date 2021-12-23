#! /usr/bin/env python3
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import scipy as sp
import lsbm

## Function that estimates the intrinsic dimension from the cumulative variance
def estim_d(Lambda, threshold):
    ''' 
    Input:
        Lambda: the eigenvalues
        threshold: the percentage of the cumulative variance
    Output:
        d: the intrinsic dimension
    '''
    if Lambda.size == 1:
        d = 1
    else:
        d = np.where(np.cumsum(Lambda) / np.sum(Lambda) > threshold)[0][0] + 1
    return d

## Function that standardises the data
def standardise_x(x):
    M = sp.mean(x,axis=0)
    S = sp.std(x,axis=0)
    xs = (x - M) / S
    return xs, M, S

## Function that calculates the RBF kernel
def RBF(x, y, sig):
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sig))

# Parsimonious Gaussian Process for Clustering (Bouveyron et al., 2015)
class PGPEM:
    
    def __init__(self, x, K, sig=None, threshold=None, standardise=False):
        ## Data
        self.K = K
        self.n = x.shape[0]
        self.eps = sp.finfo(sp.float64).eps 
        if standardise:
            self.x = standardise_x(x)[0]
        else:
            self.x = x
        ## Parameter of the RBF kernel
        if sig is None:
            self.sig = 0.5
        else:
            self.sig = sig
        ## Threshold for cumulative variance
        if threshold is None:
            self.threshold = 0.95
        else:
            self.threshold = threshold
        ## Calculate kernel
        self.K_mat = np.ones((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.K_mat[i,j] = RBF(self.x[i], self.x[j], sig=self.sig)
    
    ## E-step
    def E_step(self):
        self.D = np.ones((self.n, self.K))
        for k in range(self.K):
            Lambda_prod = (1 / self.Lambda[k] - 1 / self.Lambda_tot) / self.Lambda[k]
            t_rho_prod = np.multiply(np.sqrt(self.t[:,k]), self.rho[k])
            self.D[:,k] = np.sum(np.multiply(Lambda_prod, np.array([np.sum(np.multiply(self.Beta[k][:,j], t_rho_prod), axis=1) ** 2 for j in range(self.d[k])]).T), axis=1)
            self.D[:,k] /= self.N[k]
            self.D[:,k] += np.diag(self.rho[k]) / self.Lambda_tot + np.sum(np.log(self.Lambda[k]))
            self.D[:,k] += (self.d_max - self.d[k]) * np.log(self.Lambda_tot) - 2 * np.log(self.pi[k])
        ## self.t = 1 / np.apply_along_axis(func1d=lambda x: np.sum(np.exp(np.subtract.outer(x,x)), axis=1), axis=1, arr=self.D)
        self.t = np.exp(-self.D - logsumexp(-self.D, axis=1).reshape(-1,1))
        
    ## M-step
    def M_step(self):
        self.M = {}; self.rho = {}; self.Lambda = {}; self.Beta = {}; Lambda_tot_num = 0; Lambda_tot_den = 0; self.d = np.zeros(self.K, dtype=int)
        self.N = np.sum(self.t, axis=0)
        self.pi = self.N / self.n
        for k in range(self.K):
            out_prod_t = np.outer(self.t[:,k], self.t[:,k])
            K_sum = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    K_sum[i,j] = np.sum(np.multiply(self.t[:,k], self.K_mat[i] + self.K_mat[:,j]))
            self.rho[k] = self.K_mat - K_sum / self.N[k] + np.sum(np.multiply(out_prod_t, self.K_mat)) / (self.N[k] ** 2)
            self.M[k] = np.sqrt(out_prod_t) / self.N[k] * self.rho[k]
            # Eigenvalue decomposition  
            Lambda, Beta = np.linalg.eigh(self.M[k])
            idx = Lambda.argsort()[::-1]
            Lambda = Lambda[idx]
            Lambda[Lambda < self.eps] = self.eps
            self.Beta[k] = Beta[:,idx]
            self.d[k] = estim_d(Lambda, threshold=self.threshold)
            self.Lambda[k] = Lambda[:self.d[k]]
            self.Beta[k] = Beta[:,idx][:,:self.d[k]]
            Lambda_tot_num += np.multiply(self.pi[k], np.trace(self.M[k]) - np.sum(self.Lambda[k])) 
            Lambda_tot_den += np.multiply(self.pi[k], self.N[k] - self.d[k])
        self.Lambda_tot = Lambda_tot_num / Lambda_tot_den
        self.d_max = np.max(self.d)

    ## Initialise t
    def initialise_t(self, random=False, row_normalise=False, spherical=False, plusone=True):
        if not random:
            ## self.t = np.apply_along_axis(func1d=lambda x: (np.argsort(np.argsort(x)) + 1) / np.sum((np.arange(self.K)+1)), axis=1, 
            ##                     arr=GaussianMixture(n_components=self.K).fit(self.x).predict_proba(self.x))
            if row_normalise: 
                x_init = lsbm.row_normalise(self.x)
            else:
                if spherical:
                    x_init = lsbm.theta_transform(self.x)
                else: 
                    x_init = np.copy(self.x)
            self.t = (1 if plusone else 0) + GaussianMixture(n_components=self.K, n_init=25).fit(x_init).predict_proba(x_init)
            self.t = lsbm.row_normalise(self.t)
            ## self.t = np.apply_along_axis(func1d=lambda x: (np.argsort(np.argsort(x)) + 1) / np.sum((np.arange(self.K)+1)), axis=1, 
                        ## arr=GaussianMixture(n_components=self.K, n_init=50).fit(x_init).predict_proba(x_init))
        else:
            self.t = np.random.dirichlet(alpha=np.ones(self.K), size=self.n)
        self.M_step()

    ## Fit the PGP model using the EM algorithm
    def fit_em(self, max_iter=100):
        for _ in range(max_iter):
            print('Iteration number:', str(_+1), end='\r')
            self.E_step()
            self.M_step()
        print()