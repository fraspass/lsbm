#! /usr/bin/env python3
import numpy as np
from collections import Counter
from scipy.stats import norm
from scipy.special import logsumexp, loggamma
from scipy.stats import t
from lsbm.mvt import dmvt

################################################################################
### LSBM embeddings with weighted inner product of basis functions GP kernel ###
################################################################################

class lsbm_gibbs:

    ## Initialise the class with the number of components and embedding
    def __init__(self, X, K, W_function, fixed_function={}, K_fixed=False, first_linear=False):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.K = K
        if isinstance(K_fixed,bool):
            self.K_fixed = K_fixed
        else:
            raise TypeError('K_fixed must be a boolean value.')
        ## Function to obtain the design matrix
        self.fW = W_function 
        self.fixed_function = fixed_function
        Fs = Counter()
        for k,_ in self.fW:
            Fs[k] += 1
        self.kappa = len(Fs)
        if self.kappa != self.K and self.K_fixed:
            raise ValueError('The number of latent functions in W_function must be equal to K (when K is fixed).')
        ## Set up first_linear
        if isinstance(first_linear, bool):
            self.first_linear = (self.K if self.K_fixed else self.kappa) * [first_linear]
        else:
            self.first_linear = first_linear
            if np.sum([isinstance(k,bool) for k in first_linear]) != self.K and self.K_fixed:
                raise ValueError('If K is fixed, first_linear is either a boolean or a K-vector of booleans.')
            if np.sum([isinstance(k,bool) for k in first_linear]) != self.kappa and not self.K_fixed:
                raise ValueError('If K is not fixed, first_linear is either a boolean or a vector of booleans of size equal to the number of possible kernels.')

    ## Initialise the model parameters
    def initialise(self, z, theta, Lambda_0=1.0, a_0=1.0, b_0=1.0, nu=1.0, mu_theta=0.0, sigma_theta=1.0, omega=0.9, g_prior=True):
        ## Initial cluster configuration
        self.z = z
        if np.min(self.z) == 1:
            self.z -= 1
        self.theta = theta
        ## Theta hyperparameters
        self.mu_theta = mu_theta
        self.sigma_theta = sigma_theta
        ## K hyperparameters
        self.omega = omega
        ## Basis functions
        self.g_prior = g_prior
        self.W = {}
        ## for j in range(self.d):
        ##    for k in range(self.K):
        for k,j in self.fW:
            self.W[k,j] = np.array([self.fW[k,j](self.theta[i]) for i in range(self.n)])
        self.fixed_W = {}
        ## for j in self.fixed_function:
        ##     for k in range(self.K):
        for k,j in self.fixed_function:
            self.fixed_W[j] = np.array([self.fixed_function[k,j](self.theta[i]) for i in range(self.n)])[:,0] ## rewrite using only one coefficient (1)
        ## If K_fixed, match each cluster with a set of functions
        if not self.K_fixed:
            self.fk = np.zeros(self.K, dtype=int)
        ## Prior parameters
        self.nu = nu
        self.a0 = a_0
        self.b0 = b_0
        self.lambda_coef = Lambda_0
        self.Lambda0 = {}; self.Lambda0_inv = {}
        for j in range(self.d):
            for k in range(self.K if self.K_fixed else self.kappa):
                if g_prior:
                    self.Lambda0_inv[k,j] = Lambda_0 * np.dot(self.W[k,j].T,self.W[k,j]) 
                else:
                    self.Lambda0_inv[k,j] = np.diag(np.ones(len(self.fW[k,j](1))) * Lambda_0)
                self.Lambda0[k,j] = np.linalg.inv(self.Lambda0_inv[k,j])
        self.nu = nu
        ## Initialise hyperparameter vectors
        self.nk = np.zeros(self.K, dtype=int) ## Counter(self.z)
        for ind in self.z:
            self.nk[ind] += 1
        self.a = np.zeros(self.K)
        for k in range(self.K):
            self.a[k] = self.a0 + self.nk[k] / 2  
        self.b = {}; self.WtW = {}; self.WtX = {}; self.Lambda_inv = {}; self.Lambda = {}; self.mu = {}
        for j in range(self.d):
            self.b[j] = {}; self.WtW[j] = {}; self.WtX[j] = {}; self.Lambda_inv[j] = {}; self.Lambda[j] = {}; self.mu[j] = {}
            for k in range(self.K):
                X = self.X[self.z == k][:,j]
                if j in self.fixed_function:
                    X -= self.fixed_W[j][self.z == k]
                if j == 0 and self.first_linear[k if self.K_fixed else self.fk[k]]:
                    self.b[j][k] = self.b0 + np.sum((X - self.theta[self.z == k]) ** 2) / 2
                else:
                    W = self.W[k if self.K_fixed else self.fk[k],j][self.z == k]
                    self.WtW[j][k] = np.dot(W.T,W)
                    self.WtX[j][k] = np.dot(W.T,X)
                    self.Lambda_inv[j][k] = self.WtW[j][k] + self.Lambda0_inv[k if self.K_fixed else self.fk[k],j]
                    self.Lambda[j][k] = np.linalg.inv(self.Lambda_inv[j][k])
                    self.mu[j][k] = np.dot(self.Lambda[j][k], self.WtX[j][k])
                    self.b[j][k] = self.b0 + (np.dot(X.T,X) - np.dot(self.mu[j][k].T, np.dot(self.Lambda_inv[j][k],self.mu[j][k]))) / 2

    ########################################################
    ### a. Resample the allocations using Gibbs sampling ###
    ########################################################
    def gibbs_communities(self,l=1):    
        ## Change the value of l when too large
        if l > self.n:
            l = self.n
        ## Update the latent allocations in randomised order
        ## Loop over the indices
        WtW_old = {}; WtX_old = {}; Lambda_inv_old = {}; Lambda_old = {}; mu_old = {}; b_old = {}; position = {}
        for i in np.random.choice(self.n, size=l, replace=False):
            zold = self.z[i]
            ## Update parameters of the distribution
            self.a[zold] -= .5
            self.nk[zold] -= 1
            for j in range(self.d):
                position[j] = self.X[i,j]
                if j in self.fixed_function:
                    position[j] -= self.fixed_W[j][i]
                if j == 0 and self.first_linear[zold if self.K_fixed else self.fk[zold]]:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] - self.theta[i]) ** 2 / 2
                else:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] ** 2 - np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold]))) / 2 
                    WtW_old[j] = np.copy(self.WtW[j][zold])
                    WtX_old[j] = np.copy(self.WtX[j][zold])
                    self.WtW[j][zold] -= np.outer(self.W[zold if self.K_fixed else self.fk[zold],j][i],self.W[zold if self.K_fixed else self.fk[zold],j][i])
                    self.WtX[j][zold] -= self.W[zold if self.K_fixed else self.fk[zold],j][i] * position[j]
                    Lambda_inv_old[j] = np.copy(self.Lambda_inv[j][zold])
                    Lambda_old[j] = np.copy(self.Lambda[j][zold])
                    self.Lambda_inv[j][zold] = self.WtW[j][zold] + self.Lambda0_inv[zold if self.K_fixed else self.fk[zold],j]
                    self.Lambda[j][zold] = np.linalg.inv(self.Lambda_inv[j][zold])
                    mu_old[j] = np.copy(self.mu[j][zold])
                    self.mu[j][zold] = np.dot(self.Lambda[j][zold], self.WtX[j][zold])
                    self.b[j][zold] -= np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold])) / 2
            ## Calculate probabilities for community allocations
            community_probs = np.log((np.array([self.nk[k] for k in range(self.K)]) + self.nu/self.K) / (self.n - 1 + self.nu))
            for k in range(self.K):
                for j in range(self.d):
                    if j == 0 and self.first_linear[k if self.K_fixed else self.fk[k]]:
                        community_probs[k] += t.logpdf(position[j], df=2*self.a[k], loc=self.theta[i], scale=np.sqrt(self.b[j][k] / self.a[k])) 
                    else:
                        community_probs[k] += t.logpdf(position[j], df=2*self.a[k], loc=np.dot(self.W[k if self.K_fixed else self.fk[k],j][i],self.mu[j][k]), 
                                scale=np.sqrt(self.b[j][k] / self.a[k] * (1 + np.dot(self.W[k if self.K_fixed else self.fk[k],j][i].T, np.dot(self.Lambda[j][k], self.W[k if self.K_fixed else self.fk[k],j][i])))))              
            ## Raise error if nan probabilities are computed
            if np.isnan(community_probs).any():
                print(community_probs)
                raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
            ## Update allocation
            znew = np.random.choice(self.K, p=np.exp(community_probs - logsumexp(community_probs)))
            self.z[i] = np.copy(znew) 
            ## Update parameters
            self.a[znew] += .5
            self.nk[znew] += 1
            if znew == zold:
                ## Re-update to old values
                for j in range(self.d):
                    self.b[j][znew] = b_old[j]
                    if not (j == 0 and self.first_linear[znew if self.K_fixed else self.fk[znew]]):
                        self.WtW[j][znew] = WtW_old[j]
                        self.WtX[j][znew] = WtX_old[j]
                        self.Lambda_inv[j][znew] = Lambda_inv_old[j]
                        self.Lambda[j][znew] = Lambda_old[j]
                        self.mu[j][znew] = mu_old[j] 
            else:
                ## Update to new values
                for j in range(self.d):
                    if j == 0 and self.first_linear[znew if self.K_fixed else self.fk[znew]]:
                        self.b[j][znew] += (position[j] - self.theta[i]) ** 2 / 2
                    else:
                        self.b[j][znew] += np.dot(self.mu[j][znew].T,np.dot(self.Lambda_inv[j][znew],self.mu[j][znew])) / 2 
                        self.WtW[j][znew] += np.outer(self.W[znew if self.K_fixed else self.fk[znew],j][i],self.W[znew if self.K_fixed else self.fk[znew],j][i])
                        self.WtX[j][znew] += self.W[znew if self.K_fixed else self.fk[znew],j][i] * position[j]
                        self.Lambda_inv[j][znew] = self.WtW[j][znew] + self.Lambda0_inv[znew if self.K_fixed else self.fk[znew],j]
                        self.Lambda[j][znew] = np.linalg.inv(self.Lambda_inv[j][znew])
                        self.mu[j][znew] = np.dot(self.Lambda[j][znew], self.WtX[j][znew])
                        self.b[j][znew] += (position[j] ** 2 - np.dot(self.mu[j][znew].T,np.dot(self.Lambda_inv[j][znew],self.mu[j][znew]))) / 2
        return None

    ##############################################
    ### b. Resample the latent positions theta ###
    ##############################################
    def resample_theta(self, l=1, sigma_prop=0.1):
       ## Change the value of l when too large
        if l > self.n:
            l = self.n
        ## Update the latent allocations in randomised order
        ## Loop over the indices
        WtW_old = {}; WtX_old = {}; Lambda_inv_old = {}; Lambda_old = {}; mu_old = {}; b_old = {}; position = {}
        position_prop = {}; W_prop = {}; W_prop_fixed = {}
        for i in np.random.choice(self.n, size=l, replace=False):
            zold = self.z[i]
            theta_old = self.theta[i]
            ## Update parameters of the distribution
            self.a[zold] -= .5
            self.nk[zold] -= 1
            for j in range(self.d):
                position[j] = self.X[i,j]
                if j in self.fixed_function:
                    position[j] -= self.fixed_W[j][i]
                if j == 0 and self.first_linear[zold if self.K_fixed else self.fk[zold]]:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] - theta_old) ** 2 / 2
                else:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] ** 2 - np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold]))) / 2 
                    WtW_old[j] = np.copy(self.WtW[j][zold])
                    WtX_old[j] = np.copy(self.WtX[j][zold])
                    self.WtW[j][zold] -= np.outer(self.W[zold if self.K_fixed else self.fk[zold],j][i],self.W[zold if self.K_fixed else self.fk[zold],j][i])
                    self.WtX[j][zold] -= self.W[zold if self.K_fixed else self.fk[zold],j][i] * position[j]
                    Lambda_inv_old[j] = np.copy(self.Lambda_inv[j][zold])
                    Lambda_old[j] = np.copy(self.Lambda[j][zold])
                    self.Lambda_inv[j][zold] = self.WtW[j][zold] + self.Lambda0_inv[zold if self.K_fixed else self.fk[zold],j]
                    self.Lambda[j][zold] = np.linalg.inv(self.Lambda_inv[j][zold])
                    mu_old[j] = np.copy(self.mu[j][zold])
                    self.mu[j][zold] = np.dot(self.Lambda[j][zold], self.WtX[j][zold])
                    self.b[j][zold] -= np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold])) / 2
            ## Calculate proposal
            theta_prop = np.random.normal(loc=theta_old, scale=sigma_prop)
            for j in range(self.d):
                position_prop[j] = self.X[i,j]
                for k in range(self.K if self.K_fixed else self.kappa):
                    W_prop[k,j] = self.fW[k,j](theta_prop)
                if j in self.fixed_function:
                    W_prop_fixed[j] = self.fixed_function[j](theta_prop)
                    position_prop[j] -= W_prop_fixed[j]
            ## Calculate acceptance ratio
            numerator_accept = norm.logpdf(theta_prop,loc=self.mu_theta,scale=self.sigma_theta)
            for j in range(self.d):
                if j == 0 and self.first_linear[zold if self.K_fixed else self.fk[zold]]:
                    numerator_accept += t.logpdf(position_prop[j], df=2*self.a[zold], loc=theta_prop, scale=np.sqrt(self.b[j][zold] / self.a[zold])) 
                else:
                    numerator_accept += t.logpdf(position_prop[j], df=2*self.a[zold], loc=np.dot(W_prop[zold if self.K_fixed else self.fk[zold],j],self.mu[j][zold]), 
                            scale=np.sqrt(self.b[j][zold] / self.a[zold] * (1 + np.dot(W_prop[zold if self.K_fixed else self.fk[zold],j].T, np.dot(self.Lambda[j][zold], W_prop[zold if self.K_fixed else self.fk[zold],j])))))
            denominator_accept = norm.logpdf(theta_old,loc=self.mu_theta,scale=self.sigma_theta)
            for j in range(self.d):
                if j == 0 and self.first_linear[zold if self.K_fixed else self.fk[zold]]:
                    denominator_accept += t.logpdf(position[j], df=2*self.a[zold], loc=theta_old, scale=np.sqrt(self.b[j][zold] / self.a[zold])) 
                else:
                    denominator_accept += t.logpdf(position[j], df=2*self.a[zold], loc=np.dot(self.W[zold if self.K_fixed else self.fk[zold],j][i],self.mu[j][zold]), 
                            scale=np.sqrt(self.b[j][zold] / self.a[zold] * (1 + np.dot(self.W[zold if self.K_fixed else self.fk[zold],j][i].T, np.dot(self.Lambda[j][zold], self.W[zold if self.K_fixed else self.fk[zold],j][i])))))             
            ## Calculate acceptance probability
            accept_ratio = numerator_accept - denominator_accept
            accept = (-np.random.exponential(1) < accept_ratio)
            ## Update parameters
            self.a[zold] += .5
            self.nk[zold] += 1
            if accept:
                self.theta[i] = theta_prop
            if not accept:
                ## Re-update to old values
                for j in range(self.d):
                    self.b[j][zold] = b_old[j]
                    if not (j == 0 and self.first_linear[zold if self.K_fixed else self.fk[zold]]):
                        self.WtW[j][zold] = WtW_old[j]
                        self.WtX[j][zold] = WtX_old[j]
                        self.Lambda_inv[j][zold] = Lambda_inv_old[j]
                        self.Lambda[j][zold] = Lambda_old[j]
                        self.mu[j][zold] = mu_old[j] 
            else:
                ## Update to new values
                for j in range(self.d):
                    ## Update design matrix
                    for k in range(self.K if self.K_fixed else self.kappa):
                        self.W[k,j][i] = W_prop[k,j]
                    if j in self.fixed_function:
                        self.fixed_W[j][i] = W_prop_fixed[j]
                    if j == 0 and self.first_linear[zold if self.K_fixed else self.fk[zold]]:
                        self.b[j][zold] += (position[j] - self.theta[i]) ** 2 / 2
                    else:
                        self.b[j][zold] += np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold])) / 2 
                        self.WtW[j][zold] += np.outer(self.W[zold if self.K_fixed else self.fk[zold],j][i],self.W[zold if self.K_fixed else self.fk[zold],j][i])
                        self.WtX[j][zold] += self.W[zold if self.K_fixed else self.fk[zold],j][i] * position_prop[j]
                        self.Lambda_inv[j][zold] = self.WtW[j][zold] + self.Lambda0_inv[zold if self.K_fixed else self.fk[zold],j]
                        self.Lambda[j][zold] = np.linalg.inv(self.Lambda_inv[j][zold])
                        self.mu[j][zold] = np.dot(self.Lambda[j][zold], self.WtX[j][zold])
                        self.b[j][zold] += (position_prop[j] ** 2 - np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold]))) / 2
        return None

    #################################################
    ### c. Propose to add/remove an empty cluster ###
    #################################################
    def propose_empty(self, verbose=False): 
        if self.K_fixed:
            raise ValueError('propose_empty can only be run if K_fixed is set to False.')   
        ## Propose K
        if self.K == 1:
            K_prop = 2
        elif (self.K == self.n):
            K_prop = self.n - 1
        else:
            ## If there are no empty clusters and K_prop = K-1, reject the proposal
            if not np.any(self.nk == 0):
                K_prop = self.K+1
            else:
                K_prop = np.random.choice([self.K-1, self.K+1])
        ## Assign values to the variable remove
        if K_prop < self.K:
            remove = True
        else:
            remove = False
        ## Propose functional form for new cluster
        if not remove:
            fk_prop = np.random.choice(self.kappa)
        ## Propose a new (empty) vector of cluster allocations
        if remove:
            ## Delete empty cluster with largest index (or sample at random)
            ind_delete = np.random.choice(np.where(self.nk == 0)[0])
            nk_prop = np.delete(self.nk, ind_delete)
        else:
            nk_prop = np.append(self.nk, 0)
        ## Common term for the acceptance probability
        accept_ratio = self.K * loggamma(self.nu / self.K) - K_prop * loggamma(self.nu / K_prop) + \
                            np.sum(loggamma(nk_prop + self.nu / K_prop)) - np.sum(loggamma(self.nk + self.nu / self.K)) + \
                            (K_prop - self.K) * np.log(1 - self.omega) * np.log(.5) * int(self.K == 1) - np.log(.5) * int(self.K == self.n)
        ## Accept or reject the proposal
        accept = (-np.random.exponential(1) < accept_ratio)
        ## Scale all the values if an empty cluster is added
        if verbose:
            print('\t',['Add','Remove'][int(remove)], accept, np.exp(accept_ratio), K_prop, end='')
        if accept:
            self.nk = nk_prop
            if not remove:
                self.fk = np.append(self.fk, fk_prop)
                self.a = np.append(self.a, self.a0)
                for j in range(self.d):
                    self.b[j][K_prop-1] = self.b0
                    if not (j == 0 and self.first_linear[fk_prop]):
                        self.WtW[j][K_prop-1] = np.zeros((self.W[fk_prop,j].shape[1], self.W[fk_prop,j].shape[1]))
                        self.WtX[j][K_prop-1] = np.zeros(self.W[fk_prop,j].shape[1])
                        self.Lambda_inv[j][K_prop-1] = np.copy(self.Lambda0_inv[fk_prop,j])
                        self.Lambda[j][K_prop-1] = np.copy(self.Lambda0[fk_prop,j])
                        self.mu[j][K_prop-1] = np.zeros(self.W[fk_prop,j].shape[1])
            else:
                ## Delete old values
                fk_old = self.fk[ind_delete]
                self.fk = np.delete(self.fk, ind_delete)
                self.a = np.delete(self.a,ind_delete)
                for j in range(self.d):
                    del self.b[j][ind_delete]
                    if not (j == 0 and self.first_linear[fk_old]):
                        del self.WtW[j][ind_delete]
                        del self.WtX[j][ind_delete]
                        del self.Lambda_inv[j][ind_delete]
                        del self.Lambda[j][ind_delete]
                        del self.mu[j][ind_delete]
                ## Relabel groups
                Q = np.arange(self.K)[np.arange(self.K) > ind_delete]
                for k in Q:
                    self.z[self.z == k] = k-1
                    for j in range(self.d):
                        self.b[j][k-1] = self.b[j][k]; del self.b[j][k]
                        if not (j == 0 and self.first_linear[self.fk[k-1]]):
                            self.WtW[j][k-1] = self.WtW[j][k]; del self.WtW[j][k]
                            self.WtX[j][k-1] = self.WtX[j][k]; del self.WtX[j][k]
                            self.Lambda_inv[j][k-1] = self.Lambda_inv[j][k]; del self.Lambda_inv[j][k]
                            self.Lambda[j][k-1] = self.Lambda[j][k]; del self.Lambda[j][k]
                            self.mu[j][k-1] = self.mu[j][k]; del self.mu[j][k]
            ## Update K
            self.K = K_prop
        return None

    ##################################
    ### d. Split-merge communities ###
    ##################################
    def split_merge(self, verbose=False):
        if self.K_fixed:
            raise ValueError('propose_empty can only be run if K_fixed is set to False.')   
        # Randomly choose two nodes
        q, q_prime = np.random.choice(self.n, size=2, replace=False)
        # Propose a split or merge move according to the sampled values
        if self.z[q] == self.z[q_prime]:
            split = True
            z = self.z[q]
            z_prime = self.K
            fk_prop = self.fk[z]
            fk_temp = [fk_prop, fk_prop]
        else:
            split = False
            z = np.min([self.z[q],self.z[q_prime]])
            z_prime = np.max([self.z[q],self.z[q_prime]])
            fk_prop = np.random.choice([self.fk[z], self.fk[z_prime]])
            fk_temp = [self.fk[z], self.fk[z_prime]]
        # Proposed K
        K_prop = self.K + (1 if split else -1)
        # Preprocessing for split / merge move
        nk_prop = np.ones(2, dtype=int)
        a_prop = self.a0 + np.ones(2) / 2
        b_prop = {}; WtW_prop = {}; WtX_prop = {}; Lambda_inv_prop = {}; Lambda_prop = {}; mu_prop = {}
        for j in range(self.d):
            b_prop[j] = np.ones(2) * self.b0; WtW_prop[j] = {}; WtX_prop[j] = {}; Lambda_inv_prop[j] = {}; Lambda_prop[j] = {}; mu_prop[j] = {}
            if j == 0 and self.first_linear[self.fk[z]]:
                b_prop[j][0] += (self.X[q,j] - self.theta[q]) ** 2 / 2
            else:
                WtW_prop[j][0] = np.outer(self.W[fk_temp[0],j][q], self.W[fk_temp[0],j][q])
                WtX_prop[j][0] = np.multiply(self.W[fk_temp[0],j][q], self.X[q,j])
                Lambda_inv_prop[j][0] = self.Lambda0_inv[self.fk[z],j] + WtW_prop[j][0]
                Lambda_prop[j][0] = np.linalg.inv(Lambda_inv_prop[j][0])
                mu_prop[j][0] = np.dot(Lambda_prop[j][0], WtX_prop[j][0])
                b_prop[j][0] += (self.X[q,j] ** 2 - np.dot(mu_prop[j][0].T,np.dot(Lambda_inv_prop[j][0],mu_prop[j][0]))) / 2
            if j == 0 and self.first_linear[self.fk[z if split else z_prime]]:
                b_prop[j][1] += (self.X[q_prime,j] - self.theta[q_prime]) ** 2 / 2
            else:
                WtW_prop[j][1] = np.outer(self.W[fk_temp[1],j][q_prime], self.W[fk_temp[1],j][q_prime])
                WtX_prop[j][1] = np.multiply(self.W[fk_temp[1],j][q_prime], self.X[q_prime,j])
                Lambda_inv_prop[j][1] = self.Lambda0_inv[self.fk[z if split else z_prime],j] + WtW_prop[j][1] 
                Lambda_prop[j][1] = np.linalg.inv(Lambda_inv_prop[j][1])
                mu_prop[j][1] = np.dot(Lambda_prop[j][1], WtX_prop[j][1]) 
                b_prop[j][1] += (self.X[q_prime,j] ** 2 - np.dot(mu_prop[j][1].T,np.dot(Lambda_inv_prop[j][1],mu_prop[j][1]))) / 2
        ## Indices
        if split:
            indices = np.where(self.z == z)[0]
        else:
            indices = np.where(np.logical_or(self.z == z, self.z == z_prime))[0]
        indices = indices[np.logical_and(indices != q, indices != q_prime)]
        if not split:
            ## Calculate parameters for merge move
            nk_merge = self.nk[z] + self.nk[z_prime]
            a_merge = self.a0 + nk_merge / 2
            b_merge = {}; WtW_merge = {}; WtX_merge = {}; Lambda_inv_merge = {}; Lambda_merge = {}; mu_merge = {}
            for j in range(self.d):
                b_merge[j] = self.b0
                if j == 0 and self.first_linear[self.fk[z]]:
                    b_merge[j] += (self.X[q,j] - self.theta[q]) ** 2 / 2
                    b_merge[j] += (self.X[q_prime,j] - self.theta[q_prime]) ** 2 / 2
                    b_merge[j] += np.sum((self.X[indices,j] - self.theta[indices]) ** 2 / 2)
                else:
                    X = self.X[np.append([q,q_prime],indices)][:,j]
                    W = self.W[fk_prop,j][np.append([q,q_prime],indices)]
                    WtW_merge[j] = np.dot(W.T,W)
                    WtX_merge[j] = np.dot(W.T,X)
                    Lambda_inv_merge[j] = self.Lambda0_inv[fk_prop,j] + WtW_merge[j]
                    Lambda_merge[j] = np.linalg.inv(Lambda_inv_merge[j])
                    mu_merge[j] = np.dot(Lambda_merge[j], WtX_merge[j])
                    b_merge[j] += (self.X[q,j] ** 2 + self.X[q_prime,j] ** 2 + np.dot(self.X[indices][:,j].T,self.X[indices][:,j]) - np.dot(mu_merge[j].T,np.dot(Lambda_inv_merge[j],mu_merge[j]))) / 2
        ## Random allocation of indices
        indices = np.random.choice(indices,size=len(indices),replace=False)
        ## Calculate q probabilities
        qprob = 0
        zz = []
        for i in indices:
            ## Calculate probabilities for community allocations
            community_probs = np.log(nk_prop + self.nu/2)
            for j in range(self.d):
                if j == 0 and self.first_linear[self.fk[z]]:
                    community_probs[0] += t.logpdf(self.X[i,j], df=2*a_prop[0], loc=self.theta[i], scale=np.sqrt(b_prop[j][0] / a_prop[0])) 
                else:
                    community_probs[0] += t.logpdf(self.X[i,j], df=2*a_prop[0], loc=np.dot(self.W[fk_temp[0],j][i], mu_prop[j][0]), 
                            scale=np.sqrt(b_prop[j][0] / a_prop[0] * (1 + np.dot(self.W[fk_temp[0],j][i].T, np.dot(Lambda_prop[j][0], self.W[fk_temp[0],j][i])))))
                if j == 0 and self.first_linear[self.fk[z if split else z_prime]]:
                    community_probs[1] += t.logpdf(self.X[i,j], df=2*a_prop[1], loc=self.theta[i], scale=np.sqrt(b_prop[j][1] / a_prop[1])) 
                else:
                    community_probs[1] += t.logpdf(self.X[i,j], df=2*a_prop[1], loc=np.dot(self.W[fk_temp[1],j][i], mu_prop[j][1]), 
                            scale=np.sqrt(b_prop[j][1] / a_prop[1] * (1 + np.dot(self.W[fk_temp[1],j][i].T, np.dot(Lambda_prop[j][1], self.W[fk_temp[1],j][i])))))              
            ## Raise error if nan probabilities are computed
            if np.isnan(community_probs).any():
                print(community_probs)
                raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
            ## Update allocation
            community_probs = np.exp(community_probs - logsumexp(community_probs))
            if split:
                znew = np.random.choice(2, p=community_probs)
            else:
                znew = int(self.z[i] == z_prime)
            zz = np.append(zz, znew)
            qprob += np.log(community_probs)[znew]
            knew = fk_temp[znew]
            ## Update parameters
            a_prop[znew] += 0.5
            nk_prop[znew] += 1
            for j in range(self.d):
                if j == 0 and self.first_linear[knew]:
                    b_prop[j][znew] += (self.X[i,j] - self.theta[i]) ** 2 / 2
                else:
                    b_prop[j][znew] += np.dot(mu_prop[j][znew].T,np.dot(Lambda_inv_prop[j][znew],mu_prop[j][znew])) / 2 
                    WtW_prop[j][znew] += np.outer(self.W[knew,j][i],self.W[knew,j][i])
                    WtX_prop[j][znew] += self.W[knew,j][i] * self.X[i,j]
                    Lambda_inv_prop[j][znew] = WtW_prop[j][znew] + self.Lambda0_inv[knew,j]
                    Lambda_prop[j][znew] = np.linalg.inv(Lambda_inv_prop[j][znew])
                    mu_prop[j][znew] = np.dot(Lambda_prop[j][znew], WtX_prop[j][znew])
                    b_prop[j][znew] += (self.X[i,j] ** 2 - np.dot(mu_prop[j][znew].T,np.dot(Lambda_inv_prop[j][znew],mu_prop[j][znew]))) / 2
        ## Calculate acceptance ratio
        acceptance_ratio = self.K * loggamma(self.nu / self.K) - K_prop * loggamma(self.nu / K_prop)
        nk_ast = np.copy(self.nk)
        if split:
            nk_ast[z] = nk_prop[0]; nk_ast = np.append(nk_ast, nk_prop[1])
        else:
            nk_ast[z] = nk_merge; nk_ast = np.delete(arr=nk_ast, obj=z_prime)
        acceptance_ratio += np.sum(loggamma(nk_ast + self.nu / K_prop)) - np.sum(loggamma(self.nk + self.nu / self.K))
        acceptance_ratio += (K_prop - self.K) * np.log(1 - self.omega) 
        ## acceptance_ratio += np.log(.5) * int(self.K == 1) - np.log(.5) * int(self.K == self.n)
        acceptance_ratio -= (1 if split else -1) * qprob
        if split:
            indices0 = np.append([q],indices[zz == 0])
            indices1 = np.append([q_prime],indices[zz == 1])
            indices_all = np.append([q,q_prime],indices)
            for j in range(self.d):
                if j == 0 and self.first_linear[fk_prop]:
                    acceptance_ratio += np.sum(loggamma(a_prop) - loggamma(self.a0) + self.a0 * loggamma(self.b0) - nk_prop/2 * loggamma(2*np.pi))
                    acceptance_ratio -= a_prop[0] * loggamma(self.b0 + np.sum((self.X[:,j][indices0] - self.theta[indices0]) ** 2) / 2)
                    acceptance_ratio -= a_prop[1] * loggamma(self.b0 + np.sum((self.X[:,j][indices1] - self.theta[indices1]) ** 2) / 2)
                    acceptance_ratio -= loggamma(self.a[z]) - loggamma(self.a0) + self.a0 * loggamma(self.b0) - self.nk[z]/2 * loggamma(2*np.pi)
                    acceptance_ratio += self.a[z] * loggamma(self.b0 + np.sum((self.X[:,j][indices_all] - self.theta[indices_all]) ** 2) / 2) 
                else:
                    S0 = np.dot(self.W[fk_prop,j][indices0], np.dot(self.Lambda0[fk_prop,j], self.W[fk_prop,j][indices0].T)) + np.diag(np.ones(nk_prop[0]))
                    acceptance_ratio += dmvt(x=self.X[indices0,j], mu=np.zeros(nk_prop[0]), Sigma=self.b0 / self.a0 * S0, nu=2*self.a0)
                    S1 = np.dot(self.W[fk_prop,j][indices1], np.dot(self.Lambda0[fk_prop,j], self.W[fk_prop,j][indices1].T)) + np.diag(np.ones(nk_prop[1]))
                    acceptance_ratio += dmvt(x=self.X[indices1,j], mu=np.zeros(nk_prop[1]), Sigma=self.b0 / self.a0 * S1, nu=2*self.a0)
                    S_all = np.dot(self.W[fk_prop,j][indices_all], np.dot(self.Lambda0[fk_prop,j], self.W[fk_prop,j][indices_all].T)) + np.diag(np.ones(self.nk[z]))
                    acceptance_ratio -= dmvt(x=self.X[indices_all,j], mu=np.zeros(self.nk[z]), Sigma=self.b0 / self.a0 * S_all, nu=2*self.a0)
        else:
            indices0 = np.append([q],indices[zz == 0])
            indices1 = np.append([q_prime],indices[zz == 1])
            indices_all = np.append([q,q_prime],indices)
            for j in range(self.d):
                if j == 0 and self.first_linear[fk_prop]:
                    acceptance_ratio -= np.sum(loggamma(a_prop) - loggamma(self.a0) + self.a0 * loggamma(self.b0) - nk_prop/2 * loggamma(2*np.pi))
                    acceptance_ratio += a_prop[0] * loggamma(self.b0 + np.sum((self.X[:,j][indices0] - self.theta[indices0]) ** 2) / 2)
                    acceptance_ratio += a_prop[1] * loggamma(self.b0 + np.sum((self.X[:,j][indices1] - self.theta[indices1]) ** 2) / 2)
                    acceptance_ratio += loggamma(a_merge) - loggamma(self.a0) + self.a0 * loggamma(self.b0) - nk_merge/2 * loggamma(2*np.pi)
                    acceptance_ratio -= a_merge * loggamma(self.b0 + np.sum((self.X[:,j][indices_all] - self.theta[indices_all]) ** 2) / 2) 
                else:
                    S0 = np.dot(self.W[fk_temp[0],j][indices0], np.dot(self.Lambda0[fk_temp[0],j], self.W[fk_temp[0],j][indices0].T)) + np.diag(np.ones(nk_prop[0]))
                    acceptance_ratio -= dmvt(x=self.X[indices0,j], mu=np.zeros(nk_prop[0]), Sigma=self.b0 / self.a0 * S0, nu=2*self.a0)
                    S1 = np.dot(self.W[fk_temp[1],j][indices1], np.dot(self.Lambda0[fk_temp[1],j], self.W[fk_temp[1],j][indices1].T)) + np.diag(np.ones(nk_prop[1]))
                    acceptance_ratio -= dmvt(x=self.X[indices1,j], mu=np.zeros(nk_prop[1]), Sigma=self.b0 / self.a0 * S1, nu=2*self.a0)
                    S_all = np.dot(self.W[fk_prop,j][indices_all], np.dot(self.Lambda0[fk_prop,j], self.W[fk_prop,j][indices_all].T)) + np.diag(np.ones(nk_merge))
                    acceptance_ratio += dmvt(x=self.X[indices_all,j], mu=np.zeros(nk_merge), Sigma=self.b0 / self.a0 * S_all, nu=2*self.a0)            
        # Accept / reject using Metropolis-Hastings
        accept = (-np.random.exponential(1) < acceptance_ratio)
        if verbose:
            print('\t',['Merge','Split'][int(split)], bool(accept), z, z_prime, K_prop, end='')
        # Update if move is accepted
        if accept:
            if split:
                self.z[indices1] = z_prime
                self.fk = np.append(self.fk, values=fk_prop)
                self.nk[z] = nk_prop[0]; self.nk = np.append(self.nk, nk_prop[1])
                self.a[z] = a_prop[0]; self.a = np.append(self.a, a_prop[1])
                for j in range(self.d):
                    self.b[j][z] = b_prop[j][0]; self.b[j][self.K] = b_prop[j][1]
                    if not (j == 0 and self.first_linear[fk_prop]):
                        self.WtW[j][z] = WtW_prop[j][0]; self.WtW[j][self.K] = WtW_prop[j][1] 
                        self.WtX[j][z] = WtX_prop[j][0]; self.WtX[j][self.K] = WtX_prop[j][1] 
                        self.Lambda_inv[j][z] = Lambda_inv_prop[j][0]; self.Lambda_inv[j][self.K] = Lambda_inv_prop[j][1]
                        self.Lambda[j][z] = Lambda_prop[j][0]; self.Lambda[j][self.K] = Lambda_prop[j][1]
                        self.mu[j][z] = mu_prop[j][0]; self.mu[j][self.K] = mu_prop[j][1]
            else:
                self.z[self.z == z_prime] = z
                self.fk[z] = fk_prop; self.fk = np.delete(arr=self.fk, obj=z_prime)
                self.nk[z] = nk_merge; self.nk = np.delete(arr=self.nk, obj=z_prime)
                self.a[z] = a_merge; self.a = np.delete(arr=self.a, obj=z_prime)
                for j in range(self.d):
                    self.b[j][z] = b_merge[j]; del self.b[j][z_prime]
                    if not (j == 0 and self.first_linear[fk_prop]):
                        self.WtW[j][z] = WtW_merge[j]; del self.WtW[j][z_prime]
                        self.WtX[j][z] = WtX_merge[j]; del self.WtX[j][z_prime]
                        self.Lambda_inv[j][z] = Lambda_inv_merge[j]; del self.Lambda_inv[j][z_prime]
                        self.Lambda[j][z] = Lambda_merge[j]; del self.Lambda[j][z_prime]
                        self.mu[j][z] = mu_merge[j]; del self.mu[j][z_prime]
                ## Relabel groups
                Q = np.arange(self.K)[np.arange(self.K) > z_prime]
                for k in Q:
                    self.z[self.z == k] = k-1
                    for j in range(self.d):
                        self.b[j][k-1] = self.b[j][k]; del self.b[j][k]
                        if not (j == 0 and self.first_linear[self.fk[k-1]]):
                            self.WtW[j][k-1] = self.WtW[j][k]; del self.WtW[j][k]
                            self.WtX[j][k-1] = self.WtX[j][k]; del self.WtX[j][k]
                            self.Lambda_inv[j][k-1] = self.Lambda_inv[j][k]; del self.Lambda_inv[j][k]
                            self.Lambda[j][k-1] = self.Lambda[j][k]; del self.Lambda[j][k]
                            self.mu[j][k-1] = self.mu[j][k]; del self.mu[j][k]
            ## Update K
            self.K = K_prop 
        return None

    ######################################################
    ### e. Resample community-specific functional form ###
    ######################################################
    def resample_kernel(self, verbose=False):
        ## Sample existing community
        k_group = np.random.choice(self.K) 
        fk_old = self.fk[k_group]
        ## Initialise hyperparameter vectors
        S = {}; b_kernels = {}; WtW_kernels = {}; WtX_kernels = {}; Lambda_inv_kernels = {}; Lambda_kernels = {}; mu_kernels = {}
        X = self.X[self.z == k_group]
        theta = self.theta[self.z == k_group]
        ## Calculate vector of probabilities
        probs = np.zeros(self.kappa)
        for k in range(self.kappa):
            b_kernels[k] = {}; WtW_kernels[k] = {}; WtX_kernels[k] = {}; Lambda_inv_kernels[k] = {}; Lambda_kernels[k] = {}; mu_kernels[k] = {}
            XX = np.copy(X)
            for j in range(self.d):
                if j in self.fixed_function:
                    XX[:,j] -= self.fixed_W[j][self.z == k_group]
                if j == 0 and self.first_linear[k]:
                    b_kernels[k][j] = self.b0 + np.sum((XX[:,j] - theta) ** 2) / 2
                    probs[k] += loggamma(self.a[k_group]) - loggamma(self.a0) - self.nk[k_group] * np.log(2*np.pi) / 2
                    probs[k] += self.a0 * loggamma(self.b0) - self.a[k_group] * loggamma(self.b0 + np.sum((XX[:,j] - theta) ** 2) / 2)
                else:
                    W = self.W[k,j][self.z == k_group]
                    WtW_kernels[k][j] = np.dot(W.T,W)
                    WtX_kernels[k][j] = np.dot(W.T,XX[:,j])
                    Lambda_inv_kernels[k][j] = WtW_kernels[k][j] + self.Lambda0_inv[k,j]
                    Lambda_kernels[k][j] = np.linalg.inv(Lambda_inv_kernels[k][j])
                    mu_kernels[k][j] = np.dot(Lambda_kernels[k][j], WtX_kernels[k][j])
                    b_kernels[k][j] = self.b0 + (np.dot(XX[:,j].T,XX[:,j]) - np.dot(mu_kernels[k][j].T, np.dot(Lambda_inv_kernels[k][j],mu_kernels[k][j]))) / 2
                    S[k,j] = np.dot(W, np.dot(self.Lambda0[k,j], W.T)) + np.diag(np.ones(self.nk[k_group]))
                    probs[k] += dmvt(x=XX[:,j], mu=np.zeros(self.nk[k_group]), Sigma=self.b0 / self.a0 * S[k,j], nu=2*self.a0) 
        ## Resample new functional form
        fk_new = np.random.choice(self.kappa, p=np.exp(probs - logsumexp(probs)))
        if verbose:
            print('\t',fk_old, fk_new, k_group, end='')
        if fk_new != fk_old:
            self.fk[k_group] = fk_new
            for j in range(self.d):
                self.b[j][k_group] = b_kernels[fk_new][j]
                if not(j == 0 and self.first_linear[fk_new]):
                    self.WtW[j][k_group] = WtW_kernels[fk_new][j]
                    self.WtX[j][k_group] = WtX_kernels[fk_new][j]
                    self.Lambda_inv[j][k_group] = Lambda_inv_kernels[fk_new][j]
                    self.Lambda[j][k_group] = Lambda_kernels[fk_new][j]
                    self.mu[j][k_group] = mu_kernels[fk_new][j]
        return None

    ###############################
    ### Marginal log-likelihood ###
    ###############################
    def marginal_loglikelihood(self):
        loglik = 0
        for k in range(self.K):
            loglik -= self.d * self.nk[k] / 2 * np.log(2*np.pi) 
            loglik += self.d * (self.a0 * np.log(self.b0) - loggamma(self.a0))
            for j in range(self.d):
                if j == 0 and self.first_linear[k if self.K_fixed else self.fk[k]]:
                    loglik -= self.a[k] * np.log(self.b[j][k])
                else:
                    loglik += np.prod(np.linalg.slogdet(self.Lambda[j][k if self.K_fixed else self.fk[k]])) / 2 
                    loglik -= self.a[k if self.K_fixed else self.fk[k]] * np.log(self.b[j][k if self.K_fixed else self.fk[k]]) 
                    loglik -= np.prod(np.linalg.slogdet(self.Lambda0[k if self.K_fixed else self.fk[k],j])) / 2
        return loglik

    ###################################################################################
    ### Calculate maximum a posteriori estimate of the parameters given z and theta ###
    ###################################################################################
    def map(self,z,theta,range_values):
        mm = lsbm_gibbs(X=self.X, K=self.K, W_function=self.fW, fixed_function=self.fixed_W)
        mm.initialise(z=z, theta=theta, Lambda_0=self.lambda_coef, a_0=self.a0, b_0=self.b0, nu=self.nu, g_prior=self.g_prior)
        W = {}
        mean = {}; confint = {} 
        for j in range(mm.d):
            for k in range(mm.K):
                mean[k,j] = np.zeros(len(range_values))
                confint[k,j] = np.zeros((len(range_values),2))
        for i in range(len(range_values)):
            x = range_values[i]
            for j in range(mm.d):
                for k in range(mm.K):
                    if j == 0 and mm.first_linear[k]:
                        mean[k,j][i] = x
                        confint[k,j][i] = t.interval(0.95, df=2*mm.a[k], loc=x, scale=np.sqrt(mm.b[j][k] / mm.a[k])) 
                    else:
                        W[k,j] = mm.fW[k,j](x)
                        mean[k,j][i] = np.dot(W[k,j],mm.mu[j][k])
                        confint[k,j][i] = t.interval(0.95, df=2*mm.a[k], loc=mean[k,j][i], 
                                scale=np.sqrt(mm.b[j][k] / mm.a[k] * (1 + np.dot(W[k,j].T, np.dot(mm.Lambda[j][k], W[k,j])))))
        return mean, confint, mm.mu, mm.marginal_loglikelihood()

    #####################
    ### MCMC sampling ###
    #####################
    def mcmc(self, samples=1000, burn=100, chains=1, store_chains=True, q=100, thinning=1, sigma_prop=0.1, verbose=False):
        ## q is the number of re-allocated nodes per iteration
        self.q = q
        ## Options for move
        move = ['communities', 'parameters']
        if not self.K_fixed:
            move += ['add_remove_empty', 'split_merge']
        if self.kappa > 1:
            move += ['resample_kernel']
        ## Option to store the chains
        if store_chains:
            theta_chain = np.zeros((self.n,chains,samples // thinning))
            z_chain = np.zeros((self.n,chains,samples // thinning))
        for chain in range(chains):
            for s in range(samples+burn):
                print('\rChain:', chain+1,'/', chains, '\tBurnin:', s+1 if s<burn else burn, '/', burn, 
                    '\tSamples:', s-burn+1 if s>=burn else 0,'/', samples, end='')
                m = np.random.choice(move)
                if m == 'communities':
                    self.gibbs_communities(l=self.q)
                elif m == 'parameters':
                    self.resample_theta(l=self.q, sigma_prop=sigma_prop)
                elif m == 'add_remove_empty':
                    self.propose_empty(verbose=verbose)
                elif m == 'split_merge':
                    self.split_merge(verbose=verbose)
                else:
                    self.resample_kernel(verbose=verbose)
                if verbose:
                    print('\t',m)
                if s >= burn and store_chains and s % thinning == 0:
                    theta_chain[:,chain,(s - burn) // thinning] = self.theta
                    z_chain[:,chain,(s - burn) // thinning] = self.z
        print('')
        if store_chains:
            return theta_chain, z_chain
        else:
            return None