#! /usr/bin/env python3
import numpy as np
from collections import Counter
from scipy.stats import norm
from sklearn.cluster import KMeans as kmeans
from scipy.stats import beta as Beta
from scipy.stats import dirichlet as Diri
from scipy.special import logit, expit, logsumexp, loggamma
from scipy.stats import t

###########################
### Quadratic embedding ###
###########################

class lsbm_gibbs:

    ## Initialise the class with the number of components and embedding
    def __init__(self, X, K, W_function, fixed_function={}):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.K = K
        ## Function to obtain the design matrix
        self.fW = W_function 
        self.fixed_function = fixed_function

    ## Initialise the model parameters
    def initialise(self, z, theta, Lambda_0=1, a_0=1, b_0=1, nu=1, mu_theta=0, sigma_theta=1, g_prior=True, first_linear=True):
        ## Initial cluster configuration
        self.z = z
        if np.min(self.z) == 1:
            self.z -= 1
        self.theta = theta
        ## Theta hyperparameters
        self.mu_theta = mu_theta
        self.sigma_theta = sigma_theta
        ## Basis functions
        self.first_linear = first_linear
        self.g_prior = g_prior
        self.W = {}
        for j in range(self.d):
            self.W[j] = np.array([self.fW[j](self.theta[i]) for i in range(self.n)])
        self.fixed_W = {}
        for j in self.fixed_function:
            self.fixed_W[j] = np.array([self.fixed_function[j](self.theta[i]) for i in range(self.n)])[:,0] ## rewrite using only one coefficient (1)
        ## Prior parameters
        self.nu = nu
        self.a0 = a_0
        self.b0 = b_0
        self.lambda_coef = Lambda_0
        self.Lambda0 = {}; self.Lambda0_inv = {}
        for j in range(self.d):
            if g_prior:
                self.Lambda0_inv[j] = Lambda_0 * np.dot(self.W[j].T,self.W[j]) # np.diag(np.ones(len(self.fW[j](1))) * Lambda_0)
            else:
                self.Lambda0_inv[j] = np.diag(np.ones(len(self.fW[j](1))) * Lambda_0)
            self.Lambda0[j] = np.linalg.inv(self.Lambda0_inv[j])
        self.nu = nu
        ## Initialise hyperparameter vectors
        self.nk = Counter(self.z)
        self.a = {}
        for k in self.nk:
            self.a[k] = self.a0 + self.nk[k] / 2  
        self.b = {}; self.WtW = {}; self.WtX = {}; self.Lambda_inv = {}; self.Lambda = {}; self.mu = {}
        for j in range(self.d):
            self.b[j] = {}; self.WtW[j] = {}; self.WtX[j] = {}; self.Lambda_inv[j] = {}; self.Lambda[j] = {}; self.mu[j] = {}
            for k in range(self.K):
                X = self.X[self.z == k][:,j]
                if j in self.fixed_function:
                    X -= self.fixed_W[j][self.z == k]
                if j == 0 and self.first_linear:
                    self.b[j][k] = self.b0 + np.sum((X - self.theta[self.z == k]) ** 2) / 2
                else:
                    W = self.W[j][self.z == k]
                    self.WtW[j][k] = np.dot(W.T,W)
                    self.WtX[j][k] = np.dot(W.T,X)
                    self.Lambda_inv[j][k] = self.WtW[j][k] + self.Lambda0_inv[j]
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
            self.nk[zold] -= 1.0
            for j in range(self.d):
                position[j] = self.X[i,j]
                if j in self.fixed_function:
                    position[j] -= self.fixed_W[j][i]
                if j == 0 and self.first_linear:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] - self.theta[i]) ** 2 / 2
                else:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] ** 2 - np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold]))) / 2 
                    WtW_old[j] = np.copy(self.WtW[j][zold])
                    WtX_old[j] = np.copy(self.WtX[j][zold])
                    self.WtW[j][zold] -= np.outer(self.W[j][i],self.W[j][i])
                    self.WtX[j][zold] -= self.W[j][i] * position[j]
                    Lambda_inv_old[j] = np.copy(self.Lambda_inv[j][zold])
                    Lambda_old[j] = np.copy(self.Lambda[j][zold])
                    self.Lambda_inv[j][zold] = self.WtW[j][zold] + self.Lambda0_inv[j]
                    self.Lambda[j][zold] = np.linalg.inv(self.Lambda_inv[j][zold])
                    mu_old[j] = np.copy(self.mu[j][zold])
                    self.mu[j][zold] = np.dot(self.Lambda[j][zold], self.WtX[j][zold])
                    self.b[j][zold] -= np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold])) / 2
            ## Calculate probabilities for community allocations
            community_probs = np.log((np.array([self.nk[k] for k in range(self.K)]) + self.nu/self.K) / (self.n - 1 + self.K))
            for k in range(self.K):
                for j in range(self.d):
                    if j == 0 and self.first_linear:
                        community_probs[k] += t.logpdf(position[j], df=2*self.a[k], loc=self.theta[i], scale=np.sqrt(self.b[j][k] / self.a[k])) 
                    else:
                        community_probs[k] += t.logpdf(position[j], df=2*self.a[k], loc=np.dot(self.W[j][i],self.mu[j][k]), 
                                scale=np.sqrt(self.b[j][k] / self.a[k] * (1 + np.dot(self.W[j][i].T, np.dot(self.Lambda[j][k], self.W[j][i])))))              
            ## Raise error if nan probabilities are computed
            if np.isnan(community_probs).any():
                print(community_probs)
                raise ValueError("Error in the allocation probabilities. Check invertibility of the covariance matrices.")
            ## Update allocation
            self.z[i] = np.random.choice(self.K, p=np.exp(community_probs - logsumexp(community_probs)))
            ## Update parameters
            self.a[self.z[i]] += .5
            self.nk[self.z[i]] += 1.0
            if self.z[i] == zold:
                ## Re-update to old values
                for j in range(self.d):
                    self.b[j][self.z[i]] = b_old[j]
                    if not (j == 0 and self.first_linear):
                        self.WtW[j][self.z[i]] = WtW_old[j]
                        self.WtX[j][self.z[i]] = WtX_old[j]
                        self.Lambda_inv[j][self.z[i]] = Lambda_inv_old[j]
                        self.Lambda[j][self.z[i]] = Lambda_old[j]
                        self.mu[j][self.z[i]] = mu_old[j] 
            else:
                ## Update to new values
                for j in range(self.d):
                    if j == 0 and self.first_linear:
                        self.b[j][self.z[i]] += (position[j] - self.theta[i]) ** 2 / 2
                    else:
                        self.b[j][self.z[i]] += np.dot(self.mu[j][self.z[i]].T,np.dot(self.Lambda_inv[j][self.z[i]],self.mu[j][self.z[i]])) / 2 
                        self.WtW[j][self.z[i]] += np.outer(self.W[j][i],self.W[j][i])
                        self.WtX[j][self.z[i]] += self.W[j][i] * position[j]
                        self.Lambda_inv[j][self.z[i]] = self.WtW[j][self.z[i]] + self.Lambda0_inv[j]
                        self.Lambda[j][self.z[i]] = np.linalg.inv(self.Lambda_inv[j][self.z[i]])
                        self.mu[j][self.z[i]] = np.dot(self.Lambda[j][self.z[i]], self.WtX[j][self.z[i]])
                        self.b[j][self.z[i]] += (position[j] ** 2 - np.dot(self.mu[j][self.z[i]].T,np.dot(self.Lambda_inv[j][self.z[i]],self.mu[j][self.z[i]]))) / 2
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
            self.nk[zold] -= 1.0
            for j in range(self.d):
                position[j] = self.X[i,j]
                if j in self.fixed_function:
                    position[j] -= self.fixed_W[j][i]
                if j == 0 and self.first_linear:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] - theta_old) ** 2 / 2
                else:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] ** 2 - np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold]))) / 2 
                    WtW_old[j] = np.copy(self.WtW[j][zold])
                    WtX_old[j] = np.copy(self.WtX[j][zold])
                    self.WtW[j][zold] -= np.outer(self.W[j][i],self.W[j][i])
                    self.WtX[j][zold] -= self.W[j][i] * position[j]
                    Lambda_inv_old[j] = np.copy(self.Lambda_inv[j][zold])
                    Lambda_old[j] = np.copy(self.Lambda[j][zold])
                    self.Lambda_inv[j][zold] = self.WtW[j][zold] + self.Lambda0_inv[j]
                    self.Lambda[j][zold] = np.linalg.inv(self.Lambda_inv[j][zold])
                    mu_old[j] = np.copy(self.mu[j][zold])
                    self.mu[j][zold] = np.dot(self.Lambda[j][zold], self.WtX[j][zold])
                    self.b[j][zold] -= np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold])) / 2
            ## Calculate proposal
            theta_prop = np.random.normal(loc=theta_old, scale=sigma_prop)
            for j in range(self.d):
                position_prop[j] = self.X[i,j]
                W_prop[j] = self.fW[j](theta_prop)
                if j in self.fixed_function:
                    W_prop_fixed[j] = self.fixed_function[j](theta_prop)
                    position_prop[j] -= W_prop_fixed[j]
            ## Calculate acceptance ratio
            numerator_accept = norm.logpdf(theta_prop,loc=self.mu_theta,scale=self.sigma_theta)
            for j in range(self.d):
                if j == 0 and self.first_linear:
                    numerator_accept += t.logpdf(position_prop[j], df=2*self.a[zold], loc=theta_prop, scale=np.sqrt(self.b[j][zold] / self.a[zold])) 
                else:
                    numerator_accept += t.logpdf(position_prop[j], df=2*self.a[zold], loc=np.dot(W_prop[j],self.mu[j][zold]), 
                            scale=np.sqrt(self.b[j][zold] / self.a[zold] * (1 + np.dot(W_prop[j].T, np.dot(self.Lambda[j][zold], W_prop[j])))))
            denominator_accept = norm.logpdf(theta_old,loc=self.mu_theta,scale=self.sigma_theta)
            for j in range(self.d):
                if j == 0 and self.first_linear:
                    denominator_accept += t.logpdf(position[j], df=2*self.a[zold], loc=theta_old, scale=np.sqrt(self.b[j][zold] / self.a[zold])) 
                else:
                    denominator_accept += t.logpdf(position[j], df=2*self.a[zold], loc=np.dot(self.W[j][i],self.mu[j][zold]), 
                            scale=np.sqrt(self.b[j][zold] / self.a[zold] * (1 + np.dot(self.W[j][i].T, np.dot(self.Lambda[j][zold], self.W[j][i])))))             
            ## Calculate acceptance probability
            accept_ratio = numerator_accept - denominator_accept
            accept = (-np.random.exponential(1) < accept_ratio)
            ## Update parameters
            self.a[zold] += .5
            self.nk[zold] += 1.0
            if accept:
                self.theta[i] = theta_prop
            if not accept:
                ## Re-update to old values
                for j in range(self.d):
                    self.b[j][zold] = b_old[j]
                    if not (j == 0 and self.first_linear):
                        self.WtW[j][zold] = WtW_old[j]
                        self.WtX[j][zold] = WtX_old[j]
                        self.Lambda_inv[j][zold] = Lambda_inv_old[j]
                        self.Lambda[j][zold] = Lambda_old[j]
                        self.mu[j][zold] = mu_old[j] 
            else:
                ## Update to new values
                for j in range(self.d):
                    ## Update design matrix
                    self.W[j][i] = W_prop[j]
                    if j in self.fixed_function:
                        self.fixed_W[j][i] = W_prop_fixed[j]
                    if j == 0 and self.first_linear:
                        self.b[j][zold] += (position[j] - self.theta[i]) ** 2 / 2
                    else:
                        self.b[j][zold] += np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold])) / 2 
                        self.WtW[j][zold] += np.outer(self.W[j][i],self.W[j][i])
                        self.WtX[j][zold] += self.W[j][i] * position_prop[j]
                        self.Lambda_inv[j][zold] = self.WtW[j][zold] + self.Lambda0_inv[j]
                        self.Lambda[j][zold] = np.linalg.inv(self.Lambda_inv[j][zold])
                        self.mu[j][zold] = np.dot(self.Lambda[j][zold], self.WtX[j][zold])
                        self.b[j][zold] += (position_prop[j] ** 2 - np.dot(self.mu[j][zold].T,np.dot(self.Lambda_inv[j][zold],self.mu[j][zold]))) / 2
        return None

    ###############################
    ### Marginal log-likelihood ###
    ###############################
    def marginal_loglikelihood(self):
        loglik = 0
        for k in range(self.K):
            loglik -= self.d * self.nk[k]/2 * np.log(2*np.pi) 
            loglik += self.d * (self.a0 * np.log(self.b0) - loggamma(self.a0))
            for j in range(self.d):
                if j == 0 and self.first_linear:
                    loglik -= self.a[k] * np.log(self.b[j][k])
                else:
                    loglik += np.sqrt(np.linalg.det(self.Lambda[j][k])) - self.a[k] * np.log(self.b[j][k]) - np.sqrt(np.linalg.det(self.Lambda0[j]))
        return loglik

    ###################################################################################
    ### Calculate maximum a posteriori estimate of the parameters given z and theta ###
    ###################################################################################
    def map(self,z,theta,range_values):
        mm = lsbm_gibbs(X=self.X, K=self.K, W_function=self.fW, fixed_function=self.fixed_W)
        mm.initialise(z=z, theta=theta, Lambda_0=self.lambda_coef, a_0=self.a0, b_0=self.b0, nu=self.nu, 
                        g_prior=self.g_prior, first_linear=self.first_linear)
        W = {}
        mean = {}; confint = {} 
        for j in range(mm.d):
            for k in range(mm.K):
                mean[k,j] = np.zeros(len(range_values))
                confint[k,j] = np.zeros((len(range_values),2))
        for i in range(len(range_values)):
            x = range_values[i]
            for j in range(mm.d):
                W[j] = mm.fW[j](x)
                for k in range(mm.K):
                    if j == 0 and mm.first_linear:
                        mean[k,j][i] = x
                        confint[k,j][i] = t.interval(0.95, df=2*mm.a[k], loc=x, scale=np.sqrt(mm.b[j][k] / mm.a[k])) 
                    else:
                        mean[k,j][i] = np.dot(W[j],mm.mu[j][k])
                        confint[k,j][i] = t.interval(0.95, df=2*mm.a[k], loc=mean[k,j][i], 
                                scale=np.sqrt(mm.b[j][k] / mm.a[k] * (1 + np.dot(W[j].T, np.dot(mm.Lambda[j][k], W[j])))))
        return mean, confint, mm.mu, mm.marginal_loglikelihood()

    def bootstrapped_ci(self,z,theta,boot_samples=150):
        boot_mu = {}
        for j in range(self.d):
            boot_mu[j] = {}
            if not (j == 0 and self.first_linear):
                for k in range(self.K):
                    boot_mu[j][k] = np.zeros((boot_samples,len(self.mu[j][k])))
        for b in range(boot_samples):
            Xboot = np.copy(self.X)
            thetaboot = np.copy(theta)
            for k in range(self.K):
                posk = np.where(z == k)[0]
                boot = np.random.choice(posk, size=len(posk), replace=True)
                thetaboot[posk] = theta[boot]
                Xboot[posk] = self.X[boot]
            mm = lsbm_gibbs(X=Xboot, K=self.K, W_function=self.fW, fixed_function=self.fixed_W)
            mm.initialise(z=z, theta=thetaboot, Lambda_0=self.lambda_coef, a_0=self.a0, b_0=self.b0, nu=self.nu, 
                        g_prior=self.g_prior, first_linear=self.first_linear)
            for j in range(self.d):
                if not (j == 0 and self.first_linear):
                    for k in range(self.K):
                        boot_mu[j][k][b] = mm.mu[j][k]
        return boot_mu

    #####################
    ### MCMC sampling ###
    #####################
    def mcmc(self, samples=1000, burn=100, chains=1, store_chains=True, q=100, thinning=1, sigma_prop=0.1):
        ## q is the number of re-allocated nodes per iteration
        self.q = q
        ## Option to store the chains
        if store_chains:
            theta_chain = np.zeros((self.n,chains,samples // thinning))
            z_chain = np.zeros((self.n,chains,samples // thinning))
        for chain in range(chains):
            for s in range(samples+burn):
                print('\rChain:', chain+1,'/', chains, '\tBurnin:', s+1 if s<burn else burn, '/', burn, '\tSamples:', s-burn+1 if s>burn else 0,'/', samples, end='')
                move = ['communities','parameters']
                m = np.random.choice(move)
                if m == 'communities':
                    self.gibbs_communities(l=self.q)
                else:
                    self.resample_theta(l=self.q, sigma_prop=sigma_prop)
                if s >= burn and store_chains and s % thinning == 0:
                    theta_chain[:,chain,(s - burn) // thinning] = self.theta
                    z_chain[:,chain,(s - burn) // thinning] = self.z
        print('')
        if store_chains:
            return theta_chain, z_chain
        else:
            return None