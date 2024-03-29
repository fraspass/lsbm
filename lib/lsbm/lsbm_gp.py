#! /usr/bin/env python3
import numpy as np
from collections import Counter
from scipy.stats import norm, t
from scipy.special import logsumexp

####################################
### LSBM with Gaussian processes ###
####################################

## Utility functions to update inverses when row/columns are added/removed

# 1) Update inverse matrix M when the i-th row/column is removed
# M is already an inverse matrix
def delete_inverse(M, ind):
  A = np.delete(arr=np.delete(arr=M,obj=ind,axis=0),obj=ind,axis=1)
  B = np.delete(arr=M,obj=ind,axis=0)[:,ind]
  C = np.delete(arr=M,obj=ind,axis=1)[ind]
  D = M[ind,ind]
  return A - np.outer(B,C) / D

# 2) Update inverse matrix M when a row/column is added on position i
# M is already an inverse matrix
# This function might create numerical problems for some values of the prior parameters. 
# If fast_update=True returns errors, try fast_update=False, and this function will not be used.
def add_inverse(M, row, col, ind):
  A11_inv = M
  A12 = np.delete(arr=col,obj=ind).reshape(-1,1)
  A21 = np.delete(arr=row,obj=ind).reshape(1,-1)
  A22 = col[ind]
  ## Inverse
  F22_inv = 1 / (A22 - np.matmul(np.matmul(A21,A11_inv),A12))
  F11_inv = A11_inv + F22_inv * np.matmul(np.matmul(np.matmul(A11_inv,A12),A21),A11_inv)
  B21 = - F22_inv * np.matmul(A21,A11_inv)
  B12 = - F22_inv * np.matmul(A11_inv,A12)
  ## Set up the matrix
  inv = F11_inv
  inv = np.insert(arr=inv, obj=ind, values=B21, axis=0)
  inv = np.insert(arr=inv, obj=ind, values=np.insert(arr=B12,obj=ind, values=F22_inv), axis=1)
  return inv

class lsbm_gp_gibbs:

    ## Initialise the class with the number of components and embedding
    def __init__(self, X, K, csi):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.K = K
        ## Kernel functions
        self.csi = csi

    ## Initialise the model parameters
    def initialise(self, z, theta, a_0=1, b_0=1, nu=1, mu_theta=0, sigma_theta=1, first_linear=True):
        ## Initial cluster configuration
        self.z = z
        if np.min(self.z) == 1:
            self.z -= 1
            if np.max(self.z) == 0:
                raise ValueError('z must have at least two different labels')
        ## Theta
        self.theta = theta
        ## Theta hyperparameters
        self.mu_theta = mu_theta
        self.sigma_theta = sigma_theta
        ## Set up first_linear
        if isinstance(first_linear,bool):
            self.first_linear = self.K * [first_linear]
        else:
            self.first_linear = first_linear
            if np.sum([isinstance(first_linear[k],bool) for k in first_linear]) != self.K:
                raise ValueError('first_linear is either a boolean or a K-vector of booleans')
        ## Prior parameters
        self.nu = nu
        self.a0 = a_0
        self.b0 = b_0
        ## Initialise hyperparameter vectors
        self.nk = Counter(self.z)
        self.a = {}; self.groups = {}; self.X_groups = {}; self.theta_groups = {}
        for k in range(self.K):
            self.a[k] = self.a0 + self.nk[k] / 2.
            self.groups[k] = np.where(self.z == k)[0]
            self.X_groups[k] = self.X[self.groups[k]]
            self.theta_groups[k] = self.theta[self.groups[k]]
        self.b = {}; self.Csi_I = {}; self.X_Csi_X = {}
        for j in range(self.d):
            self.b[j] = {}
            for k in range(self.K):
                X = self.X_groups[k][:,j]
                if j == 0 and self.first_linear[k]:
                    self.b[j][k] = self.b0 + np.sum((X - self.theta[self.z == k]) ** 2) / 2
                else:
                    self.Csi_I[k,j] = np.linalg.inv(self.csi[k,j](self.theta_groups[k],self.theta_groups[k]) + np.diag(np.ones(self.nk[k])))
                    self.X_Csi_X[k,j] = np.matmul(np.matmul(np.transpose(self.X_groups[k][:,j]), self.Csi_I[k,j]),self.X_groups[k][:,j])
                    self.b[j][k] = self.b0 + self.X_Csi_X[k,j] / 2
                    
    ########################################################
    ### a. Resample the allocations using Gibbs sampling ###
    ########################################################
    def gibbs_communities(self, l=1, fast_update=True):	
        ## Change the value of l when too large
        if l > self.n:
            l = self.n
        ## Update the latent allocations in randomised order
        ## Loop over the indices
        b_old = {}; Csi_I_Old = {}; X_Csi_X_Old = {}
        for i in np.random.choice(self.n, size=l, replace=False):
            zold = self.z[i]
            thetai = self.theta[i]
            position = self.X[i]
            ## Update parameters of the distribution
            self.a[zold] -= .5
            self.nk[zold] -= 1.0
            ## Update groups, X_group and theta_group
            ind_del = int(np.where(self.groups[zold] == i)[0])
            self.groups[zold] = np.delete(self.groups[zold], obj=ind_del)
            self.X_groups[zold] = np.delete(self.X_groups[zold], obj=ind_del, axis=0)
            self.theta_groups[zold] = np.delete(self.theta_groups[zold], obj=ind_del)
            ## Loop over dimensions
            for j in range(self.d):
                if j == 0 and self.first_linear[zold]:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] - thetai) ** 2 / 2
                else:
                    ## Update Csi
                    Csi_I_Old[j] = np.copy(self.Csi_I[zold,j])
                    self.Csi_I[zold,j] = delete_inverse(Csi_I_Old[j], ind=ind_del)
                    ## Update X_Csi_X
                    X_Csi_X_Old[j] = 2 * self.b[j][zold] - self.b0
                    self.X_Csi_X[zold,j] = np.matmul(np.matmul(np.transpose(self.X_groups[zold][:,j]),self.Csi_I[zold,j]),self.X_groups[zold][:,j])
                    ## Update b
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= X_Csi_X_Old[j] / 2 
                    self.b[j][zold] += self.X_Csi_X[zold,j] / 2 
            ## Calculate probabilities for community allocations
            community_probs = np.log((np.array([self.nk[k] for k in range(self.K)]) + self.nu/self.K) / (self.n - 1 + self.K))
            for k in range(self.K):
                for j in range(self.d):
                    if j == 0 and self.first_linear[k]:
                        community_probs[k] += t.logpdf(position[j], df=2*self.a[k], loc=thetai, scale=np.sqrt(self.b[j][k] / self.a[k])) 
                    else:   
                        csi_left = self.csi[k,j](thetai, self.theta_groups[k])
                        csi_prod = np.matmul(csi_left, self.Csi_I[k,j])
                        mu_star = np.matmul(csi_prod, self.X_groups[k][:,j])
                        csi_star = self.csi[k,j](thetai,thetai) - np.matmul(csi_prod, np.transpose(csi_left))
                        community_probs[k] += t.logpdf(position[j], df=2*self.a[k], loc=mu_star, scale=np.sqrt(self.b[j][k] / self.a[k] * (1 + csi_star)))              
            ## Raise error if nan probabilities are computed
            if np.isnan(community_probs).any():
                raise ValueError("Error in the allocation probabilities.")
            ## Update allocation
            znew = np.random.choice(self.K, p=np.exp(community_probs - logsumexp(community_probs)))
            self.z[i] = znew
            ind_add = np.searchsorted(self.groups[znew], i) 
            ## Update groups, X_group and theta_group
            self.groups[znew] = np.insert(self.groups[znew], obj=ind_add, values=i) 
            self.X_groups[znew] = np.insert(self.X_groups[znew], obj=ind_add, values=self.X[i], axis=0)
            self.theta_groups[znew] = np.insert(self.theta_groups[znew], obj=ind_add, values=thetai)
            ## Update parameters
            self.a[znew] += .5
            self.nk[znew] += 1.0
            if znew == zold:
                ## Re-update to old values
                for j in range(self.d):
                    self.b[j][znew] = b_old[j]
                    if not (j == 0 and self.first_linear[znew]):
                        self.Csi_I[znew,j] = Csi_I_Old[j]
                        self.X_Csi_X[znew,j] = X_Csi_X_Old[j]
            else:
                ## Update to new values
                for j in range(self.d):
                    if j == 0 and self.first_linear[znew]:
                        self.b[j][znew] += (position[j] - thetai) ** 2 / 2
                    else:
                        ## Fast update might cause numerical errors
                        if fast_update:
                            add_row = self.csi[znew,j](thetai,self.theta_groups[znew])
                            add_row[ind_add] += 1
                            self.Csi_I[znew,j] = add_inverse(self.Csi_I[znew,j], row=add_row, col=add_row, ind=ind_add)
                        else:
                            self.Csi_I[znew,j] = np.linalg.inv(self.csi[znew,j](self.theta_groups[znew],self.theta_groups[znew]) + np.diag(np.ones(int(self.nk[znew]))))
                        self.X_Csi_X[znew,j] = np.matmul(np.matmul(np.transpose(self.X_groups[znew][:,j]), self.Csi_I[znew,j]),self.X_groups[znew][:,j])
                        self.b[j][znew] += self.X_Csi_X[znew,j] / 2 
        return None

    ##############################################
    ### b. Resample the latent positions theta ###
    ##############################################
    def resample_theta(self, l=1, sigma_prop=0.1, fast_update=True):
       ## Change the value of l when too large
        if l > self.n:
            l = self.n
        ## Update the latent allocations in randomised order
        ## Loop over the indices
        b_old = {}; Csi_I_Old = {}; X_Csi_X_Old = {}
        b_prop = {}; Csi_I_Prop = {}; X_Csi_X_Prop = {}
        for i in np.random.choice(self.n, size=l, replace=False):
            zold = self.z[i]
            theta_old = self.theta[i]
            position = self.X[i]
            indi = np.where(self.groups[zold] == i)[0][0]
            ## Update parameters of the distribution
            self.a[zold] -= .5
            self.nk[zold] -= 1.0
            ## Loop over dimensions
            for j in range(self.d):
                if j == 0 and self.first_linear[zold]:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] - theta_old) ** 2 / 2
                else:
                    ## Update Csi
                    Csi_I_Old[j] = np.copy(self.Csi_I[zold,j])
                    self.Csi_I[zold,j] = delete_inverse(Csi_I_Old[j], ind=indi)
                    X_group_indi = np.delete(self.X_groups[zold][:,j], obj=indi)
                    ## Update X_Csi_X
                    X_Csi_X_Old[j] = 2 * (self.b[j][zold] - self.b0)
                    self.X_Csi_X[zold,j] = np.matmul(np.matmul(X_group_indi.reshape(1,-1),self.Csi_I[zold,j]),X_group_indi.reshape(-1,1))
                    ## Update b
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= X_Csi_X_Old[j] / 2 
                    self.b[j][zold] += self.X_Csi_X[zold,j] / 2 
            ## Calculate proposal
            theta_prop = np.random.normal(loc=theta_old, scale=sigma_prop)
            position_prop = np.copy(position)
            ## Update to new values
            for j in range(self.d):
                b_prop[j] = float(np.copy(self.b[j][zold]))
                if j == 0 and self.first_linear[zold]:
                    b_prop[j] += (position_prop[j] - theta_prop) ** 2 / 2
                else:
                    tz = np.copy(self.theta_groups[zold])
                    np.put(tz, ind=indi, v=theta_prop)
                    if fast_update:
                        add_row = self.csi[zold,j](theta_prop,tz)
                        add_row[indi] += 1
                        Csi_I_Prop[j] = add_inverse(self.Csi_I[zold,j], row=add_row, col=add_row, ind=indi)
                    else:
                        Csi_I_Prop[j] = np.linalg.inv(self.csi[zold,j](tz,tz) + np.diag(np.ones(int(self.nk[zold]+1))))
                    X_Csi_X_Prop[j] = float(np.matmul(np.matmul(self.X_groups[zold][:,j].reshape(1,-1), Csi_I_Prop[j]),self.X_groups[zold][:,j].reshape(-1,1)))
                    b_prop[j] += X_Csi_X_Prop[j] / 2 
            ## Calculate acceptance ratio
            numerator_accept = norm.logpdf(theta_prop,loc=self.mu_theta,scale=self.sigma_theta)
            for j in range(self.d):
                if j == 0 and self.first_linear[zold]:
                    numerator_accept += t.logpdf(position_prop[j], df=2*self.a[zold], loc=theta_prop, scale=np.sqrt(self.b[j][zold] / self.a[zold])) 
                else:
                    csi_left = self.csi[zold,j](theta_prop, np.delete(self.theta_groups[zold], obj=indi))
                    csi_prod = np.matmul(csi_left.reshape(1,-1), self.Csi_I[zold,j])
                    mu_star = np.matmul(csi_prod, np.delete(self.X_groups[zold][:,j], obj=indi, axis=0))
                    csi_star = self.csi[zold,j](theta_prop,theta_prop) - np.matmul(csi_prod, csi_left.reshape(-1,1))
                    numerator_accept += t.logpdf(position_prop[j], df=2*self.a[zold], loc=mu_star, 
                            scale=np.sqrt(self.b[j][zold] / self.a[zold] * (1 + csi_star)))
            denominator_accept = norm.logpdf(theta_old,loc=self.mu_theta,scale=self.sigma_theta)
            for j in range(self.d):
                if j == 0 and self.first_linear[zold]:
                    denominator_accept += t.logpdf(position[j], df=2*self.a[zold], loc=theta_old, scale=np.sqrt(self.b[j][zold] / self.a[zold])) 
                else:
                    csi_left = self.csi[zold,j](theta_old, np.delete(self.theta_groups[zold], obj=indi))
                    csi_prod = np.matmul(csi_left.reshape(1,-1), self.Csi_I[zold,j])
                    mu_star = np.matmul(csi_prod, np.delete(self.X_groups[zold][:,j], obj=indi, axis=0))
                    csi_star = self.csi[zold,j](theta_old,theta_old) - np.matmul(csi_prod, csi_left.reshape(-1,1))
                    denominator_accept += t.logpdf(position[j], df=2*self.a[zold], loc=mu_star, 
                            scale=np.sqrt(self.b[j][zold] / self.a[zold] * (1 + csi_star)))           
            ## Calculate acceptance probability
            accept_ratio = float(numerator_accept - denominator_accept)
            accept = (-np.random.exponential(1) < accept_ratio)
            accepts = True
            ## Update parameters
            self.a[zold] += .5
            self.nk[zold] += 1.0
            if accept:
                self.theta[i] = theta_prop
                np.put(self.theta_groups[zold], ind=indi, v=theta_prop)
            if not accept:
                ## Re-update to old values
                for j in range(self.d):
                    self.b[j][zold] = b_old[j]
                    if not (j == 0 and self.first_linear[zold]):
                        self.Csi_I[zold,j] = Csi_I_Old[j]
                        self.X_Csi_X[zold,j] = X_Csi_X_Old[j]
            else:
                ## Update to new values
                for j in range(self.d):
                    self.b[j][zold] = b_prop[j]
                    if not (j == 0 and self.first_linear[zold]):
                        self.Csi_I[zold,j] = Csi_I_Prop[j]
                        self.X_Csi_X[zold,j] = X_Csi_X_Prop[j]
        return None

    ###################################################################################
    ### Calculate maximum a posteriori estimate of the parameters given z and theta ###
    ###################################################################################
    def map(self,z,theta,range_values):
        mm = lsbm_gp_gibbs(X=self.X, K=self.K, csi=self.csi)
        mm.initialise(z=z, theta=theta, a_0=self.a0, b_0=self.b0, nu=self.nu, first_linear=self.first_linear)
        mean = {}; confint = {}; Csi_X = {}
        for j in range(mm.d):
            for k in range(mm.K):
                mean[k,j] = np.zeros(len(range_values))
                confint[k,j] = np.zeros((len(range_values),2))
                if not (j == 0 and mm.first_linear[k]):
                    Csi_X[k,j] = np.matmul(mm.Csi_I[k,j], mm.X_groups[k][:,j])
        for i in range(len(range_values)):
            x = range_values[i]
            for j in range(mm.d):
                for k in range(mm.K):
                    if j == 0 and mm.first_linear[k]:
                        mean[k,j][i] = x
                        confint[k,j][i] = t.interval(0.95, df=2*mm.a[k], loc=x, scale=np.sqrt(mm.b[j][k] / mm.a[k])) 
                    else:
                        csi_left = mm.csi[k,j](x, mm.theta_groups[k])
                        mean[k,j][i] = np.matmul(csi_left, Csi_X[k,j])
                        var = mm.csi[k,j](x,x) - np.matmul(np.matmul(csi_left, mm.Csi_I[k,j]), np.transpose(csi_left))
                        confint[k,j][i] = t.interval(0.95, df=2*mm.a[k], loc=mean[k,j][i], scale=np.sqrt(mm.b[j][k] / mm.a[k] * (1 + var)))
        return mean, confint

    #####################
    ### MCMC sampling ###
    #####################
    def mcmc(self, samples=1000, burn=100, chains=1, store_chains=True, ell=100, thinning=1, sigma_prop=0.1, fast_update=True):
        ## q is the number of re-allocated nodes per iteration
        self.ell = ell
        ## Option to store the chains
        if store_chains:
            theta_chain = np.zeros((self.n,chains,samples // thinning))
            z_chain = np.zeros((self.n,chains,samples // thinning))
        for chain in range(chains):
            for s in range(samples+burn):
                print('\rChain:', chain+1,'/', chains, '\tBurnin:', s+1 if s<burn else burn, '/', burn, 
                            '\tSamples:', s-burn+1 if s>=burn else 0,'/', samples, end='')
                move = ['communities','parameters']
                m = np.random.choice(move)
                if m == 'communities':
                    self.gibbs_communities(l=self.ell, fast_update=fast_update)
                else:
                    self.resample_theta(l=self.ell, sigma_prop=sigma_prop, fast_update=fast_update)
                if s >= burn and store_chains and s % thinning == 0:
                    theta_chain[:,chain,(s - burn) // thinning] = self.theta
                    z_chain[:,chain,(s - burn) // thinning] = self.z
        print('')
        if store_chains:
            return theta_chain, z_chain
        else:
            return None