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
def delete_inverse(M, ind):
  A = np.delete(arr=np.delete(arr=M,obj=ind,axis=0),obj=ind,axis=1)
  B = np.delete(arr=M,obj=ind,axis=0)[:,ind]
  C = np.delete(arr=M,obj=ind,axis=1)[ind]
  D = M[ind,ind]
  return A - np.outer(B,C) / D

# 2) Update inverse matrix M when a row/column is added on position i
def add_inverse(M, row, col, ind):
  A11_inv = M
  A12 = np.delete(arr=col,obj=ind).reshape(-1,1)
  A21 = np.delete(arr=row,obj=ind).reshape(1,-1)
  A22 = row[ind]
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
    def __init__(self, X, K, csi, fixed_function={}):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.K = K
        ## Kernel functions
        self.csi = csi
        self.fixed_function = fixed_function

    ## Initialise the model parameters
    def initialise(self, z, theta, a_0=1, b_0=1, nu=1, mu_theta=0, sigma_theta=1, first_linear=True):
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
        self.fixed_W = {}
        for j in self.fixed_function:
            self.fixed_W[j] = np.array([self.fixed_function[j](self.theta[i]) for i in range(self.n)])[:,0] ## rewrite using only one coefficient (1)
        ## Prior parameters
        self.nu = nu
        self.a0 = a_0
        self.b0 = b_0
        ## Initialise hyperparameter vectors
        self.nk = Counter(self.z)
        self.a = {}; self.groups = {}; self.X_groups = {}; self.theta_groups = {}
        for k in range(self.K):
            self.a[k] = self.a0 + self.nk[k] / 2
            self.groups[k] = np.arange(self.n)[self.z == k]
            self.X_groups[k] = self.X[self.z == k]
            self.theta_groups[k] = self.theta[self.z == k]
        self.b = {}; self.Csi_I = {}; self.X_Csi_X = {}
        for j in range(self.d):
            self.b[j] = {}; self.Csi_I[j] = {}
            for k in range(self.K):
                X = self.X[self.z == k][:,j]
                if j in self.fixed_function:
                    X -= self.fixed_W[j][self.z == k]
                if j == 0 and self.first_linear:
                    self.b[j][k] = self.b0 + np.sum((X - self.theta[self.z == k]) ** 2) / 2
                else:
                    self.Csi_I[k][j] = np.linalg.inv(self.csi[k,j](self.theta_groups[k],self.theta_groups[k]) + np.diag(np.ones(self.nk[k])))
                    self.X_Csi_X[k][j] = np.matmul(np.matmul(np.transpose(self.X_groups[k]), self.Csi_I[k][j]),self.X_groups[k])
                    self.b[j][k] = self.b0 + self.X_Csi_X[k][j] / 2
                    
    ########################################################
    ### a. Resample the allocations using Gibbs sampling ###
    ########################################################
    def gibbs_communities(self,l=1):	
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
            ind_del = np.where(self.groups[zold] == i)[0]
            self.groups[zold] = np.delete(self.groups[zold], obj=ind_del)
            self.X_groups[zold] = np.delete(self.X_groups[zold], obj=ind_del, axis=0)
            self.theta_groups[zold] = np.delete(self.theta_groups[zold], obj=ind_del)
            ## Loop over dimensions
            for j in range(self.d):
                if j in self.fixed_function:
                    position[j] -= self.fixed_W[j][i]
                if j == 0 and self.first_linear:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] - thetai) ** 2 / 2
                else:
                    ## Update Csi
                    Csi_I_Old[j] = np.copy(self.Csi_I[zold][j])
                    self.Csi_I[zold][j] = delete_inverse(Csi_I_Old, ind=ind_del)
                    ## Update X_Csi_X
                    X_Csi_X_Old[j] = 2 * self.b[j][zold] - self.b0
                    self.X_Csi_X[zold][j] = np.matmul(np.matmul(np.transpose(self.X_groups[zold][:,j]),self.Csi_I[zold][j]),self.X_groups[zold][:,j])
                    ## Update b
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= X_Csi_X_Old[j] / 2 
                    self.b[j][zold] += self.X_Csi_X[zold][j] / 2 
            ## Calculate probabilities for community allocations
            community_probs = np.log((np.array([self.nk[k] for k in range(self.K)]) + self.nu/self.K) / (self.n - 1 + self.K))
            for k in range(self.K):
                csi_left = self.csi(thetai, self.theta_groups[k])
                for j in range(self.d):
                    if j == 0 and self.first_linear:
                        community_probs[k] += t.logpdf(position[j], df=2*self.a[k], loc=thetai, scale=np.sqrt(self.b[j][k] / self.a[k])) 
                    else:   
                        csi_prod = np.matmul(csi_left, self.Csi_I[k][j])
                        mu_star = np.matmul(csi_prod, self.X_groups[k])
                        csi_star = self.csi[k,j](thetai,thetai) - np.matmul(csi_prod, np.transpose(csi_left))
                        community_probs[k] += t.logpdf(position[j], df=2*self.a[k], loc=mu_star, scale=np.sqrt(self.b[j][k] / self.a[k] * (1 + csi_star)))              
            ## Raise error if nan probabilities are computed
            if np.isnan(community_probs).any():
                print(community_probs)
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
                    if not (j == 0 and self.first_linear):
                        self.Csi_I[znew][j] = Csi_I_Old[j]
                        self.X_Csi_X[znew][j] = X_Csi_X_Old[j]
            else:
                ## Update to new values
                for j in range(self.d):
                    if j == 0 and self.first_linear:
                        self.b[j][znew] += (position[j] - self.theta[i]) ** 2 / 2
                    else:
                        add_row = self.csi[znew,j](self.theta[i],self.theta_groups[znew]).reshape(1,-1)
                        ## add_row = np.insert(add_row, obj=ind_add, values=self.csi[znew,j](self.theta[i],self.theta[i]))
                        add_row[ind_add] += 1
                        self.Csi_I[znew][j] = add_inverse(self.Csi_I[znew][j], row=add_row, col=np.transpose(add_row), ind=ind_add)
                        self.X_Csi_X[znew][j] = np.matmul(np.matmul(np.transpose(self.X_groups[znew][:,j]), self.Csi_I[znew]),self.X_groups[znew][:,j])
                        self.b[znew][j] += self.X_Csi_X[znew][j] / 2 
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
        b_old = {}; Csi_I_Old = {}; X_Csi_X_Old = {}
        b_prop = {}; Csi_I_Prop = {}; X_Csi_X_Prop = {}
        for i in np.random.choice(self.n, size=l, replace=False):
            zold = self.z[i]
            theta_old = self.theta[i]
            position = self.X[i]
            indi = np.where(self.groups[zold] == i)[0]
            ## Update parameters of the distribution
            self.a[zold] -= .5
            self.nk[zold] -= 1.0
            ## Loop over dimensions
            for j in range(self.d):
                if j in self.fixed_function:
                    position[j] -= self.fixed_W[j][i]
                if j == 0 and self.first_linear:
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= (position[j] - theta_old) ** 2 / 2
                else:
                    ## Update Csi
                    Csi_I_Old[j] = np.copy(self.Csi_I[zold][j])
                    self.Csi_I[zold][j] = delete_inverse(Csi_I_Old, ind=indi)
                    ## Update X_Csi_X
                    X_Csi_X_Old[j] = 2 * self.b[j][zold] - self.b0
                    self.X_Csi_X[zold][j] = np.matmul(np.matmul(np.transpose(self.X_groups[zold][:,j]),self.Csi_I[zold][j]),self.X_groups[zold][:,j])
                    ## Update b
                    b_old[j] = float(np.copy(self.b[j][zold]))
                    self.b[j][zold] -= X_Csi_X_Old[j] / 2 
                    self.b[j][zold] += self.X_Csi_X[zold][j] / 2 
            ## Calculate proposal
            theta_prop = np.random.normal(loc=theta_old, scale=sigma_prop)
            position_prop = np.copy(np.position)
            for j in range(self.d):
                if j in self.fixed_function:
                    position_prop[j] -= self.fixed_function[j](theta_prop)
                    b_prop[j] = self.b0
                ## Update to new values
                for j in range(self.d):
                    if j == 0 and self.first_linear:
                        b_prop[j] += (position_prop[j] - theta_prop[i]) ** 2 / 2
                    else:
                        add_row = self.csi[zold,j](theta_prop,np.add(np.delete(self.theta_groups[zold], obj=indi), obj=indi, values=theta_prop)).reshape(1,-1)
                        add_row[indi] += 1
                        Csi_I_Prop[j] = add_inverse(self.Csi_I[zold][j], row=add_row, col=np.transpose(add_row), ind=indi)
                        X_Csi_X_Prop[j] = np.matmul(np.matmul(np.transpose(self.X_groups[zold][:,j]), self.Csi_I[zold][j]),self.X_groups[zold][:,j])
                        b_prop[j] += X_Csi_X_Prop[zold][j] / 2 
            ## Calculate acceptance ratio
            numerator_accept = norm.logpdf(theta_prop,loc=self.mu_theta,scale=self.sigma_theta)
            for j in range(self.d):
                if j == 0 and self.first_linear:
                    numerator_accept += t.logpdf(position_prop[j], df=2*self.a[zold], loc=theta_prop, scale=np.sqrt(self.b[j][zold] / self.a[zold])) 
                else:
                    csi_left = self.csi[zold,j](theta_prop, np.delete(self.theta_groups[zold], obj=indi))
                    csi_prod = np.matmul(csi_left, self.Csi_I[zold][j])
                    mu_star = np.matmul(csi_prod, np.delete(self.X_groups[zold], obj=indi, axis=0))
                    csi_star = self.csi[zold,j](theta_prop,theta_prop) - np.matmul(csi_prod, np.transpose(csi_left))
                    numerator_accept += t.logpdf(position_prop[j], df=2*self.a[zold], loc=mu_star, 
                            scale=np.sqrt(b_prop[j] / self.a[zold] * (1 + csi_star)))
            denominator_accept = norm.logpdf(theta_old,loc=self.mu_theta,scale=self.sigma_theta)
            for j in range(self.d):
                if j == 0 and self.first_linear:
                    denominator_accept += t.logpdf(position[j], df=2*self.a[zold], loc=theta_old, scale=np.sqrt(self.b[j][zold] / self.a[zold])) 
                else:
                    csi_left = self.csi[zold,j](theta_old, np.delete(self.theta_groups[zold], obj=indi))
                    csi_prod = np.matmul(csi_left, self.Csi_I[zold][j])
                    mu_star = np.matmul(csi_prod, np.delete(self.X_groups[zold], obj=indi, axis=0))
                    csi_star = self.csi[zold,j](theta_old,theta_old) - np.matmul(csi_prod, np.transpose(csi_left))
                    denominator_accept += t.logpdf(position[j], df=2*self.a[zold], loc=mu_star, 
                            scale=np.sqrt(self.b[j][zold] / self.a[zold] * (1 + csi_star)))           
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
                    if not(j == 0 and self.first_linear):
                        self.Csi_I[zold][j] = Csi_I_Old[j]
                        self.X_Csi_X[zold][j] = X_Csi_X_Old[j]
            else:
                ## Update to new values
                for j in range(self.d):
                    self.b[j][zold] = b_prop[j]
                    if not(j == 0 and self.first_linear):
                        self.Csi_I[zold][j] = Csi_I_Prop[j]
                        self.X_Csi_X[zold][j] = X_Csi_X_Prop[j]
        return None

    ###################################################################################
    ### Calculate maximum a posteriori estimate of the parameters given z and theta ###
    ###################################################################################
    def map(self,z,theta,range_values):
        mm = lsbm_gp_gibbs(X=self.X, K=self.K, csi=self.csi, fixed_function=self.fixed_W)
        mm.initialise(z=z, theta=theta, a_0=self.a0, b_0=self.b0, nu=self.nu, first_linear=self.first_linear)
        mean = {}; confint = {}; Csi_X = {}
        for j in range(mm.d):
            for k in range(mm.K):
                mean[k,j] = np.zeros(len(range_values))
                confint[k,j] = np.zeros((len(range_values),2))
                Csi_X[k,j] = np.matmul(mm.Csi_I[k,j], mm.X_groups[k,j])
        for i in range(len(range_values)):
            x = range_values[i]
            for j in range(mm.d):
                for k in range(mm.K):
                    if j == 0 and mm.first_linear:
                        mean[k,j][i] = x
                        confint[k,j][i] = t.interval(0.95, df=2*mm.a[k], loc=x, scale=np.sqrt(mm.b[j][k] / mm.a[k])) 
                    else:
                        csi_left = mm.csi[k,j](x, self.theta_groups[k]).reshape(-1,1)
                        mean[k,j][i] = np.matmul(csi_left, Csi_X[k,j])
                        var = mm.csi[k,j](x,x) - np.matmul(np.matmul(csi_left, mm.Csi_I[k,j]), np.transpose(csi_left))
                        confint[k,j][i] = t.interval(0.95, df=2*mm.a[k], loc=mean[k,j][i], scale=np.sqrt(mm.b[j][k] / mm.a[k] * (1 + var)))
        return mean, confint

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
                print('\rChain:', chain+1,'/', chains, '\tBurnin:', s+1 if s<burn else burn, '/', burn, 
                            '\tSamples:', s-burn+1 if s>burn else 0,'/', samples, end='')
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