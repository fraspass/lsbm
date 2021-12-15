#! /usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

## Run in DCSBM code folder

## Load data
X = np.load('X_icl1.npy')
lab = np.loadtxt('labs1.csv', delimiter=',', dtype=int)

from sklearn.linear_model import LinearRegression
rr = np.linspace(0,2.1,num=300)
rr_mat = np.array([rr,rr**2]).T
preds = {}
for g in [1,0]:
    ix = np.where(lab == g)
    reg = LinearRegression().fit(np.array([X[ix,0],X[ix,0]**2])[:,0].T, X[ix,1].reshape(-1,1))
    preds[g] = reg.predict(rr_mat)[:,0]

fig, ax = plt.subplots()
cdict = ['#004D40','#1E88E5']
mms = ['s','o']
group = ['Civil Engineering','Mathematics']
gls = [0,1]
xx = np.linspace(np.min(X[:,0]),np.max(X[:,0]))
for i in range(2):
    ix = np.where(lab == gls[i])[0]
    ax.scatter(X[:,0][ix], X[:,1][ix], c = cdict[i], label = group[i], marker=mms[i], edgecolor='black', linewidth=0.3)
    ax.plot(rr, preds[gls[i]], c = cdict[i])

ax.legend()
plt.xlabel('$$\\hat{\\mathbf{X}}_1$$')
plt.ylabel('$$\\hat{\\mathbf{X}}_2$$')
plt.savefig("x12_maths_cive.pdf",bbox_inches='tight')
plt.show()