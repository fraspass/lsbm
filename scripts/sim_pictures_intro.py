#! /usr/bin/env python3
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from scipy.linalg import orthogonal_procrustes as proc

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

np.random.seed(1771)
n = 1000
### Means
theta = np.random.uniform(size=n)
X = np.array([theta ** 2, (1-theta) ** 2, 2 * theta * (1-theta)]).T

A = np.zeros((n,n))
for i in range(n-1):
    for j in range(i,n):
        A[i,j] = np.random.binomial(n=1,p=np.inner(X[i],X[j]),size=1)
        A[j,i] = A[i,j]

Lambda, Gamma = np.linalg.eigh(A)
k = np.argsort(np.abs(Lambda))[::-1][:3]
X_hat = np.dot(Gamma[:,k],np.diag(np.sqrt(np.abs(Lambda[k]))))
X_tilde = np.dot(X_hat,proc(X_hat,X)[0])

### Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_tilde[:,0], X_tilde[:,1], X_tilde[:,2], c='gray', s=1, alpha=0.75)
ax.scatter(X[:,0],X[:,1],X[:,2],s=1,c='black')
ax.view_init(elev=25, azim=45)
ax.set_xlabel('$$\\hat{\\mathbf{X}}_1$$')
ax.set_ylabel('$$\\hat{\\mathbf{X}}_2$$')
ax.set_zlabel('$$\\hat{\\mathbf{X}}_3$$')
plt.savefig("../pictures/hw_sim_1000.pdf",bbox_inches='tight')
plt.show()
