# Latent structure blockmodels for Bayesian spectral graph clustering

This reposit contains *python* code used to reproduce results and simulations in *"Latent structure blockmodels for Bayesian spectral graph clustering"*.

## Understanding the code

The main tool for inference on LSBMs is the MCMC sampler `lsbm_gp_gibbs` contained in the file `lsbm_gp.py`. The class can be initialised using three objects: 
- a <img src="svgs/2103f85b8b1477f430fc407cad462224.svg?invert_in_darkmode" align=middle width=8.55596444999999pt height=22.831056599999986pt/>-dimensional embedding `X`, a <img src="svgs/fa3e74d315d8dd8569c63afaf353839c.svg?invert_in_darkmode" align=middle width=38.514031049999986pt height=22.831056599999986pt/>`numpy` array, where <img src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the number of nodes;
- the number of communities `K`, a positive integer;
- the set of kernels `csi`, a dictionary containing <img src="svgs/8c2aba4645470ea758c7014c414f8703.svg?invert_in_darkmode" align=middle width=43.784154149999985pt height=22.831056599999986pt/> kernel functions <img src="svgs/61a21a71d91d1afe29ef6c87bbcfb541.svg?invert_in_darkmode" align=middle width=24.466483799999992pt height=22.831056599999986pt/> for the Gaussian processes.

Additionally, the file `lsbm.py` contains a simplified class `lsbm_gibbs` which can be used for models where the kernel functions are assumed to be *inner products*. The latent functions in such models can be expressed in the form <img src="svgs/a47365b803cbfbae589ba00d757323c3.svg?invert_in_darkmode" align=middle width=181.69460429999998pt height=24.65753399999998pt/>, for basis functions <img src="svgs/166782492ccc9f0344cf301c405ca9fd.svg?invert_in_darkmode" align=middle width=96.20125349999998pt height=22.831056599999986pt/> and corresponding weights <img src="svgs/5628dabe825c1081c1d0ab40cb139570.svg?invert_in_darkmode" align=middle width=85.20361904999999pt height=22.648391699999998pt/> with joint normal-inverse-gamma prior with the variance parameter: 
<p align="center"><img src="svgs/b2769a868c59cee2f9ae9803842a9a1a.svg?invert_in_darkmode" align=middle width=230.74284915pt height=20.50407645pt/></p>

The class `lsbm` does *not* require a dictionary of kernel functions for initialisation, but a dictionary `W_function` containing the basis functions. The prior parameters of the NIG prior are then specified using the function `initialisation` within the same class. The class uses the Zellner's <img src="svgs/3cf4fbd05970446973fc3d9fa3fe3c41.svg?invert_in_darkmode" align=middle width=8.430376349999989pt height=14.15524440000002pt/>-prior as default. 

### Example: quadratic model

For example, consider the model where all the community-specific curves are assumed to be quadratic in <img src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg?invert_in_darkmode" align=middle width=8.17352744999999pt height=22.831056599999986pt/>, passing through the origin, and linear in the first dimension (with zero intercept and unit slope). Initialising the Gibbs sampler class in `lsbm` is easy, since the basis functions are straightforward to define:
```python3
phi = {}
for k in range(K):
    phi[k,0] = lambda theta: np.array([theta])
    for j in range(1,d):
        phi[k,j] = lambda theta: np.array([theta ** 2, theta])

m = lsbm.lsbm_gibbs(X=X, K=K, W_function=phi)
```

Defining the kernels requires a bit more effort. Using the dictionary `phi` defined in the previous code snippet, the corresponding kernels - assuming prior scale matrices `Delta` for the NIG prior - are: 
```python3
csi = {}
for k in range(K):
    csi[k,0] = lambda theta,theta_prime: Delta[k,0] * np.matmul(phi[k,0](theta).reshape(-1,1),phi[k,0](theta_prime).reshape(1,-1)) 
    for j in range(1,d):
        csi[k,j] = lambda theta,theta_prime: np.matmul(np.matmul(phi[k,j](theta),Delta[k,j]),np.transpose(phi[k,j](theta_prime)))

m = lsbm_gp.lsbm_gp_gibbs(X=X, K=K, csi=csi)
```

Note that the kernel functions in `csi` must be flexible enough to handle correctly four types of input - `(float,float)`, `(float,vector)`, `(vector,float)`, `(vector,vector)` -, returning appropriate objects - `float`, `vector` or `matrix` - in each of the four circumstances. 