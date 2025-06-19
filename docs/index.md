# Quasi-Monte Carlo Community Software in Python

[![](https://img.shields.io/badge/Docs-6b03fc)](https://QMCSoftware.github.io/QMCSoftware/)
[![](https://img.shields.io/badge/GitHub-15bfa9)](https://github.com/QMCSoftware/QMCSoftware)
[![PyPI Downloads](https://img.shields.io/pypi/dm/qmcpy.svg?label=PyPI%20downloads)](https://pypi.org/project/qmcpy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3964489.svg)](https://doi.org/10.5281/zenodo.3964489)
[![Tests](https://github.com/QMCSoftware/QMCSoftware/workflows/QMCPy_CI/badge.svg)](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/python-package-conda.yml)
[![](https://img.shields.io/badge/QMC_Blogs-fc7005)](http://qmcpy.wordpress.com/)

[![](https://img.shields.io/badge/Fred_Hickernell's_MCQMC2020_Video_Tutorial-FF0000)](https://www.youtube.com/watch?v=gL8M_7c-YUE)
[![](https://img.shields.io/badge/Aleksei_Sorokin's_PyData_Chicago_Video_Tutorial-FF0000)](https://www.youtube.com/watch?v=bRcKiLA2yBQ)


```
pip install qmcpy
```

Monte Carlo (MC) methods approximate the true mean (expectation) $\mu$ of a random variable $g(\boldsymbol{T})$ by the sample mean $\hat{\mu}_n = \frac{1}{n} \sum_{i=0}^{n-1} g(\boldsymbol{T}_i)$ for some samples $\boldsymbol{T}_0,\dots,\boldsymbol{T}_{n-1}$. We call the $d$-dimensional vector random variable $\boldsymbol{T}$ the **true measure** and we call $g$ the **integrand**. As most computer-generated random numbers are uniformly distributed, we use a transform $\boldsymbol{\psi}$ to write $\boldsymbol{T} \sim \boldsymbol{\psi}(\boldsymbol{X})$ where $\boldsymbol{X} \sim \mathcal{U}[0,1]^d$. The resulting Monte Carlo approximation is written in terms of the transformed integrand $f(\boldsymbol{x}) = g(\boldsymbol{\psi}(\boldsymbol{x}))$ as

$$\mu = \mathbb{E}[f(\boldsymbol{X})] = \int_{[0,1]^d} f(\boldsymbol{x}) \mathrm{d} \boldsymbol{x} \approx \int_{[0,1]^d} f(\boldsymbol{x}) \hat{\lambda}_n(\mathrm{d} \boldsymbol{x}) = \frac{1}{n} \sum_{i=0}^{n-1} f(\boldsymbol{x}_i) = \hat{\mu}, \qquad \boldsymbol{X} \sim \mathcal{U}[0,1]^d$$

for some **discrete distribution** $\hat{\lambda}_n$ defined by samples $\boldsymbol{x}_0,\dots,\boldsymbol{x}_{n-1} \in [0,1]^d$ (formally $\hat{\lambda}_n(A)$ measures the proporation of points $(\boldsymbol{x}_i)_{i=0}^{n-1}$ which lie in some set $A$). The *error* of this approximation is

$$E_n = \lvert \mu - \hat{\mu}_n \rvert.$$

Classic **Monte Carlo** methods choose IID (independent and identically distributed) samples $\boldsymbol{x}_0,\dots,\boldsymbol{x}_{n-1} \overset{\mathrm{IID}}{\sim} \mathcal{U}[0,1]^d$ and have error $E_n$ like $\mathcal{O}(n^{-1/2})$. **Quasi-Monte Carlo (QMC)** methods achieve a significantly better error rate of $\mathcal{O}(n^{-1})$ by using low discrepancy (LD) sequences for $(\boldsymbol{x}_i)_{i=0}^{n-1}$ which more evenly fill the unit cube than IID points.

| <img src="./assets/points.svg" alt="Alt Text" style="width:100%; height:auto;"> | 
|:--|
| The first $32$ points of each sequence are shown as purple starts, the next $32$ points are shown as green triangles, and the $64$ points after that are shown as blue circles. Notice the gaps and clusters of IID points compared to the more uniform coverage of LD sequences. |

Often practitioners would like to run their (Quasi-)Monte Carlo method until the error $E_n$ is below a desired error tolerance $\varepsilon$ and/or until they have expired their sample budget $B$. For example, one may wish to estimate the expected discounted payoff of a financial option to within a tolerance of one penny, $\varepsilon = 0.01$, or until $1$ million option paths have been simulated, $B=10^6$. **Stopping criterion** deploy (Quasi-)Monte Carlo methods under such constraints by utilizing adaptive sampling schemes and efficient error estimation procedures.  

`QMCPy` is organized into 4 main components:

## Discrete Distributions

These generates IID or LD points $\boldsymbol{x}_0,\boldsymbol{x}_1,\dots$. Supported LD sequences include

- **Lattices** with
    - extensible constructions
    - random shifts
- **Digital Nets** in base $b=2$ with
    - extensible constructions
    - digital shifts
    - linear matrix scrambling
    - nested uniform scrambling (also called Owen scrambling)
    - higher order constructions via digital interlacing
- **Halton** point sets with
    - extensible constructions
    - digital shifts
    - permutation scrambling
    - linear matrix scrambling
    - nested uniform scrambling

We can use `QMCPy` to generate the LD digital net (in base $b=2$) as follows. 

```python 
>>> import qmcpy as qp 
>>> generator = qp.DigitalNetB2(dimension=2,seed=7)
>>> generator(8) # first 8 points in the sequence
array([[0.0715562 , 0.07784108],
       [0.81420169, 0.74485558],
       [0.31409299, 0.93233913],
       [0.57163057, 0.26535753],
       [0.15541309, 0.57499661],
       [0.89830224, 0.2439158 ],
       [0.39820498, 0.43143225],
       [0.6554989 , 0.76248017]])
>>> generator(8,16) # next 8 points in the sequence 
array([[0.03088897, 0.83362275],
       [0.77280156, 0.46942063],
       [0.272731  , 0.15687463],
       [0.53100149, 0.52116877],
       [0.1970359 , 0.33483624],
       [0.93870483, 0.97247967],
       [0.43861519, 0.66002569],
       [0.69712934, 0.02229023]])
```

The same API is available for the other LD sequences: `qp.Lattice`, `qp.DigitalNetB2`, and `qp.Halton`. A similar API for IID points is available in `qp.IIDStdUniform` (essentially a wrapper around [`numpy.random.rand`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html))

## True Measures

These define $\boldsymbol{T}$, for which `QMCPy` will automatically choose an appropriate transform $\boldsymbol{\psi}$ so that $\boldsymbol{T} \sim \boldsymbol{\psi}(\boldsymbol{X})$ with $\boldsymbol{X} \sim \mathcal{U}[0,1]^d$. Some popular true measures are

- **Uniform** $\boldsymbol{T} \sim \mathcal{U}[\boldsymbol{l},\boldsymbol{u}]$ with elementwise $\boldsymbol{l} \leq \boldsymbol{u}$ for which $\boldsymbol{\psi}(\boldsymbol{x}) = \boldsymbol{l}+(\boldsymbol{u}-\boldsymbol{l}) \odot \boldsymbol{x}$ with $\odot$ the Hadamard (elementwise) product.
- **Gaussian** $\boldsymbol{T} \sim \mathcal{N}(\boldsymbol{m},\mathsf{\Sigma})$ for which $\boldsymbol{\psi}(\boldsymbol{x}) = \boldsymbol{m}+\mathsf{A}\boldsymbol{x}$ where the covariance $\mathsf{\Sigma} = \mathsf{A} \mathsf{A}^T$ may be decomposed using either
    - the Cholesky decomposition or
    - the eigendecomposition.
- **Brownian Motion** observed with an initial value $B_0$, drift $\gamma$, and diffusion $\sigma^2$ at times $\boldsymbol{t} := (t_1,\dots,t_d)^T$ satisfying $0 \leq t_1 < t_1 < \dots < t_d$ is a Gaussian with mean and covariance

$$\boldsymbol{m} = B_0 + \gamma \boldsymbol{t}$$

$$\mathsf{\Sigma} = \sigma^2 \left(\min\{t_i,t_{i'}\}\right)_{i,i'=1}^{d}$$

- **Independent Marginals** have $\boldsymbol{T} = (T_1,\dots,T_d)^T$ with $T_1,\dots,T_d$ independent. We support (continuous) marginal distributions from [`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions).

## Integrands

These define $g$, which `QMCPy` will use to define $f = g \circ \boldsymbol{\psi}$. Some popular integrands are

- **User Defined Integrands**, where the user provides a function handle for $g$
- **Financial Options**, including the *European option*, *Asian option*, and *Barrier option*
- **[`UM-Bridge`](https://um-bridge-benchmarks.readthedocs.io/en/docs/) Functions**. From their docs, `UM-Bridge` is a universal interface that makes any numerical model accessible from any programming language or higher-level software through the use of containerized environments. `UM-Bridge` also enables simulations to scale to supercomputers or the cloud with minimal effort.

## Stopping Criteria

| <img src="./assets/stopping_crit.svg" alt="Alt Text" style="width:100%; height:auto;"> | 
|:--|
| The cost of IID-Monte Carlo algorithms is $\mathcal{O}(n^2)$ in the number of samples $n$ while Quasi-Monte Carlo algorithms only cost around $\mathcal{O}(n)$. Both IID-Monte Carlo and Quasi-Monte Carlo stopping criterion consistently determine approximations which meet the desired error tolerance. |

These deploy (Quasi-)Monte Carlo methods under error tolerance and budgetary constraints by utilizing adaptive sampling schemes and efficient error estimation procedures. Common stopping criteria include

- **IID Monte Carlo** via a two step procedure using the Central Limit Theorem (CLT). Error estimates are *not guaranteed* as CLT is asymptotic in $n$ is the variance must be estimated.
- **IID Monte Carlo** via a two step procedure using Berry-Esseen inequalities to account for finite sample sizes. Error estimates are *guaranteed* for functions with bounded Kurtosis.
- **Quasi-Monte Carlo** via multiple independent randomizations of an LD point set and Student's $t$ confidence intervals.
- **Quasi-Monte Carlo** via tracking the decay of coefficients in an orthogonal basis expansion. These methods are *guaranteed* for cones of functions whose coefficients decay in a regular manner.  Efficient procedures exist to estimate coefficients when
    - pairing lattices with the Fourier expansion or
    - pairing digital nets with the Walsh expansion.
- **Quasi-Monte Carlo** via efficient Bayesian cubature methods which assume $f$ is a draw from a Gaussian process so the posterior expectation has an analytic expression. While classic Bayesian cubature would require $\mathcal{O}(n^2)$ storage and $\mathcal{O}(n^3)$ computations, when matching certain LD sequences to special kernels the Gram matrices become nicely structured to permit Bayesian cubature with only $\mathcal{O}(n)$ storage and $\mathcal{O}(n \log n)$ computations. Specifically,
    - pairing lattices with shift-invariant kernels gives circulant Gram matrices which are diagonalizable by the [Fast Fourier Transform (FFT)](https://en.wikipedia.org/wiki/Fast_Fourier_transform), and
    - pairing digital nets with digitally-shift-invariant kernels gives Gram matrices which are diagonalizable by the [Fast Walsh-Hadamard Transform (FWHT)](https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform).
- **Multilevel IID Monte Carlo and Quasi-Monte Carlo** which more efficiently integrate expensive functions by exploiting a telescoping sum over lower fidelity models. 
