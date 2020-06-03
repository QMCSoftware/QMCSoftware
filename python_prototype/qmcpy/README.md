# QMCPy

## Integrand

The function to integrate\
*Abstract class with concrete implementations*

- Linear: $g(\boldsymbol{x}) = \sum_{j=1}^{d}x_{j}$
- Keister: $g(\boldsymbol{x}) = \pi^{d/2} \, \cos(||\boldsymbol{x}||_2)$
- Asian Call
    - stock price at time $jT/d$: $~~~~~~~~~~$ $S(x_j)=S_0\exp\bigl((r-\sigma^2/2)(jT/d)+\sigma x_j\bigr)$
    - discounted call payoff $\displaystyle = \max\left(\frac{1}{d}\sum_{j=1}^{d} S(x_j)-K\;,\: 0\right) \, \exp(-rT)$
    - discounted put payoff $\displaystyle = \max\left(K-\frac{1}{d}\sum_{j=1}^{d} S(x_j)\;,\: 0\right)\, \exp(-rT)$
- QuickConstruct

<hr>

## True Measure

General measure used to define the integral\
*Abstract class with concrete implementations*

- Uniform: $\mathcal{U}(\boldsymbol{a},\boldsymbol{b})$
- Gaussian: $\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})$
- Discrete Brownian Motion: $\mathcal{N}(\boldsymbol{0},\mathsf{\Sigma})$, where $\mathsf{\Sigma} = \min(\boldsymbol{t},\boldsymbol{t}^T)$, $\boldsymbol{t} = (t_1, \ldots, t_d)^T$
- Lebesgue
- Identity Transform

<hr>

## Discrete Distribution

Sampling nodes IID or LDS (low-discrepancy sequence)\
*Abstract class with concrete implementations*

- IID Standard Uniform: $x_j \overset{iid}{\sim}   \mathcal{U}(0,1)$
- IID Standard Gaussian: $x_j \overset{iid}{\sim}   \mathcal{N}(0,1)$
- Custom IID Distribution
- Lattice (base 2): $x_j  \overset{lds}{\sim}    \mathcal{U}(0,1)$
- Sobol (base 2): $x_j \overset{lds}{\sim}    \mathcal{U}(0,1)$
- Inverse CDF Sampling
- Acceptance Rejection Sampling

<hr>

## Stopping Criterion

The stopping criterion to determine sufficient approximation\
Has class method `integrate` which preforms numerical integration\
*Abstract class with concrete implementations*

**For IID Nodes** $x_i\sim$ iid

  - Central Limit Theorem (CLT) 
  - MeanMC_g (guaranteed)

**For QMC Sequences** $\{x_{r,i}\}_{r=1}^R \sim$ lds

- CLT Repeated
- CubLattice_g (gauranteed)
- CubSobol_g (gauranteed)

<hr>   

## Accumulate Data Class

Stores data values of corresponding  stopping criterion procedure\
*Abstract class with concrete implementations*

- Mean Variance Data
- Mean Variance Repeated Data
- Cubature Data 
