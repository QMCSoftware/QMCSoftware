# QMCPy

## Integrand

The function to integrate\
*Abstract class with concrete implementations*

- Linear: $g(\boldsymbol{x}) = \sum_{j=1}^{d}x_{j}$
- Keister: $g(\boldsymbol{x}) = \pi^{d/2} \, \cos(||\boldsymbol{x}||_2)$
- European Option
- Asian Call
    - stock price at time $jT/d$: $~~~~~~~~~$ $S(x_j)=S_0\exp\bigl((r-\sigma^2/2)(jT/d)+\sigma\mathcal{B}(t_j)\bigr)$
    - discounted call payoff $= \max\left(\frac{1}{d}\sum_{j=1}^{d} S(x_j)-K\right),\: 0)  \,\exp(-rT)$
    - discounted put payoff $= \max\left(K-\frac{1}{d}\sum_{j=1}^{d} S(x_j)\right),\: 0)\,\exp(-rT)$
- Multilevel Call Options
- QuickConstruct

<hr>

## True Measure

General measure used to define the integrand\
*Abstract class with concrete implementations*

- Uniform: $\mathcal{U}(\boldsymbol{a},\boldsymbol{b})$
- Gaussian: $\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})$
- Discrete Brownian Motion: $\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})$, where $\mathsf{\Sigma} = \min(\boldsymbol{t},\boldsymbol{t})^T$, $~~~~$ $\boldsymbol{t} = (t_1, \ldots, t_d)$
- Lebesgue
- Identical to what discrete distribution mimics
- Importance sampling

<hr>

## Discrete Distribution

Sampling nodes IID or LDS (low-discrepancy sequence)\
*Abstract class with concrete implementations*

**Independent Identically Distributed (IID) Nodes**

- IID Standard Uniform: $\overset{\text{IID}}{\sim}   \mathcal{U}(0,1)^d$
- IID Standard Gaussian: $\overset{\text{IID}}{\sim}  \mathcal{N}(\boldsymbol{0}_d,\mathsf{I}_d)$
- Custom IID Distribution
- Inverse CDF Sampling
- Acceptance Rejection Sampling

**Low Discrepancy (LD) nodes**

- Lattice (base 2): $x_j  \overset{\text{LD}}{\sim}    \mathcal{U}(0,1)^d$
- Sobol (base 2): $x_j \overset{\text{LD}}{\sim}    \mathcal{U}(0,1)^d$

<hr>

## Stopping Criterion

The stopping criterion to determine sufficient approximation\
Has class method `integrate` which preforms numerical integration\
*Abstract class with concrete implementations*

**For IID Nodes** $x_i\sim$ iid

  - Mean MC (guaranteed)
  - Multilevel MC
  - Central Limit Theorem (CLT) 

**For QMC Sequences** $\{x_{r,i}\}_{r=1}^R \sim$ ld

- CubLattice_g (gauranteed)
- CubSobol_g (gauranteed)
- Multilevel QMC
- CLT Repeated