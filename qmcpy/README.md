# QMCPy

## Integrand

The function to integrate\
*Abstract class with concrete implementations*

- Linear: $g(\boldsymbol{x}) = \sum_{j=1}^{d}x_{j}$
- Keister: $g(\boldsymbol{x}) = \pi^{d/2} \, \cos(||\boldsymbol{x}||_2)$
- European Option
- Asian Call
    - stock price at time $jT/d$: $S(x_j)=S_0\exp\bigl((r-\sigma^2/2)(jT/d)+\sigma\mathcal{B}(t_j)\bigr)$
    - discounted call payoff $= \max(\frac{1}{d}\sum_{j=0}^{d-1} S(jT/d)-K)\;,\: 0)*e^{-rT}$
    - discounted put payoff $= \max(K-\frac{1}{d}\sum_{j=0}^{d-1} S(jT/d))\;,\: 0)*e^{-rT}$
- Multilevel Call Options
- QuickConstruct

<hr>

## True Measure

General measure used to define the integrand\
*Abstract class with concrete implementations*

- Uniform: $\mathcal{U}(a,b)$
- Gaussian: $\mathcal{N}(\mu,\sigma^2)$
- Discrete Brownian Motion: $\mathcal{B}(t_j)=B(t_{j-1})+Z_j\sqrt{t_j-t_{j-1}} \;$ for $\;Z_j \sim \mathcal{N}(0,1)$
- Lebesgue
- Identity Transform
- Importance sampling

<hr>

## Discrete Distribution

Sampling nodes IID or LDS (low-discrepancy sequence)\
*Abstract class with concrete implementations*

**Independent Identically Distributed (iid) Nodes**

- IID Standard Uniform: $x_j \overset{iid}{\sim}   \mathcal{U}(0,1)$
- IID Standard Gaussian: $x_j \overset{iid}{\sim}   \mathcal{N}(0,1)$
- Custom IID Distribution
- Inverse CDF Sampling
- Acceptance Rejection Sampling

**Low Discrepancy (ld) nodes**

- Lattice (base 2): $x_j  \overset{ld}{\sim}    \mathcal{U}(0,1)$
- Sobol (base 2): $x_j \overset{ld}{\sim}    \mathcal{U}(0,1)$

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
