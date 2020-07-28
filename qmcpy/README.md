# QMCPy

The QMCPy framework uses 5 abstract classes that are fleshed out in concrete implementations. Specifically, a user selects an integrand, true measure, discrete distribution, and stopping criterion specific to their Monte Carlo (MC) / quasi-Monte Carlo (qMC) problem. The $5^{th}$ abstract class accumulates data from the stopping criterion and does not need to be instantiated by the user. The following blocks give more detailed descriptions of each abstract class and the available concrete implementations. For specific class names and parameters see the [QMCPy Documentation page](https://qmcpy.readthedocs.io/en/latest/algorithms.html). 

----

## Integrand

The function to integrate.

- Keister Function: $g(\boldsymbol{x}) = \pi^{d/2} \, \cos(||\boldsymbol{x}||_2)$
- Custom Function
- European Option
    - stock price at time $jT/d=\tau_j$: $~~~~~~~~~$ $S(\tau_j)=S_0\exp\bigl((r-\sigma^2/2)\tau_j+\sigma\mathcal{B}(\tau_j)\bigr)$
    - discounted call payoff $= \max\left(S(\tau_d)-K\right),\: 0)  \,\exp(-rT)$
    - discounted put payoff $= \max\left(K-S(\tau_d)\right),\: 0)\,\exp(-rT)$
- Asian Option
    - stock price at time $jT/d=\tau_j$: $~~~~~~~~~$ $S(\tau_j)=S_0\exp\bigl((r-\sigma^2/2)\tau_j+\sigma\mathcal{B}(\tau_j)\bigr)$
    - airthmetic mean: $\gamma(\boldsymbol{\tau})= \frac{1}{2d}\sum_{j=1}^d [S(\tau_{j-1})+S(\tau_j)]$
    - geometric mean: $\gamma(\boldsymbol{\tau}) = \biggl[\prod_{j=1}^d [S(\tau_{j-1})S(\tau_j)]\biggr]^{\frac{1}{2d}}$
    
    - discounted call payoff $= \max( \gamma(\boldsymbol{\tau})-K,\: 0)\,\exp(-rT)$
    - discounted put payoff $= \max(K-\gamma(\boldsymbol{\tau}),0)\,\exp(-rT)$
- Multilevel Call Options with Milstein Discretization 
- Linear Function: $g(\boldsymbol{x}) = \sum_{j=1}^{d}x_{j}$

----

## True Measure

General measure used to define the integrand.

- Uniform: $\mathcal{U}(\boldsymbol{a},\boldsymbol{b})$
- Gaussian: $\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})$
- Discrete Brownian Motion: $\mathcal{N}(\boldsymbol{\mu},\mathsf{\Sigma})$, where $\mathsf{\Sigma} = \min(\boldsymbol{t},\boldsymbol{t})^T$, $~~~~$ $\boldsymbol{t} = (t_1, \ldots, t_d)$
- Lebesgue
- Importance sampling
- Identical to what the discrete distribution mimics

----

## Discrete Distribution

Sampling nodes.

**Low Discrepancy (LD) nodes**

- Lattice (base 2): $\overset{\text{LD}}{\sim}    \mathcal{U}(0,1)^d$
- Sobol' (base 2): $\overset{\text{LD}}{\sim}    \mathcal{U}(0,1)^d$
- Generalized Halton: $\overset{\text{LD}}{\sim}    \mathcal{U}(0,1)^d$
- Korobov: $\overset{\text{LD}}{\sim}    \mathcal{U}(0,1)^d$

**Independent Identically Distributed (IID) Nodes**

- IID Standard Uniform: $\overset{\text{IID}}{\sim}   \mathcal{U}(0,1)^d$
- IID Standard Gaussian: $\overset{\text{IID}}{\sim}  \mathcal{N}(\boldsymbol{0}_d,\mathsf{I}_d)$
- Custom IID Distribution
- Inverse CDF Sampling
- Acceptance Rejection Sampling

----

## Stopping Criterion

The stopping criterion to determine sufficient approximation.\
Has class method `integrate` which preforms numerical integration.

**qMC Algorithms**

- Gauranteed Lattice Cubature
- Guaranteed Sobol Cubature
- Multilevel qMC Cubature
- CLT qMC Cubature (with Replications)

**MC Algorithms**

  - Multilevel MC Cubature
  - Guaranteed MC Cubature
  - CLT MC Cubature 
