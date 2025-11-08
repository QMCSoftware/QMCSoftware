
# Summary

`QMCPy` is an open-source Python package for high-dimensional numerical
integration using Monte Carlo (MC) and quasi-MC (QMC) methods—collectively
"(Q)MC." Its object-oriented (OO) design enables researchers to easily implement novel
algorithms. The framework offers user-friendly APIs, diverse (Q)MC algorithms,
reliable adaptive error estimation, and integration with scientific libraries
following reproducible research practices.

# Statement of Need

High-dimensional integration and simulation are essential for computational
finance, uncertainty quantification, machine learning, and physics. `QMCPy`  implements both MC methods which use independent identically distributed
(IID) points as well as QMC methods which use low-discrepancy (LD) sequences
that more evenly cover the unit cube and therefore allow for faster rates of convergence. \autoref{fig:points} visualizes IID and LD pointsets. 

![IID points with gaps and clusters alongside LD pointsets which more evenly fill the space. IID and LD pointsets. Each of the three randomized LD sequences contain purple
stars (initial 32 points), green triangles (next 32), and blue circles
(subsequent 64). The lattice was randomly shifted; the digital net was
randomized with nested uniform scrambling (Owen scrambling); the Halton pointset was randomized with linear
matrix scrambling and permutation scrambling.](../demos/talk_paper_demos/JOSS2025/JOSS2025.outputs/points.png){width=100%}

While (Q)MC methods are well established, practical implementation demands numerical and algorithmic
expertise. Our `QMCPy` implementation follows MATLAB's Guaranteed Automatic Integration Library (GAIL), with both softwares adhering to reproducible research practices. `QMCPy` bridges theory and practice by consolidating cutting-edge (Q)MC
algorithms  into a unified framework with:

- **Intuitive APIs** for problem specification and accessing (Q)MC methods,
- **Flexible integrations** with `NumPy`  and `SciPy`, 
- **Robust and adaptive sampling** with theoretically grounded error estimation, and 
- **Extensible OO components** enabling researchers to implement and test new algorithms.

Unlike other (Q)MC modules such as  `SciPy`'s `scipy.stats.qmc`
 or `PyTorch`'s `torch.quasirandom`, `QMCPy`
provides:

- customizable LD sequences with diverse randomization techniques, 
- rigorous adaptive error estimation algorithms, and 
- automatic variable transformations for (Q)MC compatibility.

# Components

(Q)MC methods approximate the multivariate integral 
\begin{equation}
\mu := \int_{\mathcal{T}} g(\mathbf{t}) \, \lambda(\mathbf{t}) \, d\mathbf{t},
\end{equation}
where $g$ is the **integrand** and $\lambda$ a non-negative weight. If
$\lambda$ is the probability density for a random variable $\mathbf{T}$, then
$\mu = \mathbb{E}[g(\mathbf{T})]$, where $\mathbf{T}$ is called the **true
measure**. Through an appropriate transformation $\boldsymbol{\psi}$, we rewrite
$\mu$ as the expectation of a function of a standard uniform random variable
$\mathbf{X}$ over the unit hypercube:
\begin{equation}
\mu = \mathbb{E}[f(\mathbf{X})] = \int_{[0,1]^d} f(\mathbf{x}) \, d\mathbf{x}, \qquad \mathbf{X} \sim \mathcal{U}[0,1]^d.
\end{equation}

If $\boldsymbol{\psi}$ satisfies $\mathbf{T} \sim
\boldsymbol{\psi}(\mathbf{X})$, then $f = g \circ \boldsymbol{\psi}$. This
transformation accommodates IID and LD samples which are approximately uniform on $[0,1]^d$. 

(Q)MC methods estimate the population mean $\mu$ in \eqref{eq:mu-uniform} via the sample mean
\begin{equation}
\widehat{\mu} := \frac{1}{n} \sum_{i=1}^{n} f(\mathbf{X}_i).
\end{equation}

MC methods choose IID sampling nodes $\mathbf{X}_1,\dots,\mathbf{X}_n$ and have
error $|\widehat{\mu}-\mu|$ like $\mathcal{O}(n^{-1/2})$. QMC methods choose
dependent LD nodes that fill $[0,1]^d$ more evenly, achieving errors like
$\mathcal{O}(n^{-1+\delta})$ where $\delta>0$ is arbitrarily small . 
A key feature of `QMCPy` is **stopping criteria** that automatically determine $n$
so $|\mu - \widehat{\mu}| \le \varepsilon$ for a user-specified tolerance
$\varepsilon>0$, either deterministically or with high probability.

`QMCPy` contains four main abstract classes which are extensible for new (Q)MC algorithms:

**Discrete Distributions** generate IID or LD sampling nodes. Available LD
pointsets include lattices, digital nets  (including Sobol' ), and Halton 
sequences. We also support 

* robust randomization routines, including
 
  - **Lattices** with shifts modulo one.
  - **Digital Nets** with digital shifts, linear matrix scrambling (LMS), or nested uniform scrambling (NUS, also called Owen scrambling).
  - **Halton Pointsets** with digital shifts, permutation scrambling, LMS, or NUS.
* higher-order digital nets and higher order scrambling for integrands $f$ with $\alpha$ degrees of "smoothness", enabling QMC convergence like $\mathcal{O}(n^{-\alpha+\delta})$ where $\delta>0$ is arbitrarily small. 
* custom generating vectors for lattices and generating matrices for digital nets,
 available from the growing collection in the `LDData` repository, which standardizes data
  from the Magic Point Shop  and Kuo's websites on lattices
 and Sobol' points. 

Internally, our LD
generators call our C package `QMCToolsCL`.

**True Measures** $\mathbf{T}$ come with default transformations
$\boldsymbol{\psi}$ satisfying $\boldsymbol{\psi}(\mathbf{X}) \sim \mathbf{T}$
where $\mathbf{X} \sim \mathcal{U}[0,1]^d$. For example, suppose $\mathbf{T}
\sim \mathcal{N}(\mathbf{m},\Sigma)$ is a $d$-dimensional Gaussian random
variable with mean $\mathbf{m}$ and covariance $\Sigma =
\mathbf{A}\mathbf{A}^T$. Then $\boldsymbol{\psi}(\mathbf{X}) = \mathbf{A}
\Phi^{-1}(\mathbf{X}) + \mathbf{m}$ where $\Phi^{-1}$ is the inverse distribution
function of a standard normal applied elementwise. We support many measures,
including those from `SciPy`'s `scipy.stats`.

**Integrands** $g$, given a true measure $\mathbf{T}$ and
transformation $\boldsymbol{\psi}$, define the transformed integrand $f = g
\circ \boldsymbol{\psi}$ so that $\mu = \mathbb{E}[g(\mathbf{T})] =
\mathbb{E}[f(\mathbf{X})]$ where $\mathbf{X} \sim \mathcal{U}[0,1]^d$. This
change of variables is done automatically. Users only need to specify $g$ and
$\mathbf{T}$.

**Stopping Criteria (SC)** adaptively increase the sample size until (Q)MC
estimates satisfy user-defined error tolerances. SC vary
depending on properties of $f$, and can include guaranteed MC algorithms
 or QMC algorithms based on:

1. multiple randomizations of LD pointsets, 
2. quickly tracking the decay of Fourier coefficients, or
3. efficient Bayesian cubature by inducing structured Gram matrices
 . 
  
\autoref{fig:stopping_crit} compares MC and QMC SC performance. `QMCPy` is also capable of simultaneously approximate functions of multiple integrands. Inspired by Julia's `MultilevelEstimators.jl`, `QMCPy` is expanding support for multilevel (Q)MC SC
 which exploit cheaper
low-fidelity surrogates to accelerate estimates of expensive integrands, often in high or infinite dimensions.

![MC and QMC SC comparison for adaptively estimating the fair price of an Asian option. SC were ran 100 times per error
tolerance; shaded regions show 10%-90% quantiles, violins show
error distributions. MC algorithms require $n = \mathcal{O}(1/\varepsilon^2)$ samples 
while QMC algorithms require only $n = \mathcal{O}(1/\varepsilon)$ samples to meet the same error tolerance $\varepsilon$. Both consistently meet
tolerances.](../demos/talk_paper_demos/JOSS2025/JOSS2025.outputs/stopping_crit.png){
width=100% }


# Distribution and Resources

`QMCPy` can be installed using the command `pip install qmcpy`. \autoref{fig:points} and \autoref{fig:stopping_crit}
are reproducible via the Jupyter Notebook. Our project
website  features publications, presentations, blogs, documentation, and demos. Our GitHub repository  contains
open-source code, tests, and issue tracking. `QMCPy` is distributed under the Apache
(v2.0) license. Community feedback and engagement are welcome.

