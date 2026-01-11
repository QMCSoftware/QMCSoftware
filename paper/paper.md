---
title: |
  QMCPy: A Python Framework for (Quasi-)Monte Carlo Algorithms
tags:
  - (quasi-)Monte Carlo numerical integration
  - randomized low-discrepancy sequences
  - automatic error estimation
  - object oriented Python framework
authors:
  - name: Aleksei G. Sorokin
    orcid: 0000-0003-1004-4632
    affiliation: "1, 2"
  - name: Fred J. Hickernell
    orcid: 0000-0001-6677-1324
    affiliation: "1"
  - name: Sou-Cheng T. Choi
    orcid: 0000-0002-6190-2986
    affiliation:  "1, 3" # (Multiple affiliations must be quoted)
  - name: Jagadeeswaran Rathinavel
    orcid: 0009-0005-6753-4589
    affiliation: "4"
  - name: Pieterjan Robbe
    orcid: 0000-0002-6254-8245
    affiliation: "5"
  - name: Aadit Jain
    orcid: 0009-0002-4805-3665
    affiliation: "6"
affiliations:
 - name: Illinois Institute of Technology, USA
   index: 1
 - name: University of Chicago, USA
   index: 2
 - name: SouLab LLC, USA
   index: 3
 - name: Torc Robotics, USA
   index: 4
 - name: Sandia National Laboratories, USA
   index: 5
 - name: University of California San Diego, USA
   index: 6
date: 9 January 2026
bibliography: paper.bib
csl: joss-simple.csl
colorlinks: true
linkcolor: blue
citecolor: blue
urlcolor: blue
filecolor: magenta
header-includes:
  # LaTeX macros and compact section spacing for PDF output (Pandoc)
  - \usepackage{xspace,titlesec}
  # Convenience macro for author name
  - \providecommand{\HickernellFJ}{Hickernell\xspace}
  # Reduce vertical spacing around headings
  #- \titlespacing*{\section}{0pt}{1ex plus .2ex minus .2ex}{0.6ex plus .1ex}
  #- \titlespacing*{\subsection}{0pt}{0.8ex plus .2ex minus .2ex}{0.5ex plus .1ex}
  #- \titlespacing*{\subsubsection}{0pt}{0.6ex plus .2ex minus .2ex}{0.4ex plus .1ex}
---

# Summary

Monte Carlo (MC) methods estimate high-dimensional integrals by computing sample averages at independent and identically distributed (IID) random points in the integration domain. Quasi-Monte Carlo (QMC) methods replace IID samples with low-discrepancy (LD) sequences which more uniformly cover the integration domain, leading to faster convergence and reduced computational requirements. \autoref{fig:points} visualizes IID and LD sequences. 

`QMCPy` ([https://qmcsoftware.github.io/QMCSoftware](https://qmcsoftware.github.io/QMCSoftware)) [@QMCPy2026] is our Python package for high-dimensional numerical integration using MC and QMC methods, collectively "(Q)MC." Its object-oriented design enables researchers to easily implement novel (Q)MC algorithms. The framework offers user-friendly APIs, diverse (Q)MC algorithms, adaptive error estimation techniques, and integration with scientific libraries following reproducible research practices. Compared to previous versions, `QMCPy` v2.2 (which is easily installed with `pip install -U qmcpy`) includes

- improved documentation,
- strengthened tests and demos,
- expanded support for randomized LD sequences,
- enhanced option pricing capabilities, and
- new utilities for fast QMC-based kernel methods.

![An IID sequence with gaps and clusters alongside LD sequences which more evenly fill the space. Each of the three randomized LD sequences contains purple stars (initial 32 points), green triangles (next 32), and blue circles (subsequent 64). The lattice was randomly shifted; the digital sequence was randomized with nested uniform scrambling (Owen scrambling); the Halton sequence was randomized with linear matrix scrambling and permutation scrambling. Figures in this paper are reproducible via the Jupyter notebook at [https://github.com/QMCSoftware/QMCSoftware/blob/master/demos/talk_paper_demos/JOSS2026/joss2026.ipynb](https://github.com/QMCSoftware/QMCSoftware/blob/master/demos/talk_paper_demos/JOSS2026/joss2026.ipynb)\label{fig:points}](./figs/points.png){width=100%}

# Statement of Need

High-dimensional integration and simulation are essential for computational finance [@Lem04a;@wangsloan05;@giles2009multilevel;@zhang2021sentiment], uncertainty quantification [@Seelinger2023;@MUQ;@parno2021muq;@Marzouk2016;@KaaEtal21], machine learning [@DICK2021101587;@pmlr-v80-chen18f], and physics [@AB02;@LanBin14;@bernhard2015quantifying]. While (Q)MC methods are well established [@DicPil10a;@dick2013high], practical implementation demands numerical and algorithmic expertise. Our `QMCPy` implementation follows MATLAB's Guaranteed Automatic Integration Library (GAIL) [@ChoEtal21a2;@TonEtAl22a], with both adhering to reproducible research practices [@Cho14a2;@ChoEtal22a]. However, `QMCPy` consolidates a substantially broader range of cutting-edge (Q)MC algorithms [@ChoEtal22a;@ChoEtal24a2;@sorokin2022bounding;@sorokin2025unified;@HicKirSor26a] into a unified framework, bridging theory and practice. `QMCPy` features

- **intuitive APIs** for problem specification and accessing (Q)MC methods,
- **flexible integrations** with `NumPy` [@harris2020array], `SciPy` [@2020SciPy-NMeth], and `PyTorch` [@NEURIPS2019_9015],
- **robust and adaptive sampling** with theoretically grounded error estimation, and
- **extensible (Q)MC components** enabling researchers to implement and test new algorithms.

While modules like `scipy.stats.qmc` [@Roy2023] and `torch.quasirandom` [@NEURIPS2019_9015] provide basic (Q)MC sequences such as Sobol', Halton, and Latin hypercube, `QMCPy` targets (Q)MC researchers and practitioners requiring additional capabilities to enable state of the art (Q)MC techniques. Advanced features unique to `QMCPy` include

- customizable LD sequences with diverse randomization techniques,
- efficient generators of LD sequences with multiple independent randomizations,
- automatic variable transformations for (Q)MC compatibility, and
- rigorous adaptive error estimation algorithms.

# Components

(Q)MC methods approximate the multivariate integral
\begin{equation}\label{eq:mu-general}
  \mu := \mathbb{E}[g(\mathbf{T})] = \int_{\mathcal{T}} g(\mathbf{t}) \, \lambda(\mathbf{t}) \, d\mathbf{t}, \qquad \mathbf{T} \sim \lambda,
\end{equation}
where $g$ is the **integrand** and $\lambda$ is the probability density of a random variable $\mathbf{T}$ whose distribution we call the **true measure**. Through an appropriate transformation $\boldsymbol{\psi}$, we rewrite $\mu$ as the expectation of a function of a standard uniform random variable $\mathbf{X}$ over the unit hypercube:
\begin{equation}\label{eq:mu-uniform}
  \mu = \mathbb{E}[f(\mathbf{X})] = \int_{[0,1]^d} f(\mathbf{x}) \, d\mathbf{x}, \qquad \mathbf{X} \sim \mathcal{U}[0,1]^d.
\end{equation}
If $\boldsymbol{\psi}$ satisfies $\mathbf{T} \sim \boldsymbol{\psi}(\mathbf{X})$, then $f = g \circ \boldsymbol{\psi}$. This transformation accommodates IID and LD samples which are approximately uniform on $[0,1]^d$. 

(Q)MC methods estimate the population mean $\mu$ in \eqref{eq:mu-uniform} via the sample mean
\begin{equation}\label{eq:mu-hat}
  \widehat{\mu} := \frac{1}{n} \sum_{i=1}^{n} f(\mathbf{X}_i).
\end{equation}
MC methods choose IID sampling nodes $\mathbf{X}_1,\dots,\mathbf{X}_n$ and have error $|\widehat{\mu}-\mu|$ like $\mathcal{O}(n^{-1/2})$ [@Nie78]. QMC methods choose dependent LD nodes that fill $[0,1]^d$ more evenly, i.e., the discrepancy between the **discrete distribution** of $\mathbf{X}_1,\dots,\mathbf{X}_n$ and the uniform distribution is small. QMC methods can achieve errors like $\mathcal{O}(n^{-1+\delta})$ where $\delta>0$ is arbitrarily small [@WanHic00b;@Wan03a]. A key feature of `QMCPy` are **stopping criteria** that automatically determine $n$ so $|\mu - \widehat{\mu}| \le \varepsilon$ for a user-specified tolerance $\varepsilon>0$, either deterministically or with high probability.

![MC and QMC SC comparison for pricing an Asian option.\label{fig:stopping_crit}](./figs/stopping_crit.png){width=100%}

`QMCPy` contains four main abstract classes which are extensible for new (Q)MC algorithms:

1. **Discrete Distributions** generate IID or LD sequences. Available LD
sequences and randomization routines [@sorokin2025unified] include

    - **Lattices** with random shifts [@CraPat76;@HicEtal03;@Ric51;@coveyou1967fourier;@WanHic02a].
    - **Digital Sequences** (including Sobol' sequences) with digital shifts, linear matrix scrambling (LMS), or nested uniform scrambling (NUS, also called Owen scrambling) [@Sob67;@dick2005multivariate; @Mat98;@Owe95;@owen2003variance;@dick2011higher;@Nie87;@Nie92;@DicPil10a].
    - **Halton Sequences** with digital shifts, permutation scrambling, LMS, or NUS [@Hal60;@WanHic00;@Mat98;@owen2024gain;@MorCaf94].

    We support higher-order digital sequences and higher order scrambling for integrands $f$ with $\alpha$ degrees of "smoothness", enabling QMC convergence like $\mathcal{O}(n^{-\alpha+\delta})$ where $\delta>0$ is arbitrarily small [@dick2011higher]. Internally, our LD generators call our C package `QMCToolsCL` [@QMCToolsCL]. `QMCPy` also integrates with the `LDData` repository [@LDData] which stores generating vectors for lattices and generating matrices for digital sequences, primarily drawing from the Magic Point Shop [@KuoNuy16a] and Kuo's websites on lattices [@cools2006constructing;@nuyens2006fast;@KuoGenerators] and Sobol' sequences [@JoeKuo03;@joe2008constructing;@SobolDirection].

2. **True Measures** come with default transformations $\boldsymbol{\psi}$ satisfying $\boldsymbol{\psi}(\mathbf{X}) \sim \mathbf{T}$ where $\mathbf{X} \sim \mathcal{U}[0,1]^d$. For example, suppose $\mathbf{T} \sim \mathcal{N}(\mathbf{m},\Sigma)$ is a $d$-dimensional Gaussian random variable with mean $\mathbf{m}$ and covariance $\Sigma = \mathbf{A}\mathbf{A}^T$. Then $\boldsymbol{\psi}(\mathbf{X}) = \mathbf{A} \Phi^{-1}(\mathbf{X}) + \mathbf{m}$ where $\Phi^{-1}$ is the inverse distribution function of a standard normal applied elementwise. We support many measures, including those from `SciPy`'s `scipy.stats` [@2020SciPy-NMeth].

3. **Integrands** $g$, given a transformation $\boldsymbol{\psi}$, define the transformed integrand $f = g \circ \boldsymbol{\psi}$ so that $\mu = \mathbb{E}[g(\mathbf{T})] = \mathbb{E}[f(\mathbf{X})]$ where $\mathbf{X} \sim \mathcal{U}[0,1]^d$. This change of variables is performed automatically. Users only need to specify $g$ and $\mathbf{T}$.

4. **Stopping Criteria (SC)** adaptively increase the sample size $n$ until (Q)MC estimates satisfy user-defined error tolerances [@HicEtal18a;@TonEtAl22a;@owen2024error]. SC vary depending on properties of $f$, and can include guaranteed MC algorithms [@HicEtal14a] or QMC algorithms based on:

    - multiple randomizations of LD sequences [@l2023confidence],
    - quickly tracking the decay of Fourier coefficients [@HicJim16a;@JimHic16a;@HicEtal17a;@DinHic20a], or
    - efficient Bayesian cubature by inducing structured Gram matrices [@Jag19a;@RatHic19a;@JagHic22a].

    `QMCPy` is also capable of simultaneously approximating functions of multiple integrands [@sorokin2022bounding]. Inspired by `MultilevelEstimators.jl` [@MultilevelEstimators], `QMCPy` is expanding support for multilevel (Q)MC SC [@giles2009multilevel;@giles2015multilevel] that exploit cheaper, low-fidelity surrogates to accelerate expensive integrand estimates in high or infinite dimensions.

    \autoref{fig:stopping_crit} compares (Q)MC SC for adaptively estimating the fair price of an Asian option with 100 independent trials per error tolerances. In the left and middle panels, the medians across trials are plotted for each method along with shaded regions showing 10%-90% quantiles. MC SC are generally slow and require $n = \mathcal{O}(1/\varepsilon^2)$ samples (and time) for an error tolerance $\varepsilon$. QMC SC are generally faster, requiring only $n = \mathcal{O}(1/\varepsilon)$. Both MC and QMC SC consistently meet tolerances, with the right panel showing the distribution of (Q)MC errors for a single error tolerance. 

# Acknowledgements

The authors acknowledge support from the U.S. National Science Foundation grant DMS-2316011 and Department of Energy Office of Science Graduate Student Research Program. We thank the international (Q)MC research community as well as JOSS reviewers and editors for invaluable and timely feedback and support.

This article has been co-authored by employees of National Technology and Engineering Solutions of Sandia, LLC under Contract No. DE-NA0003525 with the U.S. Department of Energy (DOE). The employees co-own right, title and interest in and to the article and are responsible for its contents. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this article or allow others to do so, for United States Government purposes. The DOE will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan ([https://www.energy.gov/downloads/doe-public-access-plan](https://www.energy.gov/downloads/doe-public-access-plan)).

# References
