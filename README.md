# QMCPy: Quasi-Monte Carlo Community Software in Python

[![](https://img.shields.io/badge/qmcpy.org-15bfa9)](https://qmcpy.org/)
[![Docs](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/docs.yml/badge.svg?branch=master)](https://qmcsoftware.github.io/QMCSoftware/)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.09705/status.svg)](https://doi.org/10.21105/joss.09705)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3964489.svg)](https://doi.org/10.5281/zenodo.3964489)
[![](https://img.shields.io/badge/PyPI-fc7303)](https://pypi.org/project/qmcpy/)
[![GitHub stars](https://img.shields.io/github/stars/QMCSoftware/QMCSoftware?style=social)](https://github.com/QMCSoftware/QMCSoftware)
[![Tests](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/alltests.yml/badge.svg)](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/alltests.yml)
[![codecov on Unit Tests](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/unittests.yml/badge.svg)](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/unittests.yml)
[![codecov on All Tests](https://codecov.io/github/QMCSoftware/QMCSoftware/graph/badge.svg?token=Gqf0grDPQt)](https://codecov.io/github/QMCSoftware/QMCSoftware)
[![PEP8 score](docs/assets/pep8-badge.svg)](https://github.com/QMCSoftware/QMCSoftware/actions/workflows/pep8.yml)

QMCPy is an open-source Python framework for Monte Carlo and quasi-Monte Carlo integration. It combines low-discrepancy point generators, probability measures, integrands, and stopping criteria through a common object-oriented interface. The project supports practitioners applying modern QMC methods and researchers developing new algorithms within a shared, reproducible framework.

[Get started](https://qmcsoftware.github.io/QMCSoftware/demos/qmcpy_intro/) | [API reference](https://qmcsoftware.github.io/QMCSoftware/api/discrete_distributions/) | [Demos](https://qmcsoftware.github.io/QMCSoftware/demos/quickstart/) | [Blogs](https://qmcsoftware.github.io/QMCSoftware/blogs/) | [Publications](https://qmcsoftware.github.io/QMCSoftware/publications/)

## Installation

```bash
pip install qmcpy
```

To install from source or contribute to QMCPy, see the [contributing guide](https://qmcsoftware.github.io/QMCSoftware/CONTRIBUTING/).

## How QMCPy Fits Together

A QMCPy integration problem is assembled from four main components:

1. A **discrete distribution** generates IID or low-discrepancy samples.
2. A **true measure** transforms those samples to the desired probability law.
3. An **integrand** evaluates the quantity of interest.
4. A **stopping criterion** determines when the requested error tolerance has been met.

See the [component overview](https://qmcsoftware.github.io/QMCSoftware/components/) for the mathematical interfaces and the [introduction notebook](https://qmcsoftware.github.io/QMCSoftware/demos/qmcpy_intro/) for a complete example.

## Learn and Explore

- Start with the [quickstart notebook](https://qmcsoftware.github.io/QMCSoftware/demos/quickstart/).
- Browse the [package reference](https://qmcsoftware.github.io/QMCSoftware/api/discrete_distributions/) and the full collection of rendered notebook demos.
- Read technical articles in the [QMCPy blog archive](https://qmcsoftware.github.io/QMCSoftware/blogs/).
- Review [publications on QMCPy and related work](https://qmcsoftware.github.io/QMCSoftware/publications/).
- Compare projects in the [QMC software ecosystem](https://qmcsoftware.github.io/QMCSoftware/qmc-software/).
- Watch the [2023 PyData Chicago tutorial](https://www.youtube.com/watch?v=bRcKiLA2yBQ) or the [2020 MCQMC tutorial](https://www.youtube.com/watch?v=gL8M_7c-YUE).

## Citation

If QMCPy supports your work, please cite the project. The repository contains a [BibTeX entry](https://github.com/QMCSoftware/QMCSoftware/blob/master/cite_qmcpy.bib), and the current software paper is available from [JOSS](https://doi.org/10.21105/joss.09705).

```text
Sou-Cheng T. Choi, Fred J. Hickernell, Michael McCourt,
Jagadeeswaran Rathinavel, Aleksei G. Sorokin,
QMCPy: A Quasi-Monte Carlo Python Library. 2026.
https://qmcsoftware.github.io/QMCSoftware/
```

## Community

QMCPy is developed by an international community of researchers and software contributors. See the [community page](https://qmcsoftware.github.io/QMCSoftware/community/) for the steering council, collaborators, contributors, sponsors, and selected references. PyPI usage statistics are updated on the [download statistics page](https://qmcsoftware.github.io/QMCSoftware/stats/pypi_downloads/).

QMCPy is distributed under the [Apache 2.0 license](https://github.com/QMCSoftware/QMCSoftware/blob/master/LICENSE) from the Illinois Institute of Technology.
