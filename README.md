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

Quasi-Monte Carlo (QMC) methods are used to approximate multivariate integrals. They have four main components: a discrete distribution, a true measure of randomness, an integrand, and a stopping criterion. Information about the integrand is obtained as a sequence of values of the function sampled at the data-sites of the discrete distribution. The stopping criterion tells the algorithm when the user-specified error tolerance has been satisfied. We are developing a framework that allows collaborators in the QMC community to develop plug-and-play modules in an effort to produce more efficient and portable QMC software. Each of the above four components is an abstract class. Abstract classes specify the common properties and methods of all subclasses. The ways in which the four kinds of classes interact with each other are also specified. Subclasses then flesh out different integrands, sampling schemes, and stopping criteria. Besides providing developers a way to link their new ideas with those implemented by the rest of the QMC community, we also aim to provide practitioners with state-of-the-art QMC software for their applications.

## Start Here

Install QMCPy from PyPI:

```bash
pip install qmcpy
```

- **Use QMCPy:** read the [documentation](https://qmcsoftware.github.io/QMCSoftware/), start with the [quickstart notebook](https://qmcsoftware.github.io/QMCSoftware/demos/quickstart), or browse the [package reference](https://qmcsoftware.github.io/QMCSoftware/api/discrete_distributions/).
- **Develop QMCPy:** use the [GitHub repository](https://github.com/QMCSoftware/QMCSoftware), [contributing guidelines](https://qmcsoftware.github.io/QMCSoftware/CONTRIBUTING/), and [PyPI package](https://pypi.org/project/qmcpy/).
- **Learn the project:** browse the [migrated blog posts](https://qmcsoftware.github.io/QMCSoftware/blogs/), [publications](https://qmcsoftware.github.io/QMCSoftware/publications/), [community page](https://qmcsoftware.github.io/QMCSoftware/community/), and [QMC software table](https://qmcsoftware.github.io/QMCSoftware/qmc-software/).
- **Cite QMCPy:** see the [recommended citation](#citation) and [BibTeX file](https://github.com/QMCSoftware/QMCSoftware/blob/master/cite_qmcpy.bib).

## Resources

The [QMCPy documentation](https://QMCSoftware.github.io/QMCSoftware/) contains a detailed **package reference** documenting functions and classes including thorough doctests. A number of example notebook **demos** are also rendered into the documentation from `QMCSoftware/demos/`. We recommend the following resources to start learning more about QMCPy

- [mathematical description of QMCPy software and components](https://qmcsoftware.github.io/QMCSoftware/components).
- [Aleksei Sorokin's 2023 PyData Chicago video tutorial](https://www.youtube.com/watch?v=bRcKiLA2yBQ) and [corresponding notebook](https://qmcsoftware.github.io/QMCSoftware/demos/talk_paper_demos/pydata_chi_2023/)
- [Fred Hickernell's 2020 MCQMC video tutorial](https://www.youtube.com/watch?v=gL8M_7c-YUE) and [corresponding notebook](https://qmcsoftware.github.io/QMCSoftware/demos/talk_paper_demos/MCQMC_Tutorial_2020/MCQMC_2020_QMC_Software_Tutorial/)
- The QMCPy [introduction notebook](https://qmcsoftware.github.io/QMCSoftware/demos/qmcpy_intro) and [quickstart notebook](https://qmcsoftware.github.io/QMCSoftware/demos/quickstart)

## Installation

```bash
pip install qmcpy
```

To install from source, please see the [contributing guidelines](https://qmcsoftware.github.io/QMCSoftware/CONTRIBUTING/).

## Citation

If you find QMCPy helpful in your work, please support us by citing the following work, which is also available as a [QMCPy BibTex citation](https://github.com/QMCSoftware/QMCSoftware/blob/master/cite_qmcpy.bib)

~~~
Sou-Cheng T. Choi, Fred J. Hickernell, Michael McCourt, Jagadeeswaran Rathinavel, Aleksei G. Sorokin,
QMCPy: A Quasi-Monte Carlo Python Library. 2026.
https://qmcsoftware.github.io/QMCSoftware/
~~~

The [QMCPy publications page](https://qmcsoftware.github.io/QMCSoftware/publications/) lists papers and reports on the development and use of QMCPy. The [QMCPy community page](https://qmcsoftware.github.io/QMCSoftware/community/#select-references) includes a list of select references upon which QMCPy was built.

### Package usage stats

PyPI download statistics are tracked automatically in [stats/pypi_downloads.md](https://github.com/QMCSoftware/QMCSoftware/blob/pypi-stats/stats/pypi_downloads.md) (in the pypi-stats branch).

### Other quasi-Monte Carlo software
A page listing other QMC software is available in this branch at
[`docs/qmc-software.md`](docs/qmc-software.md).

The page content is generated automatically from
[`data/qmc-software.yml`](https://github.com/QMCSoftware/QMCSoftware/blob/develop/data/qmc-software.yml).

## Development

Want to contribute to QMCPy? Please see our [guidelines for contributors](https://qmcsoftware.github.io/QMCSoftware/CONTRIBUTING/) which includes instructions on installation for developers, running tests, and compiling documentation.

This software would not be possible without the efforts of the [QMCPy community](https://qmcsoftware.github.io/QMCSoftware/community) including our steering council, collaborators, contributors, and sponsors.

QMCPy is distributed under an [Apache 2.0 license from the Illinois Institute of Technology](https://github.com/QMCSoftware/QMCSoftware/blob/master/LICENSE).
