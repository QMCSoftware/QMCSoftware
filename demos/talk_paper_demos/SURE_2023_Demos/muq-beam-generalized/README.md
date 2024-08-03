# Euler-Bernoulli Beam

## Overview
This benchmark models the deformation of an Euler-Bernoulli beam with a spatially variable stiffness parameter.  The PDE describing the beam deformation is solved with a finite difference method and the beam stiffness is defined at each of the (typically 31) nodes in the discretization.  The displacement of the beam is also returned at these 31 points.

This docker container uses [MUQ](https://mituq.bitbucket.io/source/_site/index.html) and [UM-Bridge](https://github.com/UM-Bridge/umbridge/tree/main) to evaluate the model.  A realization of a Gaussian process is used to define a the load $f(x)$.  More details are provided below.

## Authors
- [Matthew Parno](mailto:matthew.d.parno@dartmouth.edu)

## Run
```
docker run -it -p 4242:4242 linusseelinger/model-muq-beam:latest
```

## Properties
Model | Description
---|---
forward | Forward evaluation of the beam model

### forward
Mapping | Dimensions | Description
---|---|---
inputSizes | [31] | The stiffness $E(x)$ at each finite difference node in the discretization.
outputSizes | [31] | The vertical displacement $u(x)$ at each finite difference node.

Feature | Supported
---|---
Evaluate | True
Gradient | True (via finite difference)
ApplyJacobian | True (via finite difference)
ApplyHessian | True (via finite difference)

Config | Type | Default | Description
---|---|---|---
None | | |

## Mount directories
Mount directory | Purpose
---|---
None |

## Source code

[Model sources here.](https://github.com/UM-Bridge/benchmarks/tree/main/models/muq-beam)

## Description

Let $u(x)$ denote the vertical deflection of the beam and let $f(x)$ denote the vertical force acting on the beam at point $x$ (positive for upwards, negative for downwards).  We assume that the displacement can be well approximated using Euler-Bernoulli beam theory and thus satisfies the PDE

$$\frac{\partial^2}{\partial x^2}\left[ r E(x) \frac{\partial^2 u}{\partial x^2}\right] = f(x),$$

where $E(x)$ is an effective stiffness and $r$ is the beam radius.  This model takes in $E(x)$ at $N$ finite difference nodes and returns the value of $u(x)$ at those nodes.   The beam radius is set to $r=0.1$ and the value of $f(x)$ is fixed to a precomputed realization of a Gaussian process (the value of $f$ will not change between model evaluations).

For a beam of length $L$, the cantilever boundary conditions take the form

$$u(x=0) = 0,\quad \left.\frac{\partial u}{\partial x}\right|_{x=0} = 0$$

and

$$\left.\frac{\partial^2 u}{\partial x^2}\right|_{x=L} = 0, \quad  \left.\frac{\partial^3 u}{\partial x^3}\right|_{x=L} = 0.$$

Discretizing this PDE with finite differences (or finite elements, etc...), we obtain a linear system of the form

$$K(\hat{E})\hat{u} = \hat{f},$$

where $\hat{u}\in\mathbb{R}^N$ and $\hat{f}\in\mathbb{R}^N$ are vectors containing approximations of $u(x)$ and $E(x)$ at finite difference nodes.

