---
title: A simple MLMC Example
author: HajiAli
excerpt: ""
layout: single

categories:
  - MLMC
---


[//]: # ({% include base_path %}) 

[//]: # ({% include latex_preamble %})

$$
\def\oS{\overline S}
$$

Let's say that we have the following SDE for the Geometric Brownian Motion

$$
d S_t = \mu S_t d t + \sigma S_t d W_t
$$

And we want to estimate $\E{g(S_T)} = \E{\max(S_T-K, 0)}$ for some given
constant $\mu, \sigma, K$ and $T$. Obviously in this case we know the
expression for $S_T$, but assume that we have to use Euler-Maruyama
instead to write the approximation

$$
d \oS_n = \mu \oS_n \Delta t + \sigma \oS_n \Delta W_n
$$

where $n = 0, 1, \ldots, N$ and $ T/N = \Delta t $. Moreover, here
$\p{\Delta W_n}_{n=1}^N$ are i.i.d. standard normal
variables. The we can use Monte Carlo with $M$ samples as follows

$$
\E{g(S_T)} \approx \frac{1}{M} \sum_{m=1}^M g(\overline{S}_N^{(m)})
$$

where $\oS_N^{(m)}$ is the $m$'th sample of $\oS_N$
obtained by sampling $\p{ \Delta W_n }_{n=1}^N$ then
computing $\oS_N$. Multilevel Monte Carlo is based on the
telescoping sum

$$
\E{g(\oS_{2^L})} =
\E{g(\oS_{2^{0}})} + \sum_{\ell=1}^L
\E{g(\oS_{2^{\ell}}) - g(\oS_{2^{\ell-1}}}
$$

To be continued...

In what follows, we will go through the code
[here](https://people.maths.ox.ac.uk/gilesm/files/opre_code.zip).

The main function is

```m

function [P, Nl] = mlmc(M,eps,mlmc_l)

```

To be continued...
