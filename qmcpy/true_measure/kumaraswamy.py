from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from numpy.polynomial.legendre import leggauss
from scipy.special import digamma, polygamma
from scipy.sparse import diags
import numpy as np


class Kumaraswamy(AbstractTrueMeasure):
    r"""
    Kumaraswamy distribution as described in [https://en.wikipedia.org/wiki/Kumaraswamy_distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution).

    Examples:
        >>> true_measure = Kumaraswamy(DigitalNetB2(2,seed=7),a=[1,2],b=[3,4])
        >>> true_measure(4)
        array([[0.34705366, 0.6782161 ],
               [0.0577568 , 0.36189538],
               [0.76344358, 0.0932949 ],
               [0.17065545, 0.43009386]])
        >>> true_measure
        Kumaraswamy (AbstractTrueMeasure)
            a               [1 2]
            b               [3 4]
            mean            [0.25  0.406]
            variance        [0.038 0.035]
            standard_deviation [0.194 0.187]
            covariance      [[0.038 0.   ]
                             [0.    0.035]]

        With independent replications

        >>> x = Kumaraswamy(DigitalNetB2(3,seed=7,replications=2),a=[1,2,3],b=[3,4,5])(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.09004177, 0.22144305, 0.62190133],
                [0.31710078, 0.48718217, 0.47325643],
                [0.19657641, 0.57423463, 0.25697057],
                [0.56103074, 0.28939035, 0.63654112]],
        <BLANKLINE>
               [[0.18006788, 0.62226635, 0.5083556 ],
                [0.22602452, 0.10519477, 0.42823814],
                [0.08428482, 0.28804621, 0.2414302 ],
                [0.37253319, 0.45379743, 0.63366422]]])
    """

    def __init__(self, sampler, a=2, b=2):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            a (Union[float, np.ndarray]): First parameter $\alpha > 0$.
            b (Union[float, np.ndarray]): Second parameter $\beta > 0$.
        """
        self.parameters = ["a", "b", "mean", "variance", "standard_deviation", "covariance"]
        self.domain = np.array([[0, 1]])
        self.range = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self.a = a
        self.b = b
        self.alpha = np.array(a)
        if self.alpha.size == 1:
            self.alpha = self.alpha.item() * np.ones(self.d)
        self.beta = np.array(b)
        if self.beta.size == 1:
            self.beta = self.beta.item() * np.ones(self.d)
        if not (self.alpha.shape == (self.d,) and self.beta.shape == (self.d,)):
            raise DimensionError(
                "a and b must be scalar or have length equal to dimension."
            )
        if not ((self.alpha > 0).all() and (self.beta > 0).all()):
            raise ParameterError("Kumaraswamy requires a,b>0.")

        mean, variance = self._compute_moments()
        self._set_moments(
            mean=mean,
            variance=variance,
            standard_deviation=np.sqrt(variance),
            covariance=diags(variance, format="dia"),
        )
        super(Kumaraswamy, self).__init__()
        assert self.alpha.shape == (self.d,) and self.beta.shape == (self.d,)

    def _compute_moments(self):
        r"""
        Compute the marginal mean and variance of each coordinate.

        The Kumaraswamy raw moments are $M_n = b\,B(1 + n/a, b)$ [1], so the
        mean is $M_1$ and the variance is $M_2 - M_1^2$. Forming that difference
        directly causes cancellation error once the variance is small relative
        to $M_1^2$ (e.g. large $a$).

        Instead, with the log-moment function $K(r) = \log M_r$ (so $K(0) = 0$),

        $$\text{mean} = e^{K(1)}, \qquad
          \operatorname{Var}[X] = \text{mean}^2\,(e^{q} - 1), \qquad
          q = K(2) - 2K(1).$$

        The mean is recovered as $K(1) = \int_0^1 K'(r)\,\mathrm{d}r$ and, adding
        $K(0) = 0$, $q$ becomes a second central difference with the exact
        tent-weight (linear B-spline) form

        $$q = \int_0^1 r\,K''(r)\,\mathrm{d}r
            + \int_1^2 (2 - r)\,K''(r)\,\mathrm{d}r.$$

        Here $K'$ uses the digamma and $K''$ the trigamma function [5]. $K$ is
        convex ($K''(r) > 0$, since $r \mapsto M_r$ is log-convex by Holder's
        inequality [2]), so both integrands are nonnegative and $q$ is built
        from nonnegative pieces with no cancellation. Both integrals share one
        8-node Gauss-Legendre rule on $[0, 1]$ [3] (nodes from ``leggauss`` [4]);
        ``numpy.expm1`` keeps $e^{q} - 1$ accurate for small $q$ [4]. Every
        operation is applied elementwise to the per-coordinate parameters $a$
        and $b$, so ``mean`` and ``variance`` are returned as length-``d``
        arrays.

        **References:**

        1.  Kumaraswamy distribution. Wikipedia.
            [https://en.wikipedia.org/wiki/Kumaraswamy_distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution).

        2.  G. H. Hardy, J. E. Littlewood, and G. Polya.
            Inequalities, 2nd edition, Cambridge University Press, Cambridge, 1952
            (Holder's inequality; implies log-convexity of the moment sequence).

        3.  Philip J. Davis and Philip Rabinowitz.
            Methods of Numerical Integration, 2nd edition,
            Academic Press, Orlando, FL, 1984, ISBN 0-12-206360-0
            (Gauss-Legendre quadrature).

        4.  NumPy Reference. numpy.polynomial.legendre.leggauss and numpy.expm1.
            [https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html).
            [https://numpy.org/doc/stable/reference/generated/numpy.expm1.html](https://numpy.org/doc/stable/reference/generated/numpy.expm1.html).

        5.  SciPy Reference. scipy.special.digamma and scipy.special.polygamma.
            [https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.digamma.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.digamma.html).
            [https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.polygamma.html](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.polygamma.html).

        Returns:
            tuple: Length ``d`` arrays ``(mean, variance)``.
        """
        inv_a = 1.0 / self.alpha
        beta = self.beta

        # 8-point Gauss-Legendre nodes and weights mapped from [-1, 1] to [0, 1].
        nodes, weights = leggauss(8)
        s = 0.5 * (nodes + 1.0)[:, None]  # quadrature nodes, shape (8, 1)
        w = 0.5 * weights  # quadrature weights, shape (8,)

        # K'(r) = (1/a) * (digamma(1 + r/a) - digamma(b + 1 + r/a)).
        # K(1) = integral_0^1 K'(r) dr since K(0) = 0.
        k_prime = inv_a * (
            digamma(1.0 + s * inv_a) - digamma(beta + 1.0 + s * inv_a)
        )
        mean = np.exp(w @ k_prime)

        # K''(r) = (1/a^2) * (polygamma(1, 1 + r/a) - polygamma(1, b + 1 + r/a)).
        # K'' is mathematically nonnegative. Remove only possible last-bit negative roundoff.
        def k_double_prime(r):
            value = inv_a**2 * (
                polygamma(1, 1.0 + r * inv_a)
                - polygamma(1, beta + 1.0 + r * inv_a)
            )
            return np.maximum(value, 0.0)

        # q = K(2) - 2K(1) = integral_0^1 r K''(r) dr + integral_1^2 (2-r) K''(r) dr.
        # In the second integral substitute r = 1 + s, giving weight 1 - s.
        q = w @ (s * k_double_prime(s) + (1.0 - s) * k_double_prime(1.0 + s))

        # expm1(q) accurately computes exp(q) - 1 when q is very small.
        variance = mean * mean * np.expm1(q)

        return mean, variance

    def _transform(self, x):
        return (1 - (1 - x) ** (1 / self.beta)) ** (1 / self.alpha)

    def _weight(self, x):
        return np.prod(
            self.alpha
            * self.beta
            * x ** (self.alpha - 1)
            * (1 - x**self.alpha) ** (self.beta - 1),
            -1,
        )

    def _spawn(self, sampler, dimension):
        if dimension == self.d:  # don't do anything if the dimension doesn't change
            spawn = Kumaraswamy(sampler, a=self.alpha, b=self.beta)
        else:
            a = self.alpha[0]
            b = self.beta[0]
            if not (all(self.alpha == a) and all(self.beta == b)):
                raise DimensionError(
                    """
                    In order to spawn a Kumaraswamy measure
                    a must all be the same and
                    b must all be the same"""
                )
            spawn = Kumaraswamy(sampler, a=a, b=b)
        return spawn
