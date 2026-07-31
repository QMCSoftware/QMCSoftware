from .abstract_true_measure import AbstractTrueMeasure
from ..util import DimensionError, ParameterError
from ..discrete_distribution import DigitalNetB2
from numpy.polynomial.legendre import leggauss
from scipy.special import digamma, polygamma
from scipy.special import beta as beta_function, gammaln
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
            a = np.tile(self.a, self.d)
        self.beta = np.array(b)
        if self.beta.size == 1:
            self.beta = self.beta.item() * np.ones(self.d)
        if not (self.alpha.shape == (self.d,) and self.beta.shape == (self.d,)):
            raise DimensionError(
                "a and b must be scalar or have length equal to dimension."
            )
        if not ((self.alpha > 0).all() and (self.beta > 0).all()):
            raise ParameterError("Kumaraswamy requires a,b>0.")


        # Precompute once at module import.
        _GL_X, _GL_W = leggauss(8)

        # Nodes and weights mapped from [-1, 1] to [0, 1].
        _S = 0.5 * (_GL_X + 1.0)
        _W = 0.5 * _GL_W

        inv_a = 1.0 / a

        # log(mean) = integral_0^1 K'(r) dr
        r = _S
        k_prime = inv_a * (
            digamma(1.0 + r * inv_a)
            - digamma(b + 1.0 + r * inv_a)
        )
        log_mean = np.dot(_W, k_prime)

        mean = np.exp(log_mean)

        # K''(r), the curvature of the log-moment function.
        def k_double_prime(r):
            value = inv_a**2 * (
                polygamma(1, 1.0 + r * inv_a)
                - polygamma(1, b + 1.0 + r * inv_a)
            )

            # K'' is mathematically nonnegative. Remove only possible
            # last-bit negative roundoff.
            return np.maximum(value, 0.0)

        # q = K(2) - 2K(1)
        #
        # First integral:  integral_0^1 r K''(r) dr
        # Second integral: integral_1^2 (2-r) K''(r) dr
        #
        # In the second integral put r = 1+s, giving weight 1-s.
        q = np.dot(
            _W,
            _S * k_double_prime(_S)
            + (1.0 - _S) * k_double_prime(1.0 + _S)
        )

        # expm1(q) accurately computes exp(q)-1 when q is very small.
        variance = mean * mean * np.expm1(q)

        # Defensive cleanup of signed zero or exceptional last-bit effects.
        variance = max(float(variance), 0.0)


        # def log_moment(r: float) -> float:
        #     return (
        #         gammaln(1.0 + r / a)
        #         + gammaln(b + 1.0)
        #         - gammaln(b + 1.0 + r / a)
        #     )

        # L1 = log_moment(1.0)
        # L2 = log_moment(2.0)
        # print(L1, L2)
        # print(2 * L1 - L2)
        # print(-np.exp(L2), np.expm1(2.0 * L1 - L2))
        # variance = -np.exp(L2) * np.expm1(2.0 * L1 - L2) 

        # mean = self.beta * beta_function(1 + 1 / self.alpha, self.beta)
        # second_moment = self.beta * beta_function(1 + 2 / self.alpha, self.beta)
        # variance = second_moment - mean**2
        self._set_moments(
            mean=mean,
            variance=variance,
            standard_deviation=np.sqrt(variance),
            covariance=np.diag(variance),
        )
        super(Kumaraswamy, self).__init__()
        assert self.alpha.shape == (self.d,) and self.beta.shape == (self.d,)

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
