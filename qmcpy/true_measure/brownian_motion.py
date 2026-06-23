from .gaussian import Gaussian
from ..discrete_distribution import DigitalNetB2
from ..util import ParameterError, ParameterWarning
import warnings
import numpy as np
from scipy.stats import norm


class BrownianMotion(Gaussian):
    r"""
    Brownian Motion as described in [https://en.wikipedia.org/wiki/Brownian_motion](https://en.wikipedia.org/wiki/Brownian_motion).
    For a standard Brownian Motion $W$ we define the Brownian Motion $B$ with initial value $B_0$, drift $\gamma$, and diffusion $\sigma^2$ to be

    $$B(t) = B_0 + \gamma t + \sigma W(t).$$

    Examples:
        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),t_final=2,drift=2)
        >>> true_measure(2)
        array([[0.82189263, 2.7851793 , 3.60126805, 3.98054724],
               [0.2610643 , 0.06170064, 1.06448269, 2.30990767]])
        >>> true_measure
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.5 1.  1.5 2. ]
            drift           2^(1)
            mean            [1. 2. 3. 4.]
            covariance      [[0.5 0.5 0.5 0.5]
                             [0.5 1.  1.  1. ]
                             [0.5 1.  1.5 1.5]
                             [0.5 1.  1.5 2. ]]
            decomp_type     PCA

        With independent replications

        >>> x = BrownianMotion(DigitalNetB2(3,seed=7,replications=2),t_final=2,drift=2)(4)
        >>> x.shape
        (2, 4, 3)
        >>> x
        array([[[0.66154685, 1.50620966, 3.52322901],
                [1.77064217, 3.32782204, 4.45013223],
                [1.33558688, 3.26017547, 3.40692337],
                [2.10317345, 3.78961839, 6.17948096]],
        <BLANKLINE>
               [[1.77868019, 2.75347902, 3.41161419],
                [0.44891984, 2.53987304, 4.7224811 ],
                [0.23147948, 2.25289769, 3.00039101],
                [2.06762574, 3.21756319, 4.93375923]]])

        With Brownian Bridge construction

        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),decomp_type='BrownianBridge')
        >>> true_measure(2)
        array([[-0.29376184,  0.41054648,  0.13428456,  0.3095377 ],
               [-0.32948661, -1.19527027, -1.17959535, -1.58454187]])
        >>> true_measure
        BrownianMotion (AbstractTrueMeasure)
            time_vec        [0.25 0.5  0.75 1.  ]
            drift           0
            mean            [0. 0. 0. 0.]
            covariance      [[0.25 0.25 0.25 0.25]
                             [0.25 0.5  0.5  0.5 ]
                             [0.25 0.5  0.75 0.75]
                             [0.25 0.5  0.75 1.  ]]
            decomp_type     BROWNIANBRIDGE

        With Brownian Bridge construction and independent replications

        >>> x = BrownianMotion(DigitalNetB2(4,seed=7,replications=3),decomp_type='BrownianBridge')(2)
        >>> x.shape
        (3, 2, 4)
        >>> x
        array([[[ 0.20967731,  0.52848898, -0.03955369, -0.17751616],
                [ 0.60733712,  0.96872916,  1.82256178,  2.21516041]],
        <BLANKLINE>
               [[-0.08220913, -0.48324258,  0.03000249, -0.19149823],
                [-0.13637543,  1.03215652,  0.59673736,  0.62971114]],
        <BLANKLINE>
               [[ 0.82975853,  1.10849282,  1.10891366,  1.02092441],
                [-0.12663949, -0.23324496, -0.37900074, -0.35202342]]])

        With a custom monitoring times order that reaches all four cases (first point, left anchor, right anchor, both anchors)

        >>> true_measure = BrownianMotion(DigitalNetB2(4,seed=7),decomp_type='BrownianBridge',monitoring_times=[0.6,1.0,0.3,0.8])
        >>> true_measure.time_vec
        array([0.3, 0.6, 0.8, 1. ])
        >>> true_measure(2)
        array([[-0.42678211,  0.23976687,  0.19961117,  0.56330283],
               [-0.31994843, -1.22738085, -1.29415239, -1.73713917]])
    """

    def __init__(
        self,
        sampler,
        t_final=1,
        initial_value=0,
        drift=0,
        diffusion=1,
        decomp_type="PCA",
        lazy_decomp=True,
        monitoring_times=None,
    ):
        r"""
        Args:
            sampler (Union[AbstractDiscreteDistribution, AbstractTrueMeasure]): Either

                - a discrete distribution from which to transform samples, or
                - a true measure by which to compose a transform.
            t_final (float): End time.
            initial_value (float): Initial value $B_0$.
            drift (int): Drift $\gamma$.
            diffusion (int): Diffusion $\sigma^2$.
            decomp_type (str): Method for decomposition for covariance matrix. Options include

                - `'PCA'` for principal component analysis,
                - `'Cholesky'` for cholesky decomposition, or
                - `'BrownianBridge'` for brownian bridge construction.
            lazy_decomp (bool): If True, defer expensive matrix decomposition until needed.
            monitoring_times (Union[np.ndarray, list]): Optional sampling times for `'BrownianBridge'` 
                with length d. The given order is the insertion order.
        """
        self.parameters = ["time_vec", "drift", "mean", "covariance", "decomp_type"]
        # default to transform from standard uniform
        self.domain = np.array([[0, 1]])
        self._parse_sampler(sampler)
        self.t = t_final  # exercise time
        self.initial_value = initial_value
        self.drift = drift
        self.diffusion = diffusion
        self._bridge_times = self._get_monitoring_times(monitoring_times, decomp_type)
        self.time_vec = np.sort(self._bridge_times)
        self.diffused_sigma_bm = self.diffusion * np.minimum.outer(
            self.time_vec, self.time_vec
        )
        self.drift_time_vec_plus_init = (
            self.drift * self.time_vec + self.initial_value
        )  # mean
        self._parse_gaussian_params(
            self.drift_time_vec_plus_init,
            self.diffused_sigma_bm,
            decomp_type,
            lazy_decomp,
        )
        if self.decomp_type not in ("PCA", "CHOLESKY", "BROWNIANBRIDGE"):
            raise ParameterError(
                f"decomp_type must be 'PCA', 'Cholesky', or 'BrownianBridge'. Got '{decomp_type}'."
            )
        if self.decomp_type == "BROWNIANBRIDGE":
            self._setup_bridge()  # precompute bridge parameters
        if self.decomp_type == "BROWNIANBRIDGE" and not (self.d > 0 and (self.d & (self.d - 1)) == 0):
            warnings.warn(
                f"BrownianBridge is most efficient when d is a power of 2 (e.g., 1, 2, 4, 8, 16). Got d={self.d}.", 
                ParameterWarning,
                stacklevel=2
            )
        self.range = np.array([[-np.inf, np.inf]])
        super(Gaussian, self).__init__()

    def _spawn(self, sampler, dimension):
        return BrownianMotion(
            sampler, t_final=self.t, drift=self.drift, decomp_type=self.decomp_type
        )

    def _transform(self, x):
        if self.decomp_type == "BROWNIANBRIDGE":
            z = norm.ppf(x)
            w = self._bridge_transform(z)
            return self.drift_time_vec_plus_init + np.sqrt(self.diffusion) * w
        return super()._transform(x)

    def _get_monitoring_times(self, monitoring_times, decomp_type):
        """Return d monitoring times"""
        if decomp_type.upper() != "BROWNIANBRIDGE":
            if monitoring_times is not None:
                raise ParameterError("monitoring_times is only valid with decomp_type='BrownianBridge'.")
            return np.linspace(self.t / self.d, self.t, self.d)  # evenly spaced
        if monitoring_times is None:
            return self._van_der_corput(self.d, self.t)  # default bridge ordering
        s = np.asarray(monitoring_times, dtype=float).flatten()
        if s.shape != (self.d,):
            raise ParameterError(f"monitoring_times must have length d={self.d}. Got length {s.shape[0]}.")
        if (s <= 0).any():
            raise ParameterError("monitoring_times must be positive.")
        if np.unique(s).size != self.d:
            raise ParameterError("monitoring_times must be distinct.")
        return s
    
    @staticmethod
    def _van_der_corput(d, t_final):
        """First d van der Corput points multiplied by t_final."""
        times = np.empty(d)
        for i in range(d):
            value, increment, n = 0.0, 0.5, i
            while n:
                value += (n & 1) * increment
                increment *= 0.5
                n >>= 1
            times[i] = value
        times[0] = 1.0
        return t_final * times
    
    def _setup_bridge(self):
        """Precompute parameters (Owen Algorithm 6.1)"""
        s = self._bridge_times
        d = self.d
        left = np.full(d, -1, dtype=int)
        right = np.full(d, -1, dtype=int)
        a = np.zeros(d)
        b = np.zeros(d)
        w = np.zeros(d)
        for j in range(d):
            for k in range(j):
                if s[k] < s[j] and (left[j] == -1 or s[k] > s[left[j]]):
                    left[j] = k
                elif s[k] > s[j] and (right[j] == -1 or s[k] < s[right[j]]):
                    right[j] = k
            if left[j] >= 0 and right[j] >= 0:  # both anchors
                s_left, s_right = s[left[j]], s[right[j]]
                a[j] = (s_right - s[j]) / (s_right - s_left)
                b[j] = (s[j] - s_left) / (s_right - s_left)
                w[j] = np.sqrt((s[j] - s_left) * (s_right - s[j]) / (s_right - s_left))
            elif left[j] >= 0:  # left anchor
                a[j] = 1.0
                w[j] = np.sqrt(s[j] - s[left[j]])
            elif right[j] >= 0:  # right anchor
                s_right = s[right[j]]
                b[j] = s[j] / s_right
                w[j] = np.sqrt(s[j] * (s_right - s[j]) / s_right)
            else:  # first point
                w[j] = np.sqrt(s[j])
        self._bridge_left = left
        self._bridge_right = right
        self._bridge_a = a
        self._bridge_b = b
        self._bridge_w = w
        self._bridge_order = np.argsort(s)  # increasing time 

    def _bridge_transform(self, z):
        """Build Brownian Motion paths (Owen Algorithm 6.2)"""
        left = self._bridge_left
        right = self._bridge_right
        a = self._bridge_a
        b = self._bridge_b
        w = self._bridge_w
        paths = np.empty(z.shape[:-1] + (self.d,))
        for j in range(self.d):
            paths[..., j] = w[j] * z[..., j]
            if left[j] >= 0:
                paths[..., j] += a[j] * paths[..., left[j]]
            if right[j] >= 0:
                paths[..., j] += b[j] * paths[..., right[j]]
        return paths[..., self._bridge_order]