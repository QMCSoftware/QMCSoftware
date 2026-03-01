"""
AcceptanceRejection True Measure for QMCPy
==========================================

Implements the Deterministic Acceptance-Rejection (DAR) sampler from:

    Zhu, H., & Dick, J. (2014). Discrepancy bounds for deterministic
    acceptance-rejection samplers. Electronic Journal of Statistics,
    8(1), 678-707.

Three algorithms are provided:

    - AcceptanceRejection     : Algorithm 2  (DAR)   — unit cube [0,1]^d
    - AcceptanceRejectionReal : Algorithm 3  (DAR-Real) — real space R^d
    - ReducedAcceptanceRejection : Algorithm 4 (DRAR) — hybrid inversion + A-R

Usage example (mirrors QMCPy style)
------------------------------------
    from qmcpy.discrete_distribution import Sobol
    from qmcpy.true_measure.acceptance_rejection import AcceptanceRejection
    import numpy as np

    def my_density(x):            # unnormalised density on [0,1]^d
        return 2.0 * x[0]         # psi(x) = 2*x_1 => L=2, C=integral=1

    driver = Sobol(dimension=2)   # s = target_dim + 1
    measure = AcceptanceRejection(
        target_density = my_density,
        discrete_distrib = driver,
        upper_bound = 2.0,        # L = sup psi(x)
        density_integral = 1.0,   # C = integral of psi over [0,1]^d
    )
    samples = measure.gen_samples(n=1024)   # shape (1024, 1)
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc as scipy_qmc
from scipy.integrate import quad, dblquad


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> tuple[int, int]:
    """Return (m, 2**m) where 2**m >= n.  Required for (t,m,s)-net guarantee."""
    m = int(math.ceil(math.log2(max(n, 1))))
    return m, 2 ** m


def _estimate_integral_1d(psi: Callable, n_quad: int = 1000) -> float:
    """Numerically estimate C = int_0^1 psi(np.array([x])) dx."""
    val, _ = quad(lambda x: float(psi(np.array([x]))), 0.0, 1.0)
    return val


def _vectorised_psi(psi: Callable, X: NDArray) -> NDArray:
    """Evaluate psi row-wise on an (M, d) array, return shape (M,)."""
    return np.apply_along_axis(psi, 1, X).astype(float)


# ---------------------------------------------------------------------------
# Abstract base (mirrors QMCPy AbstractTrueMeasure interface)
# ---------------------------------------------------------------------------

class _AbstractTrueMeasureMixin:
    """
    Lightweight mixin that provides the QMCPy-compatible interface:
        .d              – target dimension
        .domain         – (d, 2) array of [lower, upper] bounds
        .range          – (d, 2) array of output range
        .gen_samples(n) – generate n accepted samples, shape (n, d)
        .__call__(n)    – alias for gen_samples
        ._transform(x)  – raises NotImplementedError (handled by gen_samples)
        ._weight(x)     – returns ones (weights are implicit in filtering)

    In the real QMCPy tree this class inherits from AbstractTrueMeasure.
    The mixin keeps this file self-contained for the notebook demo while
    preserving full compatibility once dropped into qmcpy/true_measure/.
    """

    def _transform(self, x: NDArray) -> NDArray:  # noqa: D401
        raise NotImplementedError(
            f"{self.__class__.__name__} uses acceptance-rejection filtering "
            "and does not support _transform.  Call gen_samples() instead."
        )

    def _weight(self, x: NDArray) -> NDArray:
        return np.ones(x.shape[0])

    def __call__(
        self,
        n: Optional[int] = None,
        n_min: Optional[int] = None,
        n_max: Optional[int] = None,
        return_weights: bool = False,
        warn: bool = True,
    ) -> NDArray:
        N = n if n is not None else (n_max - n_min)
        samples = self.gen_samples(n=N, warn=warn)
        if return_weights:
            return samples, np.ones(len(samples))
        return samples

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__} (TrueMeasure Object)"]
        for k, v in self._repr_params().items():
            lines.append(f"    {k:<24} {v}")
        return "\n".join(lines)

    def _repr_params(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Algorithm 2 – DAR on [0,1]^d
# ---------------------------------------------------------------------------

class AcceptanceRejection(_AbstractTrueMeasureMixin):
    """
    Deterministic Acceptance-Rejection (DAR) sampler on the unit cube.

    Implements **Algorithm 2** of Zhu & Dick (2014).

    Given a non-negative density  ``psi : [0,1]^d -> R_+``  bounded by
    ``L = sup psi(x)`` the algorithm filters a (t,m,s)-net in dimension
    ``s = d + 1`` through the acceptance condition

        L * u  <=  psi(x)

    where  x = (q_1,...,q_d)  and  u = q_{d+1}  are slices of each driver
    point.  Because the driver is a proper (t,m,s)-net (M must equal b^m),
    Theorem 1 of Zhu & Dick guarantees the accepted samples have star
    discrepancy  D*_N = O( N^{-1/s} (log N)^{(s-1)/s} ).

    Parameters
    ----------
    target_density : callable
        ``psi(x)`` where ``x`` is a 1-D numpy array of length ``d``.
        Must be non-negative and bounded by ``upper_bound``.
    discrete_distrib : object with ``.d`` attribute and ``.__call__(n)``
        Any QMCPy ``DiscreteDistribution`` (Sobol, Halton, Lattice …) of
        dimension ``s = d + 1``.  **Must** be a genuine low-discrepancy
        generator — using ``IIDStdUniform`` reduces this to plain
        Monte Carlo and loses all QMC convergence guarantees.
    upper_bound : float
        ``L``, a constant satisfying  ``psi(x) <= L``  for all  x.
    density_integral : float or None
        ``C = integral_{[0,1]^d} psi(x) dx``.  If ``None`` it is estimated
        numerically (only valid for ``d=1``; supply the value for ``d>1``).
    scramble : bool
        Passed to the internal ``scipy.stats.qmc.Sobol`` engine when the
        driver does not expose a ``random_base2`` method directly.  Ignored
        when the driver already provides (t,m,s)-net points.

    Notes
    -----
    The number of driver points used is always the smallest power of 2
    that is at least  ceil(N / (C/L)).  This is required so that the
    generated points form a proper (t,m,s)-net — using an arbitrary M
    would invalidate the discrepancy bound in Theorem 1.

    The driver's dimension must satisfy  ``driver.d == target_dim + 1``.
    """

    def __init__(
        self,
        target_density: Callable,
        discrete_distrib,
        upper_bound: float,
        density_integral: Optional[float] = None,
        scramble: bool = False,
    ) -> None:
        if not hasattr(discrete_distrib, "d"):
            raise ValueError("discrete_distrib must have attribute .d (dimension).")
        self.psi = target_density
        self.driver = discrete_distrib
        self.L = float(upper_bound)
        self.s = discrete_distrib.d          # driver dim
        self.d = self.s - 1                  # target dim
        self.scramble = scramble

        if self.L <= 0:
            raise ValueError("upper_bound L must be strictly positive.")
        if self.d < 1:
            raise ValueError(
                f"driver dimension must be >= 2 (driver.d = {self.s}), "
                "so that target dimension d = s-1 >= 1."
            )

        # C = integral of psi; estimated numerically if not supplied (1-D only)
        if density_integral is not None:
            self.C = float(density_integral)
        elif self.d == 1:
            self.C = _estimate_integral_1d(self.psi)
        else:
            raise ValueError(
                "density_integral must be supplied for d > 1.  "
                "Please compute C = integral_{[0,1]^d} psi(x) dx and pass it in."
            )

        if self.C <= 0 or self.C > self.L:
            raise ValueError(
                f"density_integral C={self.C} must satisfy 0 < C <= L={self.L}."
            )

        # Acceptance rate (Section 2, Zhu & Dick 2014)
        self.acceptance_rate = self.C / self.L

        # QMCPy-style attributes
        self.domain = np.tile([0.0, 1.0], (self.d, 1))
        self.range  = np.tile([0.0, 1.0], (self.d, 1))
        self.parameters = ["psi", "L", "C", "acceptance_rate"]

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    def _get_driver_points(self, M: int) -> NDArray:
        """
        Draw M = 2^m points from the driver in s dimensions.

        Calls the driver the same way QMCPy's TrueMeasure does — via
        ``driver(n=M)`` — but also supports a ``random_base2(m)`` method
        (scipy Sobol) for exact (t,m,s)-net generation.
        """
        m = int(math.log2(M))  # M is already a power of 2
        # Prefer random_base2 if the driver exposes it (real scipy Sobol)
        if hasattr(self.driver, "random_base2"):
            return self.driver.random_base2(m=m)
        # Fallback: QMCPy-style call
        try:
            return self.driver(n=M)
        except TypeError:
            return self.driver(n=M, warn=False)

    def gen_samples(self, n: int, warn: bool = True) -> NDArray:
        """
        Generate ``n`` accepted samples from the target density ``psi``.

        Parameters
        ----------
        n : int
            Desired number of accepted points.
        warn : bool
            If True, warn when fewer than ``n`` points were accepted from
            the driver batch (happens when the initial M estimate is tight).

        Returns
        -------
        samples : ndarray, shape (n, d)
        """
        # Step 1 — choose M = 2^m >= ceil(N / acceptance_rate)
        M_min = int(math.ceil(n / self.acceptance_rate))
        m, M = _next_power_of_2(M_min)

        # Step 2 — generate driver points  Q  of shape (M, s)
        Q = self._get_driver_points(M)          # (M, s)

        # Step 3 — apply DAR filter (vectorised)
        x_cands = Q[:, : self.d]                # (M, d) candidates
        u_check = Q[:, -1]                      # (M,)   check variable

        psi_vals = _vectorised_psi(self.psi, x_cands)   # (M,)
        mask     = self.L * u_check <= psi_vals          # accept if L*u <= psi(x)
        accepted = x_cands[mask]                         # (k, d)

        if len(accepted) < n:
            if warn:
                import warnings
                warnings.warn(
                    f"AcceptanceRejection: only {len(accepted)}/{n} samples accepted "
                    f"from M={M} driver points.  Consider increasing M or checking L.",
                    stacklevel=2,
                )
            # Recurse with a larger M (double)
            extra = self.gen_samples(n - len(accepted), warn=warn)
            return np.vstack([accepted, extra])[:n]

        return accepted[:n]

    def _repr_params(self) -> dict:
        return {
            "d (target dim)":   self.d,
            "s (driver dim)":   self.s,
            "L (upper_bound)":  f"{self.L:.4g}",
            "C (integral)":     f"{self.C:.4g}",
            "acceptance_rate":  f"{self.acceptance_rate:.4g}",
        }


# ---------------------------------------------------------------------------
# Algorithm 3 – DAR-Real on R^d via inverse Rosenblatt transform
# ---------------------------------------------------------------------------

class AcceptanceRejectionReal(_AbstractTrueMeasureMixin):
    """
    DAR sampler for densities on real space  R^d.

    Implements **Algorithm 3** of Zhu & Dick (2014).

    The density ``psi : R^d -> R_+`` is related to the auxiliary function
    ``H : R^d -> R_+`` via the inverse Rosenblatt (marginal-CDF) transform
    (Lemma 4).  The unit-cube driver is mapped to real space through the
    supplied marginal quantile functions ``inv_cdfs``, and acceptance is
    decided by  ``psi(z) >= L * H(z) * v``  where  ``v``  is the last
    driver coordinate.

    Parameters
    ----------
    target_density : callable
        ``psi(z)``  for  ``z`` a 1-D real-valued array of length ``d``.
    inv_cdfs : list of callable
        ``[F_1^{-1}, ..., F_d^{-1}]``  — marginal quantile functions.
        Each  ``F_j^{-1} : [0,1] -> R``  maps a uniform deviate to the
        j-th marginal.  E.g. ``scipy.stats.norm.ppf`` for a Gaussian.
    H_func : callable
        ``H(z)``  — the auxiliary bound function (see Eq. 2 of the paper).
        Must satisfy  ``psi(z) <= L * H(z)``  for all  ``z``.
    discrete_distrib : QMCPy-compatible driver of dimension ``d + 1``.
    upper_bound : float
        ``L``, satisfying  ``psi(z) <= L * H(z)``  for all  z.
    density_integral : float
        ``C = integral_{R^d} psi(z) dz``.
    """

    def __init__(
        self,
        target_density: Callable,
        inv_cdfs: list[Callable],
        H_func: Callable,
        discrete_distrib,
        upper_bound: float,
        density_integral: float,
    ) -> None:
        self.psi      = target_density
        self.inv_cdfs = inv_cdfs
        self.H        = H_func
        self.driver   = discrete_distrib
        self.L        = float(upper_bound)
        self.C        = float(density_integral)
        self.s        = discrete_distrib.d
        self.d        = self.s - 1
        self.acceptance_rate = self.C / self.L

        self.domain = np.full((self.d, 2), [-np.inf, np.inf])
        self.range  = np.full((self.d, 2), [-np.inf, np.inf])
        self.parameters = ["psi", "L", "C", "acceptance_rate"]

    def _get_driver_points(self, M: int) -> NDArray:
        m = int(math.log2(M))
        if hasattr(self.driver, "random_base2"):
            return self.driver.random_base2(m=m)
        try:
            return self.driver(n=M)
        except TypeError:
            return self.driver(n=M, warn=False)

    def gen_samples(self, n: int, warn: bool = True) -> NDArray:
        """
        Generate ``n`` accepted samples in  R^d.

        The Rosenblatt transform  T : [0,1]^s -> R^d x R_+  (Lemma 4) maps
        driver point  (u_1,...,u_d, u_{d+1})  to

            z_j = F_j^{-1}(u_j),   j = 1,...,d
            v   = u_{d+1} * H(z)   (the scaled check variable)

        Acceptance condition (Eq. 6):   psi(z) >= L * v
        """
        M_min = int(math.ceil(n / self.acceptance_rate))
        m, M  = _next_power_of_2(M_min)

        Q = self._get_driver_points(M)   # (M, s)

        U  = Q[:, : self.d]              # (M, d) uniform part
        Us = Q[:, -1]                    # (M,)   check coord

        # Apply marginal quantile transforms  z_j = F_j^{-1}(u_j)
        # Clip to (eps, 1-eps) to avoid -inf/+inf at boundaries (e.g. Normal ppf)
        eps = 1e-8
        U_clipped = np.clip(U, eps, 1 - eps)
        Z_cols = [self.inv_cdfs[j](U_clipped[:, j]) for j in range(self.d)]
        Z = np.column_stack(Z_cols)       # (M, d)

        # Evaluate H and psi row-wise
        H_vals  = np.apply_along_axis(self.H,   1, Z).astype(float)   # (M,)
        psi_vals = np.apply_along_axis(self.psi, 1, Z).astype(float)  # (M,)

        # Acceptance: psi(z) >= L * H(z) * u_{d+1}
        mask     = psi_vals >= self.L * H_vals * Us
        accepted = Z[mask]

        if len(accepted) < n:
            if warn:
                import warnings
                warnings.warn(
                    f"AcceptanceRejectionReal: only {len(accepted)}/{n} accepted.",
                    stacklevel=2,
                )
            extra = self.gen_samples(n - len(accepted), warn=warn)
            return np.vstack([accepted, extra])[:n]

        return accepted[:n]

    def _repr_params(self) -> dict:
        return {
            "d (target dim)":  self.d,
            "s (driver dim)":  self.s,
            "L":               f"{self.L:.4g}",
            "C":               f"{self.C:.4g}",
            "acceptance_rate": f"{self.acceptance_rate:.4g}",
        }


# ---------------------------------------------------------------------------
# Algorithm 4 – DRAR (Reduced / hybrid A-R on [0,1]^d)
# ---------------------------------------------------------------------------

class ReducedAcceptanceRejection(_AbstractTrueMeasureMixin):
    """
    Deterministic Reduced Acceptance-Rejection (DRAR) sampler.

    Implements **Algorithm 4** of Zhu & Dick (2014).

    The target density  ``psi = H_1 + (psi - H_1)``  is split into:

    * Region  **L** = {x : psi(x) >= H_1(x)} — here  psi - H_1 >= 0, so both
      H_1  and  psi - H_1  can be sampled by *direct inversion*, giving
      discrepancy  O((log N)^{d-1} / N).

    * Region  **S** = {x : psi(x) < H_1(x)} — here we use standard DAR
      (Algorithm 2), giving discrepancy  O(N^{-1/s}).

    The three sub-samplers (R_{1,1}, R_{1,2}, R_{2,2}) are combined
    proportionally to their masses.

    Parameters
    ----------
    target_density : callable
        ``psi(x)``  on  ``[0,1]^d``.
    H1_func : callable
        ``H_1(x)``  — the splitting function.  Must satisfy  ``H_1(x) >= 0``
        and  ``H_1(x) >= psi(x)``  somewhere (otherwise all mass is in L
        and plain inversion suffices).
    inv_cdf_H1_S : callable
        Quantile function  F^{-1}  for  H_1  restricted to region S,
        mapping  [0,1] -> [0,1].
    inv_cdf_psi_L : callable
        Quantile function for  (psi - H_1)  on region L.
    inv_cdf_H1_L : callable
        Quantile function for  H_1  on region L.
    mass_S : float
        ``integral_{S} H_1(x) dx``  — total mass of H_1 in region S.
    mass_L_H1 : float
        ``integral_{L} H_1(x) dx``  — mass of H_1 in region L.
    mass_L_psi : float
        ``integral_{L} (psi(x)-H_1(x)) dx``  — mass of  psi-H_1 in region L.
    discrete_distrib : driver with  ``.d == d + 1``  (for R_{1,1} DAR part).
    sobol_1d : driver with  ``.d == 1``  (for direct inversion parts).
    upper_bound : float
        ``L = sup_{x in S} H_1(x) / (something)``; see paper Eq.(11).
    """

    def __init__(
        self,
        target_density: Callable,
        H1_func: Callable,
        inv_cdf_H1_S: Callable,
        inv_cdf_psi_L: Callable,
        inv_cdf_H1_L: Callable,
        mass_S: float,
        mass_L_H1: float,
        mass_L_psi: float,
        discrete_distrib,           # s-dim driver for A-R part
        sobol_1d,                   # 1-dim driver for inversion parts
        upper_bound: float,
    ) -> None:
        self.psi          = target_density
        self.H1           = H1_func
        self.Finv_H1_S    = inv_cdf_H1_S
        self.Finv_psi_L   = inv_cdf_psi_L
        self.Finv_H1_L    = inv_cdf_H1_L
        self.mass_S       = float(mass_S)
        self.mass_L_H1    = float(mass_L_H1)
        self.mass_L_psi   = float(mass_L_psi)
        self.C_total      = mass_S + mass_L_H1 + mass_L_psi
        self.driver_ar    = discrete_distrib   # dim s = d+1
        self.driver_inv   = sobol_1d           # dim 1
        self.L            = float(upper_bound)
        self.s            = discrete_distrib.d
        self.d            = self.s - 1

        # Proportional split
        self.frac_S       = self.mass_S    / self.C_total
        self.frac_L_H1    = self.mass_L_H1 / self.C_total
        self.frac_L_psi   = self.mass_L_psi / self.C_total

        self.domain = np.tile([0.0, 1.0], (self.d, 1))
        self.range  = np.tile([0.0, 1.0], (self.d, 1))
        self.parameters = ["psi", "L", "C_total", "frac_S", "frac_L_H1"]

    def _get_ar_points(self, M: int) -> NDArray:
        m = int(math.log2(M))
        if hasattr(self.driver_ar, "random_base2"):
            return self.driver_ar.random_base2(m=m)
        return self.driver_ar(n=M)

    def _get_inv_points(self, M: int) -> NDArray:
        m = int(math.log2(M))
        if hasattr(self.driver_inv, "random_base2"):
            return self.driver_inv.random_base2(m=m).ravel()
        return self.driver_inv(n=M).ravel()

    def gen_samples(self, n: int, warn: bool = True) -> NDArray:
        """
        Generate ``n`` samples by combining three sub-samplers.

        R_{1,1} — DAR on region S (acceptance-rejection)
        R_{1,2} — direct inversion of H_1 on region L
        R_{2,2} — direct inversion of (psi - H_1) on region L
        """
        N_S    = max(1, int(math.ceil(n * self.frac_S)))
        N_L_H1 = max(1, int(math.ceil(n * self.frac_L_H1)))
        N_L_p  = max(1, int(math.ceil(n * self.frac_L_psi)))

        results = []

        # --- R_{1,1}: DAR in region S ---
        acc_rate_S = self.mass_S / (self.L * 1.0)   # acceptance rate in S
        if acc_rate_S <= 0:
            acc_rate_S = 0.5
        M_min_S = int(math.ceil(N_S / max(acc_rate_S, 1e-6)))
        m_S, M_S = _next_power_of_2(M_min_S)
        Q_S = self._get_ar_points(M_S)
        x_cands = Q_S[:, : self.d]
        u_check = Q_S[:, -1]
        H1_vals  = np.apply_along_axis(self.H1, 1, x_cands).astype(float)
        psi_vals = np.apply_along_axis(self.psi, 1, x_cands).astype(float)
        in_S = psi_vals < H1_vals
        accept_S = in_S & (self.L * u_check <= H1_vals)
        R11 = x_cands[accept_S][:N_S]
        results.append(R11)

        # --- R_{1,2}: direct inversion of H_1 on region L ---
        _, M_L = _next_power_of_2(N_L_H1)
        u_L = self._get_inv_points(M_L)[:N_L_H1]
        R12 = self.Finv_H1_L(u_L).reshape(-1, self.d)
        results.append(R12)

        # --- R_{2,2}: direct inversion of (psi - H_1) on region L ---
        _, M_Lp = _next_power_of_2(N_L_p)
        u_Lp = self._get_inv_points(M_Lp)[:N_L_p]
        R22 = self.Finv_psi_L(u_Lp).reshape(-1, self.d)
        results.append(R22)

        combined = np.vstack([r for r in results if len(r) > 0])
        np.random.shuffle(combined)   # mix the three sub-streams
        return combined[:n]

    def _repr_params(self) -> dict:
        return {
            "d (target dim)":  self.d,
            "s (driver dim)":  self.s,
            "L":               f"{self.L:.4g}",
            "C_total":         f"{self.C_total:.4g}",
            "frac_S":          f"{self.frac_S:.3f}",
            "frac_L_H1":       f"{self.frac_L_H1:.3f}",
            "frac_L_psi":      f"{self.frac_L_psi:.3f}",
        }
