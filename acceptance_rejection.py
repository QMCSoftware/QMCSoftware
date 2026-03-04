"""
AcceptanceRejection True Measure for QMCPy
==========================================

Implements the Deterministic Acceptance-Rejection (DAR) sampler from:

    Zhu, H., & Dick, J. (2014). Discrepancy bounds for deterministic
    acceptance-rejection samplers. Electronic Journal of Statistics,
    8(1), 678-707.  DOI: 10.1214/14-EJS898

Three classes are provided:

    AcceptanceRejection        Algorithm 2 (DAR)      — unit cube [0,1]^d
    AcceptanceRejectionReal    Algorithm 3 (DAR-Real) — real space R^d
    ReducedAcceptanceRejection Algorithm 4 (DRAR)     — hybrid inversion + A-R

Density convention (FIX #1)
---------------------------
All callables ``psi``, ``H``, ``H1`` must accept a **batched** array of
shape ``(N, d)`` and return shape ``(N,)``.  This is consistent with
QMCPy's standard integrand convention and avoids the slow
``np.apply_along_axis`` row-by-row evaluation used in the original code.

    def my_density(x):          # x shape: (N, d)
        return 2.0 * x[:, 0]   # returns shape: (N,)

Placing in QMCPy
----------------
Drop this file into ``qmcpy/true_measure/acceptance_rejection.py`` and
replace the ``_AbstractTrueMeasureMixin`` base class with the real one::

    from .abstract_true_measure import AbstractTrueMeasure

Then change each class declaration to::

    class AcceptanceRejection(AbstractTrueMeasure):
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> tuple[int, int]:
    """Return (m, 2**m) where 2**m >= n.

    The (t,m,s)-net guarantee requires exactly M = 2^m driver points.
    Using any other M invalidates the discrepancy bound of Theorem 1.
    """
    m = max(int(math.ceil(math.log2(max(n, 1)))), 1)
    return m, 2 ** m


# ---------------------------------------------------------------------------
# Standalone base  (swap for AbstractTrueMeasure when inside QMCPy)
# ---------------------------------------------------------------------------

class _AbstractTrueMeasureMixin:
    """
    Stand-in for AbstractTrueMeasure that keeps this file self-contained.

    Provides the QMCPy TrueMeasure interface so the classes work both
    standalone (notebook, testing) and inside the QMCPy package tree.

    When placing in QMCPy, replace with::

        from .abstract_true_measure import AbstractTrueMeasure

    and inherit from ``AbstractTrueMeasure`` instead of this mixin.

    Interface
    ---------
    .d              target dimension
    .domain         (d, 2) ndarray  [lower, upper] per input dimension
    .range          (d, 2) ndarray  [lower, upper] per output dimension
    .gen_samples(n) generate n accepted samples, returns ndarray (n, d)
    .__call__(n)    alias for gen_samples
    ._transform(x)  raises NotImplementedError — A-R is variable-output
                    and cannot be expressed as a 1-to-1 transform
    ._weight(x)     returns psi(x) / C — normalised density, used as
                    importance-sampling weight (FIX #7)
    """

    def _transform(self, x: NDArray) -> NDArray:
        raise NotImplementedError(
            f"{self.__class__.__name__} uses acceptance-rejection filtering "
            "and cannot be expressed as a 1-to-1 transform. "
            "Call gen_samples(n) directly."
        )

    def _weight(self, x: NDArray) -> NDArray:
        """Importance weight w(x) = psi(x) / C at each point."""
        # FIX #7: was ``return np.ones(x.shape[0])``, which is wrong.
        # The correct weight is the normalised target density.
        return self.psi(x) / self.C

    def __call__(
        self,
        n: Optional[int] = None,
        n_min: Optional[int] = None,
        n_max: Optional[int] = None,
        return_weights: bool = False,
        warn: bool = True,
    ):
        N = n if n is not None else (n_max - n_min)
        samples = self.gen_samples(n=N, warn=warn)
        if return_weights:
            return samples, self._weight(samples)
        return samples

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__} (TrueMeasure Object)"]
        for k, v in self._repr_params().items():
            lines.append(f"    {k:<28} {v}")
        return "\n".join(lines)

    def _repr_params(self) -> dict:
        return {}


# ---------------------------------------------------------------------------
# Shared driver helper  (used by all three classes)
# ---------------------------------------------------------------------------

def _get_driver_points(driver, M: int) -> NDArray:
    """Draw exactly M = 2^m points from *driver* in its native dimension.

    Prefers ``random_base2(m)`` when available (scipy Sobol) because
    that guarantees exactly 2^m net points from a fresh sequence.
    Falls back to the QMCPy ``driver(n=M)`` calling convention.
    """
    # FIX #10: use round() not int() so floating-point near-integers
    # (e.g. log2(1024) = 9.9999...) are handled correctly.
    m = int(round(math.log2(M)))
    if hasattr(driver, "random_base2"):
        return driver.random_base2(m=m)
    try:
        return driver(n=M)
    except TypeError:
        return driver(n=M, warn=False)


# ---------------------------------------------------------------------------
# Algorithm 2 — DAR on [0,1]^d
# ---------------------------------------------------------------------------

class AcceptanceRejection(_AbstractTrueMeasureMixin):
    """
    Deterministic Acceptance-Rejection (DAR) sampler on the unit cube.

    Implements **Algorithm 2** of Zhu & Dick (2014).

    Given a non-negative density ``psi : [0,1]^d -> R_+`` bounded above
    by ``L = sup psi(x)``, the algorithm filters a (t,m,s)-net in driver
    dimension ``s = d + 1`` through the acceptance condition::

        psi(x) >= L * u

    where ``x = q_{1:d}`` are the first d coordinates of each driver
    point and ``u = q_{d+1}`` is the last.  Theorem 1 guarantees::

        D*_N(psi) = O(N^{-1/s})

    Parameters
    ----------
    target_density : callable
        ``psi(x)`` where ``x`` has shape ``(N, d)``, returns ``(N,)``.
        Must be non-negative on ``[0,1]^d`` and bounded by ``upper_bound``.
    discrete_distrib : driver with ``.d`` and ``.__call__(n)``
        Any QMCPy ``DiscreteDistribution`` of dimension ``s = d + 1``.
        Using an IID driver reduces this to standard random A-R.
    upper_bound : float
        ``L`` satisfying ``psi(x) <= L`` for all ``x`` in ``[0,1]^d``.
    density_integral : float
        ``C = integral_{[0,1]^d} psi(x) dx``.
    max_retries : int
        Maximum number of times ``gen_samples`` doubles the driver
        before giving up.  Default 4 (up to 16x the initial M).

    Notes
    -----
    Driver points count is always the smallest power of 2 satisfying
    ``M >= ceil(N / (C/L))``.  This is required for the (t,m,s)-net
    property and the discrepancy bound of Theorem 1.

    References
    ----------
    Zhu & Dick (2014), EJS 8(1), 678-707.  DOI: 10.1214/14-EJS898
    """

    def __init__(
        self,
        target_density: Callable,
        discrete_distrib,
        upper_bound: float,
        density_integral: float,
        max_retries: int = 4,
    ) -> None:
        if not hasattr(discrete_distrib, "d"):
            raise ValueError("discrete_distrib must expose a .d attribute (driver dimension).")

        self.psi         = target_density
        self.driver      = discrete_distrib
        self.L           = float(upper_bound)
        self.C           = float(density_integral)
        self.s           = int(discrete_distrib.d)
        self.d           = self.s - 1
        self.max_retries = int(max_retries)

        if self.L <= 0:
            raise ValueError(f"upper_bound L must be strictly positive, got {self.L}.")
        if self.d < 1:
            raise ValueError(
                f"driver.d = {self.s} implies target dimension d = {self.d} < 1. "
                "Driver dimension must be >= 2."
            )
        if self.C <= 0:
            raise ValueError(f"density_integral C must be strictly positive, got {self.C}.")

        # FIX #8: C > L is not mathematically impossible (it can happen in
        # degenerate cases) but strongly indicates the user has swapped C and L,
        # so warn rather than silently producing wrong results.  The original
        # hard error ``C > L`` was also wrong because it would fire on valid
        # inputs when d > 1 and psi is broad.
        if self.C > self.L:
            warnings.warn(
                f"density_integral C={self.C:.4g} > upper_bound L={self.L:.4g}. "
                "For a density on [0,1]^d this is unusual — "
                "check that C is the integral of psi and L is the supremum.",
                UserWarning,
                stacklevel=2,
            )

        self.acceptance_rate = self.C / self.L

        self.domain     = np.tile([0.0, 1.0], (self.d, 1))
        self.range      = np.tile([0.0, 1.0], (self.d, 1))
        self.parameters = ["psi", "L", "C", "acceptance_rate", "d"]

    def gen_samples(self, n: int, warn: bool = True) -> NDArray:
        """
        Generate ``n`` accepted samples from ``psi``.

        Parameters
        ----------
        n : int
            Desired number of accepted points.
        warn : bool
            Emit a RuntimeWarning when fewer than ``n`` samples are
            returned after all retries are exhausted.

        Returns
        -------
        samples : ndarray, shape (n, d)
        """
        # Step i: M = 2^m >= ceil(n / acceptance_rate)
        M_min    = int(math.ceil(n / max(self.acceptance_rate, 1e-12)))
        m_use, M = _next_power_of_2(M_min)

        accepted_batches: list[NDArray] = []
        n_collected = 0
        n_driver    = 0

        # FIX #3: retry loop instead of recursion.
        # Each iteration generates a fresh, larger (t,m,s)-net so the
        # net property is always preserved within each batch.
        # Recursion (original code) concatenated points from two separate
        # nets starting at index 0, breaking the net structure entirely.
        for _ in range(1 + self.max_retries):
            Q = _get_driver_points(self.driver, M)   # (M, s)

            # Step ii: acceptance condition  psi(x) >= L * u
            x_cands  = Q[:, :self.d]                 # (M, d)
            u_check  = Q[:, -1]                      # (M,)
            # FIX #1: direct batched call instead of np.apply_along_axis
            psi_vals = self.psi(x_cands)             # (M,)
            mask     = psi_vals >= self.L * u_check
            batch    = x_cands[mask]

            n_driver    += M
            n_collected += len(batch)
            accepted_batches.append(batch)

            if n_collected >= n:
                break

            m_use += 1
            M      = 2 ** m_use

        # Step iii: project (already done — x_cands is the projection)
        accepted = np.concatenate(accepted_batches, axis=0)
        if len(accepted) > n:
            accepted = accepted[:n]

        if warn and len(accepted) < n:
            warnings.warn(
                f"AcceptanceRejection.gen_samples: requested {n} samples "
                f"but only {len(accepted)} accepted after {n_driver} driver "
                f"points ({1 + self.max_retries} attempts). "
                f"Increase max_retries or verify upper_bound L >= sup psi.",
                RuntimeWarning,
                stacklevel=2,
            )

        return accepted

    def _repr_params(self) -> dict:
        return {
            "d (target dim)":  self.d,
            "s (driver dim)":  self.s,
            "L (upper bound)": f"{self.L:.4g}",
            "C (integral)":    f"{self.C:.4g}",
            "acceptance_rate": f"{self.acceptance_rate:.4g}",
        }


# ---------------------------------------------------------------------------
# Algorithm 3 — DAR-Real on R^d via inverse Rosenblatt transform
# ---------------------------------------------------------------------------

class AcceptanceRejectionReal(_AbstractTrueMeasureMixin):
    """
    DAR sampler for densities on real space R^d.

    Implements **Algorithm 3** of Zhu & Dick (2014).

    The unit-cube driver is mapped to R^d through the supplied marginal
    quantile functions (inverse Rosenblatt transform, Lemma 4).
    Acceptance is decided by::

        psi(z) >= L * H(z) * u_{d+1}

    where ``z_j = F_j^{-1}(u_j)`` and ``u_{d+1}`` is the last driver
    coordinate.

    Restriction
    -----------
    This implementation applies the quantile functions **independently
    per dimension** (product-form Rosenblatt transform), which is exact
    when H factors as a product over its marginals (e.g. a product of
    univariate distributions).  For a non-product H the user must apply
    the joint Rosenblatt transform themselves before calling this class.

    Parameters
    ----------
    target_density : callable
        ``psi(z)`` where ``z`` has shape ``(N, d)``, returns ``(N,)``.
        Must satisfy ``psi(z) <= L * H(z)`` for all ``z``.
    inv_cdfs : list of callable
        ``[F_1^{-1}, ..., F_d^{-1}]`` — marginal quantile functions of H.
        Each maps a 1-D array of uniforms in ``[0, 1]`` to R.
        Example: ``[scipy.stats.norm.ppf]`` for a 1-D standard Gaussian.
    H_func : callable
        ``H(z)`` where ``z`` has shape ``(N, d)``, returns ``(N,)``.
        Auxiliary bound function satisfying ``psi(z) <= L * H(z)``.
    discrete_distrib : driver of dimension ``d + 1``.
    upper_bound : float
        ``L`` satisfying ``psi(z) <= L * H(z)`` for all ``z``.
    density_integral : float
        ``C = integral_{R^d} psi(z) dz``.
    max_retries : int
        Maximum retry doublings.  Default 4.

    References
    ----------
    Zhu & Dick (2014), EJS 8(1), 678-707.  DOI: 10.1214/14-EJS898
    """

    def __init__(
        self,
        target_density: Callable,
        inv_cdfs: list[Callable],
        H_func: Callable,
        discrete_distrib,
        upper_bound: float,
        density_integral: float,
        max_retries: int = 4,
    ) -> None:
        if not hasattr(discrete_distrib, "d"):
            raise ValueError("discrete_distrib must expose a .d attribute.")
        if len(inv_cdfs) != discrete_distrib.d - 1:
            raise ValueError(
                f"len(inv_cdfs) = {len(inv_cdfs)} must equal "
                f"discrete_distrib.d - 1 = {discrete_distrib.d - 1}."
            )

        self.psi         = target_density
        self.inv_cdfs    = inv_cdfs
        self.H           = H_func
        self.driver      = discrete_distrib
        self.L           = float(upper_bound)
        self.C           = float(density_integral)
        self.s           = int(discrete_distrib.d)
        self.d           = self.s - 1
        self.max_retries = int(max_retries)

        if self.L <= 0:
            raise ValueError(f"upper_bound L must be strictly positive, got {self.L}.")
        if self.C <= 0:
            raise ValueError(f"density_integral C must be strictly positive, got {self.C}.")

        self.acceptance_rate = self.C / self.L

        self.domain     = np.full((self.d, 2), [-np.inf, np.inf])
        self.range      = np.full((self.d, 2), [-np.inf, np.inf])
        self.parameters = ["psi", "L", "C", "acceptance_rate", "d"]

    def gen_samples(self, n: int, warn: bool = True) -> NDArray:
        """
        Generate ``n`` accepted samples in R^d.

        The Rosenblatt transform T : [0,1]^s -> R^d x R_+ (Lemma 4)
        maps driver point (u_1,...,u_d, u_{d+1}) to::

            z_j = F_j^{-1}(u_j),   j = 1,...,d
            v   = u_{d+1} * H(z)

        Acceptance condition:  psi(z) >= L * v

        Parameters
        ----------
        n : int
        warn : bool

        Returns
        -------
        samples : ndarray, shape (n, d)
        """
        M_min    = int(math.ceil(n / max(self.acceptance_rate, 1e-12)))
        m_use, M = _next_power_of_2(M_min)

        accepted_batches: list[NDArray] = []
        n_collected = 0
        n_driver    = 0

        # FIX #3: retry loop instead of recursion
        for _ in range(1 + self.max_retries):
            Q  = _get_driver_points(self.driver, M)  # (M, s)
            U  = Q[:, :self.d]                       # (M, d) uniform coords
            Us = Q[:, -1]                            # (M,)   check coord

            # Apply marginal quantile transforms column-wise (vectorised).
            # Clip to (eps, 1-eps) to avoid ±inf at boundaries.
            # FIX #1: each inv_cdf receives a 1-D array, which is the
            # correct convention for scalar quantile functions.
            eps       = 1e-8
            U_clipped = np.clip(U, eps, 1.0 - eps)
            Z_cols    = [self.inv_cdfs[j](U_clipped[:, j]) for j in range(self.d)]
            Z         = np.column_stack(Z_cols)      # (M, d)

            # FIX #1: direct batched calls to H and psi
            H_vals   = self.H(Z)                     # (M,)
            psi_vals = self.psi(Z)                   # (M,)

            # Acceptance: psi(z) >= L * H(z) * u_{d+1}
            mask  = psi_vals >= self.L * H_vals * Us
            batch = Z[mask]

            n_driver    += M
            n_collected += len(batch)
            accepted_batches.append(batch)

            if n_collected >= n:
                break

            m_use += 1
            M      = 2 ** m_use

        accepted = np.concatenate(accepted_batches, axis=0)
        if len(accepted) > n:
            accepted = accepted[:n]

        if warn and len(accepted) < n:
            warnings.warn(
                f"AcceptanceRejectionReal.gen_samples: requested {n} samples "
                f"but only {len(accepted)} accepted after {n_driver} driver "
                f"points. Verify upper_bound L satisfies psi(z) <= L*H(z) everywhere.",
                RuntimeWarning,
                stacklevel=2,
            )

        return accepted

    def _repr_params(self) -> dict:
        return {
            "d (target dim)":  self.d,
            "s (driver dim)":  self.s,
            "L (upper bound)": f"{self.L:.4g}",
            "C (integral)":    f"{self.C:.4g}",
            "acceptance_rate": f"{self.acceptance_rate:.4g}",
        }


# ---------------------------------------------------------------------------
# Algorithm 4 — DRAR (Reduced / hybrid A-R on [0,1]^d)
# ---------------------------------------------------------------------------

class ReducedAcceptanceRejection(_AbstractTrueMeasureMixin):
    """
    Deterministic Reduced Acceptance-Rejection (DRAR) sampler.

    Implements **Algorithm 4** of Zhu & Dick (2014).

    The density is decomposed as ``psi = H_1 + (psi - H_1)``:

    * **Region L** = {x : psi(x) >= H_1(x)} — psi - H_1 >= 0 here, so
      both H_1 and psi - H_1 are sampled by **direct inversion**,
      giving discrepancy O((log N)^{d-1} / N).

    * **Region S** = {x : psi(x) < H_1(x)} — standard DAR (Algorithm 2)
      is applied here, giving discrepancy O(N^{-1/s}).

    Three sub-samplers combined proportionally to their masses:

    * R_{1,1} — DAR on region S
    * R_{1,2} — direct inversion of H_1 on region L
    * R_{2,2} — direct inversion of (psi - H_1) on region L

    Parameters
    ----------
    target_density : callable
        ``psi(x)`` shape ``(N, d)`` -> ``(N,)``.
    H1_func : callable
        ``H_1(x)`` shape ``(N, d)`` -> ``(N,)``.  The splitting function.
    inv_cdf_H1_S : callable
        Quantile function for H_1 restricted to region S.
        Maps 1-D uniform array to points, reshaped to ``(N, d)``.
    inv_cdf_psi_L : callable
        Quantile function for ``(psi - H_1)`` on region L.
    inv_cdf_H1_L : callable
        Quantile function for H_1 on region L.
    mass_S : float
        ``integral_{S} psi(x) dx`` — total mass of psi in region S.
    mass_L_H1 : float
        ``integral_{L} H_1(x) dx`` — mass of H_1 in region L.
    mass_L_psi : float
        ``integral_{L} (psi(x) - H_1(x)) dx`` — mass of psi - H_1 in L.
    discrete_distrib : driver of dimension ``d + 1`` — for R_{1,1}.
    sobol_1d_H1 : driver of dimension 1 — for R_{1,2} direct inversion.
    sobol_1d_psi : driver of dimension 1 — for R_{2,2} direct inversion.
        **Must be a separate driver from sobol_1d_H1** so R_{1,2} and
        R_{2,2} use independent uniform sequences.  (FIX #12)
    upper_bound : float
        ``L`` satisfying ``psi(x) <= L`` for all x in region S.
    max_retries : int
        Maximum retry doublings for the R_{1,1} DAR step.  Default 4.

    References
    ----------
    Zhu & Dick (2014), EJS 8(1), 678-707.  DOI: 10.1214/14-EJS898
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
        discrete_distrib,
        sobol_1d_H1,           # independent driver for R_{1,2}
        sobol_1d_psi,          # independent driver for R_{2,2}  (FIX #12)
        upper_bound: float,
        max_retries: int = 4,
    ) -> None:
        self.psi         = target_density
        self.H1          = H1_func
        self.Finv_H1_S   = inv_cdf_H1_S
        self.Finv_psi_L  = inv_cdf_psi_L
        self.Finv_H1_L   = inv_cdf_H1_L
        self.mass_S      = float(mass_S)
        self.mass_L_H1   = float(mass_L_H1)
        self.mass_L_psi  = float(mass_L_psi)
        self.C_total     = self.mass_S + self.mass_L_H1 + self.mass_L_psi
        self.driver_ar   = discrete_distrib
        self.driver_H1   = sobol_1d_H1
        self.driver_psi  = sobol_1d_psi
        self.L           = float(upper_bound)
        self.s           = int(discrete_distrib.d)
        self.d           = self.s - 1
        self.max_retries = int(max_retries)

        if self.C_total <= 0:
            raise ValueError("Sum of masses (C_total) must be strictly positive.")
        if self.L <= 0:
            raise ValueError("upper_bound L must be strictly positive.")

        self.frac_S      = self.mass_S    / self.C_total
        self.frac_L_H1   = self.mass_L_H1 / self.C_total
        self.frac_L_psi  = self.mass_L_psi / self.C_total

        # _weight() uses self.C and self.psi (from _AbstractTrueMeasureMixin)
        self.C           = self.C_total

        self.domain      = np.tile([0.0, 1.0], (self.d, 1))
        self.range       = np.tile([0.0, 1.0], (self.d, 1))
        self.parameters  = ["psi", "L", "C_total", "frac_S", "frac_L_H1", "frac_L_psi"]

    def gen_samples(self, n: int, warn: bool = True) -> NDArray:
        """
        Generate ``n`` samples by combining three sub-samplers.

        R_{1,1} — DAR (Algorithm 2) in region S
        R_{1,2} — direct inversion of H_1 in region L
        R_{2,2} — direct inversion of (psi - H_1) in region L

        Sub-streams are concatenated in order.  They are **not shuffled**:
        shuffling with a random seed would destroy the deterministic
        low-discrepancy structure.  (FIX #4)

        Parameters
        ----------
        n : int
        warn : bool

        Returns
        -------
        samples : ndarray, shape (n, d)
        """
        N_S    = max(1, int(math.ceil(n * self.frac_S)))
        N_L_H1 = max(1, int(math.ceil(n * self.frac_L_H1)))
        N_L_p  = max(1, int(math.ceil(n * self.frac_L_psi)))

        # ---- R_{1,1}: DAR in region S --------------------------------
        # Acceptance condition is psi(x) >= L * u  (FIX #2 — the original
        # code used H1_vals here, which is wrong; the standard DAR accept
        # criterion is psi(x) >= L * u regardless of region).
        acc_rate_S = max(self.mass_S / self.L, 1e-12)
        M_min_S    = int(math.ceil(N_S / acc_rate_S))
        m_S, M_S   = _next_power_of_2(M_min_S)
        R11_batches: list[NDArray] = []
        n_R11 = 0

        # FIX #3: retry loop (no recursion)
        for _ in range(1 + self.max_retries):
            Q_S      = _get_driver_points(self.driver_ar, M_S)
            x_cands  = Q_S[:, :self.d]               # (M_S, d)
            u_check  = Q_S[:, -1]                    # (M_S,)
            # FIX #1: batched calls
            H1_vals  = self.H1(x_cands)              # (M_S,)
            psi_vals = self.psi(x_cands)             # (M_S,)
            in_S     = psi_vals < H1_vals            # restrict to region S
            # FIX #2: correct acceptance condition — psi(x) >= L * u
            accept_S = in_S & (psi_vals >= self.L * u_check)
            batch    = x_cands[accept_S]
            R11_batches.append(batch)
            n_R11 += len(batch)
            if n_R11 >= N_S:
                break
            m_S += 1
            M_S  = 2 ** m_S

        R11 = np.concatenate(R11_batches, axis=0)[:N_S] if R11_batches \
              else np.empty((0, self.d))

        # ---- R_{1,2}: direct inversion of H_1 on region L -----------
        # FIX #12: uses self.driver_H1 — independent from self.driver_psi
        _, M_L = _next_power_of_2(N_L_H1)
        u_L    = _get_driver_points(self.driver_H1, M_L).ravel()[:N_L_H1]
        R12    = self.Finv_H1_L(u_L).reshape(-1, self.d)

        # ---- R_{2,2}: direct inversion of (psi - H_1) on region L ---
        # FIX #12: uses self.driver_psi — independent from self.driver_H1
        _, M_Lp = _next_power_of_2(N_L_p)
        u_Lp    = _get_driver_points(self.driver_psi, M_Lp).ravel()[:N_L_p]
        R22     = self.Finv_psi_L(u_Lp).reshape(-1, self.d)

        # ---- Combine sub-streams in order (no random shuffle) --------
        # FIX #4: removed np.random.shuffle which destroyed the
        # deterministic QMC structure.
        parts    = [r for r in [R11, R12, R22] if len(r) > 0]
        combined = np.concatenate(parts, axis=0)

        if len(combined) > n:
            combined = combined[:n]

        if warn and len(combined) < n:
            warnings.warn(
                f"ReducedAcceptanceRejection.gen_samples: requested {n} "
                f"samples but only {len(combined)} produced. "
                f"Check that mass_S + mass_L_H1 + mass_L_psi = C_total and "
                f"upper_bound L >= sup psi on region S.",
                RuntimeWarning,
                stacklevel=2,
            )

        return combined

    def _repr_params(self) -> dict:
        return {
            "d (target dim)":  self.d,
            "s (driver dim)":  self.s,
            "L (upper bound)": f"{self.L:.4g}",
            "C_total":         f"{self.C_total:.4g}",
            "frac_S":          f"{self.frac_S:.4f}",
            "frac_L_H1":       f"{self.frac_L_H1:.4f}",
            "frac_L_psi":      f"{self.frac_L_psi:.4f}",
        }
