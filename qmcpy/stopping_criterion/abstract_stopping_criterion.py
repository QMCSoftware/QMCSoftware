import copy
import sys

from .diagnostics import _IterationTraceLogger, print_diagnostic  # noqa: F401
from ..integrand.abstract_integrand import AbstractIntegrand
from ..util import (
    DistributionCompatibilityError,
    ParameterError,
    MethodImplementationError,
    _univ_repr,
)
import numpy as np


# Optional diagnostic hook for resume-aware stopping criteria.
IS_PRINT_DIAGNOSTIC = False


class AbstractStoppingCriterion(object):
    _RESUME_FORMAT_VERSION = 1

    def __init__(self, allowed_distribs, allow_vectorized_integrals):
        """Initialize a stopping criterion base class.

        Args:
            allowed_distribs (list): Allowed discrete distribution classes.
            allow_vectorized_integrals (bool): Whether to allow integrands with
                vectorized outputs, meaning ``integrand.d_indv != ()``.
        """
        sname = type(self).__name__
        prefix = "A concrete implementation of StoppingCriterion must have "
        # integrand check
        if (not hasattr(self, "integrand")) or (
            not isinstance(self.integrand, AbstractIntegrand)
        ):
            raise ParameterError(prefix + "self.integrand, an Integrand instance")
        # true measure check
        if (not hasattr(self, "true_measure")) or (
            self.true_measure != self.integrand.true_measure
        ):
            raise ParameterError(
                prefix + "self.true_measure=self.integrand.true_measure"
            )
        # discrete distribution check
        if (not hasattr(self, "discrete_distrib")) or (
            self.discrete_distrib != self.integrand.discrete_distrib
        ):
            raise ParameterError(
                prefix + "self.discrete_distrib=self.integrand.discrete_distrib"
            )
        if not isinstance(self.discrete_distrib, tuple(allowed_distribs)):
            raise DistributionCompatibilityError(
                "%s must have a DiscreteDistribution in %s"
                % (sname, str(allowed_distribs))
            )
        if (not allow_vectorized_integrals) and len(self.integrand.d_indv) > 0:
            raise ParameterError(
                "Vectorized integrals (with d_indv!=() outputs per sample) are not supported by this stopping criterion"
            )
        # parameter checks
        if not hasattr(self, "parameters"):
            self.parameters = []

    def integrate(self, resume=None):
        """Determine the samples needed to satisfy the target tolerance.

        Args:
            resume (Data, optional): Existing integration state to resume from,
                if supported. A valid resume checkpoint must continue the same
                numerical experiment without duplicating samples, losing
                accumulated statistics, or weakening the requested tolerance
                guarantee. Supported resume implementations validate and copy
                the checkpoint before mutating any state so the caller's saved
                object is preserved. Defaults to None.

        Returns:
            tuple[Union[float, np.ndarray], Data]: Approximation to the integral
                with shape ``integrand.d_comb`` and the corresponding data
                object.
        """
        raise MethodImplementationError(self, "integrate")

    def _make_trace_logger(self):
        """Create an iteration trace logger for this stopping criterion.

        Returns:
            _IterationTraceLogger: Trace logger configured from the stopping
                criterion's optional trace attributes.
        """
        return _IterationTraceLogger(self)

    def _prepare_resume_data(self, resume, validate_resume, restore_resume):
        """Validate and restore a resume checkpoint before integration.

        Args:
            resume (Data or None): Resume checkpoint passed to ``integrate``.
            validate_resume (callable): Validator taking ``resume`` and raising
                on incompatible state.
            restore_resume (callable): Restorer taking ``resume`` and mutating
                the current stopping criterion into a compatible resumed state.

        Returns:
            Data or None: A validated deep copy of the supplied checkpoint, or
            None when no checkpoint was supplied.
        """
        if resume is None:
            return None
        validate_resume(resume)
        data = copy.deepcopy(resume)
        restore_resume(data)
        return data

    def _restore_resume_state(self, data):
        """Optional hook for subclasses to align state before resuming.

        Subclasses that need to restore RNG state or rewrite checkpoint fields
        may override this method. The default implementation is a no-op.

        Args:
            data (Data): Deep-copied resume checkpoint that will be mutated by
                the resumed integration run.
        """
        return None

    def _capture_resume_provenance(self, resume):
        """Capture resume bookkeeping before the live ``Data`` object is mutated.

        Args:
            resume (Data or None): Resume checkpoint passed to ``integrate``.

        Returns:
            dict or None: Previous sample/time totals, or None for fresh runs.
        """
        if resume is None:
            return None
        previous_total_time = getattr(
            resume, "time_integrate_total", getattr(resume, "time_integrate", 0.0)
        )
        return {
            "n_total": int(getattr(resume, "n_total", 0)),
            "time_integrate_total": float(previous_total_time),
        }

    @staticmethod
    def _qualified_class_name(obj):
        """Return a stable fully-qualified class name for audit metadata."""
        cls = type(obj)
        return "%s.%s" % (cls.__module__, cls.__name__)

    @staticmethod
    def _qmcpy_version():
        """Best-effort lookup of the loaded QMCPy version string."""
        qmcpy_module = sys.modules.get("qmcpy")
        return getattr(qmcpy_module, "__version__", None)

    def _annotate_checkpoint_metadata(self, data):
        """Attach lightweight checkpoint metadata used for auditing/resume."""
        data._qmcpy_version = self._qmcpy_version()
        data._resume_format_version = int(self._RESUME_FORMAT_VERSION)
        data._stopping_criterion_class = self._qualified_class_name(self)
        data._integrand_class = self._qualified_class_name(self.integrand)
        data._true_measure_class = self._qualified_class_name(self.true_measure)
        data._discrete_distribution_class = self._qualified_class_name(
            self.discrete_distrib
        )

    def _finalize_integration_data(self, data, elapsed, resume_provenance=None):
        """Attach shared integration metadata before returning ``Data``.

        Args:
            data (Data): Integration state to finalize.
            elapsed (float): Wall-clock time spent in the current ``integrate``
                call.
            resume_provenance (dict or None, optional): Output of
                :meth:`_capture_resume_provenance`. Defaults to None.
        """
        data.stopping_crit = self
        data.integrand = self.integrand
        data.true_measure = self.integrand.true_measure
        data.discrete_distrib = self.true_measure.discrete_distrib
        data.time_integrate = float(elapsed)
        previous_time = 0.0
        previous_n_total = 0
        if resume_provenance is not None:
            previous_time = float(resume_provenance.get("time_integrate_total", 0.0))
            previous_n_total = int(resume_provenance.get("n_total", 0))
        data.resumed = resume_provenance is not None
        data.n_resume_from = previous_n_total
        data.time_integrate_previous = previous_time
        data.time_integrate_resume = float(elapsed)
        data.time_integrate_total = previous_time + float(elapsed)
        self._annotate_checkpoint_metadata(data)

    def _resume_value_equal(self, current, saved):
        """Deep equality check tolerant of arrays, lists, dicts, and QMCPy objects.

        Args:
            current: Value from the live stopping criterion.
            saved: Value from the resume checkpoint.

        Returns:
            bool: True when the two values are considered equal.
        """
        if isinstance(current, np.ndarray) or isinstance(saved, np.ndarray):
            try:
                return np.array_equal(
                    np.asarray(current), np.asarray(saved), equal_nan=True
                )
            except TypeError:
                return np.array_equal(np.asarray(current), np.asarray(saved))
        if isinstance(current, (list, tuple)) and isinstance(saved, (list, tuple)):
            return len(current) == len(saved) and all(
                self._resume_value_equal(curr_i, saved_i)
                for curr_i, saved_i in zip(current, saved)
            )
        if isinstance(current, dict) and isinstance(saved, dict):
            return current.keys() == saved.keys() and all(
                self._resume_value_equal(current[key], saved[key]) for key in current
            )
        if (
            type(current) is type(saved)
            and hasattr(current, "parameters")
            and hasattr(saved, "parameters")
        ):
            return str(current) == str(saved)
        try:
            is_equal = current == saved
        except Exception:
            is_equal = str(current) == str(saved)
        if isinstance(is_equal, np.ndarray):
            return bool(np.all(is_equal))
        return bool(is_equal)

    def _require_resume_attrs(self, data, attrs):
        """Raise ParameterError if any attribute in *attrs* is absent from *data*.

        Args:
            data (Data): Resume checkpoint.
            attrs (tuple[str, ...]): Required attribute names.

        Raises:
            ParameterError: If one or more attributes are missing.
        """
        missing = [attr for attr in attrs if not hasattr(data, attr)]
        if missing:
            raise ParameterError(
                "resume data missing required attribute(s): %s"
                % ", ".join(sorted(missing))
            )

    def _validate_resume_object(self, label, current, saved, attrs):
        """Validate that a saved sub-object is compatible with the current one.

        Checks type equality and then compares each attribute listed in *attrs*
        using :meth:`_resume_value_equal`.

        Args:
            label (str): Human-readable name used in error messages.
            current: Live object (integrand, true measure, etc.).
            saved: Saved object from the resume checkpoint.
            attrs (tuple[str, ...]): Attribute names to compare.

        Raises:
            ParameterError: If the objects are incompatible.
        """
        if saved is None:
            raise ParameterError("resume data missing %s state." % label)
        if type(saved) is not type(current):
            raise ParameterError(
                "resume data has incompatible %s type %s; expected %s."
                % (label, type(saved).__name__, type(current).__name__)
            )
        missing = [
            attr for attr in attrs if not hasattr(current, attr) or not hasattr(saved, attr)
        ]
        if missing:
            raise ParameterError(
                "resume data missing %s attribute(s): %s"
                % (label, ", ".join(sorted(missing)))
            )
        for attr in attrs:
            if not self._resume_value_equal(getattr(current, attr), getattr(saved, attr)):
                raise ParameterError(
                    "resume data has incompatible %s.%s." % (label, attr)
                )

    def _validate_resume_data(self, data, required_fields=()):
        """Run standard cross-cutting resume compatibility checks.

        Validates stopping criterion type, integrand, true measure, discrete
        distribution, checkpoint format version, and ``n_total`` against
        ``n_limit``.

        Args:
            data (Data): Resume checkpoint to validate.
            required_fields (tuple[str, ...], optional): Additional attribute
                names that must be present on *data*. Defaults to ``()``.

        Raises:
            ParameterError: If any compatibility check fails.
        """
        self._require_resume_attrs(data, ("stopping_crit", "integrand", "true_measure", "discrete_distrib", "n_total") + tuple(required_fields))
        resume_format_version = getattr(
            data, "_resume_format_version", self._RESUME_FORMAT_VERSION
        )
        try:
            resume_format_version = int(resume_format_version)
        except (TypeError, ValueError):
            raise ParameterError("resume data has invalid _resume_format_version.")
        if resume_format_version != self._RESUME_FORMAT_VERSION:
            raise ParameterError(
                "resume data uses checkpoint format version %d; expected %d."
                % (resume_format_version, self._RESUME_FORMAT_VERSION)
            )
        if type(data.stopping_crit) is not type(self):
            raise ParameterError("resume data was generated by %s, not %s." % (type(data.stopping_crit).__name__, type(self).__name__))
        self._validate_resume_object("integrand", self.integrand, data.integrand, ("d", "d_indv", "d_comb") + tuple(getattr(self.integrand, "parameters", [])))
        self._validate_resume_object("true_measure", self.true_measure, data.true_measure, ("d", "domain", "range") + tuple(getattr(self.true_measure, "parameters", [])))
        self._validate_resume_object(
            "discrete_distrib",
            self.discrete_distrib,
            data.discrete_distrib,
            ("d", "replications", "mimics", "entropy", "spawn_key")
            + tuple(getattr(self.discrete_distrib, "parameters", [])),
        )
        if int(data.n_total) > int(self.n_limit):
            raise ParameterError("resume data n_total=%d exceeds current n_limit=%d." % (int(data.n_total), int(self.n_limit)))

    def _validate_resume_with_state(self, data, required_fields=(), state_fields=()):
        """Validate resume data including algorithm-specific state fields.

        Calls :meth:`_validate_resume_data` and additionally checks that all
        *state_fields* are present and that ``n_total >= n_init``.

        Args:
            data (Data): Resume checkpoint to validate.
            required_fields (tuple[str, ...], optional): Extra data attributes
                required beyond the standard set. Defaults to ``()``.
            state_fields (tuple[str, ...], optional): Algorithm-state attributes
                that must also be present. Defaults to ``()``.

        Raises:
            ParameterError: If any compatibility check fails.
        """
        self._validate_resume_data(data, required_fields=required_fields)
        self._require_resume_attrs(data, state_fields)
        if hasattr(self, "n_init") and int(data.n_total) < int(self.n_init):
            raise ParameterError("resume data must include at least n_init samples.")

    def _validate_resume_shape(self, label, value, expected_shape):
        """Raise ParameterError when a resume array has the wrong shape."""
        actual_shape = np.shape(value)
        expected_shape = tuple(expected_shape)
        if actual_shape != expected_shape:
            raise ParameterError(
                "resume data %s shape must be %s; got %s."
                % (label, expected_shape, actual_shape)
            )

    @staticmethod
    def _is_power_of_two(n):
        """Return True when *n* is a positive power of two."""
        return n > 0 and (n & (n - 1)) == 0

    def _restore_resume_rng_state(self, data):
        """Deep-copy the saved RNG state into the live discrete distribution.

        Ensures that samples drawn after resuming are independent of those
        already stored in the checkpoint.

        Args:
            data (Data): Resume checkpoint carrying the saved distribution in
                ``data.discrete_distrib``.

        Raises:
            ParameterError: If either distribution lacks an ``rng`` attribute.
        """
        saved_distrib = data.discrete_distrib
        if not hasattr(saved_distrib, "rng") or not hasattr(self.discrete_distrib, "rng"):
            raise ParameterError("resume data is missing discrete distribution RNG state.")
        self.discrete_distrib.rng.bit_generator.state = copy.deepcopy(
            saved_distrib.rng.bit_generator.state
        )

    def _compute_indv_alphas(self, alphas_comb):
        """Distribute combined confidence levels to individual integrand dimensions.

        Uses the integrand dependency map to allocate the per-combined-output
        alpha budget down to each individual output dimension.

        Args:
            alphas_comb (np.ndarray): Per-combined-output confidence levels with
                shape ``integrand.d_comb``.

        Returns:
            tuple[np.ndarray, bool]: ``(alphas_indv, identity_dependency)``
                where *alphas_indv* has shape ``integrand.d_indv`` and
                *identity_dependency* is True when each combined output depends
                on exactly its matching individual output.
        """
        alphas_indv = np.tile(1, self.integrand.d_indv)
        identity_dependency = True
        for k in np.ndindex(self.integrand.d_comb):
            comb_flags = np.tile(True, self.integrand.d_comb)
            comb_flags[k] = False
            flags_indv = self.integrand.dependency(comb_flags)
            if (
                self.integrand.d_indv != self.integrand.d_comb
                or (flags_indv != comb_flags).any()
            ):
                identity_dependency = False
            dependents_k = ~flags_indv
            n_dep_k = dependents_k.sum()
            alpha_k = alphas_comb[k] / n_dep_k
            alpha_k_mat = alpha_k * dependents_k
            alphas_indv = np.where(alpha_k_mat == 0, alphas_indv, np.minimum(alpha_k_mat, alphas_indv))
        return alphas_indv, identity_dependency

    def set_tolerance(self, abs_tol=None, rel_tol=None, rmse_tol=None):
        """Reset the tolerances.

        Args:
            abs_tol (float): Absolute tolerance, when supported. If supplied,
                reset it; otherwise ignore it.
            rel_tol (float): Relative tolerance, when supported. If supplied,
                reset it; otherwise ignore it.
            rmse_tol (float): RMSE tolerance, when supported. If supplied,
                reset it; otherwise ignore it. If ``rmse_tol`` is not supplied
                but ``abs_tol`` is, then ``rmse_tol = abs_tol / norm.ppf(1 -
                alpha / 2)``.
        """
        raise MethodImplementationError(self, "set_tolerance")

    def __repr__(self):
        return _univ_repr(self, "AbstractStoppingCriterion", self.parameters)
