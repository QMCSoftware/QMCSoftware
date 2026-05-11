import numpy as np

from ..util import DimensionError, ParameterError


def _clip_unit_interval(u):
    eps = np.finfo(float).eps
    return np.clip(u, eps, 1.0 - eps)


def _validate_marginals(marginals):
    try:
        parsed = list(marginals)
    except TypeError as exc:
        raise ParameterError(
            "marginals must be a length d list of distributions with a 'ppf' method."
        ) from exc

    if len(parsed) == 0:
        raise ParameterError("marginals must contain at least one distribution.")

    for j, marginal in enumerate(parsed):
        if not hasattr(marginal, "ppf") or not callable(marginal.ppf):
            raise ParameterError(
                "Each copula marginal must implement a callable 'ppf' method; "
                f"marginal {j} does not."
            )

    return parsed


def _validate_dimension(distribution, marginals):
    d = getattr(distribution, "d", distribution)
    try:
        d = int(d)
    except (TypeError, ValueError) as exc:
        raise DimensionError("distribution must expose integer dimension d.") from exc

    if len(marginals) != d:
        raise DimensionError("Length of marginals must match sampler dimension.")

    return d


def _apply_marginal_ppfs(v, marginals):
    v = _clip_unit_interval(np.asarray(v, dtype=float))
    _validate_dimension(v.shape[-1], marginals)

    t = np.empty_like(v, dtype=float)
    for j, marginal in enumerate(marginals):
        t[..., j] = marginal.ppf(v[..., j])
    return t


def _validate_correlation_matrix(correlation, d):
    corr = np.asarray(correlation, dtype=float)

    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation must be a square matrix.")
    if corr.shape != (d, d):
        raise ValueError(
            f"correlation shape {corr.shape} must match sampler dimension {d}."
        )
    if not np.all(np.isfinite(corr)):
        raise ValueError("correlation must contain only finite values.")
    if not np.allclose(corr, corr.T, rtol=1e-12, atol=1e-12):
        raise ValueError("correlation must be symmetric.")
    if not np.allclose(np.diag(corr), 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("correlation must have ones on the diagonal.")

    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation must be positive definite.") from exc

    return corr


def _build_marginal_range(marginals):
    ranges = []
    eps = np.finfo(float).eps

    for marginal in marginals:
        if hasattr(marginal, "interval") and callable(marginal.interval):
            try:
                ranges.append(marginal.interval(1.0))
                continue
            except Exception:
                pass

        try:
            ranges.append((marginal.ppf(eps), marginal.ppf(1.0 - eps)))
        except Exception:
            ranges.append((-np.inf, np.inf))

    return np.asarray(ranges, dtype=float)


def _marginal_cdfs_and_logpdf(x, marginals):
    x = np.asarray(x, dtype=float)
    _validate_dimension(x.shape[-1], marginals)

    u = np.empty_like(x, dtype=float)
    log_marginal_density = np.zeros(x.shape[:-1], dtype=float)

    for j, marginal in enumerate(marginals):
        if not hasattr(marginal, "cdf") or not callable(marginal.cdf):
            raise ParameterError(
                "Each marginal must implement 'cdf' to compute copula weights."
            )

        if hasattr(marginal, "logpdf") and callable(marginal.logpdf):
            log_pdf_j = marginal.logpdf(x[..., j])
        elif hasattr(marginal, "pdf") and callable(marginal.pdf):
            pdf_j = marginal.pdf(x[..., j])
            with np.errstate(divide="ignore", invalid="ignore"):
                log_pdf_j = np.log(pdf_j)
        else:
            raise ParameterError(
                "Each marginal must implement 'pdf' or 'logpdf' to compute copula weights."
            )

        u[..., j] = marginal.cdf(x[..., j])
        log_marginal_density += np.asarray(log_pdf_j, dtype=float)

    return _clip_unit_interval(u), log_marginal_density
