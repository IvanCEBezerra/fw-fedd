import numpy as np


def _prepare_residual(residual, max_lag):
    """
    Validate, center and rescale the residual.

    Rescaling preserves autocorrelation and partial autocorrelation
    while reducing the risk of numerical overflow.
    """
    if (
        isinstance(max_lag, bool)
        or not isinstance(max_lag, (int, np.integer))
        or max_lag < 1
    ):
        raise ValueError("max_lag must be a positive integer.")

    x = np.asarray(residual, dtype=float)

    if x.ndim != 1:
        raise ValueError(
            "residual must be a one-dimensional array."
        )

    if x.size <= max_lag:
        raise ValueError(
            f"residual must contain more than {max_lag} observations; "
            f"received {x.size}."
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "residual must contain only finite values."
        )

    with np.errstate(over="raise", invalid="raise"):
        centered = x - np.mean(x)

    scale = np.max(np.abs(centered))

    if scale == 0.0:
        return np.zeros_like(centered)

    return centered / scale


def _autocorrelation_sequence(centered, max_lag):
    """
    Compute autocorrelations from lag zero through max_lag.

    Uses a common denominator for every lag, corresponding to
    biased autocovariances normalized by the lag-zero covariance.

    For a constant residual, return zeros as an explicit
    convention for otherwise undefined correlations.
    """
    denominator = np.dot(centered, centered)
    correlations = np.zeros(max_lag + 1, dtype=float)

    if denominator == 0.0:
        return correlations

    correlations[0] = 1.0

    for lag in range(1, max_lag + 1):
        numerator = np.dot(
            centered[lag:],
            centered[:-lag],
        )

        correlations[lag] = numerator / denominator

    if not np.isfinite(correlations).all():
        raise FloatingPointError(
            "Non-finite values while computing autocorrelation."
        )

    return correlations


def calculate_acf(residual, max_lag=5):
    """
    Compute residual autocorrelation for lags 1 through max_lag.

    Formula:
        ACF(k) =
            sum((x[t] - mean) * (x[t-k] - mean))
            / sum((x[t] - mean)**2)

    Lag zero is excluded from the returned feature vector.

    Parameters
    ----------
    residual : array-like
        Residual component from STL decomposition.
    max_lag : int, default=5
        Maximum lag. Must be smaller than the residual length.

    Returns
    -------
    np.ndarray
        Array of shape (max_lag,).

        For a constant residual, returns zeros as an explicit
        convention because its autocorrelation is undefined.
    """
    centered = _prepare_residual(residual, max_lag)

    correlations = _autocorrelation_sequence(
        centered,
        max_lag,
    )

    return correlations[1:].copy()


def calculate_pacf(residual, max_lag=5):
    """
    Compute residual partial autocorrelation using Yule-Walker.

    Each lag k is estimated by solving the order-k Yule-Walker
    system and selecting its last autoregressive coefficient.

    Uses biased autocovariances with a common denominator n,
    normalized by the lag-zero covariance. This normalization
    preserves the Yule-Walker solution.

    Lag zero is excluded from the returned feature vector.

    Parameters
    ----------
    residual : array-like
        Residual component from STL decomposition.
    max_lag : int, default=5
        Maximum lag. Must be smaller than the residual length.

    Returns
    -------
    np.ndarray
        Array of shape (max_lag,).

        For a constant residual, returns zeros as an explicit
        convention because its partial autocorrelation is undefined.

    Raises
    ------
    np.linalg.LinAlgError
        If a Yule-Walker system cannot be solved.
    """
    centered = _prepare_residual(residual, max_lag)

    correlations = _autocorrelation_sequence(
        centered,
        max_lag,
    )

    if correlations[0] == 0.0:
        return np.zeros(max_lag, dtype=float)

    partial_correlations = np.empty(max_lag, dtype=float)

    for lag in range(1, max_lag + 1):
        indices = np.abs(
            np.subtract.outer(
                np.arange(lag),
                np.arange(lag),
            )
        )

        correlation_matrix = correlations[indices]
        target_correlations = correlations[1:lag + 1]

        try:
            coefficients = np.linalg.solve(
                correlation_matrix,
                target_correlations,
            )
        except np.linalg.LinAlgError as error:
            raise np.linalg.LinAlgError(
                f"Could not solve the Yule-Walker system at lag {lag}."
            ) from error

        partial_correlations[lag - 1] = coefficients[-1]

    if not np.isfinite(partial_correlations).all():
        raise FloatingPointError(
            "Non-finite values while computing partial autocorrelation."
        )

    return partial_correlations