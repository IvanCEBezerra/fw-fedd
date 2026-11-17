import numpy as np


def _validate_integer(value, name, minimum=1):
    """Validate an integer parameter and return a Python integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(
            f"{name} must be an integer greater than or equal "
            f"to {minimum}."
        )

    return int(value)


def _validate_vector(values, name):
    """Validate a nonempty, finite, one-dimensional array."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array."
        )

    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} must contain only finite values."
        )

    return values


def calculate_bicorrelation(residual, max_lag=3):
    """
    Compute the residual third-order joint moment.

    For each lag k:
        B(k) = mean(x[t] * x[t-k] * x[t-2k])

    Uses the supplied STL residual directly, without additional
    centering or variance normalization. This preserves the
    stated feature definition and its dependence on scale.

    Parameters
    ----------
    residual : array-like
        Residual component from STL decomposition.
    max_lag : int, default=3
        Maximum lag. The residual length must exceed 2 * max_lag.

    Returns
    -------
    np.ndarray
        Array of shape (max_lag,) for lags 1 through max_lag.

    Raises
    ------
    ValueError
        If inputs are invalid or the residual is too short.
    FloatingPointError
        If the calculation produces a numerical overflow.
    """
    max_lag = _validate_integer(max_lag, "max_lag")
    x = _validate_vector(residual, "residual")

    if x.size <= 2 * max_lag:
        raise ValueError(
            f"residual must contain more than {2 * max_lag} "
            f"observations; received {x.size}."
        )

    values = np.empty(max_lag, dtype=float)

    with np.errstate(over="raise", invalid="raise"):
        for lag in range(1, max_lag + 1):
            current = x[2 * lag:]
            previous = x[lag:-lag]
            earlier = x[:-2 * lag]

            values[lag - 1] = np.mean(
                current * previous * earlier
            )

    if not np.isfinite(values).all():
        raise FloatingPointError(
            "Non-finite values while computing bicorrelation."
        )

    return values


def _histogram_mutual_information(x, y, n_bins=10):
    """
    Estimate mutual information using a joint histogram.

    Formula:
        I(X; Y) =
            sum(p_xy * log(p_xy / (p_x * p_y)))

    Equal-width bins are estimated separately for each variable
    from the supplied paired observations. Only occupied joint
    bins contribute to the sum.

    Uses the natural logarithm, so the result is expressed in nats.

    Parameters
    ----------
    x, y : array-like
        Finite, equally sized, one-dimensional paired observations.
    n_bins : int, default=10
        Number of bins per variable. Must be at least 2.

    Returns
    -------
    float
        Nonnegative estimated mutual information.
        Returns zero if either variable is constant.

    Notes
    -----
    This is a finite-sample histogram estimate, not an unbiased
    estimator. Ten bins are an explicit reconstruction choice.
    """
    n_bins = _validate_integer(
        n_bins,
        "n_bins",
        minimum=2,
    )

    x = _validate_vector(x, "x")
    y = _validate_vector(y, "y")

    if x.shape != y.shape:
        raise ValueError(
            "x and y must have the same shape."
        )

    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0

    with np.errstate(over="raise", invalid="raise"):
        joint_counts, _, _ = np.histogram2d(
            x,
            y,
            bins=n_bins,
        )

    total = joint_counts.sum()

    if total <= 0.0:
        raise FloatingPointError(
            "The joint histogram contains no observations."
        )

    joint_probability = joint_counts / total

    probability_x = joint_probability.sum(
        axis=1,
        keepdims=True,
    )

    probability_y = joint_probability.sum(
        axis=0,
        keepdims=True,
    )

    independent_probability = (
        probability_x * probability_y
    )

    occupied = joint_probability > 0.0

    # Occupied joint bins necessarily have positive marginals.
    with np.errstate(
        divide="raise",
        over="raise",
        invalid="raise",
    ):
        contributions = joint_probability[occupied] * (
            np.log(joint_probability[occupied])
            - np.log(independent_probability[occupied])
        )

        mutual_information = float(
            np.sum(contributions)
        )

    if not np.isfinite(mutual_information):
        raise FloatingPointError(
            "Non-finite mutual information estimate."
        )

    # Remove small negative values caused by floating-point rounding.
    return max(0.0, mutual_information)


def calculate_mutual_information(
    residual,
    max_lag=3,
    n_bins=10,
):
    """
    Estimate residual mutual information at successive lags.

    For each lag k, pairs:
        current = residual[k:]
        lagged = residual[:-k]

    Parameters
    ----------
    residual : array-like
        Residual component from STL decomposition.
    max_lag : int, default=3
        Maximum lag. Must be smaller than the residual length.
    n_bins : int, default=10
        Number of equal-width histogram bins per variable.

    Returns
    -------
    np.ndarray
        Array of shape (max_lag,) containing mutual information
        in nats for lags 1 through max_lag.
    """
    max_lag = _validate_integer(max_lag, "max_lag")

    n_bins = _validate_integer(
        n_bins,
        "n_bins",
        minimum=2,
    )

    x = _validate_vector(residual, "residual")

    if x.size <= max_lag:
        raise ValueError(
            f"residual must contain more than {max_lag} "
            f"observations; received {x.size}."
        )

    values = np.empty(max_lag, dtype=float)

    for lag in range(1, max_lag + 1):
        values[lag - 1] = _histogram_mutual_information(
            x=x[lag:],
            y=x[:-lag],
            n_bins=n_bins,
        )

    return values