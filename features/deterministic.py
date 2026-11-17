import numpy as np


def _validate_components(window, residual, other_component):
    """Validate three finite, equally sized, one-dimensional arrays."""
    arrays = []

    for name, values in (
        ("window", window),
        ("residual", residual),
        ("other_component", other_component),
    ):
        values = np.asarray(values, dtype=float)

        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional array.")

        if values.size == 0:
            raise ValueError(f"{name} must not be empty.")

        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values.")

        arrays.append(values)

    if not all(values.shape == arrays[0].shape for values in arrays):
        raise ValueError("All components must have the same shape.")

    return tuple(arrays)


def _variance_strength(residual, adjusted_series):
    """
    Compute a variance-ratio strength using population variances.

    Common rescaling preserves the variance ratio while reducing
    the risk of numerical overflow.

    Implementation conventions:
        - Return zero when the denominator variance is zero.
        - Clip negative estimated strength to zero.
    """
    if not np.isfinite(adjusted_series).all():
        raise FloatingPointError(
            "Non-finite values after subtracting the other component."
        )

    scale = max(
        float(np.max(np.abs(residual))),
        float(np.max(np.abs(adjusted_series))),
    )

    if scale == 0.0:
        return 0.0

    residual_variance = np.var(residual / scale, ddof=0)
    adjusted_variance = np.var(adjusted_series / scale, ddof=0)

    if adjusted_variance == 0.0:
        return 0.0

    strength = 1.0 - residual_variance / adjusted_variance

    return float(np.clip(strength, 0.0, 1.0))


def trend_degree(window, residual, seasonal):
    """
    Compute trend strength after removing seasonality.

    Formula:
        trend_strength = 1 - var(residual) / var(window - seasonal)

    Parameters
    ----------
    window : array-like
        Original time series window.
    residual : array-like
        Residual component from the same STL decomposition.
    seasonal : array-like
        Seasonal component from the same STL decomposition.

    Returns
    -------
    float
        Strength clipped to [0, 1]. Zero denotes either no estimated
        strength or a degenerate denominator.
    """
    window, residual, seasonal = _validate_components(
        window, residual, seasonal
    )

    adjusted_series = window - seasonal

    return _variance_strength(residual, adjusted_series)


def seasonal_degree(window, residual, trend):
    """
    Compute seasonal strength after removing the trend.

    Formula:
        seasonal_strength = 1 - var(residual) / var(window - trend)

    Parameters
    ----------
    window : array-like
        Original time series window.
    residual : array-like
        Residual component from the same STL decomposition.
    trend : array-like
        Trend component from the same STL decomposition.

    Returns
    -------
    float
        Strength clipped to [0, 1]. Zero denotes either no estimated
        strength or a degenerate denominator.
    """
    window, residual, trend = _validate_components(
        window, residual, trend
    )

    adjusted_series = window - trend

    return _variance_strength(residual, adjusted_series)