import numpy as np
from scipy.signal import find_peaks


def _validate_series(values, name):
    """Convert input to a nonempty, finite, one-dimensional array."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")

    return values


def _center_and_scale(values):
    """
    Center and rescale values for stable dimensionless calculations.

    Rescaling does not change skewness, kurtosis or autocorrelation.
    Constant series are represented by a vector of zeros.
    """
    centered = values - np.mean(values)

    if not np.isfinite(centered).all():
        raise FloatingPointError("Numerical overflow while centering values.")

    scale = np.max(np.abs(centered))

    if scale == 0.0:
        return np.zeros_like(centered)

    return centered / scale


def std_residual(residual):
    """
    Return the population standard deviation of the STL residual.

    Uses ddof=0, preserving the original variance estimator.
    A constant residual has standard deviation zero.
    """
    residual = _validate_series(residual, "residual")

    value = float(np.std(residual, ddof=0))

    if not np.isfinite(value):
        raise FloatingPointError(
            "Numerical overflow while computing residual standard deviation."
        )

    return value


def skewness_residual(residual):
    """
    Return the third standardized central moment of the STL residual.

    Uses population central moments without finite-sample correction.
    A symmetric distribution has skewness zero.

    Skewness is undefined for a constant residual; this implementation
    explicitly represents that degenerate case by zero.
    """
    residual = _validate_series(residual, "residual")
    centered = _center_and_scale(residual)

    second_moment = np.mean(centered**2)

    if second_moment == 0.0:
        return 0.0

    third_moment = np.mean(centered**3)

    return float(third_moment / second_moment**1.5)


def kurtosis_residual(residual):
    """
    Return conventional Pearson kurtosis of the STL residual.

    Uses the fourth standardized central moment:
        kurtosis = E[(X - mean)^4] / E[(X - mean)^2]^2

    A normal distribution has kurtosis 3, rather than excess
    kurtosis zero. No finite-sample correction is applied.

    Kurtosis is undefined for a constant residual; this implementation
    explicitly represents that degenerate case by zero.
    """
    residual = _validate_series(residual, "residual")
    centered = _center_and_scale(residual)

    second_moment = np.mean(centered**2)

    if second_moment == 0.0:
        return 0.0

    fourth_moment = np.mean(centered**4)

    return float(fourth_moment / second_moment**2)


def turning_points_rate(window):
    """
    Return the percentage of turning points in the ORIGINAL series.

    A turning point is a strict local maximum or minimum.
    Equal neighboring values do not form a turning point.

    The denominator is the complete window length n:
        rate = 100 * turning_point_count / n
    """
    window = _validate_series(window, "window")

    if window.size < 3:
        return 0.0

    previous = window[:-2]
    current = window[1:-1]
    following = window[2:]

    local_maxima = (current > previous) & (current > following)
    local_minima = (current < previous) & (current < following)

    count = np.count_nonzero(local_maxima | local_minima)

    return float(100.0 * count / window.size)


def periodicity(window, min_peak_trough_difference=0.1, max_lag=None):
    """
    Return the first qualifying positive autocorrelation peak lag.

    For the thesis feature, pass the decomposition residual. The causal
    seasonal-period estimator also uses this peak-finding primitive on a
    linearly detrended reference window; that is a separate operation.

    A qualifying peak:
        1. Occurs after an autocorrelation trough.
        2. Has positive autocorrelation.
        3. Exceeds the preceding trough by at least the specified
           peak-to-trough difference.

    Returns zero when no qualifying peak exists.

    Reconstruction choices:
        - Search up to half the window length by default.
        - Use biased autocorrelation with a common denominator.
        - Use the immediately preceding local trough.
        - Represent absence of periodicity by zero.

    These implementation details are explicit assumptions and do
    not establish equivalence to the author's original code.
    """
    window = _validate_series(window, "window")

    if (
        not np.isfinite(min_peak_trough_difference)
        or min_peak_trough_difference <= 0
    ):
        raise ValueError(
            "min_peak_trough_difference must be positive and finite."
        )

    if max_lag is not None:
        if (
            isinstance(max_lag, bool)
            or not isinstance(max_lag, (int, np.integer))
            or max_lag < 2
        ):
            raise ValueError("max_lag must be an integer of at least 2.")

    if window.size < 4:
        return 0.0

    lag_limit = (
        window.size // 2
        if max_lag is None
        else min(int(max_lag), window.size - 1)
    )

    centered = _center_and_scale(window)
    denominator = np.dot(centered, centered)

    if denominator == 0.0:
        return 0.0

    autocorrelation = np.empty(lag_limit + 1, dtype=float)
    autocorrelation[0] = 1.0

    for lag in range(1, lag_limit + 1):
        autocorrelation[lag] = (
            np.dot(centered[lag:], centered[:-lag]) / denominator
        )

    peaks = find_peaks(autocorrelation)[0]
    troughs = find_peaks(-autocorrelation)[0]

    for peak in peaks:
        preceding_troughs = troughs[troughs < peak]

        if preceding_troughs.size == 0:
            continue

        trough = preceding_troughs[-1]
        difference = autocorrelation[peak] - autocorrelation[trough]

        if (
            autocorrelation[peak] > 0.0
            and difference >= min_peak_trough_difference
        ):
            return float(peak)

    return 0.0
