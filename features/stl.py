import numpy as np
from statsmodels.tsa.seasonal import STL


def estimate_stl_period(window):
    """Estimate a period causally from a linearly detrended reference window.

    This rule is a reconstruction choice. It never reads concept labels or
    future samples. Two is the explicit STL fallback when no peak is found.
    Call once per concept and keep the result fixed until the next reset.
    """
    from fw_fedd.features.descriptive import periodicity

    values = np.asarray(window, dtype=float)
    if values.ndim != 1 or values.size < 7 or not np.isfinite(values).all():
        raise ValueError("Period estimation requires at least 7 finite values.")
    coordinates = np.linspace(-1.0, 1.0, values.size)
    slope, intercept = np.polyfit(coordinates, values, 1)
    detrended = values - (slope * coordinates + intercept)
    detected = int(periodicity(detrended))
    return detected if 2 <= detected < values.size else 2


def decompose_stl(window, period=None, robust=False):
    """
    Decompose a time series window using STL.

    The decomposition follows:
        window = trend + seasonal + residual

    Parameters
    ----------
    window : array-like
        Nonempty, finite, one-dimensional time series window.
        Its length must exceed the seasonal period.
    period : int or None, default=None
        Number of observations in one seasonal cycle.
        Must be an integer greater than or equal to 2. None uses the causal
        reference-window estimator; the pipeline freezes it per concept.
    robust : bool, default=False
        Whether to use robust fitting to reduce the influence
        of outliers.

    Returns
    -------
    trend : np.ndarray
        Estimated trend component.
    seasonal : np.ndarray
        Estimated seasonal component.
    residual : np.ndarray
        Residual component.

    Notes
    -----
    The period estimator and robust setting are explicit
    reconstruction choices, not recovered author parameters.

    Other STL smoothing settings use the installed statsmodels
    defaults. Record the library version for reproducibility.

    When called on a sliding window, the caller must provide
    only observations available at the current timestamp.
    """
    window = np.array(window, dtype=float, copy=True)

    if window.ndim != 1:
        raise ValueError(
            "window must be a one-dimensional array."
        )

    if window.size == 0:
        raise ValueError("window must not be empty.")

    if not np.isfinite(window).all():
        raise ValueError(
            "window must contain only finite values."
        )

    if period is None:
        period = estimate_stl_period(window)

    if (
        isinstance(period, bool)
        or not isinstance(period, (int, np.integer))
        or period < 2
    ):
        raise ValueError(
            "period must be an integer greater than or equal to 2."
        )

    period = int(period)

    if window.size <= period:
        raise ValueError(
            f"window must contain more than {period} observations; "
            f"received {window.size}."
        )

    if not isinstance(robust, (bool, np.bool_)):
        raise ValueError("robust must be a boolean.")

    decomposition = STL(
        window,
        period=period,
        robust=bool(robust),
    ).fit()

    trend = np.asarray(
        decomposition.trend,
        dtype=float,
    )

    seasonal = np.asarray(
        decomposition.seasonal,
        dtype=float,
    )

    residual = np.asarray(
        decomposition.resid,
        dtype=float,
    )

    for name, component in (
        ("trend", trend),
        ("seasonal", seasonal),
        ("residual", residual),
    ):
        if component.shape != window.shape:
            raise RuntimeError(
                f"STL returned an unexpected shape for {name}."
            )

        if not np.isfinite(component).all():
            raise FloatingPointError(
                f"STL produced non-finite values in {name}."
            )

    return trend, seasonal, residual
