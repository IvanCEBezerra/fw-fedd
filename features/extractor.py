"""Integrate the project's updated 23-feature functions."""

import numpy as np

from fw_fedd.features.stl import decompose_stl
from fw_fedd.features.deterministic import (
    trend_degree,
    seasonal_degree,
)
from fw_fedd.features.descriptive import (
    std_residual,
    skewness_residual,
    kurtosis_residual,
    turning_points_rate,
    periodicity,
)
from fw_fedd.features.linear import (
    calculate_acf,
    calculate_pacf,
)
from fw_fedd.features.nonlinear import (
    calculate_bicorrelation,
    calculate_mutual_information,
)


FEATURE_NAMES = (
    "trend_strength",
    "seasonal_strength",
    "residual_std",
    "residual_skewness",
    "residual_kurtosis",
    "turning_points_pct",
    "periodicity",
) + tuple(
    f"acf_{k}" for k in range(1, 6)
) + tuple(
    f"pacf_{k}" for k in range(1, 6)
) + tuple(
    f"bicorrelation_{k}" for k in range(1, 4)
) + tuple(
    f"mutual_information_{k}" for k in range(1, 4)
)


def extract_features(
    window,
    period=None,
    *,
    robust=False,
    n_bins=10,
):
    """
    Return 23 features from one causal time series window.

    STL settings and histogram bins are reconstruction choices,
    not recovered parameters of the author's original program.
    """
    window = np.asarray(window, dtype=float)

    if window.ndim != 1 or window.size < 7:
        raise ValueError(
            "window must be one-dimensional with at least "
            "7 observations."
        )

    if not np.isfinite(window).all():
        raise ValueError(
            "window must contain only finite values."
        )

    trend, seasonal, residual = decompose_stl(
        window,
        period=period,
        robust=robust,
    )

    descriptive_features = np.array(
        [
            trend_degree(window, residual, seasonal),
            seasonal_degree(window, residual, trend),
            std_residual(residual),
            skewness_residual(residual),
            kurtosis_residual(residual),
            turning_points_rate(window),
            periodicity(residual),
        ],
        dtype=float,
    )

    acf_features = calculate_acf(
        residual,
        max_lag=5,
    )

    pacf_features = calculate_pacf(
        residual,
        max_lag=5,
    )

    bicorrelation_features = calculate_bicorrelation(
        residual,
        max_lag=3,
    )

    mutual_information_features = calculate_mutual_information(
        residual,
        max_lag=3,
        n_bins=n_bins,
    )

    features = np.concatenate(
        (
            descriptive_features,
            acf_features,
            pacf_features,
            bicorrelation_features,
            mutual_information_features,
        )
    )

    if features.shape != (23,):
        raise RuntimeError(
            f"Expected 23 features, got shape {features.shape}."
        )

    if not np.isfinite(features).all():
        raise FloatingPointError(
            "Feature extraction produced non-finite values."
        )

    return features
