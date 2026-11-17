import numpy as np


def _weighted_centered_unit_vector(vector, normalized_weights):
    """
    Return a weighted, centered vector with Euclidean norm one.

    Positive rescaling preserves Pearson correlation while
    reducing numerical overflow and underflow.

    Returns None when no weighted variation can be represented.
    """
    scale = np.max(np.abs(vector))

    if scale == 0.0:
        return None

    scaled = vector / scale

    # Subtracting an anchor before the weighted mean helps when
    # values have a large common offset and small differences.
    shifted = scaled - scaled[0]

    centered = (
        shifted - np.dot(normalized_weights, shifted)
    )

    centered_scale = np.max(np.abs(centered))

    if centered_scale == 0.0:
        return None

    centered = centered / centered_scale

    weighted = (
        np.sqrt(normalized_weights) * centered
    )

    weighted_scale = np.max(np.abs(weighted))

    if weighted_scale == 0.0:
        return None

    weighted = weighted / weighted_scale
    norm = np.linalg.norm(weighted)

    if not np.isfinite(norm) or norm == 0.0:
        raise FloatingPointError(
            "Could not normalize the weighted feature vector."
        )

    return weighted / norm


def weighted_pearson_distance(fv0, fvt, weights):
    """
    Compute weighted Pearson distance between two feature vectors.

    Formula:
        distance = 1 - weighted_correlation(fv0, fvt)

    Parameters
    ----------
    fv0 : array-like
        Reference feature vector.
    fvt : array-like
        Current feature vector.
    weights : array-like
        Finite, nonnegative feature weights.
        At least one weight must be positive.

    Returns
    -------
    float
        Distance in [0, 2]:
            0: perfect positive correlation.
            1: zero correlation.
            2: perfect negative correlation.

    Notes
    -----
    Features with zero weight do not contribute.

    Pearson correlation is undefined when either vector has
    no variation over the positively weighted features.

    Explicit conventions for degenerate cases:
        - Identical active vectors return distance zero.
        - Different vectors with undefined correlation return one.

    These conventions are implementation choices and do not
    establish equivalence to the author's original code.
    """
    a = np.asarray(fv0, dtype=float)
    b = np.asarray(fvt, dtype=float)
    w = np.asarray(weights, dtype=float)

    for name, vector in (
        ("fv0", a),
        ("fvt", b),
        ("weights", w),
    ):
        if vector.ndim != 1:
            raise ValueError(
                f"{name} must be a one-dimensional array."
            )

        if vector.size == 0:
            raise ValueError(
                f"{name} must not be empty."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"{name} must contain only finite values."
            )

    if not (a.shape == b.shape == w.shape):
        raise ValueError(
            "fv0, fvt and weights must have the same shape."
        )

    if np.any(w < 0.0):
        raise ValueError(
            "weights must be nonnegative."
        )

    active = w > 0.0

    if not np.any(active):
        raise ValueError(
            "At least one weight must be positive."
        )

    # Zero-weight features have no influence on the distance.
    a = a[active]
    b = b[active]
    w = w[active]

    if np.array_equal(a, b):
        return 0.0

    if np.all(a == a[0]) or np.all(b == b[0]):
        return 1.0

    # Normalize without summing potentially very large weights.
    w = w / np.max(w)
    w = w / np.sum(w)

    unit_a = _weighted_centered_unit_vector(a, w)
    unit_b = _weighted_centered_unit_vector(b, w)

    if unit_a is None or unit_b is None:
        return 1.0

    # For unit vectors, 1 - dot(a, b) = ||a - b||^2 / 2.
    # Avoid subtracting nearly equal numbers: the previous expression could
    # alternate between zero and round-off-sized positive distances for
    # nearly identical vectors, feeding artificial jumps to a low-variance ICI.
    difference = unit_a - unit_b
    distance = float(0.5 * np.dot(difference, difference))

    if not np.isfinite(distance):
        raise FloatingPointError(
            "Weighted Pearson distance is non-finite."
        )

    # Protect the theoretical bounds against floating-point rounding.
    return float(np.clip(distance, 0.0, 2.0))
