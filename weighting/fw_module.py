import numpy as np


class FeatureWeighter:
    """
    Compute feature weights using inverse standard deviation.

    Weights follow:
        w_i = 1 / (std(f_i) + epsilon)

    Feature statistics are updated with Welford's algorithm.
    Call reset() after a detected concept drift.
    """

    def __init__(self, n_features=23, epsilon=1e-6):
        if (
            isinstance(n_features, bool)
            or not isinstance(n_features, (int, np.integer))
            or n_features < 1
        ):
            raise ValueError("n_features must be a positive integer.")

        epsilon = float(epsilon)

        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be positive and finite.")

        self.n_features = int(n_features)
        self.epsilon = epsilon

        self.reset()

    def reset(self):
        """Reset the accumulated feature statistics."""
        self.n = 0
        self.mean = np.zeros(self.n_features, dtype=float)
        self.M2 = np.zeros(self.n_features, dtype=float)

    def update(self, feature_vector):
        """
        Update statistics using one finite feature vector.

        Parameters
        ----------
        feature_vector : array-like of shape (n_features,)
        """
        x = np.asarray(feature_vector, dtype=float)

        if x.shape != (self.n_features,):
            raise ValueError(
                f"Expected shape ({self.n_features},), got {x.shape}."
            )

        if not np.isfinite(x).all():
            raise ValueError(
                "feature_vector must contain only finite values."
            )

        # Reject failed updates before they contaminate later distances.
        next_n = self.n + 1
        with np.errstate(over="raise", invalid="raise"):
            delta = x - self.mean
            next_mean = self.mean + delta / next_n
            next_M2 = self.M2 + delta * (x - next_mean)
        if not np.isfinite(next_mean).all() or not np.isfinite(next_M2).all():
            raise FloatingPointError("Non-finite feature-weight statistics.")
        self.n = next_n
        self.mean = next_mean
        self.M2 = next_M2

    def get_weights(self):
        """
        Return inverse-standard-deviation weights.

        Population variance uses M2 / n, preserving the original
        estimator. Before two observations, return uniform weights.
        A common positive rescaling preserves weighted Pearson correlation.
        """
        if self.n < 2:
            return np.ones(self.n_features, dtype=float)

        variance = self.M2 / self.n

        # Protect against tiny negative values from round-off.
        variance = np.maximum(variance, 0.0)

        std = np.sqrt(variance)

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            denominators = std + self.epsilon
            # Equivalent to reciprocal weights divided by their maximum,
            # without forming reciprocals that can overflow for tiny epsilon.
            weights = np.min(denominators) / denominators
        if not np.isfinite(weights).all() or not np.any(weights > 0.0):
            raise FloatingPointError("Invalid feature weights.")
        # A common positive factor leaves weighted Pearson unchanged.
        return weights / np.max(weights)
