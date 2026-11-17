"""Zeroth-order, cumulative intersection-of-confidence-intervals test.

Implements the minimum-variance estimator in Alippi, Boracchi and Roveri
(2011), Sections 3.2 and 3.4, applied to the FW-FEDD distance sequence.
This is a reference-based reconstruction, not recovered author code. The
thesis does not fully specify whether a temporal LPA kernel was used.
"""

import numpy as np


METHOD = "ICI_CUMULATIVE_ORDER_ZERO_MINIMUM_VARIANCE"

class ICIDetector:
    """Monitor a scalar sequence using cumulative zeroth-order fits.

    The first training_size distances initialize the sample mean and sample
    standard deviation (ddof=1). Monitoring starts on the next distance.
    The mean includes calibration and all subsequent distances in the same
    concept. The calibration standard deviation remains fixed, and the
    standard error is reference_std / sqrt(n).

    Each new interval is intersected with the complete previous intersection.
    An empty intersection signals a two-sided change. No warning threshold,
    fixed-block cadence, covariance correction or extra cooldown is used.
    The caller must reset after a drift and supply a new stable calibration.

    Gaussian/independence assumptions of the reference estimator are not
    guaranteed for distances derived from overlapping feature windows.
    gamma is an interval multiplier, not a sequential false-alarm guarantee.

    """

    def __init__(self, training_size=300, gamma=2.5):
        if (isinstance(training_size, bool)
                or not isinstance(training_size, (int, np.integer))
                or training_size < 2):
            raise ValueError("training_size must be an integer >= 2.")
        self.training_size = int(training_size)
        self.gamma = float(gamma)
        if not np.isfinite(self.gamma) or self.gamma <= 0.0:
            raise ValueError("gamma must be finite and positive.")
        self.method = METHOD
        self.reset()

    def reset(self):
        """Clear calibration, cumulative mean and historical intersection."""
        self.n = 0
        self.n_decisions = 0
        self.ready = False
        self.test_performed = False
        self.drift_detected = False
        self.last_distance = None
        self.current_mean = None
        self.reference_mean = None
        self.reference_std = None
        self.standard_error = None
        self.reference_lower = None
        self.reference_upper = None
        self.interval_lower = None
        self.interval_upper = None
        self.lower_bound = None
        self.upper_bound = None
        self._training_M2 = 0.0
        self._arithmetic_mean = 0.0

    @property
    def statistics(self):
        """Return a detached snapshot that survives an external reset."""
        return {
            "weighted_distance": self.last_distance,
            "internal_mean": self.current_mean,
            "intersection_lower": self.lower_bound,
            "intersection_upper": self.upper_bound,
            "test_performed": self.test_performed,
            "reference_std": self.reference_std,
            "standard_error": self.standard_error,
            "current_interval_lower": self.interval_lower,
            "current_interval_upper": self.interval_upper,
            "reference_mean": self.reference_mean,
            "reference_lower": self.reference_lower,
            "reference_upper": self.reference_upper,
            "distance_count": self.n,
            "decision_count": self.n_decisions,
            "ready": self.ready,
        }

    def _set_interval(self):
        self.standard_error = self.reference_std / np.sqrt(self.n)
        radius = self.gamma * self.standard_error
        self.interval_lower = self.current_mean - radius
        self.interval_upper = self.current_mean + radius
        if not np.isfinite((self.current_mean, self.reference_std,
                            self.standard_error, self.interval_lower,
                            self.interval_upper)).all():
            raise FloatingPointError("ICI produced non-finite interval statistics.")

    def update(self, distance):
        """Return (warning, drift, below_warning) for one finite distance.

        Calibration never emits an alarm. After detection the alarm remains
        latched until reset, without consuming additional distances.
        Exactly constant calibration produces a zero-width interval; no
        arbitrary variance floor is added to the reference formula.
        """
        distance = float(distance)
        if not np.isfinite(distance):
            raise ValueError("distance must be finite.")
        self.test_performed = False
        if self.drift_detected:
            return False, True, False

        # Compute candidate cumulative statistics before committing them.
        next_n = self.n + 1
        with np.errstate(over="raise", invalid="raise"):
            delta = np.float64(distance) - self._arithmetic_mean
            next_mean = self._arithmetic_mean + delta / next_n
            next_M2 = self._training_M2
            if not self.ready:
                next_M2 += delta * (distance - next_mean)
        if not np.isfinite((next_mean, next_M2)).all():
            raise FloatingPointError("ICI produced non-finite cumulative statistics.")

        self.last_distance = distance
        self.n = next_n
        self._arithmetic_mean = float(next_mean)
        self.current_mean = self._arithmetic_mean

        if not self.ready:
            self._training_M2 = float(next_M2)
            if self.n == self.training_size:
                self.reference_mean = self.current_mean
                self.reference_std = float(np.sqrt(
                    max(0.0, self._training_M2 / (self.n - 1))))
                self._set_interval()
                self.reference_mean = self.current_mean
                self.reference_lower = self.interval_lower
                self.reference_upper = self.interval_upper
                self.lower_bound = self.interval_lower
                self.upper_bound = self.interval_upper
                self.ready = True
            return False, False, True

        self._set_interval()
        self.lower_bound = max(self.lower_bound, self.interval_lower)
        self.upper_bound = min(self.upper_bound, self.interval_upper)
        self.test_performed = True
        self.n_decisions += 1
        self.drift_detected = bool(self.lower_bound > self.upper_bound)
        return False, self.drift_detected, not self.drift_detected

