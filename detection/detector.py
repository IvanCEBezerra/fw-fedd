"""Online FW-FEDD with inverse-standard-deviation weights and cumulative ICI."""
from collections import deque

import numpy as np

from fw_fedd.features.extractor import extract_features
from fw_fedd.features.stl import estimate_stl_period
from fw_fedd.weighting.fw_module import FeatureWeighter
from fw_fedd.detection.pearson import weighted_pearson_distance
from fw_fedd.detection.ici import ICIDetector


def _validate_integer(value, name, minimum=1):
    if (isinstance(value, bool)
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


class FWFEDDDetector:
    """Process one finite univariate observation at a time.

    Defaults match the selected ici_gamma25 configuration: window 150,
    training 300 distances, gamma 2.5, causal period estimation and std weights.
    Event indices are global and zero-based. Calibration emits no alarms.
    After an alarm, a new reference and calibration start on the next input.
    """

    def __init__(self, window_size=150, period=None, epsilon=1e-6,
                 training_size=300, gamma=2.5):
        self.window_size = _validate_integer(window_size, "window_size", 7)
        self.period = (None if period is None
                       else _validate_integer(period, "period", 2))
        if self.period is not None and self.period >= self.window_size:
            raise ValueError("period must be smaller than window_size.")
        self.epsilon = float(epsilon)
        self.weighter = FeatureWeighter(n_features=23, epsilon=self.epsilon)
        self.ici = ICIDetector(training_size=training_size, gamma=gamma)
        self.method = f"FW-FEDD / {self.ici.method}"
        self.reset()

    def reset(self):
        """
        Reset the complete detector.

        Clears events and restarts global observation indexing.
        Internal drift handling uses a separate concept reset,
        preserving global indices and recorded events.
        """
        self._buffer = deque(maxlen=self.window_size)

        self._t = -1
        self._s = 0

        self._fv0 = None
        self._active_period = None
        self.last_phase = "reference_window_collection"
        self.last_distance = None
        self.last_test_performed = False
        self.last_active_period = None
        self.last_test_statistics = {}

        self._state = "normal"
        self._detections = []
        self._n_drifts = 0

        self.weighter.reset()
        self.ici.reset()

    def _reset_concept(self):
        """
        Reset concept-specific statistics after a drift.

        The next reference window starts at the next observation.
        Global indexing and the event history are preserved.
        """
        self._s = self._t + 1

        self._buffer.clear()
        self._fv0 = None
        self._active_period = None

        self.weighter.reset()
        self.ici.reset()

    def update(self, value):
        """
        Process one observation.

        Parameters
        ----------
        value : float
            A finite observation available at the current timestamp.

        Returns
        -------
        str
            "normal" or "drift".

            A detected drift is returned on the same call that
            records it. Processing resumes in the normal state
            on the next observation.
        """
        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                "value must be a finite observation."
            )

        # A drift is a one-observation event.
        if self._state == "drift":
            self._state = "normal"

        self._t += 1
        self.last_test_performed = False
        self.last_distance = None
        self.last_phase = "reference_window_collection"
        self.last_active_period = self._active_period
        self.last_test_statistics = {}
        self._buffer.append(value)

        points_since_reset = self._t - self._s + 1

        if points_since_reset < self.window_size:
            return self._state

        current_window = np.asarray(
            self._buffer,
            dtype=float,
        )
        if self._active_period is None:
            self._active_period = (
                estimate_stl_period(current_window)
                if self.period is None else self.period)
        self.last_active_period = self._active_period

        feature_vector = np.asarray(
            extract_features(
                current_window,
                period=self._active_period,
            ),
            dtype=float,
        )

        if feature_vector.shape != (23,):
            raise ValueError(
                "extract_features must return an array "
                f"with shape (23,), got {feature_vector.shape}."
            )

        if not np.isfinite(feature_vector).all():
            raise FloatingPointError(
                "Feature extraction produced non-finite values."
            )

        # The first complete window defines the concept reference.
        if self._fv0 is None:
            # The reference is the first feature observation of this concept.
            # Include it so the first distance already has two observations
            # for the default inverse-standard-deviation weighting policy.
            self.weighter.update(feature_vector)
            self._fv0 = feature_vector.copy()
            return self._state

        self.weighter.update(feature_vector)
        weights = self.weighter.get_weights()

        distance = float(
            weighted_pearson_distance(
                self._fv0,
                feature_vector,
                weights,
            )
        )

        if not np.isfinite(distance):
            raise FloatingPointError(
                "Weighted Pearson distance is non-finite."
            )

        self.last_distance = distance
        ready_before = self.ici.ready
        self.last_phase = "monitoring" if ready_before else "internal_test_training"
        _, drift_signal, _ = (
            self.ici.update(distance)
        )
        self.last_test_performed = self.ici.test_performed
        # Preserve the actual decision limits before _reset_concept clears
        # the test. The next input clears this detached public snapshot.
        self.last_test_statistics = self.ici.statistics

        if drift_signal:
            self._detections.append(
                (self._t, "drift")
            )

            self._n_drifts += 1

            self._reset_concept()

            # Preserve the event in the public return value.
            self._state = "drift"
            return self._state

        return self._state

    @property
    def state(self):
        """Return the current detector state."""
        return self._state

    @property
    def detections(self):
        """Return a copy of the recorded (global_index, event) pairs."""
        return list(self._detections)

    @property
    def n_drifts(self):
        """Return the total number of recorded drift events."""
        return self._n_drifts

    @property
    def n_samples(self):
        """Return the number of observations processed."""
        return self._t + 1

