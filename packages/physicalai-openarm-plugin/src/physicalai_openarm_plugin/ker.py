"""KER exoskeleton angle filtering and retargeting onto OpenArm joint targets.

Ported from ``dora-openarm-ker``'s ``KerPoseProcessor``, kept in degrees since
``OpenArmFollower.send_action`` takes degrees and clips them to its limits.
"""

from __future__ import annotations

import numpy as np

# Number of encoder channels the KER device streams: 8 right, then 8 left.
KER_NUM_ANGLES = 16


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map a value from one range to another, with clipping.

    Returns:
        ``x`` clipped to the input range and linearly mapped onto the output range.
    """
    if in_max == in_min:
        return out_min
    x = max(min(x, in_max), in_min) if in_min < in_max else max(min(x, in_min), in_max)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class OnlineHampelFilter:
    """Real-time Hampel filter to eliminate hardware spikes with zero phase delay."""

    def __init__(self, window_size: int = 5, n_sigmas: float = 3.0, min_threshold: float = 5.0) -> None:
        """Initialize the Hampel filter parameters and historical buffer."""
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.min_threshold = min_threshold
        self.scale_factor = 1.4826
        self.history: np.ndarray | None = None

    def process(self, current_values: list[float]) -> list[float]:
        """Process streaming multi-channel data to detect and suppress sudden spikes.

        Returns:
            The input values with detected spikes replaced by the window median.
        """
        curr = np.array(current_values, dtype=np.float64)

        if self.history is None:
            self.history = np.tile(curr, (self.window_size, 1))
            return current_values

        medians = np.median(self.history, axis=0)
        mads = np.median(np.abs(self.history - medians), axis=0)
        thresholds = np.maximum(self.n_sigmas * self.scale_factor * mads, self.min_threshold)

        filtered_values = np.where(np.abs(curr - medians) > thresholds, medians, curr)

        # Track raw samples, not the suppressed output: feeding filtered_values
        # back in would let a real, sustained change (e.g. a fast gripper snap)
        # get permanently mistaken for a spike, since the median would then
        # never move away from the stale baseline.
        self.history[:-1] = self.history[1:]
        self.history[-1] = curr

        return filtered_values.tolist()


class KerTeleopProcessor:
    """Turn raw KER encoder degrees into left/right OpenArm 8-joint degree targets."""

    def __init__(self, *, use_hampel: bool = False) -> None:
        """Initialize the retargeting processor with an optional Hampel filter."""
        self.hampel = OnlineHampelFilter(window_size=5, n_sigmas=3.0, min_threshold=5.0) if use_hampel else None

    def process(self, raw_angles: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Filter raw encoder degrees and split into (left, right) 8-joint degree vectors.

        Returns:
            The left and right arm targets in canonical OpenArm joint order.

        Raises:
            ValueError: If the device did not report all 16 encoder angles.
        """
        if len(raw_angles) < KER_NUM_ANGLES:
            msg = f"Expected {KER_NUM_ANGLES} KER angles, got {len(raw_angles)}"
            raise ValueError(msg)
        filtered = self.hampel.process(raw_angles) if self.hampel else raw_angles

        grip_r_deg = map_range(filtered[7], 0.0, -60.0, -60.0, 10.0)
        grip_l_deg = map_range(filtered[15], 0.0, 60.0, 60.0, -10.0)

        pos_right = np.array([*filtered[:7], grip_r_deg], dtype=np.float32)
        pos_left = np.array([*filtered[8:15], grip_l_deg], dtype=np.float32)

        return pos_left, pos_right
