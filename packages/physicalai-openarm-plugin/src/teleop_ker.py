"""Teleoperate the bimanual OpenArm follower directly from a KER exoskeleton leader.

Unlike ``67.py`` (camera + ``PoseEstimator``), this reads joint angles straight
from the KER leader device via ``openarm_ker`` and drives the followers with
them -- no pose estimation involved. Angle filtering and the raw-degrees ->
per-side 8-joint retargeting are ported from ``dora-openarm-ker``'s
``KerPoseProcessor`` (see /home/intel/dora-openarm-ker), just kept in degrees
since ``OpenArmFollower.send_action`` already takes degrees and clips them to
the URDF limits itself.
"""

import argparse
import time

import numpy as np
from openarm_ker.ker_stream import CMD_STANDBY, CMD_STREAM, KERStream

from physicalai_openarm_plugin import BimanualOpenArmFollower, OpenArmFollower

CONTROL_HZ = 100.0
CONTROL_PERIOD = 1.0 / CONTROL_HZ


def map_range(x, in_min, in_max, out_min, out_max):
    """Map a value from one range to another, with clipping."""
    if in_max == in_min:
        return out_min
    x = max(min(x, in_max), in_min) if in_min < in_max else max(min(x, in_min), in_max)
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class OnlineHampelFilter:
    """Real-time Hampel filter to eliminate hardware spikes with zero phase delay."""

    def __init__(self, window_size: int = 5, n_sigmas: float = 3.0, min_threshold: float = 5.0):
        """Initialize the Hampel filter parameters and historical buffer."""
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.min_threshold = min_threshold
        self.scale_factor = 1.4826
        self.history = None

    def process(self, current_values: list[float]) -> list[float]:
        """Process streaming multi-channel data to detect and suppress sudden spikes."""
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

    def __init__(self, use_hampel: bool = False):
        """Initialize the retargeting processor with an optional Hampel filter."""
        self.hampel = (
            OnlineHampelFilter(window_size=5, n_sigmas=3.0, min_threshold=5.0) if use_hampel else None
        )

    def process(self, raw_angles: list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Filter raw encoder degrees and split into (left, right) 8-joint degree vectors."""
        filtered = self.hampel.process(raw_angles) if self.hampel else raw_angles

        grip_r_deg = map_range(filtered[7], 0.0, -60.0, -60.0, 10.0)
        grip_l_deg = map_range(filtered[15], 0.0, 60.0, 60.0, -10.0)

        pos_right = np.array([*filtered[:7], grip_r_deg], dtype=np.float32)
        pos_left = np.array([*filtered[8:15], grip_l_deg], dtype=np.float32)

        return pos_left, pos_right


class Robot:
    """Bimanual OpenArm follower pair driven by direct CAN joint targets."""

    def __init__(self, left_port: str = "can1", right_port: str = "can0"):
        """Connect both follower arms over their SocketCAN interfaces."""
        left = OpenArmFollower(left_port, side="left")
        right = OpenArmFollower(right_port, side="right")
        self.robot = BimanualOpenArmFollower(left, right)
        self.robot.connect()

    def move_to_pos(self, pos: np.ndarray, goal_time: float = CONTROL_PERIOD) -> None:
        """Set the target position for the background control loop. Returns immediately."""
        self.robot.send_action(pos, goal_time=goal_time)

    def disconnect(self):
        """Disconnect both follower arms."""
        self.robot.disconnect()


def main():
    """Teleoperate the OpenArm follower pair from a connected KER leader device."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hampel", action="store_true", help="Enable Hampel filter", default=False)
    parser.add_argument("--left-port", default="can1", help="SocketCAN interface for the left follower")
    parser.add_argument("--right-port", default="can0", help="SocketCAN interface for the right follower")
    args = parser.parse_args()

    print("Connecting to KER leader device...")
    stream = KERStream()
    stream.connect()
    stream.send_command(CMD_STREAM)

    print("\n=== Verified Device Metadata ===")
    print(f" Hardware : {stream.metadata.get('hw')}")
    print(f" Firmware : {stream.metadata.get('fw')}")
    print(f" Updated  : {stream.metadata.get('updated')}")
    print("================================\n")

    processor = KerTeleopProcessor(use_hampel=args.hampel)
    robot = Robot(left_port=args.left_port, right_port=args.right_port)

    print(f"KER Teleop Running at {CONTROL_HZ:g} Hz. Press Ctrl+C to stop.\n")
    try:
        while True:
            tick_start = time.perf_counter()

            data = stream.latest()
            if data is not None:
                pos_left, pos_right = processor.process(data["angles"])
                combined = np.concatenate((pos_left, pos_right))
                robot.move_to_pos(combined, goal_time=CONTROL_PERIOD)

            elapsed = time.perf_counter() - tick_start
            time.sleep(max(0.0, CONTROL_PERIOD - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down: Sending STANDBY command...")
        try:
            stream.send_command(CMD_STANDBY)
            time.sleep(0.1)
        except Exception:
            pass
        stream.close()
        robot.disconnect()
        print("KER Teleop Disconnected safely.")


if __name__ == "__main__":
    main()
