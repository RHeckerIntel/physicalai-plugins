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
from physicalai_openarm_plugin.ker import KerTeleopProcessor

CONTROL_HZ = 100.0
CONTROL_PERIOD = 1.0 / CONTROL_HZ


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
