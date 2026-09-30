import threading
import time

import cv2
import numpy as np
from physicalai.capture import RealSenseCamera
from physicalai.robot import RobotObservation

from physicalai_openarm_plugin import OpenArmFollower, BimanualOpenArmFollower
from pose_estimate import PoseEstimator


class Robot:
    def __init__(self, max_velocity_deg_s: float = 60.0, control_hz: float = 200.0):
        left = OpenArmFollower("can1", side="left")
        right = OpenArmFollower("can0", side="right")
        self.robot = BimanualOpenArmFollower(left, right)
        self.robot.connect()

    def move_to_pos(self, pos: np.ndarray) -> None:
        """Set the target position for the background control loop. Returns immediately."""
        self.robot.send_action(pos, goal_time=0.5)

    def disconnect(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.robot.disconnect()


if __name__ == '__main__':
    estimator = PoseEstimator(swap_arms=False, gripper=0.0)

    cameras = RealSenseCamera.discover()
    print(cameras)
    camera = RealSenseCamera(serial_number="234322303853")
    camera.connect()
    robot = Robot()

    while True:
        frame = camera.read().data
        left, right = estimator.estimate(frame)
        left = left * np.asarray([-1, 1, -1, 1, 1, 1, 1, 1])
        combined = np.concatenate((left, right))

        vis = estimator.visualize(frame)

        cv2.imshow("Image", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(combined)
        robot.move_to_pos(combined)

    robot.disconnect()
