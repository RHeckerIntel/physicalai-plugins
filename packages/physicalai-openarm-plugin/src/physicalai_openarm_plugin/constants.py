"""OpenArm hardware constants based on its documented Damiao configuration."""

from __future__ import annotations

from typing import Final

OPENARM_JOINT_ORDER: Final[tuple[str, ...]] = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_yaw",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
)
NUM_OPENARM_JOINTS: Final[int] = len(OPENARM_JOINT_ORDER)
NUM_BIMANUAL_OPENARM_JOINTS: Final[int] = NUM_OPENARM_JOINTS * 2

OPENARM_MOTOR_CONFIG: Final[dict[str, tuple[int, int, str]]] = {
    "shoulder_pitch": (0x01, 0x11, "dm8009"),
    "shoulder_roll": (0x02, 0x12, "dm8009"),
    "shoulder_yaw": (0x03, 0x13, "dm4340"),
    "elbow": (0x04, 0x14, "dm4340"),
    "wrist_yaw": (0x05, 0x15, "dm4310"),
    "wrist_pitch": (0x06, 0x16, "dm4310"),
    "wrist_roll": (0x07, 0x17, "dm4310"),
    "gripper": (0x08, 0x18, "dm4310"),
}

MOTOR_LIMITS: Final[dict[str, tuple[float, float, float]]] = {
    "dm4310": (12.5, 30.0, 10.0),
    "dm4340": (12.5, 8.0, 28.0),
    "dm8009": (12.5, 45.0, 54.0),
}

DEFAULT_POSITION_KP: Final[tuple[float, ...]] = (240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 25.0)
DEFAULT_POSITION_KD: Final[tuple[float, ...]] = (5.0, 5.0, 5.0, 5.0, 0.5, 0.5, 0.5, 0.5)

LEFT_JOINT_LIMITS_DEG: Final[dict[str, tuple[float, float]]] = {
    "shoulder_pitch": (-175.0, 175.0),
    "shoulder_roll": (-90.0, 90.0),
    "shoulder_yaw": (-85.0, 85.0),
    "elbow": (0.0, 135.0),
    "wrist_yaw": (-85.0, 85.0),
    "wrist_pitch": (-40.0, 40.0),
    "wrist_roll": (-80.0, 80.0),
    "gripper": (0.0, 65.0),
}
RIGHT_JOINT_LIMITS_DEG: Final[dict[str, tuple[float, float]]] = {
    "shoulder_pitch": (-175.0, 175.0),
    "shoulder_roll": (-90.0, 90.0),
    "shoulder_yaw": (-85.0, 85.0),
    "elbow": (0.0, 135.0),
    "wrist_yaw": (-85.0, 85.0),
    "wrist_pitch": (-40.0, 40.0),
    "wrist_roll": (-80.0, 80.0),
    "gripper": (-65.0, 0.0),
}

CAN_CMD_ENABLE: Final[int] = 0xFC
CAN_CMD_DISABLE: Final[int] = 0xFD
CAN_CMD_REFRESH: Final[int] = 0xCC
CAN_PARAM_ID: Final[int] = 0x7FF
