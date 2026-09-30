# ruff: noqa: DOC201, DOC501, PLR6301, S101, UP046

"""Bimanual OpenArm drivers: paired CAN followers and a KER exoskeleton leader."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import numpy as np
from openarm_ker.ker_stream import CMD_STREAM, KERStream
from physicalai.config import export_config

from physicalai_openarm_plugin.constants import NUM_BIMANUAL_OPENARM_JOINTS, NUM_OPENARM_JOINTS, OPENARM_JOINT_ORDER
from physicalai_openarm_plugin.ker import KerTeleopProcessor
from physicalai_openarm_plugin.openarm import OpenArmFollower, OpenArmObservation

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

ArmT = TypeVar("ArmT", bound=OpenArmFollower)
KerTransport = Literal["usb", "serial"]


@dataclass
class BimanualOpenArmObservation:
    """Combined left-then-right OpenArm observation."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Combined primary state vector."""
        return self.joint_positions


class _BimanualOpenArm(Generic[ArmT]):
    NUM_JOINTS = NUM_BIMANUAL_OPENARM_JOINTS

    def __init__(self, left: ArmT, right: ArmT) -> None:
        if type(left) is not type(right):
            msg = "Both OpenArm instances must have the same driver type"
            raise ValueError(msg)
        if left.port == right.port:
            msg = "Left and right OpenArms must use distinct SocketCAN interfaces"
            raise ValueError(msg)
        self.left = left
        self.right = right

    @property
    def joint_names(self) -> list[str]:
        """Left then right prefixed canonical joint names."""
        return [f"left_{name}" for name in self.left.joint_names] + [f"right_{name}" for name in self.right.joint_names]

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Stable identities for both independent CAN interfaces."""
        return tuple(sorted(self.left.device_ids + self.right.device_ids))

    def connect(self) -> None:
        """Connect both arms and roll back left if right fails."""
        self.left.connect()
        try:
            self.right.connect()
        except Exception:
            self.left.disconnect()
            raise

    def disconnect(self) -> None:
        """Disconnect both arms even when one teardown fails."""
        first_error: Exception | None = None
        for arm in (self.left, self.right):
            try:
                arm.disconnect()
            except Exception as error:  # noqa: BLE001  # pragma: no cover - hardware defensive path
                first_error = first_error or error
        if first_error is not None:
            raise first_error

    def is_connected(self) -> bool:
        """Return whether both arms are connected."""
        return self.left.is_connected() and self.right.is_connected()

    def get_observation(self) -> RobotObservation:
        """Read and concatenate both arm states."""
        left = self.left.get_observation()
        right = self.right.get_observation()
        assert isinstance(left, OpenArmObservation)
        assert isinstance(right, OpenArmObservation)
        sensor_data = None
        if left.sensor_data is not None and right.sensor_data is not None:
            sensor_data = {
                name: np.concatenate((left.sensor_data[name], right.sensor_data[name]))
                for name in left.sensor_data.keys() & right.sensor_data.keys()
            }
        return BimanualOpenArmObservation(
            np.concatenate((left.joint_positions, right.joint_positions)),
            left.timestamp,
            sensor_data,
        )


@export_config(class_path="physicalai_openarm_plugin.BimanualOpenArmFollower")
class BimanualOpenArmFollower(_BimanualOpenArm[OpenArmFollower]):
    """Two directly controlled OpenArm followers."""

    def __init__(self, left: OpenArmFollower, right: OpenArmFollower) -> None:
        """Initialize the paired follower drivers."""
        super().__init__(left, right)

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Split a 16-element target vector and send it to both followers."""
        if action.shape != (self.NUM_JOINTS,):
            msg = f"Expected action shape ({self.NUM_JOINTS},), got {action.shape}"
            raise ValueError(msg)
        self.left.send_action(action[:NUM_OPENARM_JOINTS], goal_time=goal_time)
        self.right.send_action(action[NUM_OPENARM_JOINTS:], goal_time=goal_time)


@export_config(class_path="physicalai_openarm_plugin.BimanualOpenArmLeader")
class BimanualOpenArmLeader:
    """Read-only bimanual leader driven by a KER exoskeleton over ``KERStream``.

    The KER device streams 16 encoder angles (right arm then left arm), which
    are filtered and retargeted onto the same left-then-right 16-joint layout
    that ``BimanualOpenArmFollower`` accepts as its action.
    """

    NUM_JOINTS = NUM_BIMANUAL_OPENARM_JOINTS

    def __init__(
        self,
        *,
        transport: KerTransport = "usb",
        port: str = "/dev/ttyACM0",
        use_hampel: bool = False,
        first_frame_timeout: float = 2.0,
        _stream: KERStream | None = None,
    ) -> None:
        """Configure the KER stream; the device is not opened until ``connect``."""
        if transport not in {"usb", "serial"}:
            msg = "transport must be 'usb' or 'serial'"
            raise ValueError(msg)
        if first_frame_timeout <= 0:
            msg = "first_frame_timeout must be positive"
            raise ValueError(msg)
        self._transport = transport
        self._port = port
        self._first_frame_timeout = first_frame_timeout
        self._stream = _stream or KERStream(transport=transport, port=port)
        self._processor = KerTeleopProcessor(use_hampel=use_hampel)
        self._connected = False

    @property
    def joint_names(self) -> list[str]:
        """Left then right prefixed canonical joint names."""
        return [f"left_{name}" for name in OPENARM_JOINT_ORDER] + [f"right_{name}" for name in OPENARM_JOINT_ORDER]

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Stable identity of the exclusively owned KER device."""
        if self._transport == "usb":
            return ("openarm-ker:usb",)
        return (f"openarm-ker:serial:{self._port}",)

    def connect(self) -> None:
        """Open the KER device, start streaming, and wait for the first frame."""
        self._stream.connect()
        try:
            self._stream.send_command(CMD_STREAM)
            self._wait_for_first_frame()
        except Exception:
            self._stream.close()
            raise
        self._connected = True

    def _wait_for_first_frame(self) -> None:
        deadline = time.monotonic() + self._first_frame_timeout
        while self._stream.latest() is None:
            if time.monotonic() > deadline:
                msg = f"No data from KER device within {self._first_frame_timeout:g}s"
                raise TimeoutError(msg)
            time.sleep(0.01)

    def disconnect(self) -> None:
        """Put the KER device in standby and release it."""
        self._connected = False
        self._stream.close()

    def is_connected(self) -> bool:
        """Return whether the KER stream session is open."""
        return self._connected and self._stream.is_connected

    def get_observation(self) -> RobotObservation:
        """Return the retargeted left-then-right joint targets in degrees."""
        if not self._stream.is_link_up:
            msg = "KER device link is down"
            raise RuntimeError(msg)
        data = self._stream.latest()
        if data is None:
            msg = "No data received from KER device"
            raise RuntimeError(msg)
        left, right = self._processor.process(data["angles"])
        return BimanualOpenArmObservation(np.concatenate((left, right)), time.monotonic())

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Reject feedback writes because the KER leader is read-only."""
        _ = action, goal_time
        msg = "Cannot send actions to the KER leader. Bilateral feedback is not implemented."
        raise RuntimeError(msg)
