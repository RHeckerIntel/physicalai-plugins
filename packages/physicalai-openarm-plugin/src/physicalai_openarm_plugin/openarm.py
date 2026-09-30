# ruff: noqa: DOC501, D107, PLR6301

"""PhysicalAI robot implementations for direct OpenArm control."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from physicalai.config import export_config

from physicalai_openarm_plugin.constants import (
    DEFAULT_POSITION_KD,
    DEFAULT_POSITION_KP,
    LEFT_JOINT_LIMITS_DEG,
    NUM_OPENARM_JOINTS,
    OPENARM_JOINT_ORDER,
    OPENARM_MOTOR_CONFIG,
    RIGHT_JOINT_LIMITS_DEG,
)
from physicalai_openarm_plugin.damiao import DamiaoSerial, DamiaoSocketCAN

if TYPE_CHECKING:
    from physicalai.capture.frame import Frame
    from physicalai.robot.interface import RobotObservation

OpenArmSide = Literal["left", "right"]
OpenArmCANAdapter = Literal["socketcan", "damiao"]


@dataclass
class OpenArmObservation:
    """OpenArm state in canonical joint order and degrees."""

    joint_positions: np.ndarray
    timestamp: float
    sensor_data: dict[str, np.ndarray] | None = None
    images: dict[str, Frame] | None = None

    @property
    def state(self) -> np.ndarray:
        """Primary position state vector."""
        return self.joint_positions


class _OpenArmBase:
    JOINT_ORDER: ClassVar[tuple[str, ...]] = OPENARM_JOINT_ORDER
    NUM_JOINTS: ClassVar[int] = NUM_OPENARM_JOINTS

    def __init__(
        self,
        port: str,
        *,
        can_adapter: OpenArmCANAdapter = "socketcan",
        dm_serial_baud: int = 921_600,
        use_can_fd: bool = True,
        can_bitrate: int = 1_000_000,
        can_data_bitrate: int = 5_000_000,
        response_timeout: float = 0.02,
        _transport: DamiaoSocketCAN | DamiaoSerial | None = None,
    ) -> None:
        if not port:
            msg = "port must be a non-empty CAN interface or Damiao USB serial device"
            raise ValueError(msg)
        if can_bitrate <= 0 or can_data_bitrate <= 0 or response_timeout <= 0:
            msg = "CAN bitrates and response_timeout must be positive"
            raise ValueError(msg)
        if can_adapter not in {"socketcan", "damiao"}:
            msg = "can_adapter must be 'socketcan' or 'damiao'"
            raise ValueError(msg)
        self._port = port
        self._can_adapter = can_adapter
        self._transport = _transport or self._make_transport(
            port=port,
            can_adapter=can_adapter,
            dm_serial_baud=dm_serial_baud,
            use_can_fd=use_can_fd,
            can_bitrate=can_bitrate,
            can_data_bitrate=can_data_bitrate,
            response_timeout=response_timeout,
        )
        # Guards every direct call into `self._transport`: a follower's background
        # motion loop (see OpenArmFollower) drives it from its own thread while
        # get_observation() may be called concurrently from the caller's thread.
        self._transport_lock = threading.Lock()

    @staticmethod
    def _make_transport(
        *,
        port: str,
        can_adapter: OpenArmCANAdapter,
        dm_serial_baud: int,
        use_can_fd: bool,
        can_bitrate: int,
        can_data_bitrate: int,
        response_timeout: float,
    ) -> DamiaoSocketCAN | DamiaoSerial:
        if can_adapter == "damiao":
            return DamiaoSerial(port, OPENARM_MOTOR_CONFIG, baud=dm_serial_baud)
        return DamiaoSocketCAN(
            port,
            OPENARM_MOTOR_CONFIG,
            use_can_fd=use_can_fd,
            bitrate=can_bitrate,
            data_bitrate=can_data_bitrate,
            response_timeout=response_timeout,
        )

    @property
    def port(self) -> str:
        """Configured SocketCAN channel or Damiao USB serial device."""
        return self._port

    @property
    def joint_names(self) -> list[str]:
        """Fixed arm-first joint order in degrees."""
        return list(self.JOINT_ORDER)

    @property
    def device_ids(self) -> tuple[str, ...]:
        """Stable identity of the exclusively owned CAN interface."""
        return (f"openarm:{self._can_adapter}:{self.port}",)

    def is_connected(self) -> bool:
        """Return whether the CAN transport is connected."""
        return self._transport.is_connected

    def _observation(self) -> OpenArmObservation:
        with self._transport_lock:
            states = self._transport.read_states()
        positions = np.array([states[name].position for name in self.JOINT_ORDER], dtype=np.float32)
        velocities = np.array([states[name].velocity for name in self.JOINT_ORDER], dtype=np.float32)
        torques = np.array([states[name].torque for name in self.JOINT_ORDER], dtype=np.float32)
        return OpenArmObservation(positions, time.monotonic(), {"velocities": velocities, "torques": torques})


@export_config(class_path="physicalai_openarm_plugin.OpenArmFollower")
class OpenArmFollower(_OpenArmBase):
    """Direct OpenArm follower with side-specific position safety limits.

    Position targets are not written to the motors synchronously. Instead,
    ``send_action`` hands the clipped target to a background motion loop and
    returns immediately; the loop ramps the commanded position from wherever
    it currently is to the new target over ``goal_time`` seconds, streaming
    interpolated setpoints at ``control_hz``. This keeps the arm from
    jumping at full (motor-PD-limited) speed whenever a caller's targets are
    far apart, without ever blocking the caller.
    """

    def __init__(
        self,
        port: str,
        *,
        side: OpenArmSide,
        can_adapter: OpenArmCANAdapter = "socketcan",
        dm_serial_baud: int = 921_600,
        disable_torque_on_disconnect: bool = True,
        use_can_fd: bool = True,
        can_bitrate: int = 1_000_000,
        can_data_bitrate: int = 5_000_000,
        response_timeout: float = 0.02,
        max_relative_target: float | None = None,
        position_kp: tuple[float, ...] = DEFAULT_POSITION_KP,
        position_kd: tuple[float, ...] = DEFAULT_POSITION_KD,
        control_hz: float = 200.0,
        _transport: DamiaoSocketCAN | DamiaoSerial | None = None,
    ) -> None:
        if side not in {"left", "right"}:
            msg = "side must be 'left' or 'right'; OpenArm followers require explicit safety limits"
            raise ValueError(msg)
        if max_relative_target is not None and (not math.isfinite(max_relative_target) or max_relative_target <= 0):
            msg = "max_relative_target must be a finite positive number when provided"
            raise ValueError(msg)
        if len(position_kp) != self.NUM_JOINTS or len(position_kd) != self.NUM_JOINTS:
            msg = f"position_kp and position_kd must each contain {self.NUM_JOINTS} values"
            raise ValueError(msg)
        if not math.isfinite(control_hz) or control_hz <= 0:
            msg = "control_hz must be a finite positive number"
            raise ValueError(msg)
        super().__init__(
            port,
            can_adapter=can_adapter,
            dm_serial_baud=dm_serial_baud,
            use_can_fd=use_can_fd,
            can_bitrate=can_bitrate,
            can_data_bitrate=can_data_bitrate,
            response_timeout=response_timeout,
            _transport=_transport,
        )
        self.side = side
        self.disable_torque_on_disconnect = disable_torque_on_disconnect
        self.max_relative_target = max_relative_target
        self.position_kp = position_kp
        self.position_kd = position_kd
        self._limits = LEFT_JOINT_LIMITS_DEG if side == "left" else RIGHT_JOINT_LIMITS_DEG

        self._control_period = 1.0 / control_hz
        self._motion_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._motion_thread: threading.Thread | None = None
        self._motion_error: Exception | None = None
        # All in JOINT_ORDER, degrees. `_commanded` is the loop's current setpoint;
        # `_traj_start`/`_traj_goal`/`_traj_start_time`/`_traj_goal_time` describe the
        # in-flight ramp it is interpolating along.
        self._commanded: np.ndarray | None = None
        self._traj_start: np.ndarray | None = None
        self._traj_goal: np.ndarray | None = None
        self._traj_start_time = 0.0
        self._traj_goal_time = 0.0

    def connect(self) -> None:
        """Connect, validate all eight motors, and start the motion loop."""
        self._transport.connect()
        with self._transport_lock:
            states = self._transport.read_states()
        commanded = np.array([states[name].position for name in self.JOINT_ORDER], dtype=np.float64)
        with self._motion_lock:
            self._commanded = commanded
            self._traj_start = commanded.copy()
            self._traj_goal = commanded.copy()
            self._traj_start_time = time.monotonic()
            self._traj_goal_time = 0.0
            self._motion_error = None
        if self._motion_thread is None or not self._motion_thread.is_alive():
            self._stop_event.clear()
            self._motion_thread = threading.Thread(target=self._motion_loop, daemon=True)
            self._motion_thread.start()

    def disconnect(self) -> None:
        """Stop the motion loop and release the CAN transport, disabling torque by default."""
        self._stop_event.set()
        if self._motion_thread is not None:
            self._motion_thread.join(timeout=1.0)
            self._motion_thread = None
        self._transport.disconnect(disable_torque=self.disable_torque_on_disconnect)

    def get_observation(self) -> RobotObservation:
        """Return positions, velocities, and torques from all motors."""
        return self._observation()

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Clip a full 8-element degree position target and hand it to the motion loop.

        Returns immediately without touching the transport. The background
        motion loop ramps toward ``action`` over ``goal_time`` seconds
        (``0`` requests an immediate jump, matched at the next control tick).
        """
        if action.shape != (self.NUM_JOINTS,):
            msg = f"Expected action shape ({self.NUM_JOINTS},), got {action.shape}"
            raise ValueError(msg)
        if not np.isfinite(action).all():
            msg = "OpenArm actions must contain only finite values"
            raise ValueError(msg)
        if not math.isfinite(goal_time) or goal_time < 0:
            msg = "goal_time must be a finite, non-negative number"
            raise ValueError(msg)
        if self._motion_error is not None:
            raise self._motion_error
        with self._motion_lock:
            reference = self._commanded
        if reference is None:
            msg = "OpenArmFollower must be connected before sending actions"
            raise RuntimeError(msg)

        max_relative_target = self.max_relative_target
        target = np.empty(self.NUM_JOINTS, dtype=np.float64)
        for index, name in enumerate(self.JOINT_ORDER):
            lower, upper = self._limits[name]
            value = float(np.clip(action[index], lower, upper))
            if max_relative_target is not None:
                value = float(
                    np.clip(value, reference[index] - max_relative_target, reference[index] + max_relative_target),
                )
            target[index] = value

        with self._motion_lock:
            self._traj_start = self._commanded if self._commanded is not None else reference
            self._traj_goal = target
            self._traj_start_time = time.monotonic()
            self._traj_goal_time = goal_time

    def _motion_loop(self) -> None:
        """Stream interpolated setpoints toward the latest goal at ``control_hz``."""
        while not self._stop_event.is_set():
            tick_start = time.perf_counter()
            with self._motion_lock:
                start, goal = self._traj_start, self._traj_goal
                start_time, goal_time = self._traj_start_time, self._traj_goal_time
            if goal is not None:
                fraction = 1.0 if goal_time <= 0 else min(1.0, (time.monotonic() - start_time) / goal_time)
                commanded = start + fraction * (goal - start)
                commands = {
                    name: (self.position_kp[index], self.position_kd[index], float(commanded[index]))
                    for index, name in enumerate(self.JOINT_ORDER)
                }
                try:
                    with self._transport_lock:
                        self._transport.send_positions(commands)
                except Exception as error:  # noqa: BLE001 - surfaced to the caller via send_action
                    self._motion_error = error
                    return
                with self._motion_lock:
                    self._commanded = commanded
            elapsed = time.perf_counter() - tick_start
            self._stop_event.wait(max(0.0, self._control_period - elapsed))


@export_config(class_path="physicalai_openarm_plugin.OpenArmLeader")
class OpenArmLeader(_OpenArmBase):
    """Direct, read-only OpenArm leader for hand-guided unilateral teleoperation."""

    def __init__(
        self,
        port: str,
        *,
        manual_control: bool = True,
        can_adapter: OpenArmCANAdapter = "socketcan",
        dm_serial_baud: int = 921_600,
        use_can_fd: bool = True,
        can_bitrate: int = 1_000_000,
        can_data_bitrate: int = 5_000_000,
        response_timeout: float = 0.02,
        _transport: DamiaoSocketCAN | DamiaoSerial | None = None,
    ) -> None:
        super().__init__(
            port,
            can_adapter=can_adapter,
            dm_serial_baud=dm_serial_baud,
            use_can_fd=use_can_fd,
            can_bitrate=can_bitrate,
            can_data_bitrate=can_data_bitrate,
            response_timeout=response_timeout,
            _transport=_transport,
        )
        self.manual_control = manual_control

    def connect(self) -> None:
        """Connect and leave torque disabled when configured for hand guidance."""
        self._transport.connect()
        if self.manual_control:
            self._transport.disable_torque()

    def disconnect(self) -> None:
        """Release the leader bus and retain its manual-control torque state."""
        self._transport.disconnect(disable_torque=self.manual_control)

    def get_observation(self) -> RobotObservation:
        """Return the leader's current degree positions and measured state."""
        return self._observation()

    def send_action(self, action: np.ndarray, *, goal_time: float = 0.1) -> None:
        """Ignore runtime writes because OpenArm leader feedback is unsupported."""
        _ = action, goal_time
