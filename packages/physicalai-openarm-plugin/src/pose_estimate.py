# ruff: file-ignore[print, undocumented-public-function, docstring-missing-returns, docstring-missing-exception, raise-vanilla-args, raw-string-in-exception, f-string-in-exception, too-many-locals]

"""Lightweight human-pose -> OpenArm joint retargeting.

``PoseEstimator`` runs MoveNet SinglePose Lightning (a ~9 MB ONNX model, real time)
via OpenVINO -- using an integrated/discrete GPU when one is available, falling
back to CPU otherwise -- on a single image or one webcam frame and maps the detected arm
keypoints to two 8-element joint vectors (joints 1..7 + gripper) that can be
handed straight to ``BimanualOpenArmFollower.send_action`` as ``concatenate((left,
right))``.

This is a deliberately simple frontal-plane retarget: only the shoulder swing
(joint 1), shoulder elevation (joint 2) and elbow flexion (joint 4) are actually
observed. Upper-arm roll and the three wrist joints (3, 5, 6, 7) are not
recoverable from a single 2D view, so they are left at 0; the gripper is a fixed
value you pass in. Every output is clipped to the per-side limits from
``physicalai_openarm_plugin.constants``.

Usage:
    from pose_estimate import PoseEstimator
    left8, right8 = PoseEstimator().estimate_from_camera(0)

    python pose_estimate.py --image me.jpg
    python pose_estimate.py --camera 0 --visualize /tmp/pose.png
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from physicalai_openarm_plugin.constants import (
    LEFT_JOINT_LIMITS_DEG,
    OPENARM_JOINT_ORDER,
    RIGHT_JOINT_LIMITS_DEG,
)

MODEL_PATH = Path(__file__).parent / "models" / "movenet_singlepose_lightning.onnx"
MODEL_URL = "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx"
MODEL_INPUT = 192  # MoveNet Lightning expects a 192x192 int32 RGB crop
MIN_SCORE = 0.2  # keypoint confidence below this is treated as "not seen"

# COCO-17 keypoint indices produced by MoveNet.
KP = {
    "nose": 0,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
}
BONES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
]


def _ensure_model(path: Path) -> Path:
    """Return the model path, downloading it once if missing."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading MoveNet model -> {path}", file=sys.stderr)
    with urllib.request.urlopen(MODEL_URL) as response:
        path.write_bytes(response.read())
    return path


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two 2D vectors, in degrees."""
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class PoseEstimator:
    """Estimate two 8-joint OpenArm targets from a single human-pose frame."""

    #: Full canonical joint order, including the gripper (8 entries).
    JOINT_ORDER = OPENARM_JOINT_ORDER

    def __init__(
            self,
            *,
            model_path: str | Path = MODEL_PATH,
            min_score: float = MIN_SCORE,
            swap_arms: bool = False,
            gripper: float = 0.0,
            device: str = "AUTO",
            smoothing: float = 0.5,
    ) -> None:
        """Load the ONNX model once and keep the compiled model for repeated calls.

        Args:
            model_path: Location of the MoveNet ONNX file (downloaded if absent).
            min_score: Keypoint confidence below which an arm is reported as zeros.
            swap_arms: Map the person's right arm to the robot's left, and vice
                versa (mirror-style teleoperation).
            gripper: Fixed gripper opening magnitude in degrees (>= 0). The URDF
                uses opposite signs per side, so this is applied as ``+gripper``
                on the left and ``-gripper`` on the right before clipping.
            device: OpenVINO device to run on. ``"AUTO"`` (the default) picks the
                fastest available device -- typically an integrated/discrete GPU
                -- and transparently falls back to ``"CPU"`` where none exists.
            smoothing: Exponential-moving-average factor in [0, 1) applied to
                keypoints across frames to damp per-frame jitter. ``0`` disables
                smoothing (raw per-frame output); higher values are smoother but
                laggier. A keypoint whose new detection is below ``min_score``
                keeps its last smoothed position and just has its confidence
                decay, instead of snapping to a noisy low-confidence reading.
        """
        self.min_score = min_score
        self.swap_arms = swap_arms
        self.gripper = gripper
        self.smoothing = smoothing
        self._smoothed_kps: np.ndarray | None = None
        core = ov.Core()
        model = core.read_model(str(_ensure_model(Path(model_path))))
        # Intel GPU plugins default to fp16 for speed, but this model produces
        # garbage keypoints (visible as the skeleton jumping randomly) once
        # AUTO hands inference over from its CPU warm-up to the GPU. Force f32.
        compiled = core.compile_model(model, device, {"INFERENCE_PRECISION_HINT": "f32"})
        self._infer_request = compiled.create_infer_request()
        self._input_port = compiled.input(0)
        self._output_port = compiled.output(0)
        #: (17, 3) keypoints (x_px, y_px, score) from the most recent estimate.
        self.keypoints: np.ndarray | None = None

    def estimate(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(left_joints, right_joints)``, each shape (8,), from a BGR frame."""
        kps = self._smooth(self._run(frame_bgr))
        self.keypoints = kps
        ls, rs = kps[KP["left_shoulder"]], kps[KP["right_shoulder"]]
        shoulder_width = float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])) or 1.0
        left = self._retarget_arm(kps, "left", shoulder_width)
        right = self._retarget_arm(kps, "right", shoulder_width)
        if self.swap_arms:
            left, right = right, left
        return left, right

    def visualize(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Write an annotated copy of the frame using the last estimate's keypoints."""
        if self.keypoints is None:
            raise RuntimeError("call estimate() before visualize()")
        vis = frame_bgr.copy()
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

        for a, b in BONES:
            pa, pb = self.keypoints[KP[a]], self.keypoints[KP[b]]
            if min(pa[2], pb[2]) < self.min_score:
                continue
            cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 255, 0), 2)
        for x, y, score in self.keypoints:
            if score >= self.min_score:
                cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)
        return vis

    # -- internals -----------------------------------------------------------

    def _run(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return 17 keypoints as an (17, 3) array of (x_px, y_px, score)."""
        h, w = frame_bgr.shape[:2]
        # Letterbox to a square so keypoint aspect ratio is preserved.
        size = max(h, w)
        square = np.zeros((size, size, 3), dtype=np.uint8)
        square[:h, :w] = frame_bgr
        resized = cv2.resize(square, (MODEL_INPUT, MODEL_INPUT))
        inp = resized.astype(np.int32)[None]  # (1, 192, 192, 3)

        out = self._infer_request.infer({self._input_port: inp})[self._output_port]  # (1, 1, 17, 3)
        kps = out[0, 0]  # (17, 3): y, x, score (normalised 0..1)
        return np.stack([kps[:, 1] * size, kps[:, 0] * size, kps[:, 2]], axis=1)

    def _smooth(self, kps: np.ndarray) -> np.ndarray:
        """Exponentially blend ``kps`` into the running per-keypoint average."""
        if self.smoothing <= 0.0 or self._smoothed_kps is None:
            self._smoothed_kps = kps.copy()
            return self._smoothed_kps

        alpha = self.smoothing
        prev = self._smoothed_kps
        confident = kps[:, 2] >= self.min_score
        blended = prev.copy()
        blended[confident] = alpha * prev[confident] + (1 - alpha) * kps[confident]
        blended[~confident, 2] = alpha * prev[~confident, 2]  # let confidence decay, keep last position
        self._smoothed_kps = blended
        return blended

    def _retarget_arm(self, kps: np.ndarray, side: str, shoulder_width: float) -> np.ndarray:
        """Map one arm's shoulder/elbow/wrist keypoints to an 8-joint vector."""
        s = kps[KP[f"{side}_shoulder"]]
        e = kps[KP[f"{side}_elbow"]]
        w = kps[KP[f"{side}_wrist"]]
        hip = kps[KP[f"{side}_hip"]]
        if min(s[2], e[2], w[2]) < self.min_score:
            print(f"  {side}: arm keypoints not confidently visible -> zeros", file=sys.stderr)
            return np.zeros(len(self.JOINT_ORDER), dtype=np.float32)

        # Work in a y-up frame so "down" is (0, -1).
        def pt(k: np.ndarray) -> np.ndarray:
            return np.array([k[0], -k[1]], dtype=np.float64)

        sp, ep, wp = pt(s), pt(e), pt(w)
        upper = ep - sp

        # Downward torso direction: shoulder -> hip when the hip is visible, else
        # straight down in the image.
        trunk_down = pt(hip) - sp if hip[2] >= self.min_score else np.array([0.0, -1.0])
        _ = shoulder_width  # reserved for a future depth/foreshortening estimate

        # shoulder_pitch (joint 1): in-plane swing of the upper arm away from straight-down.
        # Positive x (image-right) is +yaw for the left arm, -yaw for the right arm.
        yaw = float(np.degrees(np.arctan2(upper[0], -upper[1])))
        if side == "right":
            yaw = -yaw

        # shoulder_roll (joint 2): elevation of the upper arm from the resting (hanging) pose.
        # The two sides use opposite sign conventions in the URDF limits.
        elevation = _angle_between(upper, trunk_down)
        pitch = -elevation if side == "left" else elevation

        # elbow (joint 4): elbow flexion. 0 = straight arm, grows as the forearm folds in.
        interior = _angle_between(sp - ep, wp - ep)
        elbow = 180.0 - interior

        gripper = abs(self.gripper) if side == "left" else -abs(self.gripper)
        values = {"shoulder_pitch": yaw, "shoulder_roll": pitch, "elbow": elbow, "gripper": gripper}
        return self._clip_to_limits(values, side)

    def _clip_to_limits(self, values: dict[str, float], side: str) -> np.ndarray:
        limits = LEFT_JOINT_LIMITS_DEG if side == "left" else RIGHT_JOINT_LIMITS_DEG
        out = np.zeros(len(self.JOINT_ORDER), dtype=np.float32)
        for i, name in enumerate(self.JOINT_ORDER):
            lo, hi = limits[name]
            out[i] = float(np.clip(values.get(name, 0.0), lo, hi))
        return out
