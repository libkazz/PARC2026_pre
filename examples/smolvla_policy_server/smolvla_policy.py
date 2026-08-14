"""SmolVLA を PARC2026 Track 1 のポリシー API へ接続する。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Protocol

import numpy as np


ACTION_SIZE = 7
IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


class PolicyRuntime(Protocol):
    """テスト時にモデル実装を差し替えるための最小インターフェース。"""

    def get_action(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        ...

    def reset(self) -> None:
        ...


def _require_array(
    obs: dict[str, np.ndarray],
    key: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"観測に必須キーがありません: {key}")
    value = np.asarray(obs[key])
    if value.shape != shape:
        raise ValueError(f"{key} の shape は {shape} が必要です: {value.shape}")
    return value


def _quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """LIBERO の (x, y, z, w) quaternion を axis-angle へ変換する。"""
    quat = np.asarray(quat, dtype=np.float32)
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    return (quat[:3] / denominator * angle).astype(np.float32)


def prepare_libero_observation(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """LeRobot の LIBERO 評価と同じ画像・状態表現へ変換する。"""
    prepared: dict[str, np.ndarray] = {}
    camera_names = ("front", "wrist")
    for source_key, camera_name in zip(IMAGE_KEYS, camera_names, strict=True):
        image = _require_array(obs, source_key, (128, 128, 3))
        if image.dtype != np.uint8:
            raise ValueError(f"{source_key} は uint8 が必要です: {image.dtype}")
        image = np.flip(image, axis=(0, 1))
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        prepared[f"observation.images.{camera_name}"] = np.ascontiguousarray(image)

    eef_pos = _require_array(obs, STATE_KEYS[0], (3,)).astype(np.float32)
    eef_quat = _require_array(obs, STATE_KEYS[1], (4,)).astype(np.float32)
    gripper = _require_array(obs, STATE_KEYS[2], (2,)).astype(np.float32)
    state = np.concatenate((eef_pos, _quat_to_axis_angle(eef_quat), gripper))
    if not np.isfinite(state).all():
        raise ValueError("ロボット状態に NaN または Inf が含まれています")
    prepared["observation.state"] = state.astype(np.float32)
    return prepared


class LeRobotSmolVLARuntime:
    """同梱した LeRobot 0.6.0 とマージ済み SmolVLA をロードする。"""

    def __init__(self, model_path: str | Path, device: str = "cuda") -> None:
        root = Path(__file__).resolve().parent
        vendor_dir = root / "vendor"
        if not vendor_dir.is_dir():
            raise FileNotFoundError(f"LeRobot の vendor ディレクトリがありません: {vendor_dir}")
        sys.path.insert(0, str(vendor_dir))

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        self.torch = torch
        self.device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("POLICY_DEVICE=cuda ですが CUDA を利用できません")

        self.model_path = Path(model_path).resolve()
        vlm_assets = self.model_path / "vlm_assets"
        required = (
            self.model_path / "config.json",
            self.model_path / "model.safetensors",
            self.model_path / "policy_preprocessor.json",
            self.model_path / "policy_postprocessor.json",
            vlm_assets / "config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("モデル成果物が不足しています: " + ", ".join(missing))

        config = PreTrainedConfig.from_pretrained(self.model_path, local_files_only=True)
        config.device = device
        config.vlm_model_name = str(vlm_assets)
        config.load_vlm_weights = False
        config.pretrained_path = None
        config.use_peft = False

        self.policy = SmolVLAPolicy.from_pretrained(
            self.model_path,
            config=config,
            local_files_only=True,
            strict=False,
        )
        self.policy.to(device).eval()

        preprocessor_overrides = {
            "tokenizer_processor": {"tokenizer_name": str(vlm_assets)},
            "device_processor": {"device": device},
        }
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            self.model_path,
            config_filename="policy_preprocessor.json",
            overrides=preprocessor_overrides,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
            local_files_only=True,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            self.model_path,
            config_filename="policy_postprocessor.json",
            overrides={},
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
            local_files_only=True,
        )

    def get_action(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        prepared = prepare_libero_observation(obs)
        batch = {key: self.torch.from_numpy(value) for key, value in prepared.items()}
        batch["task"] = instruction
        batch = self.preprocessor(batch)
        action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def reset(self) -> None:
        self.policy.reset()


class SmolVLAPolicyAdapter:
    """PARC2026 の get_action/reset 契約を SmolVLA runtime へ委譲する。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        runtime: PolicyRuntime | None = None,
    ) -> None:
        if runtime is None:
            default_model = Path(__file__).resolve().parent / "model_weights" / "smolvla"
            resolved_model = Path(model_path or os.environ.get("MODEL_PATH", default_model))
            resolved_device = device or os.environ.get("POLICY_DEVICE", "cuda")
            runtime = LeRobotSmolVLARuntime(resolved_model, resolved_device)
        self.runtime = runtime
        self.instruction = ""

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        action = np.asarray(
            self.runtime.get_action(obs, self.instruction),
            dtype=np.float32,
        ).reshape(-1)
        if action.shape != (ACTION_SIZE,):
            raise ValueError(f"モデル出力は shape ({ACTION_SIZE},) が必要です: {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("モデル出力に NaN または Inf が含まれています")
        return action

    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction
        self.runtime.reset()
