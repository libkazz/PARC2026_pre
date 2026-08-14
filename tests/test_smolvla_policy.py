from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

_EXAMPLE_DIR = Path(__file__).parents[1] / "examples" / "smolvla_policy_server"
sys.path.insert(0, str(_EXAMPLE_DIR))

import build_submission as builder
import validate_submission as submission_validator
from smolvla_policy import (
    SmolVLAPolicyAdapter,
    prepare_libero_observation,
)


def _observation() -> dict[str, np.ndarray]:
    front = np.zeros((128, 128, 3), dtype=np.uint8)
    front[0, 0] = [10, 20, 30]
    wrist = np.full((128, 128, 3), 255, dtype=np.uint8)
    return {
        "agentview_image": front,
        "robot0_eye_in_hand_image": wrist,
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.array([1, 2, 3], dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 0, 1], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.1, 0.2], dtype=np.float32),
    }


def test_prepare_libero_observation_matches_expected_schema():
    prepared = prepare_libero_observation(_observation())

    assert set(prepared) == {
        "observation.images.front",
        "observation.images.wrist",
        "observation.state",
    }
    assert prepared["observation.images.front"].shape == (3, 128, 128)
    assert prepared["observation.images.front"].dtype == np.float32
    np.testing.assert_array_equal(prepared["observation.images.front"][:, -1, -1] * 255, [10, 20, 30])
    np.testing.assert_allclose(
        prepared["observation.state"],
        [1, 2, 3, 0, 0, 0, 0.1, 0.2],
    )


def test_prepare_libero_observation_rejects_wrong_image_dtype():
    obs = _observation()
    obs["agentview_image"] = obs["agentview_image"].astype(np.float32)

    with pytest.raises(ValueError, match="uint8"):
        prepare_libero_observation(obs)


class FakeRuntime:
    def __init__(self, action: np.ndarray):
        self.action = action
        self.calls = []
        self.reset_count = 0

    def get_action(self, obs, instruction):
        self.calls.append((obs, instruction))
        return self.action

    def reset(self):
        self.reset_count += 1


def test_adapter_forwards_instruction_and_resets_action_chunk():
    runtime = FakeRuntime(np.arange(7, dtype=np.float64))
    adapter = SmolVLAPolicyAdapter(runtime=runtime)

    adapter.reset("pick up the bowl")
    action = adapter.get_action(_observation())

    assert runtime.reset_count == 1
    assert runtime.calls[0][1] == "pick up the bowl"
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action, np.arange(7, dtype=np.float32))


def test_adapter_rejects_invalid_action_shape():
    adapter = SmolVLAPolicyAdapter(runtime=FakeRuntime(np.zeros(6)))

    with pytest.raises(ValueError, match="shape"):
        adapter.get_action(_observation())


def _fake_model(model_dir: Path) -> None:
    for name in builder.MODEL_FILES:
        path = model_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (model_dir / "vlm_assets/tokenizer.json").write_text("{}")


def _fake_lerobot_wheel(wheel_path: Path) -> None:
    files = {
        "lerobot/datasets/streaming_dataset.py": (
            "from pathlib import Path\n\nclass Backtrackable[T]:\n    pass\n"
        ),
        "lerobot/motors/motors_bus.py": (
            "type NameOrID = str | int\ntype Value = int | float\n"
        ),
        "lerobot/utils/io_utils.py": (
            "from typing import Any\n\n"
            "JsonLike = str | int | float | bool | None | list[\"JsonLike\"] | "
            "dict[str, \"JsonLike\"] | tuple[\"JsonLike\", ...]\n\n"
            "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:\n"
            "    return obj\n"
        ),
        "lerobot/processor/pipeline.py": (
            "from typing import Any, TypedDict, TypeVar, cast\n\n"
            "class DataProcessorPipeline[TInput, TOutput](HubMixin):\n"
            "    pass\n"
        ),
        "lerobot/processor/__init__.py": (
            "from .gym_action_processor import (\n"
            "    Numpy2TorchActionProcessorStep,\n"
            "    Torch2NumpyActionProcessorStep,\n"
            ")\n"
            "from .hil_processor import (\n"
            "    AddTeleopActionAsComplimentaryDataStep,\n"
            "    AddTeleopEventsAsInfoStep,\n"
            "    GripperPenaltyProcessorStep,\n"
            "    GymHILAdapterProcessorStep,\n"
            "    ImageCropResizeProcessorStep,\n"
            "    InterventionActionProcessorStep,\n"
            "    RewardClassifierProcessorStep,\n"
            "    TimeLimitProcessorStep,\n"
            ")\n"
        ),
        "lerobot/configs/video.py": "from typing import Any, ClassVar, Self\n",
        "lerobot/configs/__init__.py": (
            "from .dataset import DatasetRecordConfig\n"
            "from .policies import PreTrainedConfig\n"
        ),
        "lerobot/policies/__init__.py": (
            "from .act.configuration_act import ACTConfig as ACTConfig\n"
            "from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig\n"
        ),
        "lerobot/policies/pretrained.py": (
            "from typing import TYPE_CHECKING, TypedDict, TypeVar, Unpack\n"
            "from lerobot.configs.train import TrainPipelineConfig\n\n"
            "if TYPE_CHECKING:\n"
            "    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata\n"
        ),
        "lerobot/policies/smolvla/modeling_smolvla.py": "from typing import TypedDict, Unpack\n",
        "lerobot/policies/smolvla/smolvlm_with_expert.py": (
            "import copy\n"
            "        AutoProcessor,\n"
            "    AutoProcessor = None\n"
            "        self.processor = AutoProcessor.from_pretrained(model_id)\n"
        ),
        "lerobot-0.6.0.dist-info/METADATA": "Name: lerobot\nVersion: 0.6.0\n",
    }
    with zipfile.ZipFile(wheel_path, "w") as archive:
        for name, source in files.items():
            archive.writestr(name, source)


def test_builder_vendors_hash_verified_python310_compatible_lerobot(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    _fake_model(model_dir)
    wheel_path = tmp_path / "lerobot.whl"
    _fake_lerobot_wheel(wheel_path)
    wheel_hash = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    monkeypatch.setattr(builder, "LEROBOT_WHEEL_SHA256", wheel_hash)
    output = tmp_path / "submission.zip"

    builder.build_submission(model_dir, output, wheel_path)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "policy_server.py" in names
        assert "model_weights/smolvla/model.safetensors" in names
        streaming = archive.read("vendor/lerobot/datasets/streaming_dataset.py").decode()
        io_utils = archive.read("vendor/lerobot/utils/io_utils.py").decode()
        pipeline = archive.read("vendor/lerobot/processor/pipeline.py").decode()
        processor_init = archive.read("vendor/lerobot/processor/__init__.py").decode()
        video_config = archive.read("vendor/lerobot/configs/video.py").decode()
        configs_init = archive.read("vendor/lerobot/configs/__init__.py").decode()
        policies_init = archive.read("vendor/lerobot/policies/__init__.py").decode()
        pretrained = archive.read("vendor/lerobot/policies/pretrained.py").decode()
        smolvla_model = archive.read("vendor/lerobot/policies/smolvla/modeling_smolvla.py").decode()
        smolvlm = archive.read("vendor/lerobot/policies/smolvla/smolvlm_with_expert.py").decode()
    assert "class Backtrackable(Generic[T]):" in streaming
    assert "def deserialize_json_into_object(fpath: Path, obj: TJsonLike)" in io_utils
    assert "class DataProcessorPipeline(Generic[TInput, TOutput], HubMixin):" in pipeline
    assert "from .hil_processor import" not in processor_init
    assert "from .gym_action_processor import" not in processor_init
    assert "from typing_extensions import Self" in video_config
    assert "DatasetRecordConfig" not in configs_init
    assert "ACTConfig" not in policies_init
    assert "if TYPE_CHECKING:\n    from lerobot.configs.train" in pretrained
    assert "from typing_extensions import Unpack" in pretrained
    assert "from typing_extensions import Unpack" in smolvla_model
    assert "AutoProcessor" not in smolvlm
    assert "AutoTokenizer.from_pretrained" in smolvlm
    report = submission_validator.validate_zip(output)
    assert report.ok, report


def test_builder_rejects_model_without_offline_assets(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in builder.MODEL_FILES[:-1]:
        path = model_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({}))

    with pytest.raises(FileNotFoundError, match="vlm_assets"):
        builder.build_submission(model_dir, tmp_path / "submission.zip")
