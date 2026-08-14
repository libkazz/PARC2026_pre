"""マージ済み SmolVLA からオフライン実行可能な提出 ZIP を作る。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path


LEROBOT_VERSION = "0.6.0"
LEROBOT_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/5d/20/"
    "9a96311c19e9d256e65584ca83c49c5782d0f204836e84ceeb420d4d493e/"
    "lerobot-0.6.0-py3-none-any.whl"
)
LEROBOT_WHEEL_SHA256 = "b38a564fbc441d98380576863bf68635dde5fc2c42ddc2a39d0486640dc9e9a8"
SOURCE_FILES = ("policy_server.py", "smolvla_policy.py", "requirements.txt")
MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "vlm_assets/config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_wheel(destination: Path) -> None:
    print(f"LeRobot {LEROBOT_VERSION} wheel を取得しています...")
    with urllib.request.urlopen(LEROBOT_WHEEL_URL, timeout=60) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def _safe_extract_wheel(wheel_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel_path) as archive:
        for info in archive.infolist():
            path = Path(info.filename)
            mode = info.external_attr >> 16
            if path.is_absolute() or ".." in path.parts or stat.S_ISLNK(mode):
                raise ValueError(f"安全でない wheel エントリです: {info.filename}")
        archive.extractall(destination)


def _replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"LeRobot 互換パッチの対象が想定外です: {path} ({count}件)")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _replace_module(path: Path, required_fragments: tuple[str, ...], new: str) -> None:
    source = path.read_text(encoding="utf-8")
    for fragment in required_fragments:
        if source.count(fragment) != 1:
            raise RuntimeError(f"LeRobot runtime 限定の対象が想定外です: {path}")
    path.write_text(new, encoding="utf-8")


def _patch_python310(vendor_dir: Path) -> None:
    """LeRobot 0.6.0 の Python 3.12 型構文を 3.10 互換へ変換する。"""
    streaming = vendor_dir / "lerobot/datasets/streaming_dataset.py"
    _replace_once(
        streaming,
        "from pathlib import Path\n",
        "from pathlib import Path\nfrom typing import Generic, TypeVar\n\nT = TypeVar(\"T\")\n",
    )
    _replace_once(streaming, "class Backtrackable[T]:", "class Backtrackable(Generic[T]):")

    motors = vendor_dir / "lerobot/motors/motors_bus.py"
    _replace_once(
        motors,
        "type NameOrID = str | int\ntype Value = int | float",
        "NameOrID = str | int\nValue = int | float",
    )

    io_utils = vendor_dir / "lerobot/utils/io_utils.py"
    _replace_once(
        io_utils,
        "from typing import Any\n",
        "from typing import Any, TypeVar\n",
    )
    _replace_once(
        io_utils,
        "JsonLike = str | int | float | bool | None | list[\"JsonLike\"] | "
        "dict[str, \"JsonLike\"] | tuple[\"JsonLike\", ...]\n",
        "JsonLike = str | int | float | bool | None | list[\"JsonLike\"] | "
        "dict[str, \"JsonLike\"] | tuple[\"JsonLike\", ...]\n"
        "TJsonLike = TypeVar(\"TJsonLike\", bound=JsonLike)\n",
    )
    _replace_once(
        io_utils,
        "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:",
        "def deserialize_json_into_object(fpath: Path, obj: TJsonLike) -> TJsonLike:",
    )

    pipeline = vendor_dir / "lerobot/processor/pipeline.py"
    _replace_once(
        pipeline,
        "from typing import Any, TypedDict, TypeVar, cast\n",
        "from typing import Any, Generic, TypedDict, TypeVar, cast\n",
    )
    _replace_once(
        pipeline,
        "class DataProcessorPipeline[TInput, TOutput](HubMixin):",
        "class DataProcessorPipeline(Generic[TInput, TOutput], HubMixin):",
    )

    processor_init = vendor_dir / "lerobot/processor/__init__.py"
    hil_import = """from .hil_processor import (
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    GripperPenaltyProcessorStep,
    GymHILAdapterProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    RewardClassifierProcessorStep,
    TimeLimitProcessorStep,
)
"""
    _replace_once(
        processor_init,
        hil_import,
        "# HIL processors are not imported in the submission runtime.\n",
    )
    gym_import = """from .gym_action_processor import (
    Numpy2TorchActionProcessorStep,
    Torch2NumpyActionProcessorStep,
)
"""
    _replace_once(
        processor_init,
        gym_import,
        "# Gym action processors depend on HIL and are not imported here.\n",
    )

    video_config = vendor_dir / "lerobot/configs/video.py"
    _replace_once(
        video_config,
        "from typing import Any, ClassVar, Self\n",
        "from typing import Any, ClassVar\nfrom typing_extensions import Self\n",
    )

    configs_init = vendor_dir / "lerobot/configs/__init__.py"
    _replace_module(
        configs_init,
        ("from .dataset import DatasetRecordConfig", "from .policies import PreTrainedConfig"),
        """\"\"\"SmolVLA submission runtime configuration exports.\"\"\"

from .policies import PreTrainedConfig
from .types import (
    FeatureType,
    NormalizationMode,
    PipelineFeatureType,
    PolicyFeature,
    RTCAttentionSchedule,
)

__all__ = [
    \"FeatureType\",
    \"NormalizationMode\",
    \"PipelineFeatureType\",
    \"PolicyFeature\",
    \"PreTrainedConfig\",
    \"RTCAttentionSchedule\",
]
""",
    )

    policies_init = vendor_dir / "lerobot/policies/__init__.py"
    _replace_module(
        policies_init,
        ("from .act.configuration_act import ACTConfig", "from .smolvla.configuration_smolvla import"),
        """\"\"\"SmolVLA submission runtime policy exports.\"\"\"

from .pretrained import PreTrainedPolicy
from .smolvla.configuration_smolvla import SmolVLAConfig

__all__ = [\"PreTrainedPolicy\", \"SmolVLAConfig\"]
""",
    )

    pretrained = vendor_dir / "lerobot/policies/pretrained.py"
    _replace_once(pretrained, "from lerobot.configs.train import TrainPipelineConfig\n", "")
    _replace_once(
        pretrained,
        "if TYPE_CHECKING:\n    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata\n",
        "if TYPE_CHECKING:\n    from lerobot.configs.train import TrainPipelineConfig\n"
        "    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata\n",
    )
    _replace_once(
        pretrained,
        "from typing import TYPE_CHECKING, TypedDict, TypeVar, Unpack\n",
        "from typing import TYPE_CHECKING, TypedDict, TypeVar\n"
        "from typing_extensions import Unpack\n",
    )

    smolvla_model = vendor_dir / "lerobot/policies/smolvla/modeling_smolvla.py"
    _replace_once(
        smolvla_model,
        "from typing import TypedDict, Unpack\n",
        "from typing import TypedDict\nfrom typing_extensions import Unpack\n",
    )

    smolvlm = vendor_dir / "lerobot/policies/smolvla/smolvlm_with_expert.py"
    _replace_once(smolvlm, "import copy\n", "import copy\nfrom types import SimpleNamespace\n")
    _replace_once(smolvlm, "        AutoProcessor,\n", "        AutoTokenizer,\n")
    _replace_once(smolvlm, "    AutoProcessor = None\n", "    AutoTokenizer = None\n")
    _replace_once(
        smolvlm,
        "        self.processor = AutoProcessor.from_pretrained(model_id)\n",
        "        self.processor = SimpleNamespace(\n"
        "            tokenizer=AutoTokenizer.from_pretrained(model_id)\n"
        "        )\n",
    )


def _validate_model(model_dir: Path) -> None:
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    tokenizer_files = ("tokenizer.json", "tokenizer.model")
    if not any((model_dir / "vlm_assets" / name).is_file() for name in tokenizer_files):
        missing.append("vlm_assets/tokenizer.json または tokenizer.model")
    if missing:
        raise FileNotFoundError("モデル成果物が不足しています: " + ", ".join(missing))
    symlinks = [str(path) for path in model_dir.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError("モデル成果物に symlink は含められません: " + ", ".join(symlinks))


def build_submission(
    model_dir: Path,
    output_path: Path,
    wheel_path: Path | None = None,
) -> Path:
    source_dir = Path(__file__).resolve().parent
    model_dir = model_dir.resolve()
    output_path = output_path.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"モデルディレクトリがありません: {model_dir}")
    if output_path.is_relative_to(model_dir):
        raise ValueError("出力 ZIP はモデルディレクトリの外に指定してください")
    _validate_model(model_dir)

    with tempfile.TemporaryDirectory(prefix="parc2026-smolvla-") as temporary:
        temporary_dir = Path(temporary)
        build_dir = temporary_dir / "submission"
        build_dir.mkdir()
        for name in SOURCE_FILES:
            shutil.copy2(source_dir / name, build_dir / name)
        shutil.copytree(model_dir, build_dir / "model_weights/smolvla")

        resolved_wheel = wheel_path.resolve() if wheel_path else temporary_dir / "lerobot.whl"
        if wheel_path is None:
            _download_wheel(resolved_wheel)
        actual_hash = _sha256(resolved_wheel)
        if actual_hash != LEROBOT_WHEEL_SHA256:
            raise ValueError(
                "LeRobot wheel の SHA-256 が一致しません: "
                f"expected={LEROBOT_WHEEL_SHA256}, actual={actual_hash}"
            )
        vendor_dir = build_dir / "vendor"
        vendor_dir.mkdir()
        _safe_extract_wheel(resolved_wheel, vendor_dir)
        _patch_python310(vendor_dir)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            output_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for path in sorted(build_dir.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(build_dir).as_posix())

    print(f"提出 ZIP を作成しました: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lerobot-wheel",
        type=Path,
        help="事前取得済みの LeRobot 0.6.0 wheel（省略時は PyPI から取得）",
    )
    args = parser.parse_args()
    build_submission(args.model_dir, args.output, args.lerobot_wheel)


if __name__ == "__main__":
    main()
