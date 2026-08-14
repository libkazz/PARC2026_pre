import ast
import json
from pathlib import Path

import numpy as np


_NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "smolvla_libero_spatial_lora.ipynb"
)


def _notebook() -> dict:
    return json.loads(_NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells() -> list[str]:
    return [
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def _trajectory_selection_helpers() -> dict:
    source = next(
        source
        for source in _code_cells()
        if "BEGIN_TRAJECTORY_SELECTION_HELPERS" in source
    )
    helper_source = source.split(
        "# BEGIN_TRAJECTORY_SELECTION_HELPERS\n",
        1,
    )[1].split(
        "# END_TRAJECTORY_SELECTION_HELPERS",
        1,
    )[0]
    namespace: dict = {}
    exec(
        compile(
            helper_source,
            "notebook-trajectory-selection-helpers",
            "exec",
        ),
        namespace,
    )
    return namespace


def _actions_with_gripper(*runs: tuple[int, int]) -> np.ndarray:
    values = []
    for value, length in runs:
        values.extend(
            [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, value]]
            * length
        )
    return np.asarray(values, dtype=np.float32)


def test_notebook_code_cells_have_valid_python_syntax():
    for index, source in enumerate(_code_cells()):
        ast.parse(source, filename=f"notebook-cell-{index}")


def test_notebook_supports_colab_and_local_paths():
    source = "\n".join(_code_cells())
    normalized_source = " ".join(source.split())

    assert "IS_COLAB" in source
    assert "PARC2026_SMOLVLA_WORKDIR" in source
    assert 'Path("/content") if IS_COLAB' in normalized_source
    assert 'WORK_DIR / "lerobot"' in source
    assert 'WORK_DIR / "LIBERO-plus"' in source
    assert 'Path("/content/lerobot")' not in source


def test_colab_only_operations_are_guarded():
    code_cells = _code_cells()
    system_setup = next(
        source for source in code_cells if '"apt-get"' in source
    )
    artifact_export = next(
        source
        for source in code_cells
        if "from google.colab import files" in source
    )

    assert system_setup.index("if IS_COLAB:") < system_setup.index(
        'run_quiet(["apt-get"'
    )
    assert artifact_export.index("if IS_COLAB:") < artifact_export.index(
        "from google.colab import files"
    )


def test_notebook_selects_an_available_torch_device():
    source = "\n".join(_code_cells())

    assert 'TORCH_DEVICE = "cuda"' in source
    assert 'TORCH_DEVICE = "mps"' in source
    assert 'TORCH_DEVICE = "cpu"' in source
    assert 'f"--policy.device={TORCH_DEVICE}"' in source


def test_notebook_configures_trajectory_selection_for_200_episodes():
    source = "\n".join(_code_cells())

    assert "TRAIN_EPISODES_PER_TASK = 20" in source
    assert "GRIPPER_QUALITY_TASK_IDS = {3, 6, 7}" in source
    assert "download_videos=False" in source
    assert '"action_state_trajectory_fingerprint"' in source
    assert 'WORK_DIR / "training_episode_selection.json"' in source
    assert "choose_evenly_spaced" not in source


def test_notebook_selects_distinct_action_state_trajectories():
    helpers = _trajectory_selection_helpers()
    trajectory_by_episode = {}
    episode_indices = []

    for behavior in range(5):
        for variant in range(3):
            episode_index = behavior * 10 + variant
            actions = np.full(
                (4, 7), behavior + 0.1, dtype=np.float32
            )
            actions[:, -1] = -1
            states = np.full(
                (4, 8), behavior + 0.1, dtype=np.float32
            )
            trajectory_by_episode[episode_index] = (actions, states)
            episode_indices.append(episode_index)

    result = helpers["select_episodes_by_trajectory"](
        episode_indices,
        trajectory_by_episode,
        3,
    )

    assert result["selected_episode_indices"] == [0, 21, 42]
    assert result["unique_trajectory_count"] == 5
    assert len(set(result["selected_trajectory_fingerprints"])) == 3


def test_notebook_applies_gripper_quality_only_when_requested():
    helpers = _trajectory_selection_helpers()
    bad_actions = _actions_with_gripper((-1, 4), (1, 5))
    trajectories = {
        1: (
            bad_actions,
            np.full((len(bad_actions), 8), 1, dtype=np.float32),
        )
    }

    for episode_index in range(2, 5):
        actions = _actions_with_gripper((-1, 4), (1, 5), (-1, 3))
        trajectories[episode_index] = (
            actions,
            np.full(
                (len(actions), 8),
                episode_index,
                dtype=np.float32,
            ),
        )

    unfiltered = helpers["select_episodes_by_trajectory"](
        list(trajectories),
        trajectories,
        1,
    )
    filtered = helpers["select_episodes_by_trajectory"](
        list(trajectories),
        trajectories,
        1,
        require_gripper_quality=True,
    )

    assert unfiltered["quality_rejections"] == []
    assert filtered["quality_rejections"][0]["episode_index"] == 1
    assert filtered["quality_candidate_count"] == 3
