import json
from pathlib import Path


NOTEBOOK_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "smolvla_libero_spatial_lora_mac.ipynb"
)


def load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def notebook_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
    )


def test_mac_notebook_code_cells_compile():
    notebook = load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])
        compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec")


def test_mac_notebook_does_not_depend_on_colab_or_cuda():
    notebook = load_notebook()
    source = notebook_source(notebook)

    forbidden_fragments = (
        "/content",
        "apt-get",
        "google.colab",
        "policy.device=cuda",
        'MUJOCO_GL"] = "egl"',
    )

    for fragment in forbidden_fragments:
        assert fragment not in source


def test_mac_notebook_uses_mps_and_local_artifact_paths():
    notebook = load_notebook()
    source = notebook_source(notebook)

    assert "torch.backends.mps.is_available()" in source
    assert 'DEVICE = "mps"' in source
    assert 'f"--policy.device={DEVICE}"' in source
    assert '"lerobot.scripts.lerobot_train"' in source
    assert '"lerobot.scripts.lerobot_eval"' in source
    assert 'RUN_DIR / "training.log"' in source
    assert 'RUN_DIR / f"evaluation_{log_name}"' in source
    assert 'os.environ["HF_HUB_DISABLE_XET"] = "1"' in source
    assert 'os.environ.get("PARC2026_RUN_DIR")' in source
    assert 'eval_env["MUJOCO_GL"] = "glfw"' in source
    assert 'os.environ["MAGICK_HOME"] = homebrew_prefix' in source
    assert 'os.environ["DYLD_FALLBACK_LIBRARY_PATH"]' in source
    assert '"future": "1.0.0"' in source
    assert '"--no-deps"' in source
    assert "dependency_specs" in source
    assert "PARC2026_STANDARD_SPATIAL_TASKS" in source
    assert "standard_asset_prefixes" in source
    assert "asset_manifest_path" in source
    assert "Lifelong-Robot-Learning/LIBERO/git/trees/" in source
    assert 'repo_id="Sylvest/LIBERO-plus"' not in source
    assert 'Path.home() / "PARC2026_outputs"' in source


def test_mac_notebook_has_no_saved_execution_outputs():
    notebook = load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
