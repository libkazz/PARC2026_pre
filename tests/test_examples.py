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
    assert 'eval_env["MUJOCO_GL"] = "glfw"' in source
    assert 'Path.home() / "PARC2026_outputs"' in source


def test_mac_notebook_has_no_saved_execution_outputs():
    notebook = load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
