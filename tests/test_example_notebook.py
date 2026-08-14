import ast
import json
from pathlib import Path


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
