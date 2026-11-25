import ast
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "MELBA_demo.ipynb"


def _bootstrap_source(notebook):
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and "SOURCE_FILES =" in "".join(cell.get("source", []))
    )


def _snapshot_from_source(source):
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "SOURCE_FILES" for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def test_notebook_snapshot_uses_relative_keys_and_matches_package_sources():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    snapshot = _snapshot_from_source(_bootstrap_source(notebook))

    assert snapshot
    for path in snapshot:
        posix_path = PurePosixPath(path)
        windows_path = PureWindowsPath(path)
        assert not posix_path.is_absolute()
        assert not windows_path.is_absolute()
        assert not windows_path.drive
        assert ".." not in posix_path.parts
    source_paths = {path.relative_to(ROOT).as_posix() for path in (ROOT / "melba").rglob("*.py")}
    assert set(snapshot) == source_paths
    for relative_path, source in snapshot.items():
        assert source == (ROOT / PurePosixPath(relative_path)).read_text(encoding="utf-8")


def test_bootstrap_imports_bundled_source_from_an_isolated_temporary_directory():
    with tempfile.TemporaryDirectory(prefix="melba-bootstrap-") as temporary_root:
        runner = r'''
import json
import os
from pathlib import Path
import sys
import tempfile

notebook_path = Path(sys.argv[1])
temporary_root = Path(sys.argv[2])
isolated_cwd = temporary_root / "without_checkout"
isolated_cwd.mkdir()
os.chdir(isolated_cwd)
tempfile.tempdir = str(temporary_root)
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
bootstrap = next(
    "".join(cell["source"])
    for cell in notebook["cells"]
    if cell["cell_type"] == "code" and "SOURCE_FILES =" in "".join(cell.get("source", []))
)
namespace = {"__name__": "__main__"}
exec(compile(bootstrap, str(notebook_path), "exec"), namespace)
import melba
from melba import MELBA
assert Path(namespace["source_root"]).resolve() in Path(melba.__file__).resolve().parents
assert (Path(namespace["source_root"]) / "melba" / "model.py").is_file()
model = MELBA(epochs=5, random_state=13).fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 2.0, 3.0, 4.0])
assert model.predict([[4.0]]).shape == (1,)
'''
        result = subprocess.run(
            [sys.executable, "-c", runner, str(NOTEBOOK_PATH), temporary_root],
            cwd=temporary_root,
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
