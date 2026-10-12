"""The built wheel is importable on its own and its entry points are runnable."""

import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def wheel_dir(tmp_path_factory):
    if shutil.which("uv") is None:
        pytest.skip("uv not installed")
    out = tmp_path_factory.mktemp("wheel")
    subprocess.run(["uv", "build", "--wheel", "-o", str(out)], cwd=_ROOT,
                   check=True, capture_output=True)
    unpacked = out / "unpacked"
    with zipfile.ZipFile(next(out.glob("*.whl"))) as z:
        z.extractall(unpacked)
    return unpacked


def test_wheel_imports_outside_the_checkout(wheel_dir):
    code = ("from src.codeflow.synthetic.gen_code import generate_training_examples\n"
            "from src.codeflow.synthetic.prompts import CODE_SYSTEM\n"
            "from src.codeflow.assembly.validator import validate_example\n"
            "assert 'nREPL' in CODE_SYSTEM\n")
    r = subprocess.run([sys.executable, "-I", "-c",
                        f"import sys; sys.path.insert(0, {str(wheel_dir)!r})\n" + code],
                       cwd=wheel_dir, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_entry_points_take_no_arguments():
    import importlib
    import inspect

    scripts = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"].get("scripts", {})
    for name, target in scripts.items():
        mod, fn = target.split(":")
        sig = inspect.signature(getattr(importlib.import_module(mod), fn))
        required = [p for p in sig.parameters.values()
                    if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        assert not required, f"{name} -> {target} needs {required}"
