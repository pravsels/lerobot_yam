"""yam_common hardware API must import without lerobot or torch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

YAM_COMMON_SRC = Path(__file__).resolve().parents[1]


ISOLATION_SCRIPT = r"""
import sys
from importlib.abc import MetaPathFinder


class _BlockHeavyDeps(MetaPathFinder):
    blocked = {"lerobot", "torch"}

    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".", 1)[0]
        if root in self.blocked:
            raise ImportError(f"blocked import of {fullname}")
        return None


for key in list(sys.modules):
    root = key.split(".", 1)[0]
    if root in {"lerobot", "torch", "yam_common"}:
        del sys.modules[key]

sys.meta_path.insert(0, _BlockHeavyDeps())

from yam_common import YAMArm, YAMArmConfig, probe_motors

assert YAMArm is not None
assert YAMArmConfig is not None
assert probe_motors is not None
assert "lerobot" not in sys.modules
assert "torch" not in sys.modules
print("ok")
"""


def test_yam_common_public_api_imports_without_lerobot_or_torch() -> None:
    env = dict(**{k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"})
    pythonpath = str(YAM_COMMON_SRC)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = pythonpath if not existing else f"{pythonpath}:{existing}"
    result = subprocess.run(
        [sys.executable, "-c", ISOLATION_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ok" in result.stdout
