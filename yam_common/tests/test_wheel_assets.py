"""Built yam_common wheel must ship gravity-comp assets and the I2RT license."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path


REQUIRED_WHEEL_FILES = (
    "yam_common/robot_models/LICENSE-I2RT",
    "yam_common/robot_models/yam/yam.xml",
    "yam_common/robot_models/yam/assets/base_link_collision.stl",
    "yam_common/robot_models/yam/assets/link_1_collision.stl",
    "yam_common/robot_models/yam/assets/link_2_collision.stl",
    "yam_common/robot_models/yam/assets/link_3_collision.stl",
    "yam_common/robot_models/yam/assets/link_4_collision.stl",
    "yam_common/robot_models/yam/assets/link_5_collision.stl",
    "yam_common/robot_models/yam/assets/link_6_collision.stl",
)


def test_built_wheel_includes_model_assets_and_license(tmp_path: Path) -> None:
    package_dir = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=package_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(tmp_path.glob("yam_common-*.whl"))
    assert wheels, "uv build did not produce a yam_common wheel"
    names = set(zipfile.ZipFile(wheels[0]).namelist())
    missing = [path for path in REQUIRED_WHEEL_FILES if path not in names]
    assert missing == [], missing
