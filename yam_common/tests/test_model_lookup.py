"""Packaged gravity-comp models must resolve without workstation i2rt paths."""

from __future__ import annotations

import os
from pathlib import Path


def test_packaged_crank_model_is_found_without_i2rt_home(monkeypatch, tmp_path: Path) -> None:
    from yam_common.mujoco_kdl import get_yam_mujoco_kdl

    monkeypatch.delenv("YAM_MUJOCO_MODEL_DIR", raising=False)
    real_expanduser = os.path.expanduser

    def _hide_workstation_i2rt(path: str) -> str:
        expanded = real_expanduser(path)
        if "i2rt" in expanded.replace("\\", "/"):
            return str(tmp_path / "missing-i2rt")
        return expanded

    monkeypatch.setattr(os.path, "expanduser", _hide_workstation_i2rt)

    kdl = get_yam_mujoco_kdl("crank_4310")
    xml_path = Path(kdl.xml_path)
    assert xml_path.is_file()
    assert xml_path.name == "yam.xml"
    assert "robot_models" in xml_path.parts
    assert not str(xml_path).startswith(str(tmp_path / "missing-i2rt"))


def test_packaged_model_dir_contains_required_collision_meshes() -> None:
    from yam_common.mujoco_kdl import packaged_yam_model_dir

    model_dir = Path(packaged_yam_model_dir())
    assert (model_dir / "yam.xml").is_file()
    assets = model_dir / "assets"
    for name in (
        "base_link_collision.stl",
        "link_1_collision.stl",
        "link_2_collision.stl",
        "link_3_collision.stl",
        "link_4_collision.stl",
        "link_5_collision.stl",
        "link_6_collision.stl",
    ):
        assert (assets / name).is_file(), name
