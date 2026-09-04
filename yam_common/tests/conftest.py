"""Pytest fixtures for yam_common hardware-free tests."""

from __future__ import annotations

import sys
from pathlib import Path

# The repo folder is named yam_common/, which would otherwise shadow the
# installable package at yam_common/yam_common/. Prefer the package root.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from fakes import FakeRobot, fake_robot

__all__ = ["FakeRobot", "fake_robot"]
