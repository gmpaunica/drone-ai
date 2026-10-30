from __future__ import annotations

from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_known_good_lock_matches_installed_environment() -> None:
    locked: dict[str, str] = {}
    for raw_line in (ROOT / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        assert separator == "==", f"lock entry is not exact: {line}"
        assert name.lower() not in locked
        locked[name.lower()] = version
    assert locked
    for name, version in locked.items():
        assert metadata.version(name) == version


def test_readiness_script_only_invokes_non_flying_doctor() -> None:
    script = (ROOT / "scripts" / "ready.ps1").read_text(encoding="utf-8").lower()
    assert '"doctor"' in script
    assert "takeoff" not in script
    assert "move_forward" not in script
    assert "emergency" not in script
