from spot_check import __version__
from spot_check._version import __version__ as module_version


def test_version_is_semver_like() -> None:
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:3])
    assert __version__ == module_version
