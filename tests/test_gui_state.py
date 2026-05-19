from spot_check.gui.state import bool_from_saved, sanitize_geometry


def test_bool_from_saved() -> None:
    assert bool_from_saved(True) is True
    assert bool_from_saved("yes") is True
    assert bool_from_saved("0") is False
    assert bool_from_saved(None, default=True) is True


def test_sanitize_geometry_rejects_bad() -> None:
    assert sanitize_geometry("") == "1400x900"
    assert sanitize_geometry("100x100") == "1400x900"
