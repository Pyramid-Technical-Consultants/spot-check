"""Static audit: analysis submodules must bind every bare name they use."""

from __future__ import annotations

import pytest

from tests.analysis_import_audit import analysis_source_paths, undefined_bare_names

# Known intentional dynamic / platform-only names (keep empty unless justified).
_PER_FILE_ALLOW: dict[str, frozenset[str]] = {}


@pytest.mark.parametrize(
    "path",
    analysis_source_paths(),
    ids=lambda p: p.relative_to(p.parent.parent).as_posix(),
)
def test_analysis_module_has_no_undefined_bare_names(path) -> None:
    rel = path.name
    allowed = _PER_FILE_ALLOW.get(rel, frozenset())
    bad = [(ln, name) for ln, name in undefined_bare_names(path) if name not in allowed]
    if bad:
        lines = "\n".join(f"  line {ln}: {name}" for ln, name in sorted(bad))
        pytest.fail(f"{path.name} uses undefined bare names:\n{lines}")
