#!/usr/bin/env bash
# Bump the semantic version in src/spot_check/_version.py
#
# Usage:
#   ./scripts/bump-version.sh patch
#   ./scripts/bump-version.sh minor
#   ./scripts/bump-version.sh major
#   ./scripts/bump-version.sh 2.0.0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="$ROOT/src/spot_check/_version.py"
PART="${1:-patch}"

read_version() {
  python -c "
import pathlib, re
text = pathlib.Path('$VERSION_FILE').read_text(encoding='utf-8')
m = re.search(r'__version__\\s*=\\s*[\"\\']([^\"\\']+)[\"\\']', text)
if not m:
    raise SystemExit('Could not parse version from $VERSION_FILE')
print(m.group(1))
"
}

bump_semver() {
  local current="$1" part="$2"
  python -c "
import sys
current, part = sys.argv[1], sys.argv[2]
parts = [int(x) for x in current.split('.')]
while len(parts) < 3:
    parts.append(0)
major, minor, patch = parts[:3]
if part == 'major':
    major += 1; minor = 0; patch = 0
elif part == 'minor':
    minor += 1; patch = 0
elif part == 'patch':
    patch += 1
else:
    raise SystemExit(f'Unknown part: {part}')
print(f'{major}.{minor}.{patch}')
" "$current" "$part"
}

CURRENT="$(read_version)"

if [[ "$PART" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]]; then
  NEW="$PART"
else
  case "$PART" in
    major | minor | patch) NEW="$(bump_semver "$CURRENT" "$PART")" ;;
    *)
      echo "Usage: $0 {patch|minor|major|X.Y.Z}" >&2
      exit 1
      ;;
  esac
fi

cat >"$VERSION_FILE" <<EOF
"""Single source of truth for the SpotCheck release version."""

__version__ = "$NEW"
EOF

echo "$CURRENT -> $NEW"
echo "Next: git add $VERSION_FILE && git commit -m \"chore: release v$NEW\" && git tag v$NEW"
