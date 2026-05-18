#!/usr/bin/env bash
# Run the same lint step as GitHub Actions CI.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ruff check src tests packaging
