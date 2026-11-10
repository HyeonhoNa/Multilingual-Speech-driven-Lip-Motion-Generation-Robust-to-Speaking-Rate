#!/usr/bin/env bash
set -euo pipefail
repo=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Python is only used to verify/extract archives; no conda installation or network is needed.
python_bootstrap=${PYTHON_BOOTSTRAP:-python3}
exec "$python_bootstrap" "$repo/scripts/setup_envs.py" "$@"
