#!/usr/bin/env bash
set -euo pipefail

extra=""
if python -c "import nvidia" >/dev/null 2>&1; then
  extra="$(python - <<'PY'
from pathlib import Path
try:
    import nvidia
except ImportError:
    raise SystemExit(0)
root = Path(nvidia.__path__[0])
libs = sorted({str(p) for p in root.rglob("lib") if p.is_dir()})
print(":".join(libs))
PY
)"
fi

export LD_LIBRARY_PATH="${extra:+$extra}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Single process: extra uvicorn workers would load the model again.
exec uvicorn main:app --host 0.0.0.0 --port 8002
