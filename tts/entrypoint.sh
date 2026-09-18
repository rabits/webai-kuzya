#!/usr/bin/env bash
set -euo pipefail

# CUDA wheels ship libs under site-packages/nvidia/. ROCm uses /opt/rocm.
# Do not mix them: injecting nvidia-* into LD_LIBRARY_PATH breaks HIP.
extra=""
torch_ver="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"

case "${torch_ver}" in
  *rocm*|*hip*)
    for d in /opt/rocm/lib /opt/rocm/lib64 /opt/rocm/hip/lib; do
      if [[ -d "$d" ]]; then
        extra="${extra:+$extra:}$d"
      fi
    done
    ;;
  *)
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
    ;;
esac

export LD_LIBRARY_PATH="${extra:+$extra}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Single process: extra uvicorn workers would load the model again.
exec uvicorn main:app --host 0.0.0.0 --port 8002
