# Open WebUI with Qwen 3.8 thinking controls in Advanced Parameters.
# Frontend is rebuilt from the matching upstream tag; backend stays the official image.

FROM ghcr.io/open-webui/open-webui:main AS upstream

FROM node:22-alpine3.20 AS frontend
RUN apk add --no-cache git python3
WORKDIR /src

COPY --from=upstream /app/package.json /tmp/upstream-package.json
COPY patch_advanced_params.py /tmp/patch_advanced_params.py

ARG OPEN_WEBUI_REF=
ENV NODE_OPTIONS="--max-old-space-size=8192"

RUN set -eu; \
    VERSION="$(python3 -c "import json; print(json.load(open('/tmp/upstream-package.json'))['version'])")"; \
    REF="${OPEN_WEBUI_REF:-v${VERSION}}"; \
    echo "cloning open-webui ${REF}"; \
    if ! git clone --depth 1 --branch "$REF" https://github.com/open-webui/open-webui.git /tmp/owui; then \
      echo "tag ${REF} missing, falling back to main"; \
      git clone --depth 1 --branch main https://github.com/open-webui/open-webui.git /tmp/owui; \
    fi; \
    cp -a /tmp/owui/. /src/; \
    python3 /tmp/patch_advanced_params.py src/lib/components/chat/Settings/Advanced/AdvancedParams.svelte; \
    npm ci --force; \
    npm run build

FROM upstream
COPY --from=frontend /src/build /app/build
COPY qwen_params.py /app/backend/open_webui/utils/qwen_params.py
RUN python3 - <<'PY'
from pathlib import Path
path = Path("/app/backend/open_webui/utils/payload.py")
text = path.read_text()
marker = "from open_webui.utils.qwen_params import wrap_openai"
if marker not in text:
    path.write_text(
        text
        + "\n\n"
        + "from open_webui.utils.qwen_params import wrap_openai, wrap_form\n"
        + "apply_model_params_to_body_openai = wrap_openai(apply_model_params_to_body_openai)\n"
        + "if 'apply_params_to_form_data' in globals():\n"
        + "    apply_params_to_form_data = wrap_form(apply_params_to_form_data)\n"
    )
print("payload.py patched")
PY
