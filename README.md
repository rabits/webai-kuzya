# Kuzya — STT / LLM / TTS on DGX Spark

Open WebUI in front of an external vLLM endpoint, with local Whisper STT and F5-TTS on the Spark GPU.

```text
Browser
   │
   ▼
lb (nginx)          :80 → :443, TLS
   │
   ▼
open-webui          127.0.0.1:3000, CPU only, ~1 GB
   ├── chat    →  vLLM on the host / another compose  (the large allocation)
   ├── STT     →  stt:8001   Whisper turbo on GPU     (~2 GB)
   └── TTS     →  tts:8002   F5-TTS on GPU            (~4 GB)
```

vLLM is intentionally not in this compose file. It already owns most of the 128 GB unified memory; this stack only needs a reserved slice of it.

## Why three containers, not one

Docker itself is cheap: a few hundred MB of process overhead. What actually costs memory on GB10 is:

| Consumer | Typical resident | Notes |
| --- | --- | --- |
| vLLM | tens of GB | KV cache + weights. Set `--gpu-memory-utilization` with headroom. |
| F5-TTS | ~3–5 GB | One CUDA context + weights |
| Whisper turbo | ~2 GB | One CUDA context + weights |
| Open WebUI | ~1 GB CPU | Must not get a GPU |
| Two extra CUDA contexts vs one merged speech process | ~1–2 GB | Not worth coupling STT and TTS for this |

Merging Whisper and F5-TTS into one process would save roughly one CUDA context and one copy of `libtorch`. That is noise next to vLLM. Separate services keep the existing TTS image, let STT and TTS restart independently, and match Open WebUI’s two OpenAI-compatible audio URLs.

The setting that *does* starve this stack is vLLM grabbing the whole unified memory pool. Leave about 16–24 GB for OS + STT + TTS, for example:

```bash
vllm serve "$LLM_MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.75
```

Start STT/TTS before a tight vLLM process, or lower utilization if either speech container OOMs.

Do **not** use Open WebUI’s built-in Whisper or the `:cuda` WebUI image here. Built-in faster-whisper is CPU-only on aarch64 (CTranslate2 has no CUDA ARM wheels), and a GPU WebUI image would open a third CUDA context for no benefit.

## Prerequisites

- NVIDIA DGX Spark (GB10, aarch64, CUDA 13)
- Docker with NVIDIA container toolkit (`deploy.resources` GPU reservations; do not set `runtime: nvidia` on DGX OS)
- F5-TTS files and a `default` voice as described in [models/README.md](models/README.md)
- vLLM already serving an OpenAI-compatible API

## Run

As LLM you can use any setup that supports OpenAI protocol. I use https://github.com/blazux/qwen3.8-Flash-DGX with model "lychee888/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE":
```
$ MODEL=lychee888/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE ./flash setup
$ MODEL=lychee888/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE ./flash serve published PORT=8000
$ PORT=8000 ./flash wait
```

```bash
cp .env.example .env
# set LLM_URL / LLM_MODEL / WEBUI_SECRET_KEY
# set WEBUI_URL to the public origin, e.g. https://example.com

docker compose up -d --build
```

First STT start downloads Whisper into `models/whisper/` (turbo is ~1.6 GB). TTS warmup needs `models/voices/default/ref.wav`.

Open **https://localhost** (or your `WEBUI_URL`). HTTP on port 80 redirects to HTTPS.

On first start nginx writes a self-signed cert into `certs/` for the host in `WEBUI_URL`. Browsers will warn until you accept it, or until you drop a real cert there:

```bash
# Let's Encrypt live directory, or any dir with these filenames
# TLS_CERT_DIR=/etc/letsencrypt/live/example.com
# TLS_CERT=fullchain.pem
# TLS_KEY=privkey.pem
```

A non-443 port in `WEBUI_URL` (for example `https://example.com:8443`) is used in the redirect and in the cert marker. Nginx inside the container still listens on 443; map the host port in compose (`"8443:443"`) if you actually want that port on the host.

Changing `WEBUI_URL` regenerates the self-signed pair only if `certs/.selfsigned-for` is present (i.e. nginx created the files). Your own certs are never overwritten. To force a new self-signed cert, delete `certs/*.pem` and `certs/.selfsigned-for` and recreate `lb`.

If audio settings in the UI disagree with `.env` after the first launch, Open WebUI persisted them in its volume. Either set Admin → Settings → Audio, or recreate the `open-webui-data` volume.

### How to setup Kuzya

Create new Workspace and specify system prompt from `system_prompt_kuzya.txt` and set voice to `kuzya_calm`, then save and pick it in your chat as the model.

## Smoke test

```bash
chmod +x scripts/smoke.sh
./scripts/smoke.sh
```

This checks `/health` on STT, TTS, local WebUI, and HTTPS via nginx, then transcribes `models/voices/default/ref.wav` if it exists.

Manual checks:

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8002/health
curl -s http://127.0.0.1:8002/v1/audio/voices

curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -F file=@models/voices/default/ref.wav \
  -F language=ru

curl -s http://127.0.0.1:8002/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"f5-tts","input":"Привет, это проверка.","voice":"default","response_format":"mp3"}' \
  --output /tmp/kuzya.mp3
```

## Voice chat in Open WebUI

Admin → Settings → Audio should already be seeded from compose:

- STT engine: OpenAI, base URL `http://stt:8001/v1`, model `turbo`
- TTS engine: OpenAI, base URL `http://tts:8002/v1`, model `f5-tts`, voice `default`
- Split on punctuation (F5-TTS is happier with short clauses)

Then use the microphone / call controls in a chat against the vLLM model.

## Qwen 3.8 thinking options

Open WebUI is built from `webui/Dockerfile`, which adds three Advanced Parameters (workspace model defaults and chat overrides):

| Control | Sent to vLLM as |
| --- | --- |
| `enable_thinking` (Qwen) | `chat_template_kwargs.enable_thinking` bool |
| `preserve_thinking` (Qwen) | `chat_template_kwargs.preserve_thinking` bool |
| Reasoning Effort | `reasoning_effort`: `xhigh` → `medium` → `low` |

Each control is Default / On / Off (or Default / xhigh / medium / low), same pattern as `keep_alive`. Default leaves the field out so vLLM uses its own defaults.

First `docker compose up --build` rebuilds the Open WebUI frontend; later starts reuse `kuzya-open-webui:local`.

## Ports

| Service | Port | Bind |
| --- | --- | --- |
| nginx HTTP → HTTPS | 80 | public |
| nginx HTTPS | 443 | public |
| Open WebUI | 3000 | `127.0.0.1` only |
| STT | 8001 | host (docker network for WebUI) |
| TTS | 8002 | host (docker network for WebUI) |
| vLLM (external) | 8000 by default | |
