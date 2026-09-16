# Kuzya — STT / LLM / TTS on DGX Spark

Open WebUI in front of an external vLLM endpoint, with local Whisper STT and F5-TTS on the Spark GPU.

```text
Browser
   │
   ▼
open-webui          CPU only, ~1 GB
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

```bash
cp .env.example .env
# set LLM_URL / LLM_MODEL / WEBUI_SECRET_KEY

docker compose up -d --build
```

First STT start downloads Whisper into `models/whisper/` (turbo is ~1.6 GB). TTS warmup needs `models/voices/default/ref.wav`.

Open http://localhost:3000

If audio settings in the UI disagree with `.env` after the first launch, Open WebUI persisted them in its volume. Either set Admin → Settings → Audio, or recreate the `open-webui-data` volume.

## Smoke test

```bash
chmod +x scripts/smoke.sh
./scripts/smoke.sh
```

This checks `/health` on STT, TTS, and Open WebUI, then transcribes `models/voices/default/ref.wav` if it exists.

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

## Ports

| Service | Port |
| --- | --- |
| Open WebUI | 3000 |
| STT | 8001 |
| TTS | 8002 |
| vLLM (external) | 8000 by default |
