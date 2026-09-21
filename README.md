# Kuzya — STT / LLM / TTS on DGX Spark

Open WebUI in front of an external vLLM endpoint, with local Whisper STT and F5-TTS on the Spark GPU.

```text
Browser
   │
   ▼
 nginx          :80 → :443, TLS
   │
   ▼
open-webui          127.0.0.1:3000, CPU only, ~1 GB
   ├── chat    →  vLLM on the host / another compose  (the large allocation)
   ├── STT     →  stt:8001   Whisper turbo on GPU     (~2 GB)
   └── TTS     →  tts:8002   F5-TTS on GPU            (~4 GB)
```

vLLM is intentionally not in this compose file. It already owns most of the 128 GB unified memory; this stack only needs a reserved slice of it.

The idea of f5-tts with stress marks I got from https://github.com/korenko-git/voice-service - thanks to Dmytro it works great!

## Prerequisites

NVIDIA (default):
- DGX Spark (GB10, aarch64, CUDA 13) or any NVIDIA box with the NVIDIA Container Toolkit
- Do not set `runtime: nvidia` on DGX OS; use `deploy.resources` GPU reservations

AMD:
- ROCm 7.2.x, `/dev/kfd` and `/dev/dri`, host groups `video` / `render`

Both:
- F5-TTS files and a `default` voice as described in [models/README.md](models/README.md)
- vLLM already serving an OpenAI-compatible API

## Usage

You need just to copy the env, configure it and run the fitting to you compose file.

### DGX Spark / Nvidia

```bash
cp .env.example .env
# set LLM_URL / LLM_MODEL / WEBUI_SECRET_KEY
# set WEBUI_URL to the public origin, e.g. https://example.com

docker compose -f docker-compose.nvidia.yml up -d --build
```

### LLM setup

You need to run LLM separately, by default on 8000 port and support OpenAI protocol.

#### Qwen 3.8 27B Uncensored

On DGX Spark I think it's a good choise to have quick responses for voice assistant, but not much to reasoning.

```
$ docker run -d --name qwen38-27b --gpus all --ipc=host -p 8000:8000 \
    -v "$HOME/.cache/huggingface:/hf" -e HF_HOME=/hf \
    vllm/vllm-openai:qwen38-flash-next \
    --model lfitoto/Qwen3.8-27B-Uncensored-NVFP4 --served-model-name qwen3.8-27B \
    --kv-cache-dtype fp8 --gpu-memory-utilization 0.7 --max-model-len 262144 \
    --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \
    --enable-prefix-caching --load-format fastsafetensors --tensor-parallel-size 1 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 --mm-encoder-tp-mode data
```

#### Qwen 3.8 Flash Next

I use https://github.com/blazux/qwen3.8-Flash-DGX as vLLM - by default it fits to DGX Spark and allows to run all the components as side-services:
```
$ ./flash setup
$ ./flash serve PORT=8000
$ PORT=8000 ./flash wait
```

#### Qwen 3.8 Flash Next Uncensored

https://github.com/blazux/qwen3.8-Flash-DGX - with custom model it will not fit DGX Spark RAM. So with that I use second host for STT and TTS:
```
$ MODEL=lychee888/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE ./flash setup
$ MODEL=lychee888/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE ./flash serve published PORT=8000
$ PORT=8000 ./flash wait
```

### How to setup Kuzya

Create new Workspace and specify system prompt from `system_prompt_kuzya.txt`, and set voice to `kuzya_calm`.

If you want to use Kuzya as voice assistant (in call mode to have minimal delays) - open `Advanced Params` and turn `enable_thinking` to Off position, so he will answer right away.

Then save and pick it in your chat as the model.

## Open WebUI

After setup you can visit **https://localhost** (or your `WEBUI_URL`) to see the UI.

### Nginx

You need a way to proxy 443 port to Open-WebUI because mic will work only on HTTPS. For that you can use:
* [WebAI-Router](https://github.com/rabits/webai-router) - special system to automate switch between different configurations
* System nginx server - Install nginx, use nginx.example.conf and generate self-singned certs using the next command:
   ```
   $ sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/openwebui.key -out /etc/ssl/certs/openwebui.crt
   ```

### Voice chat

Admin → Settings → Audio should already be seeded from compose:

- STT engine: OpenAI, base URL `http://stt:8001/v1`, model `turbo`
- TTS engine: OpenAI, base URL `http://tts:8002/v1`, model `f5-tts`, voice `default`
- Split on punctuation (F5-TTS is happier with short clauses)

Then use the microphone / call controls in a chat against the vLLM model.

### Qwen 3.8 thinking options

Open WebUI is built from `webui/Dockerfile`, which adds three Advanced Parameters (workspace model defaults and chat overrides):

| Control | Sent to vLLM as |
| --- | --- |
| `enable_thinking` (Qwen) | `chat_template_kwargs.enable_thinking` bool |
| `preserve_thinking` (Qwen) | `chat_template_kwargs.preserve_thinking` bool |
| Reasoning Effort | `reasoning_effort`: `xhigh` → `medium` → `low` |

Each control is Default / On / Off (or Default / xhigh / medium / low), same pattern as `keep_alive`. Default leaves the field out so vLLM uses its own defaults.

First `docker compose -f docker-compose.nvidia.yml up --build` rebuilds the Open WebUI frontend; later starts reuse `kuzya-open-webui:local`.
