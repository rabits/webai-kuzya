import asyncio
import gc
import io
import os
import subprocess
import wave
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field


def _prefer_blocking_gpu_wait() -> str:
    """ROCm HIP busy-spins one CPU core after the first GPU op (AsyncEventsLoop).

    hipSetDeviceFlags(BlockingSync) MUST run before `import torch`. Calling it
    afterwards still returns rc=0 but the spinner is already live.
    HIP_SCHEDULE=off|spin|yield|blocking (default blocking).
    """
    import ctypes
    import glob

    choice = os.environ.get("HIP_SCHEDULE", "blocking").strip().lower()
    flags = {"auto": 0, "spin": 1, "yield": 2, "blocking": 4}
    if choice in {"off", "0", "none"}:
        return "skipped"
    value = flags.get(choice, 4)
    candidates = glob.glob("/opt/venv/lib/python*/site-packages/torch/lib/libamdhip64.so")
    candidates.extend(
        (
            "libamdhip64.so",
            "/opt/rocm/lib/libamdhip64.so",
            "/opt/rocm/lib64/libamdhip64.so",
        )
    )
    for soname in candidates:
        try:
            hip = ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        fn = getattr(hip, "hipSetDeviceFlags", None)
        if fn is None:
            continue
        fn.argtypes = [ctypes.c_uint]
        fn.restype = ctypes.c_int
        rc = fn(value)
        return f"{soname} {choice}({value}) rc={rc}"
    return "libamdhip64 not found"


_GPU_WAIT_MODE = _prefer_blocking_gpu_wait()

import torch
from silero_stress import load_accentor
from f5_tts.api import F5TTS


def _patch_torchaudio_io() -> None:
    """torchaudio 2.9+ routes load/save through torchcodec, which links CUDA (libnvrtc)."""
    import soundfile as sf
    import torch
    import torchaudio

    def load(uri, *args, **kwargs):
        frame_offset = int(kwargs.get("frame_offset", 0) or 0)
        num_frames = kwargs.get("num_frames", -1)
        frames = -1 if num_frames in (-1, None) else int(num_frames)
        data, sr = sf.read(
            str(uri),
            dtype="float32",
            always_2d=True,
            start=frame_offset,
            frames=frames,
        )
        return torch.from_numpy(np.ascontiguousarray(data.T)), sr

    def save(uri, src, sample_rate, *args, **kwargs):
        arr = src.detach().cpu().numpy()
        if arr.ndim == 2:
            arr = arr.T
        sf.write(str(uri), arr, int(sample_rate))

    torchaudio.load = load
    torchaudio.save = save


_patch_torchaudio_io()


F5_MODEL_NAME   = os.getenv("F5_MODEL_NAME",   "F5TTS_v1_Base")
F5_MODEL_PATH   = os.getenv("F5_MODEL_PATH",   "/app/models/f5tts-russian/model_last_inference.safetensors")
F5_VOCAB_PATH   = os.getenv("F5_VOCAB_PATH",   "/app/models/f5tts-russian/vocab.txt")
VOICES_DIR      = os.getenv("VOICES_DIR",      "/app/models/voices")
DEFAULT_VOICE   = os.getenv("DEFAULT_VOICE",   "default")
TTS_MODEL_ID    = os.getenv("TTS_MODEL",       "f5-tts")
F5_NFE_STEP     = int(os.getenv("F5_NFE_STEP", "32"))

ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

CONTENT_TYPES = {
    "mp3":  "audio/mpeg",
    "opus": "audio/ogg",
    "aac":  "audio/aac",
    "flac": "audio/flac",
    "wav":  "audio/wav",
    "pcm":  "application/octet-stream",
}

FFMPEG_FORMATS = {
    "mp3":  "mp3",
    "opus": "opus",
    "aac":  "adts",
    "flac": "flac",
}

models: dict = {}
infer_lock = asyncio.Lock()


def _voice_dir(voice: str) -> str:
    return os.path.join(VOICES_DIR, voice)


def _voice_ref(voice: str) -> tuple[str, str]:
    d = _voice_dir(voice)
    wav = os.path.join(d, "ref.wav")
    txt = os.path.join(d, "ref.txt")
    if not os.path.isfile(wav):
        raise FileNotFoundError(voice)
    ref_text = open(txt, encoding="utf-8").read().strip() if os.path.exists(txt) else ""
    return wav, ref_text


def list_voice_names() -> list[str]:
    if not os.path.isdir(VOICES_DIR):
        return []
    return [
        name for name in sorted(os.listdir(VOICES_DIR))
        if os.path.isfile(os.path.join(VOICES_DIR, name, "ref.wav"))
    ]


def _require_gpu() -> None:
    import torch

    ver = torch.__version__
    available = torch.cuda.is_available()
    print(f"   torch {ver} gpu={available}")
    if not available:
        if "rocm" not in ver.lower():
            raise RuntimeError(
                f"PyTorch cannot see a GPU ({ver} looks like CUDA). "
                "AMD: docker compose -f docker-compose.amd.yml build --no-cache stt tts"
            )
        raise RuntimeError(
            f"PyTorch cannot see a GPU ({ver}). "
            "Check /dev/kfd, /dev/dri, HIP_VISIBLE_DEVICES, and HSA_OVERRIDE_GFX_VERSION."
        )
    extra = ""
    try:
        extra = f" cap={torch.cuda.get_device_capability(0)}"
    except Exception:
        pass
    print(f"   torch device: {torch.cuda.get_device_name(0)}{extra}")
    print(f"   hip wait: {_GPU_WAIT_MODE}")
    if "rocm" in ver.lower() and os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") != "1":
        print(
            "   WARNING: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL is unset; "
            "RDNA4 flash attention stays off and F5-TTS uses the slow math kernel."
        )


def _configure_cpu_threads() -> None:
    import torch

    n = max(1, int(os.getenv("OMP_NUM_THREADS", "4")))
    torch.set_num_threads(n)
    torch.set_num_interop_threads(min(n, 4))
    print(f"   torch cpu threads: {torch.get_num_threads()}")


def _release_gpu(*, trim: bool = False) -> None:
    import torch

    # hipFree/empty_cache on every request is expensive on ROCm and does not
    # help latency. Only return cached blocks when the process is going away.
    if not trim:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def encode_audio(wav: bytes, response_format: str) -> tuple[bytes, str]:
    fmt = response_format.lower()
    if fmt not in CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=openai_error(
            f"Unsupported response_format '{response_format}'",
            param="response_format",
        ))
    if fmt == "wav":
        return wav, CONTENT_TYPES[fmt]
    if fmt == "pcm":
        with wave.open(io.BytesIO(wav), "rb") as wf:
            return wf.readframes(wf.getnframes()), CONTENT_TYPES[fmt]

    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "wav", "-i", "pipe:0",
            "-f", FFMPEG_FORMATS[fmt], "pipe:1",
        ],
        input=wav,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise HTTPException(status_code=500, detail=openai_error(f"ffmpeg failed: {err}"))
    return proc.stdout, CONTENT_TYPES[fmt]


def openai_error(message: str, *, err_type: str = "invalid_request_error", param: str | None = None) -> dict:
    return {"message": message, "type": err_type, "param": param, "code": None}


def synthesize(text: str, voice: str, speed: float, accentuate: bool) -> bytes:
    try:
        ref_wav, ref_text = _voice_ref(voice)
    except FileNotFoundError:
        known = ", ".join(list_voice_names()) or "(none)"
        raise HTTPException(
            status_code=400,
            detail=openai_error(f"Unknown voice '{voice}'. Available: {known}", param="voice"),
        )

    gen_text = models["accentor"](text) if accentuate else text
    audio, sr, _ = models["f5tts"].infer(
        ref_file=ref_wav,
        ref_text=ref_text,
        gen_text=gen_text,
        speed=speed,
        nfe_step=F5_NFE_STEP,
        progress=None,
        show_info=lambda *_args, **_kwargs: None,
    )
    return wav_bytes(audio, sr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Checking GPU...")
    _configure_cpu_threads()
    _require_gpu()

    print("⏳ Loading silero-stress...")
    models["accentor"] = load_accentor()

    print("⏳ Loading F5-TTS...")
    models["f5tts"] = F5TTS(
        model=F5_MODEL_NAME,
        ckpt_file=F5_MODEL_PATH,
        vocab_file=F5_VOCAB_PATH,
        device="cuda",
    )

    print("⏳ Warming up F5-TTS...")
    await asyncio.to_thread(synthesize, "Привет.", DEFAULT_VOICE, 1.0, True)

    print("✅ TTS ready")
    yield
    models.clear()
    _release_gpu(trim=True)


app = FastAPI(title="F5-TTS", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "message" in detail:
        body = {"error": detail}
    else:
        body = {"error": openai_error(str(detail))}
    return JSONResponse(status_code=exc.status_code, content=body)


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, exc: RequestValidationError):
    loc = exc.errors()[0].get("loc", ["body"])[-1] if exc.errors() else "body"
    msg = exc.errors()[0].get("msg", str(exc)) if exc.errors() else str(exc)
    return JSONResponse(
        status_code=400,
        content={"error": openai_error(msg, param=str(loc))},
    )


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = TTS_MODEL_ID
    input: str = Field(..., min_length=1)
    voice: str = DEFAULT_VOICE
    response_format: ResponseFormat = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
    accentuate: bool = True


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": TTS_MODEL_ID,
        "voices": list_voice_names(),
        "cuda": True,
    }


@app.get("/voices")
async def voices():
    return {"voices": list_voice_names()}


@app.get("/v1/audio/voices")
async def openai_voices():
    # Open WebUI probes this path and falls back to alloy/echo/... if it 404s.
    return {"voices": [{"id": name, "name": name} for name in list_voice_names()]}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": TTS_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.get("/v1/audio/models")
async def openai_audio_models():
    return {"models": [{"id": TTS_MODEL_ID, "name": TTS_MODEL_ID}]}


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    text = req.input.strip()
    if not text:
        raise HTTPException(status_code=400, detail=openai_error("input is empty", param="input"))

    async with infer_lock:
        wav = await asyncio.to_thread(
            synthesize, text, req.voice, float(req.speed), req.accentuate,
        )

    body, media = encode_audio(wav, req.response_format)
    return Response(content=body, media_type=media)
