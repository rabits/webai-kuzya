import asyncio
import base64
import gc
import io
import os
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict

import whisper


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "turbo")
WHISPER_DOWNLOAD_ROOT = os.getenv("WHISPER_DOWNLOAD_ROOT", "/app/models/whisper")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "ru")
STT_MODEL_ID = os.getenv("STT_MODEL", WHISPER_MODEL)

# Open WebUI / OpenAI clients send various aliases for the same checkpoint.
MODEL_ALIASES = {
    "whisper-1": "turbo",
    "large-v3-turbo": "turbo",
    "whisper-large-v3-turbo": "turbo",
    "openai/whisper-large-v3-turbo": "turbo",
    "whisper-large-v3": "large-v3",
    "openai/whisper-large-v3": "large-v3",
}

models: dict = {}
infer_lock = asyncio.Lock()


def resolve_model_name(name: str) -> str:
    key = (name or "").strip()
    return MODEL_ALIASES.get(key, key) or "turbo"


def openai_error(message: str, *, err_type: str = "invalid_request_error", param: str | None = None) -> dict:
    return {"message": message, "type": err_type, "param": param, "code": None}


def _require_cuda() -> None:
    import torch

    print(f"   torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch was installed without CUDA. On DGX Spark install torch from "
            "https://download.pytorch.org/whl/cu130/ (see stt/Dockerfile)."
        )
    print(f"   torch device: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")


def _release_cuda() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _silent_wav(seconds: float = 0.4, sr: int = 16000) -> bytes:
    pcm = np.zeros(int(sr * seconds), dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _write_temp(data: bytes, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return path


def transcribe_path(path: str, language: str | None, prompt: str | None, temperature: float) -> dict:
    kwargs = {
        "fp16": True,
        "temperature": temperature,
        "task": "transcribe",
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["initial_prompt"] = prompt
    return models["whisper"].transcribe(path, **kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Checking CUDA...")
    _require_cuda()

    os.makedirs(WHISPER_DOWNLOAD_ROOT, exist_ok=True)
    model_name = resolve_model_name(WHISPER_MODEL)
    print(f"⏳ Loading Whisper '{model_name}'...")
    models["whisper"] = whisper.load_model(
        model_name,
        device="cuda",
        download_root=WHISPER_DOWNLOAD_ROOT,
    )

    print("⏳ Warming up Whisper...")
    tmp = _write_temp(_silent_wav(), ".wav")
    try:
        await asyncio.to_thread(transcribe_path, tmp, WHISPER_LANGUAGE or None, None, 0.0)
    finally:
        os.unlink(tmp)
        _release_cuda()

    print("✅ STT ready")
    yield
    models.clear()
    _release_cuda()


app = FastAPI(title="Whisper STT", lifespan=lifespan)
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


class JsonTranscriptionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    language: str | None = None
    prompt: str | None = None
    response_format: str = "json"
    temperature: float = 0.0
    input_audio: dict | None = None


def _language(value: str | None) -> str | None:
    lang = (value or "").strip() or WHISPER_LANGUAGE
    return lang or None


def _format_result(result: dict, response_format: str):
    fmt = (response_format or "json").lower()
    text = (result.get("text") or "").strip()
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "verbose_json":
        return JSONResponse(result)
    if fmt != "json":
        raise HTTPException(
            status_code=400,
            detail=openai_error(f"Unsupported response_format '{response_format}'", param="response_format"),
        )
    return JSONResponse({"text": text})


async def _run_transcribe(data: bytes, suffix: str, language: str | None, prompt: str | None, temperature: float) -> dict:
    if not data:
        raise HTTPException(status_code=400, detail=openai_error("audio file is empty", param="file"))
    path = _write_temp(data, suffix)
    try:
        async with infer_lock:
            result = await asyncio.to_thread(
                transcribe_path, path, language, prompt, float(temperature or 0.0),
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=openai_error(f"transcription failed: {exc}")) from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
        _release_cuda()
    return result


@app.get("/health")
async def health():
    import torch

    return {
        "status": "ok",
        "model": STT_MODEL_ID,
        "whisper": resolve_model_name(WHISPER_MODEL),
        "language": WHISPER_LANGUAGE or None,
        "cuda": torch.cuda.is_available(),
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": STT_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    content_type = (request.headers.get("content-type") or "").lower()

    if "application/json" in content_type:
        body = JsonTranscriptionRequest.model_validate(await request.json())
        audio = body.input_audio or {}
        b64 = audio.get("data")
        if not b64:
            raise HTTPException(status_code=400, detail=openai_error("input_audio.data is required", param="input_audio"))
        try:
            data = base64.b64decode(b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=openai_error(f"invalid base64 audio: {exc}", param="input_audio")) from exc
        fmt = audio.get("format") or "wav"
        suffix = f".{str(fmt).lstrip('.')}"
        result = await _run_transcribe(data, suffix, _language(body.language), body.prompt, body.temperature)
        return _format_result(result, body.response_format)

    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail=openai_error("file is required", param="file"))

    data = await upload.read()
    filename = getattr(upload, "filename", None) or "audio.webm"
    suffix = Path(filename).suffix or ".webm"
    language = _language(form.get("language"))
    prompt = form.get("prompt") or None
    response_format = form.get("response_format") or "json"
    try:
        temperature = float(form.get("temperature") or 0.0)
    except (TypeError, ValueError):
        temperature = 0.0

    result = await _run_transcribe(data, suffix, language, prompt, temperature)
    return _format_result(result, str(response_format))
