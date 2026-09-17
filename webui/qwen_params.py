"""Map Qwen 3.8 thinking flags onto the vLLM OpenAI-compatible body.

Open WebUI already forwards `reasoning_effort` at the JSON root (same place as
the Python SDK's `reasoning_effort=`). `enable_thinking` / `preserve_thinking`
must live under `chat_template_kwargs` — the SDK's `extra_body` is just how
those extra keys are merged into the same HTTP JSON.
"""

from __future__ import annotations

from typing import Any, Callable


def _as_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off", ""}:
            return False
    return value


def fold_qwen_thinking(form_data: dict) -> dict:
    if not isinstance(form_data, dict):
        return form_data

    kwargs = form_data.get("chat_template_kwargs")
    if isinstance(kwargs, str):
        try:
            import json

            kwargs = json.loads(kwargs)
        except Exception:
            kwargs = {}
    if not isinstance(kwargs, dict):
        kwargs = {}
    else:
        kwargs = dict(kwargs)

    moved = False
    for key in ("enable_thinking", "preserve_thinking"):
        if key in form_data and form_data[key] is not None:
            kwargs[key] = _as_bool(form_data.pop(key))
            moved = True
        elif key in kwargs and kwargs[key] is not None:
            kwargs[key] = _as_bool(kwargs[key])
            moved = True

    if moved or kwargs:
        form_data["chat_template_kwargs"] = kwargs

    effort = form_data.get("reasoning_effort")
    if isinstance(effort, str):
        form_data["reasoning_effort"] = effort.strip()

    return form_data


def wrap_openai(orig: Callable) -> Callable:
    def wrapped(params, form_data):
        return fold_qwen_thinking(orig(params, form_data))

    return wrapped


def wrap_form(orig: Callable) -> Callable:
    def wrapped(*args, **kwargs):
        return fold_qwen_thinking(orig(*args, **kwargs))

    return wrapped
