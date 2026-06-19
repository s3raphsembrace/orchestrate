"""Thin Gemini wrapper: image prep, JSON calls, retry/backoff, usage metering.

Isolates every dependency on the google-genai SDK so the rest of the pipeline
is provider-agnostic and unit-testable (see evaluation/ for a stub-driven dry
run that never touches the network).
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Best-effort: register an AVIF/HEIF decoder so PIL can read .avif test images
# instead of silently dropping them. Either optional package works; if neither
# is installed (and Pillow lacks native AVIF) the load simply fails gracefully.
try:  # pip install pillow-avif-plugin
    import pillow_avif  # noqa: F401
except Exception:
    try:  # pip install pillow-heif
        from pillow_heif import register_heif_opener, register_avif_opener  # type: ignore
        register_heif_opener()
        try:
            register_avif_opener()
        except Exception:
            pass
    except Exception:
        pass

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class RateLimiter:
    """Per-key minimum-interval throttle, shared across worker threads.

    Spaces calls for a given key (here, the model name) at least
    ``min_interval_s`` apart so the pipeline stays under the provider's
    requests-per-minute ceiling instead of bursting into 429s. Each model has
    its own quota pool, so we throttle per model rather than globally.
    """

    def __init__(self, min_interval_s: float):
        self.min = max(0.0, float(min_interval_s or 0.0))
        self._next: Dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: str) -> None:
        if self.min <= 0:
            return
        with self._lock:
            now = time.monotonic()
            target = max(now, self._next.get(key, 0.0))
            self._next[key] = target + self.min
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)


@dataclass
class Usage:
    """Process-wide token / call accounting, thread-safe."""
    calls_by_model: Dict[str, int] = field(default_factory=dict)
    input_tokens_by_model: Dict[str, int] = field(default_factory=dict)
    output_tokens_by_model: Dict[str, int] = field(default_factory=dict)
    images_processed: int = 0
    cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, in_tok: int, out_tok: int) -> None:
        with self._lock:
            self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1
            self.input_tokens_by_model[model] = (
                self.input_tokens_by_model.get(model, 0) + in_tok
            )
            self.output_tokens_by_model[model] = (
                self.output_tokens_by_model.get(model, 0) + out_tok
            )

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def add_images(self, n: int) -> None:
        with self._lock:
            self.images_processed += n


def load_and_prepare_image(path: str | Path, max_long_edge: int) -> bytes:
    """Open, downscale, re-encode as JPEG. Returns raw bytes for upload."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        long_edge = max(w, h)
        if long_edge > max_long_edge:
            scale = max_long_edge / float(long_edge)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


class GeminiClient:
    def __init__(self, config, usage: Usage):
        self.config = config
        self.usage = usage
        self._client = None  # lazy init so dry-runs need no key
        self.limiter = RateLimiter(
            config.runtime.get("min_request_interval_s", 0.0)
        )

    def _ensure_client(self):
        if self._client is None:
            from google import genai  # imported lazily

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            if not api_key:
                raise RuntimeError(
                    "Set GEMINI_API_KEY (or GOOGLE_API_KEY) to call the model. "
                    "Use --dry-run for a no-network smoke test."
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def generate_json(
        self,
        model: str,
        prompt: str,
        image_bytes: List[bytes],
        max_output_tokens: int,
    ) -> Tuple[dict, int, int]:
        """Call Gemini, return (parsed_json, input_tokens, output_tokens)."""
        from google.genai import types

        client = self._ensure_client()
        retry_cfg = self.config.retry
        gen = self.config.generation

        parts = [types.Part.from_text(text=prompt)]
        for b in image_bytes:
            parts.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))

        @retry(
            reraise=True,
            stop=stop_after_attempt(retry_cfg["max_attempts"]),
            wait=wait_exponential(
                multiplier=retry_cfg["initial_backoff_s"],
                max=retry_cfg["max_backoff_s"],
            ),
            retry=retry_if_exception_type(Exception),
        )
        def _call():
            # Throttle before every attempt (including retries) so a retry
            # storm doesn't itself trip the rate limit.
            self.limiter.wait(model)
            return client.models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=gen["temperature"],
                    response_mime_type=gen["response_mime_type"],
                    max_output_tokens=max_output_tokens,
                ),
            )

        resp = _call()
        in_tok, out_tok = self._extract_usage(resp)
        self.usage.record(model, in_tok, out_tok)
        parsed = self._parse_json(getattr(resp, "text", "") or "")
        return parsed, in_tok, out_tok

    @staticmethod
    def _extract_usage(resp) -> Tuple[int, int]:
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return 0, 0
        return (
            int(getattr(meta, "prompt_token_count", 0) or 0),
            int(getattr(meta, "candidates_token_count", 0) or 0),
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if 0 <= start < end:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
        return {}
