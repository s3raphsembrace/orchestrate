"""Tiny disk cache for model responses.

Keyed on (model, prompt text, image bytes). Because generation is
deterministic (temperature 0), an identical (prompt, images) pair always maps
to the same answer, so caching makes re-runs free and lets you iterate on
downstream logic without re-billing the vision calls.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional


class ResponseCache:
    def __init__(self, enabled: bool, cache_dir: str | Path):
        self.enabled = enabled
        self.dir = Path(cache_dir)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(model: str, prompt: str, image_bytes: List[bytes]) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        for b in image_bytes:
            h.update(b"\x00")
            h.update(hashlib.sha256(b).digest())
        return h.hexdigest()

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        f = self.dir / f"{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        try:
            (self.dir / f"{key}.json").write_text(
                json.dumps(value), encoding="utf-8"
            )
        except OSError:
            pass
