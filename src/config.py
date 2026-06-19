"""Load and expose runtime configuration from config.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class Config:
    raw: Dict[str, Any] = field(default_factory=dict)

    # --- convenience accessors -------------------------------------------
    @property
    def stage1_model(self) -> str:
        return self.raw["models"]["stage1"]

    @property
    def stage2_model(self) -> str:
        return self.raw["models"]["stage2"]

    @property
    def generation(self) -> Dict[str, Any]:
        return self.raw["generation"]

    @property
    def runtime(self) -> Dict[str, Any]:
        return self.raw["runtime"]

    @property
    def retry(self) -> Dict[str, Any]:
        return self.raw["retry"]

    @property
    def cache(self) -> Dict[str, Any]:
        return self.raw["cache"]

    @property
    def pricing(self) -> Dict[str, Any]:
        return self.raw["pricing"]


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw)
