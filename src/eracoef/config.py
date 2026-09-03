"""Load config.yaml."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Windows consoles default to cp1252; player names are UTF-8 (Schröder, Ginóbili).
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else ROOT / "config.yaml"
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(ROOT)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] relative to the repo root."""
    return Path(cfg["_root"]) / cfg["paths"][key]
