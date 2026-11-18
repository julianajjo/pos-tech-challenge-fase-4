"""Carregamento de configurações (YAML + variáveis de ambiente)."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"

try:  # python-dotenv é opcional
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lê o arquivo YAML de configuração e acrescenta as credenciais do ambiente."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = deepcopy(cfg)
    az = cfg.setdefault("azure", {})
    az["speech_key"] = os.getenv("AZURE_SPEECH_KEY", "")
    az["speech_region"] = os.getenv("AZURE_SPEECH_REGION", "")
    az["language_key"] = os.getenv("AZURE_LANGUAGE_KEY", "")
    az["language_endpoint"] = os.getenv("AZURE_LANGUAGE_ENDPOINT", "")
    cfg.setdefault("alerts", {})["webhook_url"] = os.getenv("ALERT_WEBHOOK_URL", "")
    return cfg


def output_dir(cfg: dict[str, Any], *parts: str) -> Path:
    base = Path(cfg.get("paths", {}).get("output_dir", "outputs"))
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    p = base.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
