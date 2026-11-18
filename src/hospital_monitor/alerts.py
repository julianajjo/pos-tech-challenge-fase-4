"""Geração, deduplicação e envio de alertas para a equipe médica."""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)


class Severity(IntEnum):
    INFO = 0
    BAIXA = 1
    MEDIA = 2
    ALTA = 3
    CRITICA = 4

    @classmethod
    def parse(cls, value: "Severity | str | int") -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).upper()]


SEVERITY_COLORS = {
    Severity.INFO: "#6b7280",
    Severity.BAIXA: "#2563eb",
    Severity.MEDIA: "#d97706",
    Severity.ALTA: "#dc2626",
    Severity.CRITICA: "#7f1d1d",
}


@dataclass
class Alert:
    patient_id: str
    source: str  # video | audio | vitals | prescription | movement | fusion
    severity: Severity
    title: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_action: str = ""
    score: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.name
        d["timestamp"] = self.timestamp.isoformat(timespec="seconds")
        return d

    def short(self) -> str:
        return (
            f"[{self.severity.name}] {self.timestamp:%d/%m %H:%M} | {self.patient_id} | "
            f"{self.source} | {self.title}"
        )


class AlertChannel(Protocol):
    def send(self, alert: Alert) -> None: ...


class ConsoleChannel:
    def send(self, alert: Alert) -> None:
        print(alert.short())


class JsonlChannel:
    """Persiste os alertas em JSON Lines (trilha de auditoria)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, alert: Alert) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert.to_dict(), ensure_ascii=False, default=str) + "\n")


class WebhookChannel:
    """Envia o alerta para um webhook (ex.: Microsoft Teams ou Slack)."""

    def __init__(self, url: str, min_severity: Severity = Severity.ALTA, timeout: float = 5.0):
        self.url = url
        self.min_severity = min_severity
        self.timeout = timeout

    def payload(self, alert: Alert) -> dict[str, Any]:
        text = (
            f"**{alert.severity.name} - {alert.title}**\n\n"
            f"Paciente: {alert.patient_id} | Origem: {alert.source} | "
            f"{alert.timestamp:%d/%m/%Y %H:%M}\n\n{alert.description}"
        )
        if alert.recommended_action:
            text += f"\n\nConduta sugerida: {alert.recommended_action}"
        return {"text": text}

    def send(self, alert: Alert) -> None:
        if not self.url or alert.severity < self.min_severity:
            return
        import requests

        try:
            requests.post(self.url, json=self.payload(alert), timeout=self.timeout)
        except Exception as exc:  # falha no webhook não pode derrubar o monitoramento
            logger.warning("Falha ao enviar alerta para webhook: %s", exc)


class AlertManager:
    """Centraliza os alertas de todas as modalidades.

    - aplica *cooldown* para evitar fadiga de alarmes (mesmo paciente/origem/título);
    - distribui o alerta para os canais configurados;
    - mantém o histórico para o painel e para a fusão multimodal.
    """

    def __init__(self, channels: Iterable[AlertChannel] | None = None, cooldown_minutes: float = 15):
        self.channels = list(channels or [])
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.history: list[Alert] = []
        self.suppressed = 0
        self._last: dict[tuple[str, str, str], datetime] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any], console: bool = True) -> "AlertManager":
        a = cfg.get("alerts", {})
        channels: list[AlertChannel] = []
        if console:
            channels.append(ConsoleChannel())
        if a.get("jsonl_path"):
            from .config import PROJECT_ROOT

            p = Path(a["jsonl_path"])
            channels.append(JsonlChannel(p if p.is_absolute() else PROJECT_ROOT / p))
        if a.get("webhook_url"):
            channels.append(
                WebhookChannel(a["webhook_url"], Severity.parse(a.get("min_severity_webhook", "ALTA")))
            )
        return cls(channels, a.get("cooldown_minutes", 15))

    def emit(self, alert: Alert) -> bool:
        key = (alert.patient_id, alert.source, alert.title)
        last = self._last.get(key)
        if last is not None and abs(alert.timestamp - last) < self.cooldown:
            self.suppressed += 1
            return False
        self._last[key] = alert.timestamp
        self.history.append(alert)
        for ch in self.channels:
            ch.send(alert)
        return True

    def emit_many(self, alerts: Iterable[Alert]) -> int:
        return sum(self.emit(a) for a in sorted(alerts, key=lambda a: a.timestamp))

    def by_patient(self) -> dict[str, list[Alert]]:
        out: dict[str, list[Alert]] = {}
        for a in self.history:
            out.setdefault(a.patient_id, []).append(a)
        return out

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([a.to_dict() for a in self.history], f, ensure_ascii=False, indent=2, default=str)
