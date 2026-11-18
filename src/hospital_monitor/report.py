"""Gráficos e painel HTML da equipe médica."""
from __future__ import annotations

import base64
import html
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .alerts import SEVERITY_COLORS, Alert, Severity


SOURCE_LABELS = {"vitals": "Sinais vitais", "prescription": "Prescrição", "movement": "Movimentação",
                 "audio": "Áudio", "video": "Vídeo", "fusion": "FUSÃO"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_vitals(scored: pd.DataFrame, episodes, patient_id: str, path: Path) -> str:
    plt = _plt()
    g = scored[scored["patient_id"] == patient_id]
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    spec = [("hr", "FC (bpm)", "#dc2626"), ("sbp", "PAS (mmHg)", "#2563eb"), ("spo2", "SpO2 (%)", "#0891b2"),
            ("rr", "FR (irpm)", "#7c3aed"), ("news2", "NEWS2", "#111827")]
    for ax, (col, label, color) in zip(axes, spec):
        ax.plot(g["timestamp"], g[col], color=color, lw=0.8)
        ax.set_ylabel(label, fontsize=8)
        for e in [e for e in episodes if e.patient_id == patient_id]:
            ax.axvspan(e.start, e.end, color=SEVERITY_COLORS[e.severity], alpha=0.18)
        ax.tick_params(labelsize=7)
    iso = g[g["iso_flag"]]
    axes[-1].scatter(iso["timestamp"], iso["news2"], s=6, color="#d97706", label="Isolation Forest")
    art = g[g["artifact"]]
    axes[0].scatter(art["timestamp"], g.loc[art.index, "hr"], marker="x", color="#6b7280", s=20,
                    label="artefato removido")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[-1].legend(fontsize=7, loc="upper left")
    axes[0].set_title(f"Sinais vitais – {patient_id} (faixas = episódios anômalos detectados)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


def plot_movement(scored: pd.DataFrame, alerts: list[Alert], path: Path) -> str:
    plt = _plt()
    pids = sorted(scored["patient_id"].unique())
    fig, axes = plt.subplots(len(pids), 1, figsize=(12, 1.8 * len(pids)), sharex=True)
    for ax, pid in zip(axes, pids):
        g = scored[scored["patient_id"] == pid]
        ax.fill_between(g["timestamp"], g["activity"], color="#94a3b8", lw=0)
        ax.set_ylabel(pid, fontsize=8)
        for a in alerts:
            if a.patient_id == pid:
                ax.axvline(a.timestamp, color=SEVERITY_COLORS[a.severity], lw=2)
                ax.text(a.timestamp, 85, " " + a.title, fontsize=7, color=SEVERITY_COLORS[a.severity])
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=7)
    axes[0].set_title("Índice de atividade durante a internação (linhas = alertas)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


def plot_alert_timeline(alerts: list[Alert], path: Path) -> str:
    plt = _plt()
    sources = ["vitals", "prescription", "movement", "audio", "video", "fusion"]
    labels = {"vitals": "Sinais vitais", "prescription": "Prescrições", "movement": "Movimentação",
              "audio": "Áudio", "video": "Vídeo", "fusion": "FUSÃO"}
    pids = sorted({a.patient_id for a in alerts})
    fig, axes = plt.subplots(len(pids), 1, figsize=(12, 1.9 * len(pids)), sharex=True)
    axes = axes if len(pids) > 1 else [axes]
    for ax, pid in zip(axes, pids):
        for a in [a for a in alerts if a.patient_id == pid]:
            y = sources.index(a.source) if a.source in sources else 0
            ax.scatter(a.timestamp, y, s=30 + 25 * int(a.severity), color=SEVERITY_COLORS[a.severity],
                       marker="*" if a.source == "fusion" else "o", zorder=3)
        ax.set_yticks(range(len(sources)), [labels[s] for s in sources], fontsize=7)
        ax.set_ylabel(pid, fontsize=9, rotation=0, labelpad=25)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_title("Linha do tempo dos alertas por paciente e modalidade (cor = severidade)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return str(path)


def _img(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    return f'<img src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}" />'


def build_dashboard(alerts: list[Alert], summary: dict[str, dict[str, Any]], images: dict[str, str],
                    path: Path, title: str = "Central de Monitoramento Multimodal") -> str:
    alerts = sorted(alerts, key=lambda a: (a.timestamp, -int(a.severity)))
    counts = {s.name: sum(a.severity == s for a in alerts) for s in Severity if s != Severity.INFO}
    cards = []
    for pid, s in sorted(summary.items(), key=lambda x: -x[1]["peak_risk"]):
        sev = Severity[s["severity"]]
        mods = "".join(f"<li>{SOURCE_LABELS.get(m, m)}: {v:.2f}</li>" for m, v in
                       sorted(s["modalities"].items(), key=lambda x: -x[1]))
        when = s["peak_time"].strftime("%d/%m %H:%M") if s["peak_time"] else "-"
        cards.append(f"""<div class="card" style="border-top:6px solid {SEVERITY_COLORS[sev]}">
<h3>{pid}</h3><div class="risk" style="color:{SEVERITY_COLORS[sev]}">{s['peak_risk']:.2f}</div>
<div class="muted">risco multimodal máximo ({sev.name}) em {when}</div><ul>{mods}</ul>
<div class="muted">{s['n_alerts']} alertas de origem</div></div>""")
    rows = []
    for a in alerts:
        color = SEVERITY_COLORS[a.severity]
        rows.append(f"""<tr class="{'fusion' if a.source == 'fusion' else ''}">
<td>{a.timestamp:%d/%m %H:%M}</td><td>{a.patient_id}</td>
<td><span class="badge" style="background:{color}">{a.severity.name}</span></td>
<td>{SOURCE_LABELS.get(a.source, a.source)}</td><td><b>{html.escape(a.title)}</b><br><span class="muted">{html.escape(a.description)}</span></td>
<td>{html.escape(a.recommended_action)}</td></tr>""")
    gallery = "".join(f"<figure>{_img(p)}<figcaption>{html.escape(k)}</figcaption></figure>"
                      for k, p in images.items() if Path(p).exists())
    doc = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f1f5f9;color:#0f172a}}
header{{background:#0f172a;color:white;padding:18px 28px}} header h1{{margin:0;font-size:22px}}
.wrap{{padding:20px 28px}} .kpis{{display:flex;gap:12px;margin-bottom:18px}}
.kpi{{background:white;border-radius:8px;padding:12px 18px;min-width:120px;box-shadow:0 1px 3px #0002}}
.kpi b{{font-size:26px;display:block}} .cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:white;border-radius:8px;padding:12px 16px;width:220px;box-shadow:0 1px 3px #0002}}
.card h3{{margin:0}} .risk{{font-size:34px;font-weight:700}} .muted{{color:#64748b;font-size:12px}}
.card ul{{padding-left:18px;font-size:13px;margin:6px 0}}
table{{border-collapse:collapse;width:100%;background:white;font-size:13px;box-shadow:0 1px 3px #0002}}
th,td{{padding:7px 9px;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}}
th{{background:#e2e8f0}} tr.fusion{{background:#fef2f2}}
.badge{{color:white;border-radius:4px;padding:2px 6px;font-size:11px;font-weight:600}}
figure{{background:white;margin:0 0 16px;padding:10px;border-radius:8px;box-shadow:0 1px 3px #0002}}
figure img{{max-width:100%}} figcaption{{font-size:12px;color:#475569}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px}}
</style></head><body>
<header><h1>{html.escape(title)}</h1><div>Gerado em {datetime.now():%d/%m/%Y %H:%M} · {len(alerts)} alertas</div></header>
<div class="wrap">
<div class="kpis">{''.join(f'<div class="kpi" style="border-left:5px solid {SEVERITY_COLORS[Severity[k]]}"><b>{v}</b>{k}</div>' for k, v in counts.items())}</div>
<h2>Pacientes</h2><div class="cards">{''.join(cards)}</div>
<h2>Alertas para a equipe médica</h2>
<table><tr><th>Data/hora</th><th>Paciente</th><th>Severidade</th><th>Origem</th><th>Alerta</th><th>Conduta sugerida</th></tr>
{''.join(rows)}</table>
<h2>Evidências</h2><div class="gallery">{gallery}</div>
</div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return str(path)
