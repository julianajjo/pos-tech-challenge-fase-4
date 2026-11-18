"""Análise de texto local (fallback do Azure Text Analytics) e léxico de termos críticos.

O léxico de termos críticos também é aplicado sobre o resultado do Azure, para
mapear sintomas a níveis de gravidade definidos pela equipe clínica.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..alerts import Severity

# termo (sem acento, minúsculo) -> (categoria, severidade)
CRITICAL_TERMS: dict[str, tuple[str, Severity]] = {
    "dor no peito": ("cardiovascular", Severity.CRITICA),
    "dor toracica": ("cardiovascular", Severity.CRITICA),
    "aperto no peito": ("cardiovascular", Severity.CRITICA),
    "falta de ar": ("respiratorio", Severity.ALTA),
    "dispneia": ("respiratorio", Severity.ALTA),
    "dificuldade para respirar": ("respiratorio", Severity.ALTA),
    "chiado": ("respiratorio", Severity.MEDIA),
    "desmaio": ("neurologico", Severity.ALTA),
    "desmaiei": ("neurologico", Severity.ALTA),
    "sincope": ("neurologico", Severity.ALTA),
    "fala enrolada": ("neurologico", Severity.CRITICA),
    "boca torta": ("neurologico", Severity.CRITICA),
    "formigamento": ("neurologico", Severity.MEDIA),
    "fraqueza de um lado": ("neurologico", Severity.CRITICA),
    "confusao": ("neurologico", Severity.ALTA),
    "convulsao": ("neurologico", Severity.CRITICA),
    "tontura": ("neurologico", Severity.MEDIA),
    "febre": ("infeccioso", Severity.MEDIA),
    "calafrio": ("infeccioso", Severity.MEDIA),
    "sangramento": ("hemorragico", Severity.ALTA),
    "vomito com sangue": ("hemorragico", Severity.CRITICA),
    "palpitacao": ("cardiovascular", Severity.MEDIA),
    "cansada": ("fadiga", Severity.BAIXA),
    "cansado": ("fadiga", Severity.BAIXA),
    "cansaco": ("fadiga", Severity.BAIXA),
    "fadiga": ("fadiga", Severity.BAIXA),
    "fraqueza": ("fadiga", Severity.MEDIA),
    "nao tenho vontade de comer": ("nutricional", Severity.BAIXA),
    "sem apetite": ("nutricional", Severity.BAIXA),
    "quero morrer": ("saude_mental", Severity.CRITICA),
    "me machucar": ("saude_mental", Severity.CRITICA),
}
NEGATIONS = {"nao", "nega", "sem", "nenhum", "nenhuma", "nunca"}

POSITIVE = {"bem", "melhor", "otimo", "tranquilo", "aliviado", "aliviada", "dormi", "diminuiu", "normalmente", "bom"}
NEGATIVE = {"cansada", "cansado", "dor", "falta", "fraqueza", "febre", "tontura", "piorou", "pior", "medo",
            "ruim", "mal", "calafrios", "nao"}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", t)


def find_critical_terms(text: str, health_entities: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Localiza termos críticos considerando negação simples (até 3 palavras antes)."""
    norm = normalize(text)
    found = []
    for term, (cat, sev) in CRITICAL_TERMS.items():
        for m in re.finditer(r"\b" + re.escape(term) + r"s?\b", norm):
            before = norm[:m.start()].split()[-3:]
            negated = any(w in NEGATIONS for w in before) and term not in {"nao tenho vontade de comer"}
            found.append({"term": term, "category": cat, "severity": sev.name, "negated": negated,
                          "position": m.start()})
    # entidades do Text Analytics for Health não cobertas pelo léxico
    for e in health_entities or []:
        if e.get("category") in {"SymptomOrSign", "Diagnosis"} and not any(
                normalize(e["text"]) in f["term"] or f["term"] in normalize(e["text"]) for f in found):
            found.append({"term": e["text"], "category": e["category"], "severity": Severity.MEDIA.name,
                          "negated": e.get("negated", False), "position": -1, "source": "azure_health"})
    # remove termos contidos em outro já encontrado (ex.: "fraqueza" dentro de "fraqueza de um lado")
    found.sort(key=lambda f: -len(f["term"]))
    kept: list[dict[str, Any]] = []
    for f in found:
        if f["position"] >= 0 and any(k["position"] >= 0 and k["position"] <= f["position"] <
                                      k["position"] + len(k["term"]) for k in kept):
            continue
        kept.append(f)
    return sorted(kept, key=lambda f: f["position"])


class LocalTextAnalyzer:
    provider = "local"

    def analyze(self, text: str) -> dict[str, Any]:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        out_sent = []
        tot_pos = tot_neg = 0
        for s in sentences:
            words = normalize(s).replace(",", " ").replace(".", " ").split()
            p = sum(w in POSITIVE for w in words)
            n = sum(w in NEGATIVE for w in words)
            tot_pos += p
            tot_neg += n
            label = "positive" if p > n else "negative" if n > p else "neutral"
            out_sent.append({"text": s, "sentiment": label, "negative": n / max(p + n, 1), "opinions": []})
        total = max(tot_pos + tot_neg, 1)
        neg, pos = tot_neg / total, tot_pos / total
        sent = {"positive": pos, "negative": neg, "neutral": max(0.0, 1 - pos - neg)}
        sent["label"] = max(("positive", "neutral", "negative"), key=lambda k: sent[k])
        return {"provider": "local", "sentiment": sent, "sentences": out_sent, "key_phrases": [],
                "entities": [], "health_entities": []}
