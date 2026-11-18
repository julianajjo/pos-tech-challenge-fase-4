"""Integração com o Azure AI Language (Text Analytics).

Recursos utilizados sobre a transcrição da consulta:
- análise de sentimento com *opinion mining* (frase a frase);
- extração de frases-chave;
- reconhecimento de entidades (NER);
- **Text Analytics for Health**: sintomas, diagnósticos, medicamentos, com
  asserções de negação ("nega dor torácica") e vínculo a ontologias (UMLS, CID...).

Variáveis de ambiente: ``AZURE_LANGUAGE_KEY`` e ``AZURE_LANGUAGE_ENDPOINT``.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)
MAX_CHARS = 5000


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks or [""]


class AzureTextAnalyzer:
    provider = "azure"

    def __init__(self, key: str, endpoint: str, language: str = "pt-BR", use_health: bool = True):
        if not key or not endpoint:
            raise ValueError("AZURE_LANGUAGE_KEY/AZURE_LANGUAGE_ENDPOINT não configurados.")
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        self.client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        self.language = language
        self.use_health = use_health

    def analyze(self, text: str) -> dict[str, Any]:
        docs = [{"id": str(i), "language": self.language, "text": c} for i, c in enumerate(chunk_text(text))]
        out: dict[str, Any] = {"provider": "azure", "sentences": [], "key_phrases": [], "entities": [],
                               "health_entities": []}
        pos = neu = neg = 0.0
        for doc in self.client.analyze_sentiment(docs, show_opinion_mining=True):
            if doc.is_error:
                logger.warning("Sentimento: %s", doc.error)
                continue
            pos += doc.confidence_scores.positive
            neu += doc.confidence_scores.neutral
            neg += doc.confidence_scores.negative
            for s in doc.sentences:
                out["sentences"].append({
                    "text": s.text, "sentiment": s.sentiment,
                    "negative": s.confidence_scores.negative,
                    "opinions": [{"target": o.target.text,
                                  "assessments": [a.text for a in o.assessments],
                                  "sentiment": o.target.sentiment} for o in (s.mined_opinions or [])],
                })
        n = max(len(docs), 1)
        out["sentiment"] = {"positive": pos / n, "neutral": neu / n, "negative": neg / n}
        out["sentiment"]["label"] = max(("positive", "neutral", "negative"), key=lambda k: out["sentiment"][k])

        for doc in self.client.extract_key_phrases(docs):
            if not doc.is_error:
                out["key_phrases"] += list(doc.key_phrases)
        for doc in self.client.recognize_entities(docs):
            if not doc.is_error:
                out["entities"] += [{"text": e.text, "category": e.category, "subcategory": e.subcategory,
                                     "confidence": e.confidence_score} for e in doc.entities]
        if self.use_health:
            try:
                poller = self.client.begin_analyze_healthcare_entities(docs)
                for doc in poller.result():
                    if doc.is_error:
                        continue
                    for e in doc.entities:
                        a = e.assertion
                        out["health_entities"].append({
                            "text": e.text, "category": e.category, "normalized": e.normalized_text,
                            "confidence": e.confidence_score,
                            "negated": bool(a and a.certainty and "negative" in str(a.certainty).lower()),
                            "codes": [f"{d.name}:{d.entity_id}" for d in (e.data_sources or [])][:3],
                        })
            except Exception as exc:  # recurso pode não estar habilitado na região/plano
                logger.warning("Text Analytics for Health indisponível: %s", exc)
        return out


def build_text_analyzer(azure_cfg: dict[str, Any]):
    if azure_cfg.get("language_key") and azure_cfg.get("language_endpoint"):
        return AzureTextAnalyzer(azure_cfg["language_key"], azure_cfg["language_endpoint"],
                                 azure_cfg.get("text_language", "pt-BR"), azure_cfg.get("use_healthcare_entities", True))
    from ..audio.local_text import LocalTextAnalyzer

    logger.info("Azure Language não configurado: usando análise de texto local.")
    return LocalTextAnalyzer()
