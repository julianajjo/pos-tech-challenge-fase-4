from types import SimpleNamespace

import numpy as np

from hospital_monitor.audio.features import extract_features, vocal_indicators
from hospital_monitor.audio.local_text import LocalTextAnalyzer, find_critical_terms
from hospital_monitor.azure.language import AzureTextAnalyzer, chunk_text


def _speech_like(sr=16000, syl_rate=5.0, pause_every=2.5, pause_len=0.3, dur=12.0, f0=180.0):
    t = np.arange(int(sr * dur)) / sr
    f0_t = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.4 * t))
    phase = 2 * np.pi * np.cumsum(f0_t) / sr
    voice = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.5 * (1 + np.sin(2 * np.pi * syl_rate * t)) ** 2
    gate = ((t % pause_every) < (pause_every - pause_len)).astype(float)
    return (voice * env * gate * 0.3).astype(np.float32), sr


def test_slow_choppy_speech_scores_higher_for_respiratory_difficulty():
    fluent = vocal_indicators(extract_features(*_speech_like()), {})
    choppy = vocal_indicators(extract_features(*_speech_like(syl_rate=2.5, pause_every=1.0, pause_len=0.55)), {})
    assert choppy["dificuldade_respiratoria"]["score"] > fluent["dificuldade_respiratoria"]["score"] + 0.3
    assert choppy["fadiga"]["score"] > fluent["fadiga"]["score"]


def test_critical_terms_with_negation():
    terms = find_critical_terms("Estou com falta de ar e febre, mas não tenho dor no peito.")
    d = {t["term"]: t for t in terms}
    assert not d["falta de ar"]["negated"]
    assert d["dor no peito"]["negated"]


def test_local_sentiment():
    a = LocalTextAnalyzer()
    assert a.analyze("Estou me sentindo bem melhor, dormi bem.")["sentiment"]["label"] == "positive"
    assert a.analyze("Estou muito cansada, com febre e dor.")["sentiment"]["label"] == "negative"


def test_chunking_respects_limit():
    chunks = chunk_text("Frase curta. " * 1000, max_chars=500)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) > 10


class _FakeClient:
    """Imita as respostas do azure-ai-textanalytics."""

    def analyze_sentiment(self, docs, show_opinion_mining=True):
        sent = SimpleNamespace(text=docs[0]["text"], sentiment="negative",
                               confidence_scores=SimpleNamespace(positive=0.05, neutral=0.1, negative=0.85),
                               mined_opinions=[])
        return [SimpleNamespace(is_error=False, sentences=[sent],
                                confidence_scores=SimpleNamespace(positive=0.05, neutral=0.1, negative=0.85))]

    def extract_key_phrases(self, docs):
        return [SimpleNamespace(is_error=False, key_phrases=["falta de ar"])]

    def recognize_entities(self, docs):
        return [SimpleNamespace(is_error=False, entities=[])]

    def begin_analyze_healthcare_entities(self, docs):
        ent = SimpleNamespace(text="dispneia", category="SymptomOrSign", normalized_text="Dyspnea",
                              confidence_score=0.98, assertion=None,
                              data_sources=[SimpleNamespace(name="UMLS", entity_id="C0013404")])
        return SimpleNamespace(result=lambda: [SimpleNamespace(is_error=False, entities=[ent])])


def test_azure_text_analyzer_parsing():
    a = AzureTextAnalyzer.__new__(AzureTextAnalyzer)
    a.client, a.language, a.use_health = _FakeClient(), "pt-BR", True
    res = a.analyze("Tenho dispneia desde ontem.")
    assert res["sentiment"]["label"] == "negative"
    assert res["health_entities"][0]["codes"] == ["UMLS:C0013404"]
    terms = find_critical_terms("Tenho dispneia desde ontem.", res["health_entities"])
    assert terms[0]["term"] == "dispneia"
