from datetime import datetime, timedelta

from hospital_monitor.alerts import Alert, AlertManager, Severity, WebhookChannel
from hospital_monitor.fusion import MultimodalFusion

T0 = datetime(2026, 11, 11, 8, 0)


def test_cooldown_suppresses_repeated_alerts():
    m = AlertManager([], cooldown_minutes=15)
    a = Alert("P1", "vitals", Severity.ALTA, "x", "d", timestamp=T0)
    b = Alert("P1", "vitals", Severity.ALTA, "x", "d", timestamp=T0 + timedelta(minutes=5))
    c = Alert("P1", "vitals", Severity.ALTA, "x", "d", timestamp=T0 + timedelta(minutes=30))
    assert [m.emit(x) for x in (a, b, c)] == [True, False, True]
    assert m.suppressed == 1


def test_webhook_respects_min_severity(monkeypatch):
    sent = []
    import requests

    monkeypatch.setattr(requests, "post", lambda url, json, timeout: sent.append(json))
    ch = WebhookChannel("https://exemplo/webhook", Severity.ALTA)
    ch.send(Alert("P1", "audio", Severity.MEDIA, "baixo", "d"))
    ch.send(Alert("P1", "audio", Severity.CRITICA, "alto", "d"))
    assert len(sent) == 1 and "alto" in sent[0]["text"]


def test_fusion_requires_corroboration():
    f = MultimodalFusion()
    single = [Alert("P1", "vitals", Severity.ALTA, "v", "d", timestamp=T0, score=0.75)]
    assert f.run(single) == []
    multi = single + [Alert("P1", "audio", Severity.ALTA, "a", "d", timestamp=T0 + timedelta(hours=1), score=0.7)]
    fused = f.run(multi)
    assert len(fused) == 1 and fused[0].severity >= Severity.ALTA
    far = single + [Alert("P1", "audio", Severity.ALTA, "a", "d", timestamp=T0 + timedelta(hours=12), score=0.7)]
    assert f.run(far) == []
