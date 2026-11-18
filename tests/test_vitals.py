import pandas as pd

from hospital_monitor.alerts import Severity
from hospital_monitor.anomaly.vitals import (StreamingVitalsMonitor, VitalSignsAnomalyDetector, clean_artifacts,
                                             news2_score)
from hospital_monitor.data.synthetic import generate_vitals


def test_news2_reference_values():
    df = pd.DataFrame([{"rr": 16, "spo2": 97, "sbp": 120, "hr": 75, "temp": 36.8},
                       {"rr": 26, "spo2": 90, "sbp": 88, "hr": 135, "temp": 39.5}])
    assert list(news2_score(df)) == [0, 3 + 3 + 3 + 3 + 2]


def test_artifacts_are_removed():
    df, _ = generate_vitals(hours=8)
    clean, mask = clean_artifacts(df[df.patient_id == "P003"].reset_index(drop=True))
    assert mask.values.sum() >= 3
    assert clean["spo2"].min() > 80


def test_injected_episodes_detected_without_false_positives():
    df, injected = generate_vitals()
    _, episodes = VitalSignsAnomalyDetector().detect(df)
    for inj in injected:
        hits = [e for e in episodes if e.patient_id == inj.patient_id and e.start <= inj.end and e.end >= inj.start]
        assert hits, f"episódio não detectado: {inj}"
    assert not [e for e in episodes if e.patient_id == "P003"]  # paciente estável
    p2 = [e for e in episodes if e.patient_id == "P002"]
    assert max(e.severity for e in p2) == Severity.CRITICA


def test_streaming_monitor_alerts_on_desaturation():
    df, _ = generate_vitals()
    g = df[df.patient_id == "P002"].reset_index(drop=True)
    mon = StreamingVitalsMonitor("P002", g.iloc[:360], {})
    alerts = [a for _, r in g.iloc[19 * 60:20 * 60 + 30].iterrows() if (a := mon.update(r.to_dict()))]
    assert alerts and alerts[-1].severity >= Severity.ALTA
    assert alerts[0].timestamp >= pd.Timestamp("2026-11-10 20:00")
