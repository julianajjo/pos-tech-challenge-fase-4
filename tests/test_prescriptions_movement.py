from hospital_monitor.alerts import Severity
from hospital_monitor.anomaly.movement import MovementAnomalyDetector
from hospital_monitor.anomaly.prescriptions import PrescriptionAnomalyDetector
from hospital_monitor.data.synthetic import generate_movement, generate_prescriptions


def test_prescription_rules():
    _, alerts = PrescriptionAnomalyDetector().detect(generate_prescriptions())
    titles = {(a.patient_id, a.title) for a in alerts}
    assert ("P001", "Dose acima do máximo diário: Morfina") in titles
    assert ("P002", "Interação medicamentosa: Varfarina + AAS") in titles
    assert any(a.patient_id == "P004" and "Insulina" in a.title for a in alerts)
    assert not [a for a in alerts if a.patient_id == "P003" and a.severity >= Severity.MEDIA]


def test_movement_patterns():
    _, alerts = MovementAnomalyDetector().detect(generate_movement())
    got = {(a.patient_id, a.title) for a in alerts}
    assert ("P001", "Imobilidade prolongada") in got
    assert ("P002", "Agitação noturna") in got
    assert ("P004", "Possível queda") in got
    assert not [a for a in alerts if a.patient_id == "P003"]
