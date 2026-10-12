"""Tests for routing and the scoring API. Run the pipeline first, then: pytest"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import app
from triage import assign_queue

client = TestClient(app)

# A routine office visit for a real member and provider in the synthetic data.
NORMAL_CLAIM = {
    "claim_id": 1,
    "member_id": -10000010286183,
    "provider_id": "050796",
    "service_date": "2022-06-01",
    "received_date": "2022-06-03",
    "billed_amount": 230.00,
    "procedure_codes": ["99241"],
    "diagnosis_codes": ["E785", "E119", "I10"],
}


def score(**changes):
    response = client.post("/score", json={**NORMAL_CLAIM, **changes})
    assert response.status_code == 200
    return response.json()


def test_assign_queue_uses_both_thresholds():
    # columns: clean, denied, fwa
    proba = np.array([[0.98, 0.01, 0.01],    # low risk
                      [0.60, 0.35, 0.05],    # likely denial
                      [0.20, 0.10, 0.70]])   # likely FWA
    queue = assign_queue(proba, review_threshold=0.2, high_risk_threshold=0.5)
    assert list(queue) == ["auto_approve", "review", "high_risk"]


def test_normal_claim_is_auto_approved_without_reasons():
    result = score()
    assert result["queue"] == "auto_approve"
    assert result["reasons"] == []
    assert 0 <= result["score"] <= 1


def test_resubmitted_claim_is_flagged_as_duplicate():
    # Same member, provider, procedure, date and amount as a claim already in history.
    result = score(service_date="2022-03-03", received_date="2022-03-20", billed_amount=223.67)
    assert result["queue"] in ("review", "high_risk")
    assert any("duplicate" in reason.lower() for reason in result["reasons"])


def test_overbilled_claim_is_flagged_with_amount_reason():
    result = score(billed_amount=2500.00)
    assert result["queue"] in ("review", "high_risk")
    assert any("times the typical" in reason for reason in result["reasons"])
    assert result["score"] > score()["score"]


@pytest.mark.parametrize("bad_field", [
    {"procedure_codes": []},
    {"billed_amount": -5},
    {"service_date": "not a date"},
])
def test_invalid_claim_is_rejected(bad_field):
    response = client.post("/score", json={**NORMAL_CLAIM, **bad_field})
    assert response.status_code == 422
