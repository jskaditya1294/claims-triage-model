"""Routing: turn model probabilities into a queue, for many claims or for one.

Routing rule (thresholds are chosen in step 6):
    high_risk     if P(fwa) >= high_risk threshold
    review        else if P(denied) + P(fwa) >= review threshold
    auto_approve  otherwise
"""
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

import config as cfg
from model_utils import FEATURES
from reasons import reasons_for_claims

FEATURE_SQL = (Path(__file__).parent / "features.sql").read_text()
THRESHOLDS_PATH = cfg.OUTPUT_DIR / "thresholds.json"
MODEL_PATH = cfg.OUTPUT_DIR / "final_model.joblib"


def assign_queue(proba, review_threshold, high_risk_threshold):
    """`proba` has one column per class: clean, denied, fwa."""
    bad_score = 1 - proba[:, 0]
    fwa_score = proba[:, 2]
    queue = np.where(bad_score >= review_threshold, "review", "auto_approve")
    return np.where(fwa_score >= high_risk_threshold, "high_risk", queue)


def explain(model, features, proba, queue):
    """Reasons for claims in the review and high-risk queues; none for auto-approve."""
    # Explain the FWA score for high-risk claims, otherwise the more likely problem.
    explain_class = np.where((queue == "high_risk") | (proba[:, 2] > proba[:, 1]), 2, 1)
    flagged = queue != "auto_approve"
    reasons = [[] for _ in range(len(features))]
    if flagged.any():
        found = reasons_for_claims(model, features[flagged].reset_index(drop=True), explain_class[flagged])
        for position, claim_reasons in zip(np.where(flagged)[0], found):
            reasons[position] = claim_reasons
    return reasons


class TriageScorer:
    """Scores one new claim: features -> probabilities -> queue -> reasons."""

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.thresholds = json.loads(THRESHOLDS_PATH.read_text())
        # Claim history and reference tables. Read-only: scoring never changes them.
        self.con = duckdb.connect(str(cfg.DB_PATH), read_only=True)

    def build_features(self, claim):
        """Run the same feature SQL used in training, for a single claim."""
        codes = claim["procedure_codes"]
        claim_row = pd.DataFrame([{
            "claim_id": claim["claim_id"],
            "member_id": claim["member_id"],
            "provider_id": claim["provider_id"],
            "service_date": claim["service_date"],
            "received_date": claim["received_date"],
            "claim_days": claim["claim_days"],
            "billed_amount": claim["billed_amount"],
            "n_lines": len(codes),
            "primary_hcpcs": codes[0],
            "proc_set": "+".join(sorted(set(codes))),
            "principal_dx": claim["diagnosis_codes"][0],
            "n_dx": len(claim["diagnosis_codes"]),
        }])
        cursor = self.con.cursor()   # one cursor per request, safe across threads
        cursor.register("claim_row", claim_row)
        cursor.execute("""
            CREATE OR REPLACE TEMP VIEW target AS
            SELECT * REPLACE (service_date::DATE AS service_date,
                              received_date::DATE AS received_date)
            FROM claim_row
        """)
        return cursor.sql(FEATURE_SQL).df()

    def score(self, claim):
        features = self.build_features(claim)
        proba = self.model.predict_proba(features[FEATURES])
        queue = assign_queue(proba, self.thresholds["review"], self.thresholds["high_risk"])
        reasons = explain(self.model, features, proba, queue)
        return {
            "claim_id": claim["claim_id"],
            "score": round(float(1 - proba[0, 0]), 4),      # probability the claim is not clean
            "probability_denied": round(float(proba[0, 1]), 4),
            "probability_fwa": round(float(proba[0, 2]), 4),
            "queue": str(queue[0]),
            "reasons": reasons[0],
        }
