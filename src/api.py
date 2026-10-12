"""Scoring API: one claim in as JSON, score + queue + reasons out.

Run from the src folder:
    uvicorn api:app --reload
Then open http://127.0.0.1:8000/docs to try it.
"""
from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel, Field

from triage import TriageScorer

app = FastAPI(title="Claims triage")
scorer = TriageScorer()


class Claim(BaseModel):
    claim_id: int
    member_id: int
    provider_id: str
    service_date: date
    received_date: date
    billed_amount: float = Field(gt=0)
    procedure_codes: list[str] = Field(min_length=1)    # HCPCS / CPT, main procedure first
    diagnosis_codes: list[str] = Field(min_length=1)    # ICD-10, principal diagnosis first
    claim_days: int = Field(default=0, ge=0)            # last service day minus first
    place_of_service: str = "outpatient"
    claim_type: str = "40"


class TriageResult(BaseModel):
    claim_id: int
    score: float
    probability_denied: float
    probability_fwa: float
    queue: str
    reasons: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "thresholds": scorer.thresholds}


@app.post("/score", response_model=TriageResult)
def score(claim: Claim):
    return scorer.score(claim.model_dump())
