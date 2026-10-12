"""All settings in one place, so every assumption is easy to find and change."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "claims.duckdb"
OUTPUT_DIR = ROOT / "outputs"

SEED = 42

# Time-based split on RECEIVED date: train on the past, test on the future.
TRAIN_END = "2021-01-01"   # train: received 2015-2020
VALID_END = "2022-01-01"   # validation: received 2021, test: received 2022-2023

# --- Labelling assumptions (step 2) ---
# Share of claims (or members) that receive each injected pattern.
COVERAGE_ENDED_MEMBER_RATE = 0.15   # members with a coverage lapse spanning 1-6 service dates
DX_MISMATCH_RATE = 0.022            # diagnosis replaced by one with a different first letter
OVERBILLING_RATE = 0.008            # billed amount multiplied by 2x-6x
DUPLICATE_RATE = 0.009              # claim re-submitted 0-2 days later
SPIKE_PROVIDER_DAYS = 300           # provider-days that get 4-12 extra claims

# Label noise, so labels are not a perfect function of the rules.
MISSED_RATE = 0.12                  # injected bad claims that were paid anyway
UNEXPLAINED_DENY_RATE = 0.010       # clean claims denied for reasons not in the data
UNEXPLAINED_FWA_RATE = 0.0015       # clean claims flagged FWA for reasons not in the data

# --- Cost assumptions for choosing thresholds (step 6) ---
REVIEW_COST = 15.0          # dollars for one standard review
SENIOR_REVIEW_COST = 60.0   # dollars for one senior / investigations review
# Loss from paying a bad claim automatically = its billed amount.
# Chance that a reviewer stops a bad claim, by queue and by type of bad claim.
CATCH_RATE = {
    "review":    {"denied": 0.90, "fwa": 0.60},   # FWA is harder for a standard reviewer
    "high_risk": {"denied": 0.95, "fwa": 0.95},
}
# Reviewer capacity, as a share of all claims in the period.
REVIEW_CAPACITY = 0.08
HIGH_RISK_CAPACITY = 0.02

LABEL_NAMES = {0: "clean", 1: "denied", 2: "fwa"}
PATTERN_TO_LABEL = {
    "none": 0,
    "dx_mismatch": 1,
    "coverage_ended": 1,
    "duplicate": 2,
    "overbilling": 2,
    "provider_day_spike": 2,
}
