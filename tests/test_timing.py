"""Timing tests on a tiny hand-made dataset. No real data needed.

They pin down the scoring contract:
  - member history only counts claims received on an EARLIER date
  - provider volume for the day also counts claims received the SAME date
    (end-of-day batch scoring), but never a later date
  - reference tables ignore claims received on or after the training cutoff
"""
from pathlib import Path

import duckdb
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
FEATURE_SQL = (SRC / "features.sql").read_text()
REFERENCE_SQL = (SRC / "reference_tables.sql").read_text()
TRAIN_END = "2021-01-01"

# claim_id, received_date, billed_amount, procedure. All claims: member 1, provider P.
CLAIMS = [
    # Two old claims for procedure X, serviced before the cutoff.
    (10, "2020-12-31", 100.0, "X"),    # received before the cutoff: available
    (11, "2021-01-02", 900.0, "X"),    # received after the cutoff: NOT available
    # Four claims for the same service date, received on different days.
    (1, "2021-03-12", 100.0, "99241"),   # the claim being scored
    (2, "2021-03-11", 100.0, "99241"),   # received yesterday
    (3, "2021-03-12", 100.0, "99241"),   # received today
    (4, "2021-03-13", 100.0, "99241"),   # received tomorrow
]


@pytest.fixture
def con():
    con = duckdb.connect()
    con.execute("""
        CREATE TABLE claims (claim_id BIGINT, member_id BIGINT, provider_id VARCHAR,
            service_date DATE, received_date DATE, claim_days INT, billed_amount DOUBLE,
            n_lines INT, primary_hcpcs VARCHAR, proc_set VARCHAR, principal_dx VARCHAR, n_dx INT)
    """)
    for claim_id, received, amount, procedure in CLAIMS:
        service = "2020-12-30" if procedure == "X" else "2021-03-10"
        con.execute("INSERT INTO claims VALUES (?, 1, 'P', ?, ?, 0, ?, 1, ?, ?, 'E785', 1)",
                    [claim_id, service, received, amount, procedure, procedure])
    con.execute("""
        CREATE TABLE members AS
        SELECT 1::BIGINT AS member_id, DATE '1950-01-01' AS birth_date, '1' AS sex,
               NULL::DATE AS coverage_end, NULL::DATE AS coverage_restart
    """)
    con.execute(REFERENCE_SQL.replace("{train_end}", TRAIN_END))
    con.execute("CREATE VIEW target AS SELECT * FROM claims WHERE claim_id = 1")
    return con


def test_member_history_only_counts_claims_received_earlier(con):
    features = con.sql(FEATURE_SQL).df().iloc[0]
    # Only "yesterday" counts: "today" and "tomorrow" had not arrived before this claim.
    assert features["member_claims_30d"] == 1
    assert features["same_day_repeats"] == 1
    assert features["member_same_proc_365d"] == 1


def test_provider_day_volume_follows_the_end_of_day_batch_contract(con):
    features = con.sql(FEATURE_SQL).df().iloc[0]
    # The claim itself + yesterday + today. Never tomorrow.
    assert features["provider_claims_same_day"] == 3


def test_reference_tables_ignore_claims_received_after_the_cutoff(con):
    count, median = con.sql(
        "SELECT n, median_amount FROM ref_hcpcs_amount WHERE primary_hcpcs = 'X'").fetchone()
    # Claim 11 was serviced before the cutoff but received after it, so it is excluded.
    assert count == 1
    assert median == 100.0


def test_changing_a_future_claim_does_not_change_the_features(con):
    before = con.sql(FEATURE_SQL).df()
    con.execute("UPDATE claims SET billed_amount = 99999, principal_dx = 'Z000' WHERE claim_id = 4")
    after = con.sql(FEATURE_SQL).df()
    assert before.equals(after)


def test_changing_a_late_claim_does_not_change_the_reference_tables(con):
    tables = ["ref_proc_amount", "ref_hcpcs_amount", "ref_dx_proc", "ref_provider"]
    before = [con.sql(f"SELECT * FROM {t} ORDER BY ALL").df() for t in tables]
    con.execute("UPDATE claims SET billed_amount = 99999, principal_dx = 'Z000' WHERE claim_id = 11")
    con.execute(REFERENCE_SQL.replace("{train_end}", TRAIN_END))
    after = [con.sql(f"SELECT * FROM {t} ORDER BY ALL").df() for t in tables]
    assert all(b.equals(a) for b, a in zip(before, after))
