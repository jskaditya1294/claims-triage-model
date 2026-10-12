"""Leakage audit: explicit checks on timing and label use.

These checks cover timing correctness. They do not show that a feature would
work on real claims; see the note for which features the synthetic data flatters.

Run from the project folder:  python experiments/leakage_audit.py
"""
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config as cfg                                                # noqa: E402
from model_utils import FEATURES, load_split, make_preprocessor     # noqa: E402

FEATURE_SQL = (ROOT / "src" / "features.sql").read_text()
REFERENCE_SQL = (ROOT / "src" / "reference_tables.sql").read_text()
REFERENCE_TABLES = ["ref_proc_amount", "ref_hcpcs_amount", "ref_dx_proc", "ref_provider"]
LGBM = dict(n_estimators=150, learning_rate=0.05, num_leaves=8, min_child_samples=200,
            random_state=cfg.SEED, verbose=-1)


def sql_without_comments(sql):
    return re.sub(r"--.*", "", sql)


def check_1_no_label_columns_in_feature_code():
    """The feature SQL and the feature list never mention the label columns."""
    code = sql_without_comments(FEATURE_SQL + REFERENCE_SQL).lower()
    forbidden = ["label", "injected_pattern", "label_noise"]
    found = [word for word in forbidden if word in code]
    found += [f for f in FEATURES if f in ("label", "injected_pattern", "claim_id", "split")]
    return not found, f"forbidden names found: {found or 'none'}"


def check_2_features_do_not_change_when_the_future_is_deleted():
    """Point-in-time test.

    Recompute features for claims received on a given day using ONLY claims
    received up to and including that day (the end-of-day batch contract).
    Reference tables and member data are kept as stored; check 3 and the
    fixture tests in tests/test_timing.py cover those. If any feature used later information, it would
    change. We compare against the stored features built with all claims present.
    """
    con = duckdb.connect()
    con.execute(f"ATTACH '{cfg.DB_PATH.as_posix()}' AS db (READ_ONLY)")
    for table in REFERENCE_TABLES + ["members"]:
        con.execute(f"CREATE VIEW {table} AS SELECT * FROM db.{table}")

    days = con.sql("""
        SELECT received_date FROM db.claims
        WHERE received_date >= DATE '2021-01-01'
        GROUP BY 1 ORDER BY COUNT(*) DESC, 1 LIMIT 8
    """).df()["received_date"]

    checked, mismatches = 0, 0
    for day in days:
        day = str(day)[:10]
        con.execute(f"CREATE OR REPLACE VIEW claims AS SELECT * FROM db.claims WHERE received_date <= DATE '{day}'")
        con.execute(f"CREATE OR REPLACE VIEW target AS SELECT * FROM db.claims WHERE received_date = DATE '{day}'")
        truncated = con.sql(FEATURE_SQL).df().set_index("claim_id").sort_index()
        stored = con.sql(f"""
            SELECT * FROM db.model_data
            WHERE claim_id IN (SELECT claim_id FROM target)
        """).df().set_index("claim_id").sort_index()
        for feature in FEATURES:
            a, b = truncated[feature], stored[feature]
            if a.dtype == object:
                different = (a != b)
            else:
                different = ~np.isclose(a.astype(float), b.astype(float), equal_nan=True)
            mismatches += int(different.sum())
        checked += len(stored)
    con.close()
    return mismatches == 0, f"{checked:,} claims on {len(days)} received dates, {mismatches} feature values changed"


def check_3_reference_tables_only_use_claims_received_before_cutoff():
    """Rebuild the reference tables from claims received before the cutoff and compare."""
    con = duckdb.connect()
    con.execute(f"ATTACH '{cfg.DB_PATH.as_posix()}' AS db (READ_ONLY)")
    con.execute(f"CREATE VIEW claims AS SELECT * FROM db.claims WHERE received_date < DATE '{cfg.TRAIN_END}'")
    con.execute(REFERENCE_SQL.replace("{train_end}", cfg.TRAIN_END))
    differences = 0
    for table in REFERENCE_TABLES:
        differences += con.sql(f"""
            SELECT COUNT(*) FROM (
                (SELECT * FROM {table} EXCEPT SELECT * FROM db.{table})
                UNION ALL
                (SELECT * FROM db.{table} EXCEPT SELECT * FROM {table})
            )
        """).fetchone()[0]
    # No training row may have arrived on or after the cutoff.
    late = con.sql(f"""
        SELECT COUNT(*) FROM db.model_data
        WHERE split = 'train' AND received_date >= DATE '{cfg.TRAIN_END}'
    """).fetchone()[0]
    con.close()
    return differences == 0 and late == 0, (f"{differences} differing rows; "
                                            f"{late} training claims received on or after the cutoff")


def check_4_splits_do_not_overlap():
    """No claim is in two splits, and the splits are in order of received date."""
    con = duckdb.connect(str(cfg.DB_PATH), read_only=True)
    duplicates = con.sql("SELECT COUNT(*) - COUNT(DISTINCT claim_id) FROM model_data").fetchone()[0]
    ranges = con.sql("""
        SELECT split, MIN(received_date) AS first, MAX(received_date) AS last
        FROM model_data GROUP BY split ORDER BY first
    """).df()
    con.close()
    ordered = all(ranges["last"].iloc[i] < ranges["first"].iloc[i + 1] for i in range(len(ranges) - 1))
    text = ", ".join(f"{r.split} {str(r.first)[:10]} to {str(r.last)[:10]}" for r in ranges.itertuples())
    return duplicates == 0 and ordered, f"{duplicates} repeated claims; {text}"


def check_5_sanity_shuffled_labels_give_no_skill():
    """Sanity check only: with shuffled labels the pipeline must score at random.

    This catches bugs such as rows and labels being misaligned. It does NOT rule
    out leakage, because shuffling would also break a leaked relationship.
    """
    train, valid = load_split("train"), load_split("valid")
    shuffled = np.random.default_rng(cfg.SEED).permutation(train["label"].to_numpy())
    model = Pipeline([("prepare", make_preprocessor()), ("classifier", LGBMClassifier(**LGBM))])
    model.fit(train[FEATURES], shuffled)
    is_bad = valid["label"] > 0
    pr_auc = average_precision_score(is_bad, 1 - model.predict_proba(valid[FEATURES])[:, 0])
    return pr_auc < 1.5 * is_bad.mean(), (f"validation PR-AUC {pr_auc:.3f} with shuffled labels; "
                                          f"random would be {is_bad.mean():.3f}")


def main():
    checks = [check_1_no_label_columns_in_feature_code,
              check_2_features_do_not_change_when_the_future_is_deleted,
              check_3_reference_tables_only_use_claims_received_before_cutoff,
              check_4_splits_do_not_overlap,
              check_5_sanity_shuffled_labels_give_no_skill]
    lines = []
    for check in checks:
        passed, detail = check()
        lines.append(f"{'PASS' if passed else 'FAIL'}  {check.__name__}\n      {detail}")
        print(lines[-1], flush=True)
    (cfg.OUTPUT_DIR / "leakage_audit.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
