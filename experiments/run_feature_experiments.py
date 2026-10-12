"""Improvement experiments: add feature blocks one at a time and compare.

Uses TRAIN and VALIDATION only. The test set is never loaded here.
Run from the project folder:  python experiments/run_feature_experiments.py
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config as cfg                      # noqa: E402
from evaluate import EVAL_REVIEW_SHARE, evaluate, top_share   # noqa: E402
from model_utils import CATEGORICAL, NUMERIC as FINAL_NUMERIC  # noqa: E402

# The features the first version of the model had, before these experiments.
ADOPTED = ["log_amount_vs_member_history", "log_amount_vs_provider_history",
           "member_claims_365d", "member_same_proc_365d", "member_same_dx_share",
           "member_proc_dx_match_share"]
NUMERIC = [f for f in FINAL_NUMERIC if f not in ADOPTED]

BLOCKS = {
    "B dx compatibility": ["dx_share_smoothed", "dx_log_lift", "dx_pair_count_log", "proc_count_log",
                           "dx_count_log", "dx_share_in_proc_group", "dx_share_max_over_procs",
                           "dx_share_mean_over_procs"],
    "B2 dx vs sex and age": ["dx_sex_log_lift", "dx_age_log_lift"],
    "C member dx history": ["m_claims_365d", "m_same_dx3_365d", "m_same_chapter_365d",
                            "m_same_dx3_share", "m_same_proc_365d", "m_proc_dx_match_share"],
    "D amount spread and peers": ["amount_robust_deviation", "peer_log_iqr", "peer_count_log",
                                  "log_amount_vs_line_peers"],
    "D2 amount vs own history": ["log_amount_vs_member_history", "log_amount_vs_provider_history",
                                 "p_same_proc_365d"],
    "E provider volume": ["p_day_other_members", "p_day_same_proc", "p_claims_7d"],
}
LGBM = dict(n_estimators=150, learning_rate=0.05, num_leaves=8, min_child_samples=200,
            random_state=cfg.SEED, verbose=-1)


def load(split):
    con = duckdb.connect(str(cfg.DB_PATH), read_only=True)
    data = con.sql(f"""
        SELECT m.*, e.* EXCLUDE (claim_id, log_amount_vs_member_history, log_amount_vs_provider_history)
        FROM model_data m JOIN experiment_features e USING (claim_id)
        WHERE split = '{split}' ORDER BY claim_id
    """).df()
    con.close()
    return data


def fit_and_score(train, valid, numeric, classifier):
    prepare = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    model = Pipeline([("prepare", prepare), ("classifier", classifier)])
    model.fit(train[numeric + CATEGORICAL], train["label"])
    metrics, caught = evaluate(valid, model.predict_proba(valid[numeric + CATEGORICAL]))
    keep = ["pr_auc_bad", "pr_auc_fwa", "precision_bad_at_capacity", "recall_bad_at_capacity"]
    return {**{k: metrics[k] for k in keep}, **caught.drop("none").to_dict()}


def new_diagnosis_false_alarms(train, valid):
    """The failure mode of the diagnosis history features: a legitimate new diagnosis.

    Among NON-INJECTED validation claims, how often is a claim selected into the top
    6% when its diagnosis is new for the member, compared with when it is not?
    """
    prepare = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), FINAL_NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    model = Pipeline([("prepare", prepare), ("classifier", LGBMClassifier(**LGBM))])
    model.fit(train[FINAL_NUMERIC + CATEGORICAL], train["label"])
    proba = model.predict_proba(valid[FINAL_NUMERIC + CATEGORICAL])
    selected = top_share(1 - proba[:, 0], EVAL_REVIEW_SHARE)

    has_history = (valid["injected_pattern"] == "none") & (valid["member_claims_365d"] >= 3)
    is_new = valid["member_same_dx_share"] == 0
    table = pd.DataFrame({
        "group": ["non-injected, 3+ prior claims, diagnosis NEW for member",
                  "non-injected, 3+ prior claims, diagnosis seen before"],
        "claims": [int((has_history & is_new).sum()), int((has_history & ~is_new).sum())],
        "selected_in_top_6pct": [round(float(selected[has_history & is_new].mean()), 3),
                                 round(float(selected[has_history & ~is_new].mean()), 3)],
    })
    table.to_csv(cfg.OUTPUT_DIR / "new_diagnosis_check_validation.csv", index=False)
    print(table.to_string(index=False), flush=True)


def main():
    # Build the candidate feature table.
    con = duckdb.connect(str(cfg.DB_PATH))
    sql = (ROOT / "experiments" / "candidate_features.sql").read_text()
    con.execute(sql.replace("{train_end}", cfg.TRAIN_END))
    con.close()

    train, valid = load("train"), load("valid")
    rows = []

    def run(name, numeric, classifier=None):
        result = fit_and_score(train, valid, numeric, classifier or LGBMClassifier(**LGBM))
        rows.append({"experiment": name, "n_features": len(numeric) + len(CATEGORICAL), **result})
        print(f"done: {name}", flush=True)

    run("0 current features", NUMERIC)

    # A: strict receipt timing for provider-day volume (swap, not add).
    strict = [f for f in NUMERIC if f not in ("provider_claims_same_day", "provider_day_ratio")] \
        + ["provider_claims_same_day_strict", "provider_day_ratio_strict"]
    run("A strict provider-day timing", strict)

    # B to E: add blocks cumulatively on top of the current features.
    numeric = list(NUMERIC)
    for name, block in BLOCKS.items():
        numeric = numeric + block
        run(f"+ {name}", numeric)
    all_features = list(numeric)

    # Each block alone, to see which ones matter.
    for name, block in BLOCKS.items():
        run(f"only {name}", NUMERIC + block)

    # Variants on the full feature set.
    run("all, leave-one-out dx lift",
        [f if f != "dx_log_lift" else "dx_log_lift_loo" for f in all_features])
    run("all + receipt lag (generator artifact?)", all_features + ["receipt_lag_days"])
    run("all, 15 leaves", all_features, LGBMClassifier(**{**LGBM, "num_leaves": 15}))
    run("all, 15 leaves, min_child 100", all_features,
        LGBMClassifier(**{**LGBM, "num_leaves": 15, "min_child_samples": 100}))
    run("all, 300 trees", all_features, LGBMClassifier(**{**LGBM, "n_estimators": 300}))
    run("all, logistic regression", all_features,
        LogisticRegression(class_weight="balanced", max_iter=3000))

    # What was adopted into the main pipeline, and the artifact check.
    run("ADOPTED: member dx history + amount vs own history", FINAL_NUMERIC)
    run("adopted + strict provider-day features (generator artifact)",
        FINAL_NUMERIC + ["p_day_other_members", "p_claims_7d"])
    run("adopted, 15 leaves", FINAL_NUMERIC, LGBMClassifier(**{**LGBM, "num_leaves": 15}))

    # Controlled ablation: remove exactly the four member diagnosis history features
    # from the final set and retrain. Everything else is identical.
    dx_history = ["member_claims_365d", "member_same_dx_share", "member_same_proc_365d",
                  "member_proc_dx_match_share"]
    run("ABLATION: adopted minus the 4 member dx history features",
        [f for f in FINAL_NUMERIC if f not in dx_history])
    new_diagnosis_false_alarms(train, valid)

    results = pd.DataFrame(rows)
    results.to_csv(cfg.OUTPUT_DIR / "feature_experiments_validation.csv", index=False)
    pd.set_option("display.width", 250)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
