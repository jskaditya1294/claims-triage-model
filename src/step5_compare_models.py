"""Step 5: compare simple and complex models on the same data and metrics.

Simple models:  logistic regression, shallow decision tree.
Complex models: random forest, LightGBM (gradient boosted trees).

Each model is also tried with and without class weights, because that is our
class-imbalance decision. `brier_bad` measures how honest the probabilities are
(lower is better); it matters because thresholds are chosen from probabilities.
"""
import time

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config as cfg
from evaluate import evaluate
from model_utils import FEATURES, load_split, make_preprocessor

# Small trees on purpose. Larger ones (31 leaves, 300 trees) ranked slightly worse
# on validation and were overconfident: predicted 20% risk where the real rate was 8%.
LGBM_SETTINGS = dict(n_estimators=150, learning_rate=0.05, num_leaves=8,
                     min_child_samples=200, random_state=cfg.SEED, verbose=-1)

MODELS = {
    "logistic (balanced weights)": LogisticRegression(class_weight="balanced", max_iter=2000),
    "logistic (no weights)": LogisticRegression(max_iter=2000),
    "decision tree depth 6 (balanced)": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=cfg.SEED),
    "decision tree depth 6 (no weights)": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, random_state=cfg.SEED),
    "random forest (no weights)": RandomForestClassifier(
        n_estimators=200, max_depth=14, min_samples_leaf=20, n_jobs=-1, random_state=cfg.SEED),
    "lightgbm (balanced weights)": LGBMClassifier(class_weight="balanced", **LGBM_SETTINGS),
    "lightgbm (no weights)": LGBMClassifier(**LGBM_SETTINGS),
}
FINAL_MODEL = "lightgbm (no weights)"


def main():
    train, valid, test = load_split("train"), load_split("valid"), load_split("test")
    rows, patterns = [], {}

    for name, classifier in MODELS.items():
        start = time.time()
        model = Pipeline([("prepare", make_preprocessor()), ("classifier", classifier)])
        model.fit(train[FEATURES], train["label"])

        for split_name, data in [("valid", valid), ("test", test)]:
            proba = model.predict_proba(data[FEATURES])
            metrics, caught = evaluate(data, proba)
            metrics["brier_bad"] = round(brier_score_loss(data["label"] > 0, 1 - proba[:, 0]), 4)
            rows.append({"model": name, "split": split_name, **metrics})
            if split_name == "test":
                patterns[name] = caught
        print(f"{name}: done in {time.time() - start:.0f}s")

        if name == FINAL_MODEL:
            joblib.dump(model, cfg.OUTPUT_DIR / "final_model.joblib")

    columns = ["model", "split", "pr_auc_bad", "pr_auc_fwa", "precision_bad_at_capacity",
               "recall_bad_at_capacity", "precision_fwa_at_capacity", "recall_fwa_at_capacity",
               "brier_bad"]
    comparison = pd.DataFrame(rows)[columns]
    comparison.to_csv(cfg.OUTPUT_DIR / "model_comparison.csv", index=False)
    caught_table = pd.DataFrame(patterns).T
    caught_table.to_csv(cfg.OUTPUT_DIR / "caught_by_pattern_test.csv")

    pd.set_option("display.width", 250)
    for split_name in ["valid", "test"]:
        print(f"\n=== {split_name} ===")
        print(comparison[comparison["split"] == split_name].drop(columns="split").to_string(index=False))
    print("\n=== test: share of each injected pattern selected in the top 6% ===")
    print(caught_table.to_string())


if __name__ == "__main__":
    main()
