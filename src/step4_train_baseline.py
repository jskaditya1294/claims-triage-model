"""Step 4: baseline model, a logistic regression with three classes.

Classes: 0 = clean, 1 = denied, 2 = fwa.
Class imbalance: class_weight="balanced" makes a mistake on a rare class cost
more than a mistake on a clean claim. No rows are dropped or invented.
"""
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import config as cfg
from evaluate import evaluate
from model_utils import FEATURES, load_split, make_preprocessor


def main():
    train, valid, test = load_split("train"), load_split("valid"), load_split("test")

    model = Pipeline([
        ("prepare", make_preprocessor()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    model.fit(train[FEATURES], train["label"])

    results = {}
    for name, data in [("valid", valid), ("test", test)]:
        proba = model.predict_proba(data[FEATURES])
        metrics, caught = evaluate(data, proba)
        results[name] = {"metrics": metrics, "caught_by_pattern": caught.to_dict()}
        print(f"\n=== {name} ===")
        print(pd.Series(metrics).to_string())
        print("share of each injected pattern selected inside review capacity:")
        print(caught.to_string())

    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, cfg.OUTPUT_DIR / "baseline_logreg.joblib")
    (cfg.OUTPUT_DIR / "baseline_metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
