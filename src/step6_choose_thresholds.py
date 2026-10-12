"""Step 6: choose the two thresholds from stated costs, then route the test set.

Cost of a routing decision (all assumptions are in config.py):
    auto_approve : a bad claim is paid, we lose its billed amount
    review       : costs REVIEW_COST; the reviewer stops most bad claims
    high_risk    : costs SENIOR_REVIEW_COST; the senior reviewer stops almost all

Thresholds are chosen on the VALIDATION set (lowest total cost that fits
reviewer capacity) and then reported on the TEST set.

Limits of this cost model: the whole billed amount is counted as the loss (for
overbilling the real loss is only the excess), and capacity is checked over the
whole period, not day by day. Treat the dollar figures as illustrative.
"""
import json

import joblib
import numpy as np
import pandas as pd

import config as cfg
from model_utils import FEATURES, load_split
from triage import MODEL_PATH, THRESHOLDS_PATH, assign_queue, explain


def total_cost(data, queue):
    """Dollars spent on reviews plus dollars lost on bad claims that got paid."""
    amount = data["billed_amount"].to_numpy()
    label = data["label"].to_numpy()

    review_cost = (queue == "review").sum() * cfg.REVIEW_COST \
        + (queue == "high_risk").sum() * cfg.SENIOR_REVIEW_COST

    # Chance each claim's problem is missed: 1 if auto-approved, lower if reviewed.
    miss = np.ones(len(data))
    for queue_name, rates in cfg.CATCH_RATE.items():
        miss[(queue == queue_name) & (label == 1)] = 1 - rates["denied"]
        miss[(queue == queue_name) & (label == 2)] = 1 - rates["fwa"]
    loss = (amount * miss)[label > 0].sum()
    return review_cost + loss


def choose_thresholds(data, proba):
    """Try a grid of threshold pairs; keep the cheapest one that fits capacity."""
    # Fallback if no threshold pair fits capacity: auto-approve everything.
    best = {"review": 1.01, "high_risk": 1.01,
            "cost": total_cost(data, np.full(len(data), "auto_approve"))}
    for high_risk in np.arange(0.05, 0.96, 0.05):
        for review in np.arange(0.01, 0.51, 0.01):
            queue = assign_queue(proba, review, high_risk)
            if (queue == "review").mean() > cfg.REVIEW_CAPACITY:
                continue
            if (queue == "high_risk").mean() > cfg.HIGH_RISK_CAPACITY:
                continue
            cost = total_cost(data, queue)
            if cost < best["cost"]:
                best = {"review": round(float(review), 2),
                        "high_risk": round(float(high_risk), 2), "cost": cost}
    return best


def reliability_table(data, proba):
    """Do predicted probabilities match what actually happens? One row per score band."""
    bad_score = 1 - proba[:, 0]
    band = pd.cut(bad_score, [0, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0], include_lowest=True)
    table = pd.DataFrame({"band": band, "predicted": bad_score, "actual": data["label"].to_numpy() > 0})
    return table.groupby("band", observed=True).agg(
        claims=("actual", "size"), predicted_bad_rate=("predicted", "mean"),
        actual_bad_rate=("actual", "mean")).round(3)


def queue_report(data, queue):
    """Size and quality of each queue."""
    report = pd.DataFrame({"queue": queue, "label": data["label"].map(cfg.LABEL_NAMES).to_numpy()})
    table = pd.crosstab(report["queue"], report["label"])
    table["claims"] = table.sum(axis=1)
    table["share_of_claims"] = (table["claims"] / len(data)).round(4)
    table["bad_rate"] = (1 - table["clean"] / table["claims"]).round(3)
    return table.reindex(["auto_approve", "review", "high_risk"])


def main():
    model = joblib.load(MODEL_PATH)
    valid, test = load_split("valid"), load_split("test")
    valid_proba = model.predict_proba(valid[FEATURES])
    test_proba = model.predict_proba(test[FEATURES])

    best = choose_thresholds(valid, valid_proba)
    thresholds = {"review": best["review"], "high_risk": best["high_risk"]}
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))
    print("thresholds chosen on validation:", thresholds)

    calibration = reliability_table(valid, valid_proba)
    calibration.to_csv(cfg.OUTPUT_DIR / "calibration_validation.csv")
    print("\n=== validation: predicted vs actual bad rate ===")
    print(calibration.to_string())

    # Report on the test set.
    queue = assign_queue(test_proba, **{k + "_threshold": v for k, v in thresholds.items()})
    # Workload arrives by received date; spread it over the calendar days of the test period.
    days = (test["received_date"].max() - test["received_date"].min()).days + 1
    strategies = {
        "pay everything (no review)": np.full(len(test), "auto_approve"),
        "review everything (ignores capacity)": np.full(len(test), "review"),
        "model routing": queue,
    }
    costs = pd.DataFrame([
        {"strategy": name,
         "total_cost": round(total_cost(test, q)),
         "reviews_per_day": round((q != "auto_approve").sum() / days, 1)}
        for name, q in strategies.items()
    ])
    costs["saving_vs_no_review"] = costs["total_cost"].iloc[0] - costs["total_cost"]
    costs.to_csv(cfg.OUTPUT_DIR / "cost_comparison_test.csv", index=False)

    table = queue_report(test, queue)
    table.to_csv(cfg.OUTPUT_DIR / "queue_report_test.csv")

    pd.set_option("display.width", 200)
    print(f"\n=== test set: illustrative cost over {days} calendar days, {len(test):,} claims ===")
    print(costs.to_string(index=False))
    print("\n=== test set: queues ===")
    print(table.to_string())

    # Save a readable sample of routed claims with their reasons.
    reasons = explain(model, test, test_proba, queue)
    routed = test[["claim_id", "service_date", "primary_hcpcs", "principal_dx",
                   "billed_amount", "label", "injected_pattern"]].copy()
    routed["score"] = (1 - test_proba[:, 0]).round(4)
    routed["queue"] = queue
    routed["reasons"] = [" | ".join(r) for r in reasons]
    flagged = routed[routed["queue"] != "auto_approve"].sort_values("score", ascending=False)
    flagged.to_csv(cfg.OUTPUT_DIR / "flagged_claims_test.csv", index=False)

    print("\n=== example reasons ===")
    for _, row in flagged.groupby("injected_pattern").head(1).iterrows():
        print(f"[{row['queue']}] score {row['score']} (true pattern: {row['injected_pattern']})")
        for reason in row["reasons"].split(" | "):
            print("   -", reason)
    fallback_share = flagged["reasons"].str.startswith("No single strong reason").mean()
    print(f"\nflagged claims with no specific reason: {fallback_share:.1%}")


if __name__ == "__main__":
    main()
