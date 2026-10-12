"""Evaluation shared by every model, so models are compared the same way.

Accuracy is useless here (94% of claims are clean), so we use:
  - PR-AUC: how well the score ranks bad claims above clean ones.
  - Precision and recall at review capacity: if reviewers can only look at the
    top X% of claims, how many of those are bad, and how many bad ones we catch.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Fixed evaluation point for comparing models: the top 6% by bad-claim score and
# the top 2% by FWA score. Routing capacity (config.py) is a separate setting.
EVAL_REVIEW_SHARE = 0.06
EVAL_HIGH_RISK_SHARE = 0.02


def top_share(score, share):
    """True for exactly the `share` of claims with the highest score.

    Ties are broken at random (seeded). A quantile cutoff would let tied scores
    exceed capacity, which flatters decision trees because they give many
    claims the same score.
    """
    k = int(len(score) * share)
    tie_breaker = np.random.default_rng(0).random(len(score))
    order = np.lexsort((tie_breaker, -np.asarray(score)))   # by score, then at random
    selected = np.zeros(len(score), dtype=bool)
    selected[order[:k]] = True
    return selected


def evaluate(data, proba):
    """`proba` has one column per class: clean, denied, fwa."""
    is_bad = (data["label"] > 0).to_numpy()
    is_fwa = (data["label"] == 2).to_numpy()
    bad_score = 1 - proba[:, 0]     # probability the claim is not clean
    fwa_score = proba[:, 2]

    reviewed = top_share(bad_score, EVAL_REVIEW_SHARE)
    high_risk = top_share(fwa_score, EVAL_HIGH_RISK_SHARE)

    metrics = {
        "claims": len(data),
        "bad_rate": is_bad.mean(),
        "fwa_rate": is_fwa.mean(),
        "pr_auc_bad": average_precision_score(is_bad, bad_score),
        "pr_auc_fwa": average_precision_score(is_fwa, fwa_score),
        "precision_bad_at_capacity": is_bad[reviewed].mean(),
        "recall_bad_at_capacity": reviewed[is_bad].mean(),
        "precision_fwa_at_capacity": is_fwa[high_risk].mean(),
        "recall_fwa_at_capacity": high_risk[is_fwa].mean(),
    }
    # Share of each injected pattern selected inside the review capacity. This
    # counts every injected claim, including the ones noise relabelled as clean.
    caught_by_pattern = pd.Series(reviewed).groupby(data["injected_pattern"].to_numpy()).mean()
    return {k: round(float(v), 4) for k, v in metrics.items()}, caught_by_pattern.round(3)
