"""Turn model explanations into sentences a claims reviewer can act on.

How it works:
  1. LightGBM reports how much each feature pushed this claim's score up or
     down (SHAP contributions).
  2. Related features are grouped into one reason (for example the two
     duplicate features).
  3. A reason is shown only if it pushed the score UP and its plain-language
     check is true for this claim. This stops us from printing a sentence like
     "provider volume is unusual" when the volume is actually normal.
"""
import numpy as np

from model_utils import CATEGORICAL


def amount_comparisons(claim):
    """Benchmarks that the billed amount exceeds by 1.5 times or more."""
    benchmarks = [
        ("the typical ${:,.0f} for this procedure", claim["peer_median_amount"]),
        ("this provider's usual ${:,.0f}", claim["provider_usual_amount"]),
        ("this member's usual ${:,.0f}", claim["member_usual_amount"]),
    ]
    return [
        f"{claim['billed_amount'] / usual:.1f} times " + wording.format(usual)
        for wording, usual in benchmarks
        if usual == usual and claim["billed_amount"] >= 1.5 * usual   # usual == usual: not missing
    ]


# Each reason: the features behind it, a check on the claim, and the sentence.
REASONS = [
    {
        "features": ["same_day_repeats", "near_repeats_same_amount"],
        "applies": lambda c: c["same_day_repeats"] >= 1 or c["near_repeats_same_amount"] >= 1,
        "text": lambda c: (
            "Possible duplicate: an earlier claim for the same member, provider and "
            f"procedure ({c['primary_hcpcs']}) on the same or a nearby date was already received."
        ),
    },
    {
        "features": ["log_amount_vs_peers", "log_billed_amount",
                     "log_amount_vs_provider_history", "log_amount_vs_member_history"],
        "applies": lambda c: bool(amount_comparisons(c)),
        "text": lambda c: f"Billed amount ${c['billed_amount']:,.0f} is "
                          + " and ".join(amount_comparisons(c)) + ".",
    },
    {
        "features": ["dx_proc_share", "dx_proc_never_seen", "dx_chapter"],
        "applies": lambda c: c["dx_proc_share"] < 0.05,
        "text": lambda c: (
            f"Diagnosis {c['principal_dx']} rarely supports procedure {c['primary_hcpcs']}: "
            f"it appears on {c['dx_proc_share']:.1%} of past claims for this procedure."
        ),
    },
    {
        "features": ["member_same_dx_share", "member_proc_dx_match_share"],
        "applies": lambda c: c["member_claims_365d"] >= 3 and c["member_same_dx_share"] == 0,
        "text": lambda c: (
            f"Diagnosis {c['principal_dx']} is new for this member: none of their "
            f"{c['member_claims_365d']:.0f} claims in the past year had a diagnosis in the same category."
        ),
    },
    {
        "features": ["provider_claims_same_day", "provider_day_ratio"],
        "applies": lambda c: c["provider_claims_same_day"] >= 3 and c["provider_day_ratio"] >= 2,
        "text": lambda c: (
            f"Provider billed {c['provider_claims_same_day']:.0f} claims for this service date; "
            f"its usual is {c['provider_usual_daily_claims']:.0f} per day."
        ),
    },
    {
        "features": ["days_after_coverage_end"],
        "applies": lambda c: c["days_after_coverage_end"] > 0,
        "text": lambda c: (
            f"Service date is {c['days_after_coverage_end']:.0f} days after the member's coverage ended."
        ),
    },
    {
        "features": ["same_proc_prior_7d"],
        "applies": lambda c: c["same_proc_prior_7d"] >= 1,
        "text": lambda c: (
            f"Same procedure billed {c['same_proc_prior_7d']:.0f} time(s) for this member "
            "by this provider in the previous 7 days."
        ),
    },
    {
        "features": ["member_claims_30d"],
        "applies": lambda c: c["member_claims_30d"] >= 15,
        "text": lambda c: f"Member has {c['member_claims_30d']:.0f} other claims in the past 30 days.",
    },
    {
        "features": ["provider_is_new", "provider_claims_90d"],
        "applies": lambda c: c["provider_is_new"] == 1,
        "text": lambda c: "Provider has no billing history in our reference period.",
    },
]
FALLBACK = "No single strong reason: the mix of claim details resembles past problem claims."


def feature_contributions(model, features, class_index):
    """Contribution of each ORIGINAL feature to the score of one class, per claim.

    Returns a dict: feature name -> array with one value per claim.
    """
    prepared = model.named_steps["prepare"].transform(features)
    names = list(model.named_steps["prepare"].get_feature_names_out())
    booster = model.named_steps["classifier"].booster_
    raw = booster.predict(prepared, pred_contrib=True)

    # LightGBM returns one block per class: [features..., base value].
    block = len(names) + 1
    values = raw[:, class_index * block: class_index * block + len(names)]

    contributions = {}
    for position, name in enumerate(names):
        # One-hot columns such as "dx_chapter_N" are added back to "dx_chapter".
        original = next((c for c in CATEGORICAL if name.startswith(c + "_")), name)
        contributions[original] = contributions.get(original, 0) + values[:, position]
    return contributions


def top_reasons(claim, contributions, max_reasons=3):
    """Plain-language reasons for one claim, strongest first.

    `claim` is a dict of the claim's feature values; `contributions` is a dict
    feature name -> contribution for this claim.
    """
    scored = []
    for reason in REASONS:
        push = sum(contributions.get(f, 0.0) for f in reason["features"])
        if push > 0 and reason["applies"](claim):
            scored.append((push, reason["text"](claim)))
    scored.sort(reverse=True)
    return [text for _, text in scored[:max_reasons]] or [FALLBACK]


def reasons_for_claims(model, data, class_index):
    """Reasons for many claims. `class_index` is an array: 1 = denied, 2 = fwa."""
    class_index = np.asarray(class_index)
    all_reasons = [None] * len(data)
    rows = data.to_dict("records")
    for cls in (1, 2):
        positions = np.where(class_index == cls)[0]
        if len(positions) == 0:
            continue
        contributions = feature_contributions(model, data.iloc[positions], cls)
        for i, position in enumerate(positions):
            one_claim = {name: values[i] for name, values in contributions.items()}
            all_reasons[position] = top_reasons(rows[position], one_claim)
    return all_reasons
