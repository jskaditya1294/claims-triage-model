"""Step 2: create the labels.

The CMS data has no denial or fraud labels. We INJECT five realistic bad-claim
patterns into a small share of claims, then add noise. The model never sees
which claims were injected; it has to find them from the data.

    pattern              what we change                               label
    dx_mismatch          diagnosis swapped, different first letter    denied
    coverage_ended       service date falls in a coverage lapse       denied
    duplicate            claim re-submitted 0-2 days later            fwa
    overbilling          billed amount multiplied by 2x-6x            fwa
    provider_day_spike   provider bills 4-12 extra claims in one day  fwa
"""
import duckdb
import numpy as np
import pandas as pd

import config as cfg


def swap_diagnosis(claims, idx, rng):
    """Replace the diagnosis with a real code that starts with a different letter."""
    all_dx = claims["principal_dx"].to_numpy()
    old = claims.loc[idx, "principal_dx"].to_numpy()
    new = rng.choice(all_dx, size=len(idx))
    same_chapter = np.array([n[0] == o[0] for n, o in zip(new, old)])
    while same_chapter.any():   # redraw until the first letter differs
        new[same_chapter] = rng.choice(all_dx, size=same_chapter.sum())
        same_chapter = np.array([n[0] == o[0] for n, o in zip(new, old)])
    claims.loc[idx, "principal_dx"] = new


def make_copies(source_rows, pattern, first_new_id):
    """Copy claims and give the copies new claim ids."""
    copies = source_rows.copy()
    copies["claim_id"] = np.arange(first_new_id, first_new_id + len(copies))
    copies["injected_pattern"] = pattern
    return copies


def main():
    rng = np.random.default_rng(cfg.SEED)
    con = duckdb.connect(str(cfg.DB_PATH))
    claims = con.sql("SELECT * FROM claims_raw ORDER BY claim_id").df()
    members = con.sql("SELECT * FROM members_raw ORDER BY member_id").df()
    claims["injected_pattern"] = "none"
    n_original = len(claims)

    def pick_untouched(n):
        """Sample claims that have not been modified yet."""
        pool = claims.index[claims["injected_pattern"] == "none"].to_numpy()
        return rng.choice(pool, size=n, replace=False)

    # 1. Diagnosis does not support the procedure.
    idx = pick_untouched(int(cfg.DX_MISMATCH_RATE * n_original))
    swap_diagnosis(claims, idx, rng)
    claims.loc[idx, "injected_pattern"] = "dx_mismatch"

    # 2. Overbilling: amount far above peers for the same procedure.
    idx = pick_untouched(int(cfg.OVERBILLING_RATE * n_original))
    claims.loc[idx, "billed_amount"] = (
        claims.loc[idx, "billed_amount"] * rng.uniform(2, 6, size=len(idx))
    ).round(2)
    claims.loc[idx, "injected_pattern"] = "overbilling"

    # 3. Duplicates: the same claim submitted again, received 3-30 days later.
    #    The original stays clean; only the copy is the bad claim.
    idx = pick_untouched(int(cfg.DUPLICATE_RATE * n_original))
    duplicates = make_copies(claims.loc[idx], "duplicate", first_new_id=9_000_000_000_000)
    shift = rng.choice([0, 1, 2], size=len(idx), p=[0.7, 0.15, 0.15])
    duplicates["service_date"] += pd.to_timedelta(shift, unit="D")
    duplicates["received_date"] += pd.to_timedelta(shift + rng.integers(3, 31, size=len(idx)), unit="D")
    tweak = rng.random(len(idx)) < 0.3   # 30% of copies change the amount slightly
    duplicates.loc[tweak, "billed_amount"] = (
        duplicates.loc[tweak, "billed_amount"] * rng.uniform(0.90, 1.10, size=tweak.sum())
    ).round(2)

    # 4. Impossible volume: a provider bills many extra claims on one day.
    #    We copy the provider's own claims from other days onto the spike day.
    untouched = claims[claims["injected_pattern"] == "none"]
    by_provider = untouched.groupby("provider_id").indices
    big_providers = [p for p, rows in by_provider.items() if len(rows) >= 20]
    spikes = []
    for provider in rng.choice(big_providers, size=cfg.SPIKE_PROVIDER_DAYS, replace=False):
        rows = untouched.iloc[by_provider[provider]]
        spike_day = rows["service_date"].sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        extra = rows.sample(int(rng.integers(4, 13)), random_state=int(rng.integers(1e9))).copy()
        extra["service_date"] = spike_day
        extra["received_date"] = spike_day + pd.to_timedelta(rng.integers(1, 8, size=len(extra)), unit="D")
        spikes.append(extra)
    spikes = make_copies(pd.concat(spikes), "provider_day_spike", first_new_id=9_100_000_000_000)

    claims = pd.concat([claims, duplicates, spikes], ignore_index=True)

    # 5. Service after coverage ended. Coverage is a member attribute, so we pick
    #    members and end their coverage just before a random claim. Coverage
    #    restarts 1-6 claims later (a lapse), so the pattern is spread over time
    #    instead of piling up at the end of the data.
    chosen = rng.choice(members["member_id"].to_numpy(),
                        size=int(cfg.COVERAGE_ENDED_MEMBER_RATE * len(members)), replace=False)
    dates_by_member = claims.groupby("member_id")["service_date"].apply(lambda s: np.sort(s.unique()))
    coverage_end, coverage_restart = {}, {}
    for member in chosen:
        dates = dates_by_member[member]
        start = int(rng.integers(0, len(dates)))    # first service date without coverage
        k = int(rng.integers(1, 7))                 # number of service dates without coverage
        coverage_end[member] = dates[start] - np.timedelta64(1, "D")
        if start + k < len(dates):
            coverage_restart[member] = dates[start + k]
    members["coverage_end"] = pd.to_datetime(members["member_id"].map(coverage_end))
    members["coverage_restart"] = pd.to_datetime(members["member_id"].map(coverage_restart))

    claim_end = pd.to_datetime(claims["member_id"].map(coverage_end))
    claim_restart = pd.to_datetime(claims["member_id"].map(coverage_restart))
    not_covered = (claims["service_date"] > claim_end) & ~(claims["service_date"] >= claim_restart)
    claims.loc[not_covered & (claims["injected_pattern"] == "none"), "injected_pattern"] = "coverage_ended"

    # Labels: start from the pattern, then add noise in both directions.
    claims["label"] = claims["injected_pattern"].map(cfg.PATTERN_TO_LABEL)
    claims["label_noise"] = "none"
    u = rng.random(len(claims))
    injected = claims["injected_pattern"] != "none"

    missed = injected & (u < cfg.MISSED_RATE)                 # bad claim paid anyway
    claims.loc[missed, ["label", "label_noise"]] = [0, "missed"]

    deny = ~injected & (u < cfg.UNEXPLAINED_DENY_RATE)        # denied, reason not in data
    claims.loc[deny, ["label", "label_noise"]] = [1, "unexplained"]

    fwa = ~injected & (u >= cfg.UNEXPLAINED_DENY_RATE) & (
        u < cfg.UNEXPLAINED_DENY_RATE + cfg.UNEXPLAINED_FWA_RATE)
    claims.loc[fwa, ["label", "label_noise"]] = [2, "unexplained"]

    # Save back to DuckDB.
    con.register("claims_df", claims)
    con.register("members_df", members)
    con.execute("""
        CREATE OR REPLACE TABLE claims AS
        SELECT * REPLACE (service_date::DATE AS service_date, received_date::DATE AS received_date)
        FROM claims_df
    """)
    con.execute("""
        CREATE OR REPLACE TABLE members AS
        SELECT * REPLACE (birth_date::DATE AS birth_date, coverage_end::DATE AS coverage_end,
                          coverage_restart::DATE AS coverage_restart)
        FROM members_df
    """)
    con.close()

    # Report the class balance.
    cfg.OUTPUT_DIR.mkdir(exist_ok=True)
    claims["label_name"] = claims["label"].map(cfg.LABEL_NAMES)
    summary = pd.crosstab(claims["injected_pattern"], claims["label_name"], margins=True)
    summary.to_csv(cfg.OUTPUT_DIR / "label_summary.csv")
    print(summary.to_string())
    print("\nclass balance (%):")
    print((claims["label_name"].value_counts(normalize=True) * 100).round(2).to_string())


if __name__ == "__main__":
    main()
