-- Feature query: one row of features per claim in `target`.
-- `target` is every claim (batch training) or a single new claim (API), so
-- training and serving use exactly the same feature logic.
--
-- Leakage rules followed here:
--   1. Reference tables (peer amounts, usual diagnosis-procedure pairs, provider
--      norms) only use claims received before the training cutoff.
--   2. History features only use claims RECEIVED before the claim being scored.
--      One exception, on purpose: provider volume for the service day also counts
--      claims received on the same date. The scoring contract is END-OF-DAY BATCH:
--      a claim is scored after the whole day's claims have arrived. This feature
--      would not be valid for scoring each claim the moment it arrives.
--   3. The label and the injected pattern are never used as inputs.


WITH
-- Member history: earlier-received claims of the same member.
member_history AS (
    SELECT
        t.claim_id,
        COUNT(*) AS member_claims_30d,
        COUNT(*) FILTER (WHERE h.provider_id = t.provider_id
                           AND h.primary_hcpcs = t.primary_hcpcs
                           AND h.service_date = t.service_date) AS same_day_repeats,
        COUNT(*) FILTER (WHERE h.provider_id = t.provider_id
                           AND h.primary_hcpcs = t.primary_hcpcs
                           AND h.service_date >= t.service_date - 2
                           AND ABS(h.billed_amount - t.billed_amount) <= 0.03 * t.billed_amount
                        ) AS near_repeats_same_amount,
        COUNT(*) FILTER (WHERE h.provider_id = t.provider_id
                           AND h.primary_hcpcs = t.primary_hcpcs
                           AND h.service_date BETWEEN t.service_date - 7 AND t.service_date - 1
                        ) AS same_proc_prior_7d
    FROM target t
    JOIN claims h
      ON h.member_id = t.member_id
     AND h.claim_id <> t.claim_id
     AND h.received_date < t.received_date
     AND h.service_date BETWEEN t.service_date - 30 AND t.service_date
    GROUP BY t.claim_id
),
-- Provider volume on the same service day, among claims received so far.
provider_day AS (
    SELECT
        t.claim_id,
        COUNT(*) + 1 AS provider_claims_same_day   -- +1 counts the claim itself
    FROM target t
    JOIN claims h
      ON h.provider_id = t.provider_id
     AND h.claim_id <> t.claim_id
     AND h.service_date = t.service_date
     AND h.received_date <= t.received_date
    GROUP BY t.claim_id
),
-- Provider activity in the previous 90 days.
provider_history AS (
    SELECT t.claim_id, COUNT(*) AS provider_claims_90d
    FROM target t
    JOIN claims h
      ON h.provider_id = t.provider_id
     AND h.received_date < t.received_date
     AND h.service_date BETWEEN t.service_date - 90 AND t.service_date - 1
    GROUP BY t.claim_id
),
-- Member's past year: do the diagnosis and the amount fit this member's history?
member_year AS (
    SELECT
        t.claim_id,
        COUNT(*) AS member_claims_365d,
        COUNT(*) FILTER (WHERE SUBSTR(h.principal_dx, 1, 3) = SUBSTR(t.principal_dx, 1, 3))
            AS same_dx_claims,
        COUNT(*) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs) AS member_same_proc_365d,
        COUNT(*) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs
                           AND SUBSTR(h.principal_dx, 1, 3) = SUBSTR(t.principal_dx, 1, 3))
            AS same_proc_same_dx_claims,
        MEDIAN(h.billed_amount) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs)
            AS member_usual_amount
    FROM target t
    JOIN claims h
      ON h.member_id = t.member_id
     AND h.claim_id <> t.claim_id
     AND h.received_date < t.received_date
     AND h.service_date BETWEEN t.service_date - 365 AND t.service_date
    GROUP BY t.claim_id
),
-- Provider's past year: what does this provider usually bill for this procedure?
provider_year AS (
    SELECT t.claim_id, MEDIAN(h.billed_amount) AS provider_usual_amount
    FROM target t
    JOIN claims h
      ON h.provider_id = t.provider_id
     AND h.primary_hcpcs = t.primary_hcpcs
     AND h.received_date < t.received_date
     AND h.service_date BETWEEN t.service_date - 365 AND t.service_date - 1
    GROUP BY t.claim_id
),
peer AS (
    SELECT
        t.claim_id,
        -- Peer group: same procedure combination, else same main procedure, else all claims.
        COALESCE(CASE WHEN pa.n >= 30 THEN pa.median_amount END,
                 CASE WHEN ha.n >= 30 THEN ha.median_amount END,
                 (SELECT MEDIAN(median_amount) FROM ref_hcpcs_amount)) AS peer_median_amount
    FROM target t
    LEFT JOIN ref_proc_amount pa ON pa.proc_set = t.proc_set
    LEFT JOIN ref_hcpcs_amount ha ON ha.primary_hcpcs = t.primary_hcpcs
)
SELECT
    t.claim_id,
    t.service_date,
    t.received_date,
    t.primary_hcpcs,
    t.principal_dx,

    -- Claim basics
    t.billed_amount,
    LN(1 + t.billed_amount)                                   AS log_billed_amount,
    t.n_lines,
    t.n_dx,
    t.claim_days,

    -- Code grouping: HCPCS section and ICD-10 chapter
    CASE
        WHEN t.primary_hcpcs BETWEEN '99202' AND '99499' THEN 'evaluation_management'
        WHEN t.primary_hcpcs BETWEEN '90000' AND '99199' THEN 'medicine'
        WHEN t.primary_hcpcs BETWEEN '80000' AND '89999' THEN 'lab_pathology'
        WHEN t.primary_hcpcs BETWEEN '70000' AND '79999' THEN 'radiology'
        WHEN t.primary_hcpcs BETWEEN '10000' AND '69999' THEN 'surgery'
        ELSE 'hcpcs_' || SUBSTR(t.primary_hcpcs, 1, 1)         -- level II codes by letter
    END                                                        AS procedure_group,
    SUBSTR(t.principal_dx, 1, 1)                               AS dx_chapter,

    -- Diagnosis to procedure consistency
    COALESCE(dp.pair_share, 0)                                 AS dx_proc_share,
    (dp.pair_share IS NULL)::INT                               AS dx_proc_never_seen,

    -- Peer comparison of the billed amount
    p.peer_median_amount,
    LN(t.billed_amount / p.peer_median_amount)                 AS log_amount_vs_peers,

    -- Billed amount against the member's and the provider's own past year (0 = no history)
    my.member_usual_amount,
    py.provider_usual_amount,
    COALESCE(LN(t.billed_amount / my.member_usual_amount), 0)   AS log_amount_vs_member_history,
    COALESCE(LN(t.billed_amount / py.provider_usual_amount), 0) AS log_amount_vs_provider_history,

    -- Diagnosis against the member's past year (-1 = no history to compare with)
    COALESCE(my.member_claims_365d, 0)                         AS member_claims_365d,
    COALESCE(my.member_same_proc_365d, 0)                      AS member_same_proc_365d,
    COALESCE(my.same_dx_claims / NULLIF(my.member_claims_365d, 0), -1)
                                                               AS member_same_dx_share,
    COALESCE(my.same_proc_same_dx_claims / NULLIF(my.member_same_proc_365d, 0), -1)
                                                               AS member_proc_dx_match_share,

    -- Member history (time windows)
    COALESCE(mh.member_claims_30d, 0)                          AS member_claims_30d,
    COALESCE(mh.same_day_repeats, 0)                           AS same_day_repeats,
    COALESCE(mh.near_repeats_same_amount, 0)                   AS near_repeats_same_amount,
    COALESCE(mh.same_proc_prior_7d, 0)                         AS same_proc_prior_7d,

    -- Provider behaviour
    COALESCE(pd.provider_claims_same_day, 1)                   AS provider_claims_same_day,
    COALESCE(rp.usual_daily_claims, 1)                         AS provider_usual_daily_claims,
    COALESCE(pd.provider_claims_same_day, 1)
        / COALESCE(rp.usual_daily_claims, 1)                   AS provider_day_ratio,
    COALESCE(ph.provider_claims_90d, 0)                        AS provider_claims_90d,
    (rp.provider_id IS NULL)::INT                              AS provider_is_new,

    -- Member
    date_diff('year', m.birth_date, t.service_date)            AS member_age,
    COALESCE(m.sex, 'unknown')                                 AS member_sex,
    CASE WHEN t.service_date > m.coverage_end
          AND (m.coverage_restart IS NULL OR t.service_date < m.coverage_restart)
         THEN date_diff('day', m.coverage_end, t.service_date)
         ELSE 0 END                                            AS days_after_coverage_end
FROM target t
LEFT JOIN members m          ON m.member_id = t.member_id
LEFT JOIN ref_dx_proc dp     ON dp.primary_hcpcs = t.primary_hcpcs
                            AND dp.dx_category = SUBSTR(t.principal_dx, 1, 3)
LEFT JOIN ref_provider rp    ON rp.provider_id = t.provider_id
LEFT JOIN peer p             ON p.claim_id = t.claim_id
LEFT JOIN member_history mh  ON mh.claim_id = t.claim_id
LEFT JOIN member_year my     ON my.claim_id = t.claim_id
LEFT JOIN provider_year py   ON py.claim_id = t.claim_id
LEFT JOIN provider_day pd    ON pd.claim_id = t.claim_id
LEFT JOIN provider_history ph ON ph.claim_id = t.claim_id
