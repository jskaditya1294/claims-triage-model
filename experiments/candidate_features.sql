-- Candidate features for the improvement experiments (train + validation only).
-- Same leakage rules as the main pipeline: reference tables from the training
-- period, history from claims received strictly earlier.
-- {train_end} is filled in by Python.

CREATE OR REPLACE TEMP VIEW train_claims AS
SELECT c.*, SUBSTR(c.principal_dx, 1, 3) AS dx3, m.procedure_group, m.member_sex,
       CASE WHEN m.member_age IS NULL THEN 'unknown' WHEN m.member_age < 18 THEN '0-17'
            WHEN m.member_age < 45 THEN '18-44' WHEN m.member_age < 65 THEN '45-64'
            WHEN m.member_age < 75 THEN '65-74' ELSE '75+' END AS age_band
FROM claims c JOIN model_data m USING (claim_id)
WHERE c.received_date < DATE '{train_end}';

-- ---------- Reference counts ----------
CREATE OR REPLACE TABLE x_dx AS
SELECT dx3, COUNT(*) AS n_d, COUNT(*) / SUM(COUNT(*)) OVER () AS share_d FROM train_claims GROUP BY 1;

CREATE OR REPLACE TABLE x_proc AS
SELECT primary_hcpcs, COUNT(*) AS n_p FROM train_claims GROUP BY 1;

CREATE OR REPLACE TABLE x_proc_dx AS
SELECT primary_hcpcs, dx3, COUNT(*) AS n_pd FROM train_claims GROUP BY 1, 2;

CREATE OR REPLACE TABLE x_group_dx AS
SELECT procedure_group, dx3, COUNT(*) AS n_gd,
       SUM(COUNT(*)) OVER (PARTITION BY procedure_group) AS n_g
FROM train_claims GROUP BY 1, 2;

-- Every procedure code on the claim, not only the first one.
CREATE OR REPLACE TABLE x_code_dx AS
SELECT code, dx3, COUNT(*) AS n_cd, SUM(COUNT(*)) OVER (PARTITION BY code) AS n_c
FROM (SELECT UNNEST(string_split(proc_set, '+')) AS code, dx3 FROM train_claims)
GROUP BY 1, 2;

-- Diagnosis against member sex and age band.
CREATE OR REPLACE TABLE x_dx_sex AS
SELECT dx3, member_sex, COUNT(*) AS n_ds FROM train_claims GROUP BY 1, 2;
CREATE OR REPLACE TABLE x_sex AS
SELECT member_sex, COUNT(*) / SUM(COUNT(*)) OVER () AS share_s FROM train_claims GROUP BY 1;
CREATE OR REPLACE TABLE x_dx_age AS
SELECT dx3, age_band, COUNT(*) AS n_da FROM train_claims GROUP BY 1, 2;
CREATE OR REPLACE TABLE x_age AS
SELECT age_band, COUNT(*) / SUM(COUNT(*)) OVER () AS share_a FROM train_claims GROUP BY 1;

-- Billed amount: centre and spread of log amount for each peer group.
CREATE OR REPLACE TABLE x_amt_set AS
SELECT proc_set, COUNT(*) AS n, MEDIAN(LN(1 + billed_amount)) AS med,
       QUANTILE_CONT(LN(1 + billed_amount), 0.75) - QUANTILE_CONT(LN(1 + billed_amount), 0.25) AS iqr
FROM train_claims GROUP BY 1;
CREATE OR REPLACE TABLE x_amt_set_lines AS
SELECT proc_set, LEAST(n_lines, 6) AS line_bucket, COUNT(*) AS n, MEDIAN(LN(1 + billed_amount)) AS med
FROM train_claims GROUP BY 1, 2;
CREATE OR REPLACE TABLE x_amt_proc AS
SELECT primary_hcpcs, COUNT(*) AS n, MEDIAN(LN(1 + billed_amount)) AS med,
       QUANTILE_CONT(LN(1 + billed_amount), 0.75) - QUANTILE_CONT(LN(1 + billed_amount), 0.25) AS iqr
FROM train_claims GROUP BY 1;
CREATE OR REPLACE TABLE x_amt_all AS
SELECT MEDIAN(LN(1 + billed_amount)) AS med,
       QUANTILE_CONT(LN(1 + billed_amount), 0.75) - QUANTILE_CONT(LN(1 + billed_amount), 0.25) AS iqr
FROM train_claims;

-- ---------- History (claims received strictly earlier) ----------
CREATE OR REPLACE TABLE x_member_hist AS
SELECT
    t.claim_id,
    COUNT(*) AS m_claims_365d,
    COUNT(*) FILTER (WHERE SUBSTR(h.principal_dx, 1, 3) = SUBSTR(t.principal_dx, 1, 3)) AS m_same_dx3,
    COUNT(*) FILTER (WHERE SUBSTR(h.principal_dx, 1, 1) = SUBSTR(t.principal_dx, 1, 1)) AS m_same_chapter,
    COUNT(*) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs) AS m_same_proc,
    COUNT(*) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs
                       AND SUBSTR(h.principal_dx, 1, 3) = SUBSTR(t.principal_dx, 1, 3)) AS m_same_proc_dx3,
    MEDIAN(h.billed_amount) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs) AS m_proc_median_amount
FROM claims t
JOIN claims h
  ON h.member_id = t.member_id
 AND h.claim_id <> t.claim_id
 AND h.received_date < t.received_date
 AND h.service_date BETWEEN t.service_date - 365 AND t.service_date
GROUP BY t.claim_id;

CREATE OR REPLACE TABLE x_provider_hist AS
SELECT
    t.claim_id,
    COUNT(*) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs) AS p_same_proc,
    MEDIAN(h.billed_amount) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs) AS p_proc_median_amount,
    COUNT(*) FILTER (WHERE h.service_date >= t.service_date - 7) AS p_claims_7d
FROM claims t
JOIN claims h
  ON h.provider_id = t.provider_id
 AND h.received_date < t.received_date
 AND h.service_date BETWEEN t.service_date - 365 AND t.service_date - 1
GROUP BY t.claim_id;

CREATE OR REPLACE TABLE x_provider_day AS
SELECT
    t.claim_id,
    COUNT(*) + 1 AS p_day_strict,
    COUNT(DISTINCT h.member_id) AS p_day_other_members,
    COUNT(*) FILTER (WHERE h.primary_hcpcs = t.primary_hcpcs) AS p_day_same_proc
FROM claims t
JOIN claims h
  ON h.provider_id = t.provider_id
 AND h.claim_id <> t.claim_id
 AND h.service_date = t.service_date
 AND h.received_date < t.received_date
GROUP BY t.claim_id;

CREATE OR REPLACE TABLE x_code_compat AS
SELECT l.claim_id,
       MAX(COALESCE(r.n_cd, 0) / r2.n_c) AS dx_share_max_over_procs,
       AVG(COALESCE(r.n_cd, 0) / r2.n_c) AS dx_share_mean_over_procs
FROM (SELECT claim_id, SUBSTR(principal_dx, 1, 3) AS dx3,
             UNNEST(string_split(proc_set, '+')) AS code FROM claims) l
JOIN (SELECT DISTINCT code, n_c FROM x_code_dx) r2 ON r2.code = l.code
LEFT JOIN x_code_dx r ON r.code = l.code AND r.dx3 = l.dx3
GROUP BY l.claim_id;

-- ---------- One row of candidate features per claim ----------
CREATE OR REPLACE TABLE experiment_features AS
WITH base AS (
    SELECT c.*, SUBSTR(c.principal_dx, 1, 3) AS dx3, m.procedure_group, m.member_sex,
           CASE WHEN m.member_age IS NULL THEN 'unknown' WHEN m.member_age < 18 THEN '0-17'
                WHEN m.member_age < 45 THEN '18-44' WHEN m.member_age < 65 THEN '45-64'
                WHEN m.member_age < 75 THEN '65-74' ELSE '75+' END AS age_band,
           (c.received_date < DATE '{train_end}')::INT AS in_train,
           m.provider_usual_daily_claims
    FROM claims c JOIN model_data m USING (claim_id)
)
SELECT
    b.claim_id,

    -- A. strict timing for provider-day volume
    COALESCE(pd.p_day_strict, 1)                                   AS provider_claims_same_day_strict,
    COALESCE(pd.p_day_strict, 1) / b.provider_usual_daily_claims   AS provider_day_ratio_strict,

    -- B. diagnosis compatibility: smoothing, lift, support, broader groups
    (COALESCE(xpd.n_pd, 0) + 20 * COALESCE(xd.share_d, 0)) / (COALESCE(xp.n_p, 0) + 20)
                                                                   AS dx_share_smoothed,
    LN(((COALESCE(xpd.n_pd, 0) + 20 * COALESCE(xd.share_d, 0)) / (COALESCE(xp.n_p, 0) + 20) + 1e-6)
       / (COALESCE(xd.share_d, 0) + 1e-6))                         AS dx_log_lift,
    -- leave-one-out version: a training claim does not count itself
    LN(((GREATEST(COALESCE(xpd.n_pd, 0) - b.in_train, 0) + 20 * COALESCE(xd.share_d, 0))
        / (GREATEST(COALESCE(xp.n_p, 0) - b.in_train, 0) + 20) + 1e-6)
       / (COALESCE(xd.share_d, 0) + 1e-6))                         AS dx_log_lift_loo,
    LN(1 + COALESCE(xpd.n_pd, 0))                                  AS dx_pair_count_log,
    LN(1 + COALESCE(xp.n_p, 0))                                    AS proc_count_log,
    LN(1 + COALESCE(xd.n_d, 0))                                    AS dx_count_log,
    COALESCE(xgd.n_gd / xgd.n_g, 0)                                AS dx_share_in_proc_group,
    COALESCE(cc.dx_share_max_over_procs, 0)                        AS dx_share_max_over_procs,
    COALESCE(cc.dx_share_mean_over_procs, 0)                       AS dx_share_mean_over_procs,
    -- diagnosis against member sex and age band (lift over the overall mix)
    LN(((COALESCE(xds.n_ds, 0) + 5 * xs.share_s) / (COALESCE(xd.n_d, 0) + 5)) / xs.share_s)
                                                                   AS dx_sex_log_lift,
    LN(((COALESCE(xda.n_da, 0) + 5 * xa.share_a) / (COALESCE(xd.n_d, 0) + 5)) / xa.share_a)
                                                                   AS dx_age_log_lift,

    -- C. member diagnosis history
    COALESCE(mh.m_claims_365d, 0)                                  AS m_claims_365d,
    COALESCE(mh.m_same_dx3, 0)                                     AS m_same_dx3_365d,
    COALESCE(mh.m_same_chapter, 0)                                 AS m_same_chapter_365d,
    COALESCE(mh.m_same_dx3 / NULLIF(mh.m_claims_365d, 0), -1)      AS m_same_dx3_share,
    COALESCE(mh.m_same_proc, 0)                                    AS m_same_proc_365d,
    COALESCE(mh.m_same_proc_dx3 / NULLIF(mh.m_same_proc, 0), -1)   AS m_proc_dx_match_share,

    -- D. billed amount: spread-aware and personalised benchmarks
    (LN(1 + b.billed_amount) - COALESCE(CASE WHEN aset.n >= 30 THEN aset.med END,
                                        CASE WHEN aproc.n >= 30 THEN aproc.med END, aall.med))
      / GREATEST(COALESCE(CASE WHEN aset.n >= 30 THEN aset.iqr END,
                          CASE WHEN aproc.n >= 30 THEN aproc.iqr END, aall.iqr), 0.1)
                                                                   AS amount_robust_deviation,
    COALESCE(CASE WHEN aset.n >= 30 THEN aset.iqr END,
             CASE WHEN aproc.n >= 30 THEN aproc.iqr END, aall.iqr) AS peer_log_iqr,
    LN(1 + COALESCE(aset.n, 0))                                    AS peer_count_log,
    LN(1 + b.billed_amount) - COALESCE(CASE WHEN alines.n >= 30 THEN alines.med END,
                                       CASE WHEN aset.n >= 30 THEN aset.med END,
                                       CASE WHEN aproc.n >= 30 THEN aproc.med END, aall.med)
                                                                   AS log_amount_vs_line_peers,
    COALESCE(LN(b.billed_amount / mh.m_proc_median_amount), 0)     AS log_amount_vs_member_history,
    COALESCE(LN(b.billed_amount / ph.p_proc_median_amount), 0)     AS log_amount_vs_provider_history,
    COALESCE(ph.p_same_proc, 0)                                    AS p_same_proc_365d,

    -- E. provider volume
    date_diff('day', b.service_date, b.received_date)              AS receipt_lag_days,
    COALESCE(pd.p_day_other_members, 0)                            AS p_day_other_members,
    COALESCE(pd.p_day_same_proc, 0)                                AS p_day_same_proc,
    COALESCE(ph.p_claims_7d, 0)                                    AS p_claims_7d
FROM base b
LEFT JOIN x_dx xd            ON xd.dx3 = b.dx3
LEFT JOIN x_proc xp          ON xp.primary_hcpcs = b.primary_hcpcs
LEFT JOIN x_proc_dx xpd      ON xpd.primary_hcpcs = b.primary_hcpcs AND xpd.dx3 = b.dx3
LEFT JOIN x_group_dx xgd     ON xgd.procedure_group = b.procedure_group AND xgd.dx3 = b.dx3
LEFT JOIN x_code_compat cc   ON cc.claim_id = b.claim_id
LEFT JOIN x_dx_sex xds       ON xds.dx3 = b.dx3 AND xds.member_sex = b.member_sex
LEFT JOIN x_sex xs           ON xs.member_sex = b.member_sex
LEFT JOIN x_dx_age xda       ON xda.dx3 = b.dx3 AND xda.age_band = b.age_band
LEFT JOIN x_age xa           ON xa.age_band = b.age_band
LEFT JOIN x_amt_set aset     ON aset.proc_set = b.proc_set
LEFT JOIN x_amt_set_lines alines ON alines.proc_set = b.proc_set AND alines.line_bucket = LEAST(b.n_lines, 6)
LEFT JOIN x_amt_proc aproc   ON aproc.primary_hcpcs = b.primary_hcpcs
CROSS JOIN x_amt_all aall
LEFT JOIN x_member_hist mh   ON mh.claim_id = b.claim_id
LEFT JOIN x_provider_hist ph ON ph.claim_id = b.claim_id
LEFT JOIN x_provider_day pd  ON pd.claim_id = b.claim_id;
