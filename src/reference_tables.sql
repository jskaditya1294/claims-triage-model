-- Reference tables: what is "normal", learned only from claims RECEIVED before the
-- training cutoff, so nothing unavailable when the model was fitted is used.
-- {train_end} is filled in by Python.


-- Typical billed amount for each combination of procedures on a claim.
CREATE OR REPLACE TABLE ref_proc_amount AS
SELECT proc_set, COUNT(*) AS n, MEDIAN(billed_amount) AS median_amount
FROM claims WHERE received_date < DATE '{train_end}'
GROUP BY proc_set;

-- Fallback when the exact combination is rare: typical amount per main procedure.
CREATE OR REPLACE TABLE ref_hcpcs_amount AS
SELECT primary_hcpcs, COUNT(*) AS n, MEDIAN(billed_amount) AS median_amount
FROM claims WHERE received_date < DATE '{train_end}'
GROUP BY primary_hcpcs;

-- How often each diagnosis category (first 3 characters of ICD-10) goes with
-- each procedure. A low share means the diagnosis rarely supports the procedure.
CREATE OR REPLACE TABLE ref_dx_proc AS
SELECT primary_hcpcs,
       SUBSTR(principal_dx, 1, 3) AS dx_category,
       COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY primary_hcpcs) AS pair_share
FROM claims WHERE received_date < DATE '{train_end}'
GROUP BY 1, 2;

-- Each provider's normal number of claims on a day they bill.
CREATE OR REPLACE TABLE ref_provider AS
SELECT provider_id, MEDIAN(claims_that_day) AS usual_daily_claims
FROM (
    SELECT provider_id, service_date, COUNT(*) AS claims_that_day
    FROM claims WHERE received_date < DATE '{train_end}'
    GROUP BY 1, 2
)
GROUP BY provider_id;
