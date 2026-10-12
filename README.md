# Claims triage model

Take-home assignment for Apeiro Digital, Senior Data Scientist.

Scores each outpatient claim and routes it to one of three queues, with plain-language reasons:

- **auto_approve**: pay without review
- **review**: send to a claims reviewer
- **high_risk**: send to a senior reviewer or special investigations

The short write-up is in [NOTE.md](NOTE.md).

## Results on the test set (claims received 2022 to 2023)

| Model | PR-AUC (bad claim) | PR-AUC (FWA) | Precision in top 6% | Recall in top 6% |
|---|---|---|---|---|
| Logistic regression (baseline) | 0.50 | 0.55 | 57% | 58% |
| Decision tree, depth 6 | 0.64 | 0.65 | 70% | 71% |
| Random forest | 0.70 | 0.74 | 75% | 76% |
| **LightGBM (final)** | **0.71** | **0.75** | **75%** | **76%** |

5.8% of test claims are bad, so a random ranking would score a PR-AUC of about 0.06.
18% of the bad labels are random noise, so roughly 82% of bad claims carry any signal a model could find.

These numbers measure how well the model recovers patterns I injected into synthetic data. They are not a forecast of real-world fraud detection; see the Trust section of the note.

| Strategy | Illustrative cost (455 days) | Reviews per day |
|---|---|---|
| Pay everything | $12.23M | 0 |
| Review everything (ignores capacity) | $4.16M | 175 |
| **Model routing** | **$3.38M** | **14** |

Cost assumptions are in `src/config.py` and explained in the note.

## Reproduce

1. Install the libraries:

```bash
pip install -r requirements.txt
```

2. Download the data into `data/raw/` and unzip it (see "Data" below).

3. Run the whole pipeline (about 3 minutes on a laptop with 8 GB RAM):

```bash
python run_pipeline.py
```

4. Run the tests:

```bash
python -m pytest
```

5. Start the API, then open http://127.0.0.1:8000/docs:

```bash
cd src
uvicorn api:app
```

Everything is seeded (`SEED = 42`), so the numbers above reproduce exactly.

### Example API call

```bash
curl -X POST http://127.0.0.1:8000/score -H "Content-Type: application/json" -d "{\"claim_id\": 1, \"member_id\": -10000010286183, \"provider_id\": \"050796\", \"service_date\": \"2022-06-01\", \"received_date\": \"2022-06-03\", \"billed_amount\": 2500, \"procedure_codes\": [\"99241\"], \"diagnosis_codes\": [\"E785\", \"E119\"]}"
```

```json
{
  "claim_id": 1,
  "score": 0.1085,
  "probability_denied": 0.0264,
  "probability_fwa": 0.0821,
  "queue": "review",
  "reasons": ["Billed amount $2,500 is 10.2 times the typical $245 for this procedure and 11.2 times this provider's usual $224 and 11.1 times this member's usual $225."]
}
```

## Data

Option A: CMS Synthetic Medicare Enrollment, Fee-for-Service Claims and Prescription Drug Event data.
I used the outpatient claims file and the 2018 beneficiary file: 402,653 claims, 8,591 members, 4,559 providers, 2015 to 2023, ICD-10 and HCPCS codes. No real patient data.

data.cms.gov returned "Access Denied" from my network, so I downloaded the Internet Archive snapshots of the official files:

- Outpatient claims (27 MB zip): `https://web.archive.org/web/20250510121757id_/https://data.cms.gov/sites/default/files/2023-04/c3d8a962-c6b8-4a59-adb5-f0495cc81fda/Outpatient.zip`
- Beneficiary 2018 (served gzip-compressed, so rename to `.csv.gz` and gunzip): `https://web.archive.org/web/20250508134951id_/https://data.cms.gov/sites/default/files/2023-04/bf7aac0e-3ba3-4e7e-9463-073316a17d8a/beneficiary_2018.csv`

After unzipping, `data/raw/` must contain `outpatient.csv` and `beneficiary_2018.csv`.

## How the code is organised

| File | What it does |
|---|---|
| `run_pipeline.py` | Runs steps 1 to 6 in order |
| `src/config.py` | Every assumption in one place: split dates, label rates, noise, costs, capacity |
| `src/step1_load_data.py` | Raw CSV to DuckDB, one row per claim |
| `src/step2_make_labels.py` | Injects five bad-claim patterns and label noise |
| `src/reference_tables.sql` | What is "normal", learned from claims received before the training cutoff |
| `src/features.sql` | The feature query, shared by training and the API |
| `src/step3_build_features.py` | Runs the SQL and adds the time-based split |
| `src/model_utils.py` | Feature lists, data loading, preprocessing |
| `src/evaluate.py` | PR-AUC and precision/recall at review capacity |
| `src/step4_train_baseline.py` | Baseline logistic regression |
| `src/step5_compare_models.py` | Compares simple and complex models, saves the final one |
| `src/step6_choose_thresholds.py` | Picks thresholds from costs, reports cost and queues on test |
| `src/reasons.py` | Turns model explanations into reviewer sentences |
| `src/triage.py` | Routing rule and the single-claim scorer |
| `src/api.py` | FastAPI endpoint `POST /score` |
| `tests/test_triage.py` | Seven tests for routing and the API |
| `tests/test_timing.py` | Five timing tests on a tiny hand-made dataset (no real data needed) |
| `experiments/` | Feature experiments, ablation and leakage audit (see below) |

Outputs are written to `outputs/`: model comparison, label summary, cost comparison, queue report, and `flagged_claims_test.csv` with the reasons for every flagged test claim.

## Improvement experiments

The first version scored 65% precision and 67% recall at capacity. I then tested six candidate feature blocks, on train and validation only:

```bash
python experiments/run_feature_experiments.py
```

It builds the candidate features from `experiments/candidate_features.sql` and writes `outputs/feature_experiments_validation.csv`.

- **Adopted:** member diagnosis history and billed amount against the member's and provider's own history.
- **No gain or under one point, left out:** smoothed diagnosis lift, diagnosis against age and sex, price-spread features, larger trees, a binary objective (a one-off check, not in the script).
- **Rejected as artifacts of my data generator:** strict-timing provider-day volume and receipt lag.

**Controlled ablation.** Retraining the final model without exactly the four member diagnosis history features drops validation precision at capacity from 80% to 69% and recall from 76% to 66%. So most of the gain comes from them, and the note explains why the synthetic data flatters them. `outputs/new_diagnosis_check_validation.csv` measures their false-alarm cost: 7.0% of the 2,086 non-injected claims with a diagnosis new to the member are selected, against 0.4% of the 56,072 where it is not new.

## Leakage audit

```bash
python experiments/leakage_audit.py
```

Five checks, all passing (`outputs/leakage_audit.txt`):

1. The feature code never references the label columns.
2. Point-in-time: features for 14,279 claims are identical when recomputed with every later-received claim deleted.
3. Reference tables match a rebuild from claims received before the cutoff, and no training claim was received after it.
4. Splits do not overlap and are in order of received date.
5. Sanity check: training on shuffled labels gives validation PR-AUC 0.089 against 0.063 for random and 0.728 for the real model. This catches pipeline bugs; it does not by itself rule out leakage.

What the audit does not show:

- It covers timing, not realism. A feature can pass and still be flattered by the synthetic data.
- The point-in-time check keeps the stored reference tables and member data; check 3 and `tests/test_timing.py` cover those.
- Test-set numbers were printed during development. Model, feature and threshold choices were made on validation, but the test set is not a pristine holdout.

## Assumptions

- A claim is one row; the main procedure is the first line's HCPCS code.
- "Provider" is the billing facility (`PRVDR_NUM`).
- `received_date` is the CMS weekly processing date. Splits, reference tables and history are all keyed on it.
- Scoring contract: end-of-day batch. Member history uses claims received on earlier dates; provider-day volume also counts claims received the same date.
- Generator limits: replacement diagnoses and spike copies are drawn from all periods, and a substituted diagnosis is not checked for medical incompatibility.
- The 2018 beneficiary file covers 6,986 of the 8,591 members. Age and sex are missing for the rest; age is filled with the median.
- Coverage end dates do not exist in the data, so they are simulated in step 2.
- Unbundling is not modelled.
- The API reads claim history from a static DuckDB file. In production this would be a live feature store.

## How I used AI tools

I used Claude Code (Anthropic) as a pair programmer throughout:

- **Understanding the brief**: breaking the assignment into steps and a plan.
- **Data access**: finding archived copies of the CMS files when data.cms.gov was blocked.
- **Code**: drafting the pipeline scripts, the feature SQL, the API and the tests.
- **Experiments**: running the model comparison, the LightGBM settings check and the feature experiments.
- **Second opinion**: I asked ChatGPT to review the code and results for improvement ideas, then tested each idea on validation and kept only the ones that held up.
- **Writing**: first drafts of this README and the note.

I reviewed and ran every step, and the design decisions (label patterns, noise rates, time-based split, cost assumptions, final model choice) are ones I can explain and defend.
