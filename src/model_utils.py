"""Pieces shared by all model scripts: feature lists, data loading, preprocessing."""
import duckdb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as cfg

NUMERIC = [
    "log_billed_amount", "n_lines", "n_dx", "claim_days",
    "dx_proc_share", "dx_proc_never_seen",
    "log_amount_vs_peers", "log_amount_vs_member_history", "log_amount_vs_provider_history",
    "member_claims_365d", "member_same_proc_365d", "member_same_dx_share", "member_proc_dx_match_share",
    "member_claims_30d", "same_day_repeats", "near_repeats_same_amount", "same_proc_prior_7d",
    "provider_claims_same_day", "provider_day_ratio", "provider_claims_90d", "provider_is_new",
    "member_age", "days_after_coverage_end",
]
CATEGORICAL = ["procedure_group", "dx_chapter", "member_sex"]
FEATURES = NUMERIC + CATEGORICAL


def load_split(split):
    """Load the train, valid or test rows from DuckDB."""
    con = duckdb.connect(str(cfg.DB_PATH), read_only=True)
    # ORDER BY keeps the row order fixed, so training reproduces exactly.
    data = con.sql(f"SELECT * FROM model_data WHERE split = '{split}' ORDER BY claim_id").df()
    con.close()
    return data


def make_preprocessor():
    """Same preprocessing for every model, so the comparison is fair."""
    return ColumnTransformer([
        # Fill missing age with the median, then put all numbers on the same scale.
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        # Turn each category into its own 0/1 column.
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)
