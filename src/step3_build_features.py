"""Step 3: build the reference tables and the feature table in DuckDB."""
from pathlib import Path

import duckdb

import config as cfg

SQL_DIR = Path(__file__).parent
REFERENCE_SQL = (SQL_DIR / "reference_tables.sql").read_text()
FEATURE_SQL = (SQL_DIR / "features.sql").read_text()


def main():
    con = duckdb.connect(str(cfg.DB_PATH))

    # What is "normal", learned from the training period only.
    con.execute(REFERENCE_SQL.replace("{train_end}", cfg.TRAIN_END))

    # `target` = the claims we want features for. Here it is every claim.
    con.execute("CREATE OR REPLACE TEMP VIEW target AS SELECT * FROM claims")
    con.execute(f"CREATE OR REPLACE TABLE features AS {FEATURE_SQL}")

    # Attach the label and the train / validation / test split. The split is by
    # RECEIVED date: a claim belongs to the period in which it arrived to be scored.
    con.execute(f"""
        CREATE OR REPLACE TABLE model_data AS
        SELECT
            f.*,
            c.label,
            c.injected_pattern,
            CASE WHEN f.received_date < DATE '{cfg.TRAIN_END}' THEN 'train'
                 WHEN f.received_date < DATE '{cfg.VALID_END}' THEN 'valid'
                 ELSE 'test' END AS split
        FROM features f
        JOIN claims c USING (claim_id)
    """)

    print(con.sql("""
        SELECT split,
               COUNT(*) AS claims,
               ROUND(100 * AVG((label = 1)::INT), 2) AS pct_denied,
               ROUND(100 * AVG((label = 2)::INT), 2) AS pct_fwa
        FROM model_data GROUP BY split ORDER BY MIN(received_date)
    """).df().to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
