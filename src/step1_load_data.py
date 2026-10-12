"""Step 1: load the raw CMS files into DuckDB, one row per claim.

The raw outpatient file has one row per claim LINE. A claim with three
procedures appears three times, so we collapse it to one row per claim.
"""
import duckdb

from config import DB_PATH, RAW_DIR

# The raw file has 25 diagnosis columns; count how many are filled in.
N_DX_SQL = " + ".join(
    f"(COALESCE(ANY_VALUE(ICD_DGNS_CD{i}), '') <> '')::INT" for i in range(1, 26)
)


def main():
    con = duckdb.connect(str(DB_PATH))

    con.execute(f"""
        CREATE OR REPLACE VIEW outpatient_lines AS
        SELECT * FROM read_csv('{(RAW_DIR / "outpatient.csv").as_posix()}',
                               delim='|', header=true, all_varchar=true)
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW beneficiary AS
        SELECT * FROM read_csv('{(RAW_DIR / "beneficiary_2018.csv").as_posix()}',
                               delim='|', header=true, all_varchar=true)
    """)

    # One row per claim.
    con.execute(f"""
        CREATE OR REPLACE TABLE claims_raw AS
        SELECT
            CAST(CLM_ID AS BIGINT)                                   AS claim_id,
            CAST(ANY_VALUE(BENE_ID) AS BIGINT)                       AS member_id,
            ANY_VALUE(PRVDR_NUM)                                     AS provider_id,
            ANY_VALUE(NCH_CLM_TYPE_CD)                               AS claim_type,
            ANY_VALUE(CLM_FAC_TYPE_CD)                               AS place_of_service,
            ANY_VALUE(strptime(CLM_FROM_DT, '%d-%b-%Y'))::DATE       AS service_date,
            ANY_VALUE(strptime(NCH_WKLY_PROC_DT, '%d-%b-%Y'))::DATE  AS received_date,
            ANY_VALUE(date_diff('day', strptime(CLM_FROM_DT, '%d-%b-%Y'),
                                       strptime(CLM_THRU_DT, '%d-%b-%Y'))) AS claim_days,
            ANY_VALUE(CAST(CLM_TOT_CHRG_AMT AS DOUBLE))              AS billed_amount,
            COUNT(*)                                                 AS n_lines,
            ARG_MIN(HCPCS_CD, CAST(CLM_LINE_NUM AS INT))             AS primary_hcpcs,
            STRING_AGG(DISTINCT HCPCS_CD, '+' ORDER BY HCPCS_CD)     AS proc_set,
            ANY_VALUE(PRNCPAL_DGNS_CD)                               AS principal_dx,
            {N_DX_SQL}                                               AS n_dx
        FROM outpatient_lines
        GROUP BY CLM_ID
    """)

    # One row per member seen in claims. The 2018 beneficiary file does not
    # cover every member, so birth date and sex can be missing.
    con.execute("""
        CREATE OR REPLACE TABLE members_raw AS
        SELECT
            m.member_id,
            strptime(b.BENE_BIRTH_DT, '%d-%b-%Y')::DATE AS birth_date,
            b.SEX_IDENT_CD                              AS sex
        FROM (SELECT DISTINCT member_id FROM claims_raw) m
        LEFT JOIN beneficiary b ON CAST(b.BENE_ID AS BIGINT) = m.member_id
    """)

    n_claims, n_members, n_providers, first, last = con.sql("""
        SELECT COUNT(*), COUNT(DISTINCT member_id), COUNT(DISTINCT provider_id),
               MIN(service_date), MAX(service_date)
        FROM claims_raw
    """).fetchone()
    n_known = con.sql("SELECT COUNT(*) FROM members_raw WHERE birth_date IS NOT NULL").fetchone()[0]
    print(f"claims: {n_claims:,}  members: {n_members:,}  providers: {n_providers:,}")
    print(f"service dates: {first} to {last}")
    print(f"members with demographics: {n_known:,} of {n_members:,}")
    con.close()


if __name__ == "__main__":
    main()
