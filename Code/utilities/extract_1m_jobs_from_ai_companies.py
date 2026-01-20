#!/usr/bin/env python3
"""
Build ~1M jobs from companies found in Data/ai_development_deduplicated_custom.csv
Includes ALL seed UIDs + random sample of other jobs from those companies
Exports only: uid, company_id, content_clean, duplicate_group
"""

import os
import math
import csv
import psycopg2
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

# -------------------
# Config
# -------------------
SEED_FILE = "Data/ai_development_deduplicated_custom.csv"
OUTPUT_FILE = "Data/ai_company_jobs_target1M.csv"
TARGET_TOTAL = 1_000_000  # desired total rows
UID_COL = "uid"           # must exist in seed file

# -------------------
# Load seed CSV
# -------------------
print("Loading seed file...")
seed = pd.read_csv(SEED_FILE)

if UID_COL not in seed.columns:
    raise ValueError(f"Column '{UID_COL}' not found in {SEED_FILE}")

# Prefer company_id if present; otherwise fall back to company_name
if "company_id" in seed.columns and seed["company_id"].notna().any():
    COMPANY_MODE = "id"
    COMPANY_COL = "company_id"
    seed_companies = (
        seed[COMPANY_COL]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
else:
    if "company_name" not in seed.columns:
        raise ValueError(f"Neither 'company_id' nor 'company_name' present/useful in {SEED_FILE}")
    COMPANY_MODE = "name"
    COMPANY_COL = "company_name"
    seed_companies = (
        seed[COMPANY_COL]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

# UIDs as strings (text format, not UUIDs)
seed_uids = (
    seed[UID_COL]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .tolist()
)

print(f"Seed companies ({'company_id' if COMPANY_MODE=='id' else 'company_name'}): {len(seed_companies):,}")
print(f"Seed UIDs: {len(seed_uids):,}")

# -------------------
# Connect to DB
# -------------------
print("Connecting to database...")
load_dotenv("config.env")
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
)
conn.autocommit = True
conn.set_client_encoding('UTF8')
cur = conn.cursor()

# -------------------
# Create TEMP tables & bulk load (CSV-safe)
# -------------------
print("Creating temporary tables...")
cur.execute("DROP TABLE IF EXISTS temp_companies;")
if COMPANY_MODE == "id":
    cur.execute("CREATE TEMP TABLE temp_companies(company_id text);")
else:
    cur.execute("CREATE TEMP TABLE temp_companies(company_name text);")

cur.execute("DROP TABLE IF EXISTS temp_uids;")
# Use TEXT since database uid column is text, not uuid
cur.execute("CREATE TEMP TABLE temp_uids(uid text);")

def copy_list_to_temp(cur, table, col, items):
    sio = StringIO()
    w = csv.writer(sio)  # proper CSV quoting for commas/newlines
    for x in items:
        w.writerow([("" if x is None else str(x).strip())])
    sio.seek(0)
    cur.copy_expert(f"COPY {table}({col}) FROM STDIN WITH (FORMAT csv)", sio)

# Load companies/uids
print("Loading companies and UIDs to temp tables...")
if COMPANY_MODE == "id":
    copy_list_to_temp(cur, "temp_companies", "company_id", seed_companies)
    cur.execute("CREATE INDEX ON temp_companies(company_id);")
else:
    copy_list_to_temp(cur, "temp_companies", "company_name", seed_companies)
    cur.execute("CREATE INDEX ON temp_companies(company_name);")

copy_list_to_temp(cur, "temp_uids", "uid", seed_uids)
cur.execute("CREATE INDEX ON temp_uids(uid);")

# Sanity counts (optional)
cur.execute("SELECT COUNT(*) FROM temp_companies")
print("temp_companies rows:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM temp_uids")
print("temp_uids rows:", cur.fetchone()[0])

# -------------------
# Count already-included (seed) rows
# -------------------
print("Counting seed rows...")
# Note: Changed table name from job_postings to job_postings_unified
cur.execute("""
    SELECT COUNT(*) 
    FROM job_postings_unified jp 
    JOIN temp_uids tu ON jp.uid = tu.uid
    WHERE jp.content_clean IS NOT NULL
""")
included_ct = cur.fetchone()[0]
print(f"Already-included rows (seed UIDs): {included_ct:,}")

need_more = max(TARGET_TOTAL - included_ct, 0)
if need_more == 0:
    print("We already meet or exceed the target; exporting seed rows only.")

# -------------------
# Count eligible extras (same companies, not in seed uids)
# -------------------
if COMPANY_MODE == "id":
    join_company = "tc.company_id = jp.company_id"  # Both are text now
else:
    join_company = "tc.company_name = jp.company_name"

print("Counting eligible additional rows...")
cur.execute(f"""
    SELECT COUNT(*)
    FROM job_postings_unified jp
    JOIN temp_companies tc ON {join_company}
    LEFT JOIN temp_uids tu ON tu.uid = jp.uid
    WHERE tu.uid IS NULL
      AND jp.content_clean IS NOT NULL
""")
eligible_ct = cur.fetchone()[0]
print(f"Eligible additional rows: {eligible_ct:,}")

# -------------------
# Decide sampling modulus k (0..1000), md5-based predicate
# -------------------
if need_more > 0 and eligible_ct > 0:
    frac = need_more / float(eligible_ct)
    k = min(1000, max(0, math.ceil(1000 * frac)))
    if k == 0:
        k = 1  # ensure some sampling; LIMIT will cap to need_more
else:
    k = 0
print(f"Sampling modulus k: {k} (out of 1000)")
print(f"Need {need_more:,} more rows from {eligible_ct:,} eligible")

# -------------------
# Seed SQL (ALL seed UIDs, exact match)
# -------------------
seed_sql = """
    SELECT jp.uid, jp.company_id, jp.content_clean, jp.duplicate_group::text
    FROM job_postings_unified jp
    JOIN temp_uids tu ON jp.uid = tu.uid
    WHERE jp.content_clean IS NOT NULL
"""

# -------------------
# Extra SQL (sample from same companies, excluding seed uids)
# Deterministic md5-based sampling; LIMIT caps to need_more
# -------------------
if need_more > 0 and eligible_ct > 0 and k > 0:
    extra_sql = f"""
        SELECT jp.uid, jp.company_id, jp.content_clean, jp.duplicate_group::text
        FROM job_postings_unified jp
        JOIN temp_companies tc ON {join_company}
        LEFT JOIN temp_uids tu ON tu.uid = jp.uid
        WHERE tu.uid IS NULL
          AND jp.content_clean IS NOT NULL
          AND ( ('x' || substr(md5(jp.uid), 1, 8))::bit(32)::int % 1000 ) < {k}
        LIMIT {need_more}
    """
else:
    # empty result
    extra_sql = """
        SELECT uid, company_id, content_clean, duplicate_group::text
        FROM (SELECT NULL::text uid, NULL::text company_id, NULL::text content_clean, NULL::text duplicate_group) x
        WHERE false
    """

# -------------------
# UNION ALL + DISTINCT ON (uid) so seed rows win
# -------------------
final_sql = f"""
WITH seed AS (
  {seed_sql}
),
extra AS (
  {extra_sql}
),
unioned AS (
  SELECT 1 AS src, * FROM seed
  UNION ALL
  SELECT 2 AS src, * FROM extra
)
SELECT uid, company_id, content_clean, duplicate_group
FROM (
  SELECT DISTINCT ON (uid) uid, company_id, content_clean, duplicate_group, src
  FROM unioned
  ORDER BY uid, src        -- src=1 (seed) preferred over src=2 (extra)
) t
"""

print(f"Exporting to CSV: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    cur.copy_expert(
        f"COPY ({final_sql}) TO STDOUT WITH CSV HEADER",
        f
    )

# -------------------
# Sanity checks
# -------------------
print("Running sanity checks...")
out_df = pd.read_csv(OUTPUT_FILE, usecols=["uid", "company_id", "content_clean", "duplicate_group"])
print(f"✅ Wrote {len(out_df):,} rows to {OUTPUT_FILE}")

# Check uniqueness
if not out_df["uid"].is_unique:
    print("❌ WARNING: Duplicate UIDs detected in output")
else:
    print("✅ UID uniqueness check passed")

# Check if we have both seed and additional rows
seed_in_output = out_df["uid"].isin(seed_uids).sum()
print(f"✅ Seed UIDs in output: {seed_in_output:,} of {len(seed_uids):,}")

cur.close()
conn.close()
print("🎉 Done!")