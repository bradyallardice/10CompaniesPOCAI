#!/usr/bin/env python3
"""
Unified Job Postings Table Creator - WITH VERBOSE MONITORING
Creates a single job_postings_unified table from two data folders with real-time progress
"""

import os
import re
import gzip
import json
import unicodedata
import hashlib
import sys
import time
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import pandas as pd

def log(msg):
    """Print with timestamp and flush immediately"""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def main():
    log("🚀 Starting Unified Job Postings Table Creation...")
    
    # Load environment and connect to database
    load_dotenv("config.env")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    conn.autocommit = False
    
    log(f"✅ Connected to database: {os.getenv('DB_NAME')}")
    
    # Schema and table configuration
    SCHEMA = "public"
    TABLE_FINAL = "job_postings_unified"
    TABLE_STAGE = "job_postings_unified_staging"
    
    try:
        # Check what already exists
        check_existing_progress(conn, SCHEMA, TABLE_STAGE, TABLE_FINAL)
        
        # Step 1: Create final table with proper schema
        create_final_table(conn, SCHEMA, TABLE_FINAL)
        
        # Step 2: Create staging table
        create_staging_table(conn, SCHEMA, TABLE_STAGE)
        
        # Step 3: Load data into staging table
        staging_count = load_data_to_staging(conn, SCHEMA, TABLE_STAGE)
        
        if staging_count == 0:
            log("❌ No data loaded into staging table - stopping")
            return
        
        # Step 4: Merge staging into final with type casting
        merge_staging_to_final_batched(conn, SCHEMA, TABLE_STAGE, TABLE_FINAL, staging_count)
        
        # Step 5: Run QA checks
        run_qa_checks(conn, SCHEMA, TABLE_FINAL)
        
        # Step 6: Cleanup staging table
        cleanup_staging(conn, SCHEMA, TABLE_STAGE)
        
        log("🎉 Successfully created unified job postings table!")
        
    except Exception as e:
        log(f"❌ Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
        log("🔐 Database connection closed")

def check_existing_progress(conn, schema, staging_table, final_table):
    """Check what tables exist and their current state"""
    log("🔍 Checking existing progress...")
    
    with conn.cursor() as cur:
        # Check table existence
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name IN (%s, %s)
        """, [schema, final_table, staging_table])
        
        existing_tables = [row[0] for row in cur.fetchall()]
        
        for table in [final_table, staging_table]:
            if table in existing_tables:
                cur.execute(f"SELECT COUNT(*) FROM {schema}.{table};")
                count = cur.fetchone()[0]
                log(f"📊 Found {table}: {count:,} rows")
            else:
                log(f"📊 {table}: Not found (will create)")

def create_final_table(conn, schema, table_name):
    """Create the final table with proper schema including extensions and indexes"""
    log("🗂️ Creating final table schema...")
    
    with conn.cursor() as cur:
        # Create extensions
        cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")
        
        # Create normalization function
        cur.execute("""
            CREATE OR REPLACE FUNCTION public.content_normalize(t text)
            RETURNS text
            LANGUAGE sql
            IMMUTABLE
            AS $$
              SELECT
                CASE
                  WHEN t IS NULL THEN NULL
                  ELSE
                    trim(
                      regexp_replace(
                        regexp_replace(
                          lower(unaccent(t)),
                          '[^[:alnum:]\\s]+', ' ', 'g'
                        ),
                        '\\s+', ' ', 'g'
                      )
                    )
                END
            $$;
        """)
        
        # Drop table if exists and recreate
        cur.execute(f"DROP TABLE IF EXISTS {schema}.{table_name} CASCADE;")
        
        # Create final table with TEXT uid to handle surrogate UIDs
        cur.execute(f"""
            CREATE TABLE {schema}.{table_name} (
                uid                 TEXT PRIMARY KEY,
                title               TEXT,
                company_id          TEXT,
                company_name        TEXT,
                company_is_recruiter BOOLEAN,
                company_size        TEXT,
                cantons             TEXT[],
                x28_industries      TEXT[],
                x28_occupations     TEXT[],
                location_raw        TEXT,
                url                 TEXT,
                content_clean       TEXT,
                content_norm        TEXT GENERATED ALWAYS AS (public.content_normalize(content_clean)) STORED,
                duplicate_group     UUID,
                tst_created         TIMESTAMPTZ,
                tst_deleted         TIMESTAMPTZ,
                source_folder       TEXT NOT NULL,
                source_file         TEXT NOT NULL
            );
        """)
        
    conn.commit()
    log("✅ Final table created (indexes will be added after data load)")

def create_staging_table(conn, schema, table_name):
    """Create staging table with all TEXT columns"""
    log("🗂️ Creating staging table...")
    
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {schema}.{table_name};")
        
        cur.execute(f"""
            CREATE UNLOGGED TABLE {schema}.{table_name} (
                uid                 TEXT,
                title               TEXT,
                company_id          TEXT,
                company_name        TEXT,
                company_is_recruiter TEXT,
                company_size        TEXT,
                cantons             TEXT,
                x28_industries      TEXT,
                x28_occupations     TEXT,
                location_raw        TEXT,
                url                 TEXT,
                content_clean       TEXT,
                content_norm        TEXT,
                duplicate_group     TEXT,
                tst_created         TEXT,
                tst_deleted         TEXT,
                source_folder       TEXT,
                source_file         TEXT
            );
        """)
    
    conn.commit()
    log("✅ Staging table created")

def normalize_text_for_matching(text):
    """Normalize text for matching purposes"""
    if text is None:
        return None
    t = unicodedata.normalize("NFD", str(text)).encode("ascii","ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def ndjson_stream(path):
    """Stream NDJSON from gzipped file"""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def first_non_null(d, *keys):
    """Get first non-null value from dict for given keys"""
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None

def synthesize_uid_from_fields(*parts):
    """Create surrogate UID from field values"""
    key = "|".join("" if p is None else str(p) for p in parts)
    return "surr_" + hashlib.sha1(key.encode("utf-8")).hexdigest()

def choose_surrogate_uid(obj):
    """Choose surrogate UID based on available fields"""
    company_id   = first_non_null(obj, "company_id")
    company_name = first_non_null(obj, "company_name", "companyName", "company")
    url          = first_non_null(obj, "url")
    title        = first_non_null(obj, "title")
    tst_created  = first_non_null(obj, "tst_created", "created_at", "posted_at", "createdAt")

    combos = [
        (company_id, url, tst_created),
        (company_id, url, title, tst_created),
        (company_name, url, tst_created),
        (company_name, title, url, tst_created),
    ]
    for tup in combos:
        if all(x not in (None, "") for x in tup):
            return synthesize_uid_from_fields(*tup)
    return None

def build_row(obj, source_folder, source_file):
    """Build row tuple from JSON object"""
    native_uid   = first_non_null(obj, "uid", "id")
    title        = first_non_null(obj, "title")
    company_name = first_non_null(obj, "company_name", "companyName", "company")
    company_id   = first_non_null(obj, "company_id")
    url          = first_non_null(obj, "url")
    tst_created  = first_non_null(obj, "tst_created", "created_at", "posted_at", "createdAt")

    uid = str(native_uid) if native_uid else choose_surrogate_uid(obj)
    if not uid:
        return None

    # Require critical downstream fields
    if not (title and company_name and tst_created and url):
        return None

    content_clean = obj.get("content_clean")
    content_norm  = normalize_text_for_matching(content_clean)

    # Keep as text for staging table
    raw_dup = obj.get("duplicate_group")
    dup_str = str(raw_dup) if raw_dup not in (None, "") else None

    return (
        uid, title, company_id, company_name,
        str(obj.get("company_is_recruiter")) if obj.get("company_is_recruiter") is not None else None,
        obj.get("company_size"),
        str(obj.get("cantons")) if obj.get("cantons") is not None else None,
        str(obj.get("x28_industries")) if obj.get("x28_industries") is not None else None,
        str(obj.get("x28_occupations")) if obj.get("x28_occupations") is not None else None,
        obj.get("location_raw"), url, content_clean, content_norm, dup_str,
        str(tst_created) if tst_created else None,
        str(obj.get("tst_deleted")) if obj.get("tst_deleted") else None,
        source_folder, source_file,
    )

def load_data_to_staging(conn, schema, table_name):
    """Load data from both folders into staging table"""
    log("🚀 Loading data into staging table...")
    
    BASE = Path("/Users/bradyallardice/Desktop/PhD/Projects/KurerAllardice2024/10CompaniesPOCAI/")
    FOLDERS = [
        BASE / "240826_panel_data_with_dg",
        BASE / "250826_panel_data_01012024_30062025",
    ]
    
    COLS = [
        "uid","title","company_id","company_name","company_is_recruiter","company_size",
        "cantons","x28_industries","x28_occupations","location_raw","url",
        "content_clean","content_norm","duplicate_group","tst_created","tst_deleted",
        "source_folder","source_file",
    ]
    
    insert_sql = f"""
    INSERT INTO {schema}.{table_name} ({", ".join(COLS)})
    VALUES %s;
    """
    
    def insert_batch(cur, rows):
        if not rows:
            return 0
        execute_values(cur, insert_sql, rows, page_size=5000)
        return len(rows)
    
    def list_json_gz(folder):
        return list(folder.rglob("*.json.gzip"))
    
    total_scanned = total_inserted = 0
    for fld in FOLDERS:
        if not fld.exists():
            log(f"⚠️  Folder not found: {fld} (skipping)")
            continue
            
        scanned = inserted_here = 0
        files = sorted(set(p.resolve() for p in list_json_gz(fld)))
        
        log(f"📁 Processing folder: {fld.name} ({len(files)} files)")
        
        for i, path in enumerate(files):
            if i % 10 == 0:  # More frequent updates
                log(f"   File {i+1}/{len(files)}: {path.name} (scanned: {scanned:,}, inserted: {inserted_here:,})")
                
            batch = []
            for obj in ndjson_stream(path):
                scanned += 1
                total_scanned += 1
                row = build_row(obj, source_folder=fld.name, source_file=path.name)
                if row is None:
                    continue
                batch.append(row)
                if len(batch) >= 5000:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL synchronous_commit = off;")
                        inserted_here += insert_batch(cur, batch)
                        conn.commit()
                    batch.clear()
            
            if batch:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL synchronous_commit = off;")
                    inserted_here += insert_batch(cur, batch)
                    conn.commit()
                batch.clear()
        
        total_inserted += inserted_here
        log(f"✅ [{fld.name}] Final: scanned={scanned:,}, inserted={inserted_here:,}")
    
    log(f"📊 TOTAL: scanned={total_scanned:,}, inserted={total_inserted:,}")
    
    # Check staging table
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {schema}.{table_name};")
        count = cur.fetchone()[0]
    log(f"✅ Staging table verified: {count:,} rows")
    
    return count

def merge_staging_to_final_batched(conn, schema, staging_table, final_table, total_rows):
    """Merge staging data into final table with batched progress monitoring"""
    log("🔄 Starting batched merge operation...")
    
    BATCH_SIZE = 25000  # Smaller batches for more frequent updates
    uuid_regex = r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    
    batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
    log(f"📦 Processing {batches} batches of {BATCH_SIZE:,} rows each")
    
    total_processed = 0
    
    for batch_num in range(batches):
        batch_start = time.time()
        offset = batch_num * BATCH_SIZE
        
        log(f"🔄 Batch {batch_num + 1}/{batches}: rows {offset:,} to {min(offset + BATCH_SIZE, total_rows):,}")
        
        conn.autocommit = False
        
        with conn.cursor() as cur:
            cur.execute("SET LOCAL synchronous_commit = off;")
            cur.execute("SET LOCAL work_mem = '256MB';")
            
            batch_sql = f"""
                INSERT INTO {schema}.{final_table} (
                    uid, title, company_id, company_name, company_is_recruiter, company_size,
                    cantons, x28_industries, x28_occupations, location_raw, url,
                    content_clean, duplicate_group, tst_created, tst_deleted,
                    source_folder, source_file
                )
                SELECT
                    uid, title, company_id, company_name,
                    CASE 
                        WHEN company_is_recruiter IS NULL OR company_is_recruiter = '' THEN NULL
                        ELSE company_is_recruiter::boolean
                    END,
                    company_size,
                    CASE 
                        WHEN cantons IS NULL OR cantons = '' OR cantons = '[]' OR cantons = 'null' THEN NULL
                        ELSE string_to_array(trim(both '[]"\\' ' from cantons), ',')
                    END,
                    CASE 
                        WHEN x28_industries IS NULL OR x28_industries = '' OR x28_industries = '[]' OR x28_industries = 'null' THEN NULL
                        ELSE string_to_array(trim(both '[]"\\' ' from x28_industries), ',')
                    END,
                    CASE 
                        WHEN x28_occupations IS NULL OR x28_occupations = '' OR x28_occupations = '[]' OR x28_occupations = 'null' THEN NULL
                        ELSE string_to_array(trim(both '[]"\\' ' from x28_occupations), ',')
                    END,
                    location_raw, url, content_clean,
                    CASE
                        WHEN duplicate_group IS NULL OR duplicate_group = '' OR duplicate_group = 'null' THEN NULL
                        WHEN duplicate_group ~ %s THEN duplicate_group::uuid
                        ELSE NULL
                    END,
                    CASE
                        WHEN tst_created IS NULL OR tst_created = '' THEN NULL
                        ELSE tst_created::timestamptz
                    END,
                    CASE
                        WHEN tst_deleted IS NULL OR tst_deleted = '' THEN NULL
                        ELSE tst_deleted::timestamptz
                    END,
                    source_folder, source_file
                FROM (
                    SELECT * FROM {schema}.{staging_table}
                    ORDER BY uid
                    LIMIT %s OFFSET %s
                ) batch
                ON CONFLICT (uid) DO NOTHING;
            """
            
            cur.execute(batch_sql, [uuid_regex, BATCH_SIZE, offset])
            batch_inserted = cur.rowcount
            total_processed += batch_inserted
            
            conn.commit()
            
            batch_time = time.time() - batch_start
            progress_pct = ((batch_num + 1) / batches) * 100
            
            log(f"    ✅ Batch {batch_num + 1}: {batch_inserted:,} rows in {batch_time:.1f}s | Total: {total_processed:,} ({progress_pct:.1f}%)")
    
    conn.autocommit = True
    
    log(f"🎉 Merge complete! Total processed: {total_processed:,}")
    
    # Create indexes now
    log("🔄 Creating indexes...")
    create_indexes(conn, schema, final_table)
    
    # Final count check
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {schema}.{final_table};")
        final_count = cur.fetchone()[0]
    
    log(f"📊 Final table verified: {final_count:,} rows")

def create_indexes(conn, schema, table_name):
    """Create indexes on final table"""
    log("🔧 Creating indexes...")
    
    with conn.cursor() as cur:
        index_sqls = [
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {table_name}_content_norm_trgm ON {schema}.{table_name} USING gin (content_norm gin_trgm_ops);",
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {table_name}_content_clean_trgm ON {schema}.{table_name} USING gin (content_clean gin_trgm_ops);",
            f"CREATE INDEX IF NOT EXISTS {table_name}_tst_created_brin ON {schema}.{table_name} USING brin (tst_created);",
            f"CREATE INDEX IF NOT EXISTS {table_name}_company_id_btree ON {schema}.{table_name} (company_id);",
            f"CREATE INDEX IF NOT EXISTS {table_name}_company_name_btree ON {schema}.{table_name} (company_name);",
            f"CREATE INDEX IF NOT EXISTS {table_name}_dupgroup_btree ON {schema}.{table_name} (duplicate_group);",
        ]
        
        for i, idx_sql in enumerate(index_sqls):
            log(f"   Creating index {i+1}/{len(index_sqls)}...")
            cur.execute(idx_sql)
            conn.commit()
        
        cur.execute(f"ANALYZE {schema}.{table_name};")
        conn.commit()
    
    log("✅ Indexes created and table analyzed")

def run_qa_checks(conn, schema, table_name):
    """Run quality assurance checks"""
    log("🔍 Running QA checks...")
    
    def qdf(sql_query, params=None):
        return pd.read_sql_query(sql_query, conn, params=params)
    
    try:
        # Row counts by source
        result = qdf(f"""
            SELECT source_folder, COUNT(*) AS n
            FROM {schema}.{table_name}
            GROUP BY source_folder ORDER BY source_folder;
        """)
        log("📊 Row counts by source:")
        for _, row in result.iterrows():
            log(f"   {row['source_folder']}: {row['n']:,} rows")
        
        # Total count
        result = qdf(f"SELECT COUNT(*) as total FROM {schema}.{table_name};")
        total = result['total'].iloc[0]
        log(f"📊 Total rows: {total:,}")
        
        # Sample search test
        result = qdf(f"""
            SELECT COUNT(*) as ai_count
            FROM {schema}.{table_name}
            WHERE content_clean ILIKE '%machine learning%'
               OR content_norm ~* '\\yai\\y'
            LIMIT 1000;
        """)
        ai_count = result['ai_count'].iloc[0] 
        log(f"🔍 AI-related jobs found: {ai_count:,}")
        
    except Exception as e:
        log(f"⚠️ QA check error: {e}")
    
    log("✅ QA checks complete")

def cleanup_staging(conn, schema, table_name):
    """Clean up staging table"""
    log("🧹 Cleaning up staging table...")
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {schema}.{table_name};")
    conn.commit()
    log("✅ Staging table cleaned up")

if __name__ == "__main__":
    main()