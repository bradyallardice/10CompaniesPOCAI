#!/usr/bin/env python3
"""
DIRECT TABLE CREATION - Skip all checks, just do the work
"""

import os, sys, time, gzip, json, re, unicodedata, hashlib
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

def log(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def main():
    log("🚀 DIRECT TABLE CREATION - No checks, just work!")
    
    # Quick connection
    load_dotenv("config.env")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"), connect_timeout=10
    )
    conn.autocommit = True
    log("✅ Connected")
    
    # Just create the final table and load data directly
    log("🗂️ Creating final table...")
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS unaccent;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        cur.execute("""
            CREATE OR REPLACE FUNCTION public.content_normalize(t text)
            RETURNS text LANGUAGE sql IMMUTABLE AS $$
              SELECT CASE WHEN t IS NULL THEN NULL ELSE
                trim(regexp_replace(regexp_replace(lower(unaccent(t)),
                '[^[:alnum:]\\s]+', ' ', 'g'), '\\s+', ' ', 'g')) END
            $$;
        """)
        
        cur.execute("DROP TABLE IF EXISTS public.job_postings_unified CASCADE;")
        cur.execute("""
            CREATE TABLE public.job_postings_unified (
                uid TEXT PRIMARY KEY, title TEXT, company_id TEXT, company_name TEXT,
                company_is_recruiter BOOLEAN, company_size TEXT, cantons TEXT[],
                x28_industries TEXT[], x28_occupations TEXT[], location_raw TEXT,
                url TEXT, content_clean TEXT,
                content_norm TEXT GENERATED ALWAYS AS (public.content_normalize(content_clean)) STORED,
                duplicate_group UUID, tst_created TIMESTAMPTZ, tst_deleted TIMESTAMPTZ,
                source_folder TEXT NOT NULL, source_file TEXT NOT NULL
            );
        """)
    log("✅ Table created")
    
    # Load data directly into final table
    load_data_direct(conn)
    
    # Create indexes
    log("🔧 Creating indexes...")
    with conn.cursor() as cur:
        cur.execute("CREATE INDEX IF NOT EXISTS ju_content_trgm ON public.job_postings_unified USING gin (content_clean gin_trgm_ops);")
        cur.execute("CREATE INDEX IF NOT EXISTS ju_content_norm_trgm ON public.job_postings_unified USING gin (content_norm gin_trgm_ops);")
        cur.execute("CREATE INDEX IF NOT EXISTS ju_company ON public.job_postings_unified (company_name);")
        cur.execute("CREATE INDEX IF NOT EXISTS ju_date ON public.job_postings_unified USING brin (tst_created);")
        cur.execute("ANALYZE public.job_postings_unified;")
    log("✅ Indexes created")
    
    # Final count
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.job_postings_unified;")
        count = cur.fetchone()[0]
    log(f"🎉 COMPLETE! Final table has {count:,} rows")
    
    conn.close()

def first_non_null(d, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""): return d[k]
    return None

def synthesize_uid(*parts):
    key = "|".join("" if p is None else str(p) for p in parts)
    return "surr_" + hashlib.sha1(key.encode("utf-8")).hexdigest()

def choose_uid(obj):
    company_id = first_non_null(obj, "company_id")
    company_name = first_non_null(obj, "company_name", "companyName")  
    url = first_non_null(obj, "url")
    title = first_non_null(obj, "title")
    created = first_non_null(obj, "tst_created", "created_at")
    
    combos = [(company_id, url, created), (company_name, url, created), (company_name, title, url)]
    for tup in combos:
        if all(x not in (None, "") for x in tup):
            return synthesize_uid(*tup)
    return None

def normalize_text(text):
    if not text: return None
    t = unicodedata.normalize("NFD", str(text)).encode("ascii","ignore").decode("ascii")
    return re.sub(r"\s+", " ", t.lower().strip())

def parse_array_field(value):
    """Convert string array representations to proper PostgreSQL arrays"""
    if not value or value in ('null', 'None'):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Handle string representations like "['BE', 'LU']" 
        try:
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
        # Handle comma-separated strings
        if ',' in value:
            return [x.strip().strip("'\"") for x in value.split(',') if x.strip()]
    return None

def build_row(obj, folder, file):
    uid = str(first_non_null(obj, "uid", "id") or choose_uid(obj))
    title = first_non_null(obj, "title")
    company = first_non_null(obj, "company_name", "companyName")
    url = first_non_null(obj, "url")
    created = first_non_null(obj, "tst_created", "created_at")
    
    if not all([uid, title, company, url, created]):
        return None
        
    return (
        uid, title, obj.get("company_id"), company,
        obj.get("company_is_recruiter"), obj.get("company_size"),
        parse_array_field(obj.get("cantons")),
        parse_array_field(obj.get("x28_industries")),
        parse_array_field(obj.get("x28_occupations")),
        obj.get("location_raw"), url, obj.get("content_clean"),
        str(obj.get("duplicate_group")) if obj.get("duplicate_group") else None,
        created, obj.get("tst_deleted"), folder, file
    )

def ndjson_stream(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try: yield json.loads(line)
                except: continue

def load_data_direct(conn):
    log("🚀 Loading data directly...")

    BASE = Path(__file__).parent
    folders = [
        BASE / "240826_panel_data_with_dg",
        BASE / "250826_panel_data_01012024_30062025"
    ]
    
    sql = """INSERT INTO public.job_postings_unified (
        uid, title, company_id, company_name, company_is_recruiter, company_size,
        cantons, x28_industries, x28_occupations, location_raw, url, content_clean,
        duplicate_group, tst_created, tst_deleted, source_folder, source_file
    ) VALUES %s ON CONFLICT (uid) DO NOTHING"""
    
    total_processed = 0
    
    for folder in folders:
        if not folder.exists():
            log(f"⚠️ Skipping {folder.name}")
            continue
            
        files = list(folder.rglob("*.json.gzip"))
        log(f"📁 Processing {folder.name}: {len(files)} files")
        
        batch = []
        processed = 0
        
        for i, file_path in enumerate(files):
            if i % 20 == 0:
                log(f"   File {i+1}/{len(files)}: {file_path.name} (processed: {processed:,})")
            
            for obj in ndjson_stream(file_path):
                row = build_row(obj, folder.name, file_path.name)
                if row:
                    batch.append(row)
                    processed += 1
                    total_processed += 1
                    
                if len(batch) >= 10000:
                    with conn.cursor() as cur:
                        execute_values(cur, sql, batch, page_size=5000)
                    batch.clear()
        
        if batch:
            with conn.cursor() as cur:
                execute_values(cur, sql, batch, page_size=5000)
        
        log(f"✅ {folder.name}: {processed:,} rows processed")
    
    log(f"📊 TOTAL: {total_processed:,} rows processed")

if __name__ == "__main__":
    main()