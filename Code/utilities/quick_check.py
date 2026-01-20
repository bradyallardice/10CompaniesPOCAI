#!/usr/bin/env python3
import os, psycopg2
from dotenv import load_dotenv

load_dotenv("config.env")
try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"), connect_timeout=5
    )
    conn.autocommit = True
    
    with conn.cursor() as cur:
        # Quick check with timeout
        cur.execute("SELECT 'OK' as status;")
        print("✅ Database responsive")
        
        # Check tables existence
        cur.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name IN ('job_postings_unified', 'job_postings_unified_staging');")
        table_count = cur.fetchone()[0]
        print(f"📋 Tables created: {table_count}/2")
        
    conn.close()
except Exception as e:
    print(f"❌ Database issue: {e}")