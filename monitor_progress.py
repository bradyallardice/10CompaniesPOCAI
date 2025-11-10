#!/usr/bin/env python3
"""
Monitor progress of the unified job postings table creation
"""

import os
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

def main():
    print(f"🔍 Progress Check - {datetime.now().strftime('%H:%M:%S')}")
    
    # Connect to database
    load_dotenv("config.env")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"), 
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    conn.autocommit = True
    
    try:
        # Check if tables exist
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('job_postings_unified', 'job_postings_unified_staging')
                ORDER BY table_name;
            """)
            tables = [row[0] for row in cur.fetchall()]
        
        print(f"📊 Tables found: {tables}")
        
        # Check staging table
        if 'job_postings_unified_staging' in tables:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM public.job_postings_unified_staging;")
                staging_count = cur.fetchone()[0]
            print(f"📁 Staging table: {staging_count:,} rows")
        else:
            staging_count = 0
            print("📁 Staging table: Not created yet")
        
        # Check final table
        if 'job_postings_unified' in tables:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM public.job_postings_unified;")
                final_count = cur.fetchone()[0]
            print(f"✅ Final table: {final_count:,} rows")
            
            if staging_count > 0:
                progress = (final_count / staging_count) * 100 if staging_count > 0 else 0
                print(f"📈 Progress: {progress:.1f}% ({final_count:,} / {staging_count:,})")
        else:
            final_count = 0
            print("✅ Final table: Not created yet")
        
        # Check for active operations
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pid, state, query_start, now() - query_start as duration,
                       left(query, 100) as query_snippet
                FROM pg_stat_activity 
                WHERE state = 'active' 
                AND query NOT LIKE '%pg_stat_activity%'
                AND pid != pg_backend_pid()
                ORDER BY query_start;
            """)
            
            active_queries = cur.fetchall()
            
        if active_queries:
            print(f"\n🔄 Active operations: {len(active_queries)}")
            for pid, state, start, duration, query in active_queries:
                print(f"   PID {pid}: {state} for {duration} - {query}...")
        else:
            print("\n💤 No active operations detected")
        
        # If we have data in final table, show some sample stats
        if final_count > 0:
            print(f"\n📋 Sample data from final table:")
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT source_folder, COUNT(*) as count
                    FROM public.job_postings_unified 
                    GROUP BY source_folder 
                    ORDER BY source_folder;
                """)
                results = cur.fetchall()
                for folder, count in results:
                    print(f"   {folder}: {count:,} rows")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    main()