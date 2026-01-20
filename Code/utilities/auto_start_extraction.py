#!/usr/bin/env python3
"""
Monitor index creation and automatically start extraction when complete
"""
import psycopg2
import os
import time
import subprocess
from dotenv import load_dotenv

def check_index_exists(conn, index_name):
    """Check if the specified index exists and is valid"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE indexname = %s AND tablename = 'job_postings_unified'
            """, (index_name,))
            return cur.fetchone() is not None
    except Exception as e:
        print(f"Error checking index: {e}")
        return False

def check_index_building(conn, index_name):
    """Check if the index is currently being built"""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pid, query FROM pg_stat_activity 
                WHERE state = 'active' 
                  AND query ILIKE %s
            """, (f'%{index_name}%',))
            return cur.fetchone() is not None
    except Exception as e:
        print(f"Error checking active builds: {e}")
        return False

def main():
    print("🔍 AUTO-START MONITOR: Waiting for index completion...")
    print("=" * 60)
    
    # Connect to database
    load_dotenv('config.env')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER') 
    db_password = os.getenv('DB_PASSWORD')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    
    conn = psycopg2.connect(
        dbname=db_name, user=db_user, password=db_password,
        host=db_host, port=db_port
    )
    
    target_index = 'ju_content_norm_trgm'
    check_interval = 30  # Check every 30 seconds
    
    try:
        start_time = time.time()
        
        while True:
            # Check if index exists
            index_exists = check_index_exists(conn, target_index)
            index_building = check_index_building(conn, target_index)
            
            elapsed_minutes = (time.time() - start_time) / 60
            
            if index_exists and not index_building:
                print(f"✅ INDEX COMPLETE! ({elapsed_minutes:.1f} minutes elapsed)")
                print("🚀 Starting batched extraction with enhanced monitoring...")
                
                # Start the extraction process
                result = subprocess.run([
                    'python3', 'stage_0_get_job_ads.py', '--extract-keywords-batched'
                ], capture_output=False, text=True)
                
                if result.returncode == 0:
                    print("✅ Extraction completed successfully!")
                else:
                    print(f"❌ Extraction failed with code {result.returncode}")
                break
                
            elif index_building:
                print(f"🔄 Index still building... ({elapsed_minutes:.1f} minutes elapsed)")
            else:
                print(f"⏳ Waiting for index creation to start... ({elapsed_minutes:.1f} minutes elapsed)")
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\\n🛑 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()