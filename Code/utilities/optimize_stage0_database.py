#!/usr/bin/env python3
"""
Stage 0 Database Optimization Script

Based on test_stage0_index_performance.py results, this script:
1. Drops invalid and duplicate indexes
2. Rebuilds critical indexes with proper settings
3. Adds enhanced FTS with unaccent support
4. Optimizes database settings
5. Runs comprehensive cleanup

Run with: python3 optimize_stage0_database.py
"""

import psycopg2
import os
import time
from dotenv import load_dotenv

class Stage0DatabaseOptimizer:
    """Comprehensive database optimization for Stage 0 performance."""
    
    def __init__(self):
        """Initialize database connection."""
        # Load environment
        load_dotenv('config.env')
        
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        
        self.conn = psycopg2.connect(
            dbname=self.db_name,
            user=self.db_user, 
            password=self.db_password,
            host=self.db_host,
            port=self.db_port
        )
        
        # Enable autocommit for DDL operations
        self.conn.autocommit = True
        
        self.operations_completed = []
        self.operations_failed = []
        
    def close(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def execute_with_progress(self, operation_name, sql_command, expect_error=False):
        """Execute SQL with progress reporting and error handling."""
        print(f"\\n🔄 {operation_name}...")
        print(f"   SQL: {sql_command}")
        
        start_time = time.time()
        cur = self.conn.cursor()
        
        try:
            cur.execute(sql_command)
            elapsed = time.time() - start_time
            print(f"   ✅ Completed in {elapsed:.1f}s")
            self.operations_completed.append(f"{operation_name} ({elapsed:.1f}s)")
            return True
            
        except psycopg2.Error as e:
            elapsed = time.time() - start_time
            if expect_error:
                print(f"   ⚠️  Expected error (likely already done): {e}")
                self.operations_completed.append(f"{operation_name} (expected error)")
                return True
            else:
                print(f"   ❌ Failed after {elapsed:.1f}s: {e}")
                self.operations_failed.append(f"{operation_name}: {e}")
                return False
        finally:
            cur.close()
    
    def check_index_status(self):
        """Check current index status before optimization."""
        print("\\n" + "="*70)
        print("STAGE 0 DATABASE OPTIMIZATION")
        print("="*70)
        print("Checking current index status...")
        
        cur = self.conn.cursor()
        
        # Check table size
        cur.execute("SELECT COUNT(*) FROM job_postings;")
        total_rows = cur.fetchone()[0]
        print(f"Table size: {total_rows:,} rows")
        
        # Check invalid indexes
        cur.execute("""
            SELECT indexrelid::regclass AS index_name, indisvalid, indisready
            FROM pg_index
            WHERE indrelid = 'job_postings'::regclass AND indisvalid = false;
        """)
        
        invalid_indexes = cur.fetchall()
        if invalid_indexes:
            print(f"\\n❌ Invalid indexes to fix: {len(invalid_indexes)}")
            for idx_name, valid, ready in invalid_indexes:
                print(f"   {idx_name}: valid={valid}, ready={ready}")
        else:
            print("\\n✅ No invalid indexes detected")
        
        # Check usage stats for content_norm indexes
        cur.execute("""
            SELECT indexrelid::regclass AS index_name, idx_scan
            FROM pg_stat_user_indexes
            WHERE relname = 'job_postings'
              AND indexrelid::regclass::text LIKE '%content_norm_trgm%'
            ORDER BY idx_scan DESC;
        """)
        
        usage_stats = cur.fetchall()
        print(f"\\nContent_norm trigram index usage:")
        for idx_name, scan_count in usage_stats:
            print(f"   {idx_name}: {scan_count:,} scans")
        
        cur.close()
    
    def step_1_drop_invalid_duplicate_indexes(self):
        """Step 1: Drop invalid and duplicate indexes."""
        print(f"\\n{'='*70}")
        print("STEP 1: DROPPING INVALID AND DUPLICATE INDEXES")
        print(f"{'='*70}")
        
        # Drop the unused duplicate trigram index (0 scans vs 47 scans)
        self.execute_with_progress(
            "Drop duplicate trigram index",
            "DROP INDEX CONCURRENTLY IF EXISTS idx_job_postings_content_norm_trgm;",
            expect_error=True
        )
        
        # Drop broken FTS index (16kB - clearly invalid)
        self.execute_with_progress(
            "Drop broken FTS index",
            "DROP INDEX CONCURRENTLY IF EXISTS idx_job_postings_content_clean_gin;",
            expect_error=True
        )
        
        # Drop unused trigram index on content_clean
        self.execute_with_progress(
            "Drop unused content_clean trigram",
            "DROP INDEX CONCURRENTLY IF EXISTS idx_job_postings_content_clean_trgm;",
            expect_error=True
        )
        
        # Drop unused company_name index
        self.execute_with_progress(
            "Drop unused company_name index",
            "DROP INDEX CONCURRENTLY IF EXISTS idx_job_postings_company_name;",
            expect_error=True
        )
    
    def step_2_optimize_settings(self):
        """Step 2: Optimize database settings for index operations."""
        print(f"\\n{'='*70}")
        print("STEP 2: OPTIMIZING DATABASE SETTINGS")
        print(f"{'='*70}")
        
        # Boost maintenance memory for index builds
        self.execute_with_progress(
            "Set maintenance_work_mem to 2GB",
            "SET maintenance_work_mem = '2GB';"
        )
        
        # Enable parallel maintenance workers
        self.execute_with_progress(
            "Enable parallel maintenance workers",
            "SET max_parallel_maintenance_workers = 4;"
        )
        
        # Optimize for parallel scans
        self.execute_with_progress(
            "Enable parallel query workers",
            "SET max_parallel_workers_per_gather = 4;"
        )
        
        self.execute_with_progress(
            "Optimize parallel tuple cost",
            "SET parallel_tuple_cost = 0.05;"
        )
    
    def step_3_rebuild_critical_indexes(self):
        """Step 3: Rebuild critical indexes."""
        print(f"\\n{'='*70}")
        print("STEP 3: REBUILDING CRITICAL INDEXES")
        print(f"{'='*70}")
        
        # Ensure extensions exist
        self.execute_with_progress(
            "Create pg_trgm extension",
            "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
        )
        
        self.execute_with_progress(
            "Create unaccent extension",
            "CREATE EXTENSION IF NOT EXISTS unaccent;"
        )
        
        # Reindex the working trigram index to ensure validity
        self.execute_with_progress(
            "Rebuild content_norm trigram index",
            "REINDEX INDEX CONCURRENTLY job_postings_content_norm_trgm;"
        )
        
        # Create enhanced FTS index with unaccent
        print(f"\\n🔄 Creating enhanced FTS index with unaccent support...")
        self.execute_with_progress(
            "Create enhanced FTS index",
            """CREATE INDEX CONCURRENTLY idx_job_postings_content_clean_fts
               ON job_postings 
               USING gin (to_tsvector('english', unaccent(content_clean)));"""
        )
    
    def step_4_add_brin_index(self):
        """Step 4: Add space-efficient BRIN index for date ranges."""
        print(f"\\n{'='*70}")
        print("STEP 4: ADDING BRIN INDEX FOR DATE OPTIMIZATION")
        print(f"{'='*70}")
        
        # Add BRIN index for efficient date range scans
        self.execute_with_progress(
            "Create BRIN index on tst_created",
            """CREATE INDEX CONCURRENTLY idx_job_postings_tst_created_brin
               ON job_postings 
               USING brin (tst_created) WITH (pages_per_range = 64);"""
        )
    
    def step_5_vacuum_analyze(self):
        """Step 5: Comprehensive cleanup."""
        print(f"\\n{'='*70}")
        print("STEP 5: COMPREHENSIVE CLEANUP")
        print(f"{'='*70}")
        
        # Full vacuum analyze to clean up after index changes
        print(f"\\n🔄 Running VACUUM ANALYZE (this may take several minutes)...")
        self.execute_with_progress(
            "VACUUM ANALYZE job_postings",
            "VACUUM (ANALYZE, VERBOSE) job_postings;"
        )
    
    def verify_optimization_results(self):
        """Verify the optimization results."""
        print(f"\\n{'='*70}")
        print("VERIFICATION: POST-OPTIMIZATION STATUS")
        print(f"{'='*70}")
        
        cur = self.conn.cursor()
        
        # Check final index status
        cur.execute("""
            SELECT indexrelid::regclass AS index_name, indisvalid, 
                   pg_size_pretty(pg_relation_size(indexrelid)) as size
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indrelid
            WHERE c.relname = 'job_postings'
            ORDER BY pg_relation_size(indexrelid) DESC;
        """)
        
        indexes = cur.fetchall()
        print("\\nFinal index status:")
        total_index_size = 0
        valid_indexes = 0
        
        for idx_name, valid, size in indexes:
            status = "✅ Valid" if valid else "❌ Invalid"
            print(f"   {idx_name}: {status} - {size}")
            
            if valid:
                valid_indexes += 1
                # Convert size to bytes for total calculation
                cur.execute("SELECT pg_relation_size(%s);", [idx_name])
                size_bytes = cur.fetchone()[0]
                total_index_size += size_bytes
        
        print(f"\\n📊 Summary:")
        print(f"   Valid indexes: {valid_indexes}")
        print(f"   Total index size: {total_index_size / (1024**3):.1f} GB")
        
        # Check for remaining invalid indexes
        cur.execute("""
            SELECT COUNT(*) FROM pg_index i
            JOIN pg_class c ON c.oid = i.indrelid
            WHERE c.relname = 'job_postings' AND i.indisvalid = false;
        """)
        
        invalid_count = cur.fetchone()[0]
        if invalid_count == 0:
            print(f"   ✅ No invalid indexes remaining")
        else:
            print(f"   ⚠️  {invalid_count} invalid indexes still present")
        
        cur.close()
    
    def generate_final_report(self):
        """Generate final optimization report."""
        print(f"\\n{'='*70}")
        print("OPTIMIZATION COMPLETE - FINAL REPORT")
        print(f"{'='*70}")
        
        print(f"\\n✅ Successful operations ({len(self.operations_completed)}):")
        for op in self.operations_completed:
            print(f"   ✅ {op}")
        
        if self.operations_failed:
            print(f"\\n❌ Failed operations ({len(self.operations_failed)}):")
            for op in self.operations_failed:
                print(f"   ❌ {op}")
        
        success_rate = len(self.operations_completed) / (len(self.operations_completed) + len(self.operations_failed)) * 100
        print(f"\\n📊 Success rate: {success_rate:.1f}%")
        
        if len(self.operations_failed) == 0:
            print(f"\\n🎉 ALL OPTIMIZATIONS COMPLETED SUCCESSFULLY!")
            print(f"\\nNext steps:")
            print(f"1. Run: python3 tests/test_stage0_index_performance.py")
            print(f"2. Expected improvements:")
            print(f"   - Machine learning queries: 21s → <2s")
            print(f"   - Storage reduction: ~8GB saved") 
            print(f"   - No invalid indexes")
            print(f"   - Enhanced multilingual FTS support")
            print(f"\\n💡 Consider reducing Stage 0 batch size:")
            print(f"   extract_keyword_matching_jobs_batched(batch_months=3)")
            return True
        else:
            print(f"\\n⚠️  SOME OPERATIONS FAILED - Review errors above")
            return False


def run_optimization():
    """Main optimization runner with proper error handling."""
    optimizer = Stage0DatabaseOptimizer()
    
    try:
        # Pre-flight check
        optimizer.check_index_status()
        
        # Execute optimization steps
        optimizer.step_1_drop_invalid_duplicate_indexes()
        optimizer.step_2_optimize_settings() 
        optimizer.step_3_rebuild_critical_indexes()
        optimizer.step_4_add_brin_index()
        optimizer.step_5_vacuum_analyze()
        
        # Verification
        optimizer.verify_optimization_results()
        
        # Final report
        success = optimizer.generate_final_report()
        return success
        
    except Exception as e:
        print(f"\\n❌ OPTIMIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        optimizer.close()


if __name__ == "__main__":
    print("Stage 0 Database Optimization")
    print("This will drop invalid indexes and rebuild critical ones.")
    
    response = input("\\nProceed with optimization? (y/N): ")
    if response.lower() in ['y', 'yes']:
        success = run_optimization()
        exit(0 if success else 1)
    else:
        print("Optimization cancelled.")
        exit(0)