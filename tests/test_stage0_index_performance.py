#!/usr/bin/env python3
"""
Stage 0 Index Performance Test Suite

Tight, repeatable verification that the right indexes exist, that Postgres 
actually uses them, and that they give Stage-0 speedups on ~14M row table.

Testing:
- job_postings_content_norm_trgm (GIN w/ gin_trgm_ops on content_norm)  
- job_postings_pkey on uid
- idx_job_postings_tst_created (btree on tst_created)

Verifies:
1. Indexes exist and have right operator classes
2. Queries mirror code paths and use indexes (no seq scans)
3. Performance within sane targets (cold vs warm cache)
4. Index + date filter combine via BitmapAnd (batching helps)
5. Short-token edge cases (ai → "a i") don't tank into seq scans

Run with: python3 tests/test_stage0_index_performance.py
"""

import psycopg2
import os
import time
import re
import csv
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv('config.env')

class TestStage0IndexPerformance:
    """Tight verification of Stage 0 database index performance."""
    
    def __init__(self):
        """Initialize database connection and test results storage."""
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
        
        self.results = []  # Store query_id, plan_nodes, total_time_ms, shared_hit_pct
        self.failures = []  # Store failed assertions
        
    def close(self):
        """Clean up database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def parse_explain_output(self, explain_lines):
        """Parse EXPLAIN ANALYZE output for key metrics."""
        plan_text = '\n'.join([line[0] for line in explain_lines])
        
        # Extract metrics
        has_seq_scan = 'Seq Scan on job_postings' in plan_text
        has_bitmap_index = 'Bitmap Index Scan' in plan_text
        has_bitmap_and = 'BitmapAnd' in plan_text
        
        # Extract execution time and buffer stats from the full plan
        exec_time_ms = None
        buffer_hit_pct = None
        total_hits = 0
        total_reads = 0
        
        for line in explain_lines:
            line_text = line[0]
            
            # Look for execution time at the end of the plan
            if 'Execution Time:' in line_text or 'Execution time:' in line_text:
                match = re.search(r'Execution [Tt]ime: ([\d.]+) ms', line_text)
                if match:
                    exec_time_ms = float(match.group(1))
            
            # Aggregate all buffer stats from all nodes
            if 'shared hit=' in line_text or 'shared read=' in line_text:
                hit_match = re.search(r'shared hit=(\d+)', line_text)
                read_match = re.search(r'shared read=(\d+)', line_text)
                if hit_match:
                    total_hits += int(hit_match.group(1))
                if read_match:
                    total_reads += int(read_match.group(1))
        
        # Calculate overall hit percentage
        total_buffers = total_hits + total_reads
        if total_buffers > 0:
            buffer_hit_pct = (total_hits / total_buffers * 100)
        
        return {
            'has_seq_scan': has_seq_scan,
            'has_bitmap_index': has_bitmap_index,
            'has_bitmap_and': has_bitmap_and,
            'exec_time_ms': exec_time_ms,
            'buffer_hit_pct': buffer_hit_pct,
            'plan_text': plan_text
        }
    
    def run_query_test(self, query_id, query, expect_no_seq_scan=True, expect_bitmap_and=False, 
                       max_time_ms=None, warm_run=True):
        """Run a single query test with analysis."""
        cur = self.conn.cursor()
        
        # Warm run if requested
        if warm_run:
            cur.execute(query)
            cur.fetchall()
            
        # Actual test with EXPLAIN ANALYZE 
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, TIMING ON) {query}"
        cur.execute(explain_query)
        explain_output = cur.fetchall()
        
        # Parse results
        metrics = self.parse_explain_output(explain_output)
        
        # Store results
        self.results.append({
            'query_id': query_id,
            'exec_time_ms': metrics['exec_time_ms'],
            'buffer_hit_pct': metrics['buffer_hit_pct'],
            'has_seq_scan': metrics['has_seq_scan'],
            'has_bitmap_index': metrics['has_bitmap_index'],
            'has_bitmap_and': metrics['has_bitmap_and']
        })
        
        # Check assertions and record failures
        failure_count_before = len(self.failures)
        
        if expect_no_seq_scan and metrics['has_seq_scan']:
            self.failures.append(f"{query_id}: Contains Seq Scan on job_postings - index not used")
            
        if expect_bitmap_and and not metrics['has_bitmap_and']:
            self.failures.append(f"{query_id}: Missing BitmapAnd for combined index use")
            
        if max_time_ms and metrics['exec_time_ms'] and metrics['exec_time_ms'] > max_time_ms:
            self.failures.append(f"{query_id}: Execution time {metrics['exec_time_ms']:.1f}ms > {max_time_ms}ms threshold")
        
        # Special check: if we expected index use but got seq scan, that's critical
        if not metrics['has_bitmap_index'] and not metrics['has_seq_scan']:
            self.failures.append(f"{query_id}: No recognizable index scan method - may be using unexpected plan")
        
        # Print result
        has_failures = len(self.failures) > failure_count_before
        status = "❌ FAIL" if has_failures else "✅ PASS"
        time_str = f"{metrics['exec_time_ms']:.1f}ms" if metrics['exec_time_ms'] else "N/A"
        hit_str = f"{metrics['buffer_hit_pct']:.1f}%" if metrics['buffer_hit_pct'] is not None else "N/A"
        
        scan_type = "Bitmap" if metrics['has_bitmap_index'] else ("SeqScan" if metrics['has_seq_scan'] else "Other")
        if metrics['has_bitmap_and']:
            scan_type += "+And"
        
        print(f"  {status} {query_id:25} | {time_str:>8} | Hit: {hit_str:>6} | {scan_type:>10}")
        
        # Print failure details immediately if failed
        if has_failures:
            new_failures = self.failures[failure_count_before:]
            for failure in new_failures:
                print(f"    ⚠️  {failure.split(': ', 1)[1]}")
        
        cur.close()
        return metrics
    
    def test_1_pre_flight_checks(self):
        """1) Pre-flight SQL (sanity + definitions)"""
        print(f"\n{'='*70}")
        print("1) PRE-FLIGHT: INDEX DEFINITIONS AND SIZES")
        print(f"{'='*70}")
        
        cur = self.conn.cursor()
        
        # Ensure pg_trgm extension exists
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        
        # Check index validity first
        cur.execute("""
            SELECT indexrelid::regclass AS index_name, indisvalid, indisready
            FROM pg_index
            WHERE indrelid = 'job_postings'::regclass AND indisvalid = false;
        """)
        
        invalid_indexes = cur.fetchall()
        if invalid_indexes:
            print("❌ INVALID INDEXES DETECTED:")
            for idx_name, valid, ready in invalid_indexes:
                print(f"  {idx_name}: valid={valid}, ready={ready}")
            self.failures.append("Invalid indexes found - rebuild required")
        else:
            print("✅ All indexes valid")
        
        # Check indexes exist with right definitions
        cur.execute("""
            SELECT schemaname, indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'job_postings'
            ORDER BY indexname;
        """)
        
        indexes = cur.fetchall()
        print("\nCurrent indexes on job_postings:")
        
        critical_indexes = {
            'content_norm_trgm': False,
            'uid_pkey': False,
            'tst_created_btree': False
        }
        
        # Track duplicate indexes
        content_norm_trgm_indexes = []
        
        for schema, idx_name, idx_def in indexes:
            print(f"  {idx_name}")
            print(f"    {idx_def}")
            
            # Check for expected indexes
            if 'content_norm' in idx_def and 'gin_trgm_ops' in idx_def:
                critical_indexes['content_norm_trgm'] = True
                content_norm_trgm_indexes.append(idx_name)
            if 'uid' in idx_def and ('PRIMARY KEY' in idx_def or 'UNIQUE' in idx_def):
                critical_indexes['uid_pkey'] = True
            if 'tst_created' in idx_def and 'btree' in idx_def:
                critical_indexes['tst_created_btree'] = True
        
        # Check for duplicate content_norm trigram indexes
        if len(content_norm_trgm_indexes) > 1:
            print(f"\n⚠️  DUPLICATE TRIGRAM INDEXES ON content_norm:")
            for idx_name in content_norm_trgm_indexes:
                print(f"  - {idx_name}")
            print(f"   Consider keeping only the most-used and dropping the others.")
            print(f"   Use: SELECT indexrelid::regclass, idx_scan FROM pg_stat_user_indexes WHERE relname = 'job_postings'")
            self.failures.append(f"Duplicate trigram indexes: {content_norm_trgm_indexes}")
        
        # Check index usage stats
        cur.execute("""
            SELECT indexrelid::regclass AS index_name, idx_scan, idx_tup_read, idx_tup_fetch
            FROM pg_stat_user_indexes
            WHERE relname = 'job_postings'
            AND indexrelid::regclass::text LIKE '%content_norm%'
            ORDER BY idx_scan DESC;

        """)
        
        usage_stats = cur.fetchall()
        if usage_stats:
            print(f"\nContent_norm index usage stats:")
            for idx_name, scan_count, last_vac, last_analyze in usage_stats:
                print(f"  {idx_name}: {scan_count:,} scans, vacuum: {last_vac}, analyze: {last_analyze}")
                
                # Check for suspicious FTS index
                if 'content_clean_gin' in str(idx_name):
                    cur.execute("""
                        SELECT pg_size_pretty(pg_relation_size(%s::regclass)) as size,
                               pg_relation_size(%s::regclass) as size_bytes
                    """, [idx_name, idx_name])
                    size_info = cur.fetchone()
                    if size_info and size_info[1] < 1000000:  # Less than 1MB
                        print(f"  ⚠️  {idx_name} is suspiciously small ({size_info[0]}) - may be invalid/unused")
                        self.failures.append(f"Suspicious small FTS index: {idx_name}")
        
        # Check for unused indexes (scan count = 0)
        cur.execute("""
            SELECT indexrelid::regclass AS index_name, idx_scan
            FROM pg_stat_user_indexes
            WHERE relname = 'job_postings' AND idx_scan = 0
              AND indexrelid::regclass::text NOT LIKE '%pkey%';
        """)
        
        unused_indexes = cur.fetchall()
        if unused_indexes:
            print(f"\n⚠️  UNUSED INDEXES (idx_scan = 0):")
            for idx_name, scan_count in unused_indexes:
                print(f"  {idx_name}")
            print(f"   Consider dropping if they remain unused long-term")
        
        # Check sizes
        cur.execute("""
            SELECT
              relname AS object,
              pg_size_pretty(pg_relation_size(oid)) AS size,
              pg_relation_size(oid) as size_bytes
            FROM pg_class
            WHERE relname LIKE '%job_postings%'
              OR relname LIKE '%content_norm%'  
              OR relname LIKE '%tst_created%'
            ORDER BY pg_relation_size(oid) DESC;
        """)
        
        sizes = cur.fetchall()
        print(f"\nObject sizes:")
        for obj, size_pretty, size_bytes in sizes:
            print(f"  {obj:40} | {size_pretty}")
        
        # Verify critical indexes
        print(f"\nCritical indexes check:")
        all_good = True
        for idx, exists in critical_indexes.items():
            status = "✅ EXISTS" if exists else "❌ MISSING"
            print(f"  {idx:25} | {status}")
            if not exists:
                all_good = False
                self.failures.append(f"Critical index missing: {idx}")
        
        cur.close()
        
        if not all_good:
            print("❌ CRITICAL: Missing required indexes!")
            return False
            
        print("✅ All critical indexes present")
        return True
    
    def test_2_query_plans(self):
        """2) Does the planner use my indexes? (plans, no execution)"""
        print(f"\n{'='*70}")
        print("2) QUERY PLAN VERIFICATION (EXPLAIN only)")
        print(f"{'='*70}")
        
        cur = self.conn.cursor()
        
        test_queries = [
            ("ILIKE multiword", "SELECT uid FROM job_postings WHERE content_norm ILIKE '%machine learning%'"),
            ("Regex word boundary", "SELECT uid FROM job_postings WHERE content_norm ~* '\\\\yai\\\\y'"),
            ("Spaced letters (a.i.)", "SELECT uid FROM job_postings WHERE content_norm ~* '\\\\ya i\\\\y'"),
            ("Date+content combined", """SELECT uid FROM job_postings 
                WHERE tst_created >= DATE '2024-01-01' 
                AND tst_created < DATE '2024-04-01'
                AND (content_norm ILIKE '%machine learning%' OR content_norm ~* '\\\\yai\\\\y')""")
        ]
        
        print("Plan verification (no execution):")
        for test_name, query in test_queries:
            cur.execute(f"EXPLAIN {query}")
            plan_lines = cur.fetchall()
            plan_text = '\n'.join([line[0] for line in plan_lines])
            
            has_seq_scan = 'Seq Scan on job_postings' in plan_text
            has_bitmap = 'Bitmap' in plan_text
            has_bitmap_and = 'BitmapAnd' in plan_text
            
            status = "✅" if not has_seq_scan else "❌"
            scan_type = "Bitmap" if has_bitmap else "SeqScan" if has_seq_scan else "Other"
            bitmap_and = "BitmapAnd" if has_bitmap_and else ""
            
            print(f"  {status} {test_name:20} | {scan_type:>8} {bitmap_and}")
            
            if has_seq_scan:
                self.failures.append(f"{test_name}: Plan shows Seq Scan on job_postings")
        
        cur.close()
    
    def test_3_performance_with_targets(self):
        """3) How fast is it really? (EXPLAIN ANALYZE + BUFFERS)"""  
        print(f"\n{'='*70}")
        print("3) PERFORMANCE WITH TARGETS (warm cache)")
        print(f"{'='*70}")
        
        # Warm the cache first
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM job_postings WHERE tst_created >= DATE '2024-01-01' AND tst_created < DATE '2024-04-01'")
        warm_count = cur.fetchone()[0]
        print(f"Cache warmed with Q1 2024 window: {warm_count:,} rows")
        cur.close()
        
        print(f"\nQuery performance tests:")
        print(f"{'Status':<6} {'Test Name':<25} | {'Time':>8} | {'Hit Rate':>6} | {'Plan':>8}")
        print("-" * 70)
        
        # Target: content-only often < 1.5s warm, date-batched selective < 800ms
        
        # 3.1 ILIKE common phrase
        self.run_query_test(
            "ILIKE_machine_learning",
            "SELECT COUNT(*) FROM job_postings WHERE content_norm ILIKE '%machine learning%'",
            expect_no_seq_scan=True,
            max_time_ms=1500
        )
        
        # 3.2 Regex short term with word boundary  
        self.run_query_test(
            "regex_ai_boundary",
            "SELECT COUNT(*) FROM job_postings WHERE content_norm ~* '\\\\yai\\\\y'",
            expect_no_seq_scan=True,
            max_time_ms=1500
        )
        
        # 3.3 Spaced letters (from A.I.)
        self.run_query_test(
            "regex_spaced_ai", 
            "SELECT COUNT(*) FROM job_postings WHERE content_norm ~* '\\\\ya i\\\\y'",
            expect_no_seq_scan=True,
            max_time_ms=1500
        )
        
        # 3.4 Date-batched + content (rare keyword) - should be very fast
        self.run_query_test(
            "date_batch_rare",
            """SELECT COUNT(*) FROM job_postings 
               WHERE tst_created >= DATE '2024-01-01' AND tst_created < DATE '2024-04-01'
               AND content_norm ~* '\\\\ylangzeit kurzzeit gedachtnisnetzwerk\\\\y'""",
            expect_no_seq_scan=True,
            expect_bitmap_and=True,
            max_time_ms=800
        )
        
        # 3.5 Date-batched + content (common phrase)
        self.run_query_test(
            "date_batch_common",
            """SELECT COUNT(*) FROM job_postings 
               WHERE tst_created >= DATE '2024-01-01' AND tst_created < DATE '2024-04-01'
               AND content_norm ILIKE '%data science%'""",
            expect_no_seq_scan=True,
            expect_bitmap_and=True,
            max_time_ms=2000
        )
    
    def test_4_bitmap_and_benefit(self):
        """4) Prove the BitmapAnd benefit (date + trigram)"""
        print(f"\n{'='*70}")
        print("4) BITMAP AND BENEFIT VERIFICATION")
        print(f"{'='*70}")
        
        print("Comparing content-only vs date-batched performance:")
        print(f"{'Status':<6} {'Test Name':<25} | {'Time':>8} | {'Hit Rate':>6} | {'Plan':>8}")
        print("-" * 70)
        
        # Content-only (forces scan over all 14M rows via trigram)
        metrics_full = self.run_query_test(
            "content_only_full_table",
            "SELECT COUNT(*) FROM job_postings WHERE content_norm ILIKE '%large language model%'",
            expect_no_seq_scan=True,
            max_time_ms=8000  # More lenient for full table
        )
        
        # Same term, date-batched (should be faster)
        metrics_batched = self.run_query_test(
            "content_date_batched", 
            """SELECT COUNT(*) FROM job_postings 
               WHERE tst_created >= DATE '2024-04-01' AND tst_created < DATE '2024-07-01'
               AND content_norm ILIKE '%large language model%'""",
            expect_no_seq_scan=True,
            expect_bitmap_and=True,
            max_time_ms=2000
        )
        
        # Verify batched is faster
        if (metrics_full['exec_time_ms'] and metrics_batched['exec_time_ms'] and 
            metrics_batched['exec_time_ms'] >= metrics_full['exec_time_ms']):
            self.failures.append("Date batching not providing expected speedup")
            print("  ⚠️  WARNING: Date batching not faster than full table scan")
        else:
            speedup = metrics_full['exec_time_ms'] / metrics_batched['exec_time_ms'] if metrics_batched['exec_time_ms'] else 1
            print(f"  ✅ Batching speedup: {speedup:.1f}x faster")
    
    def test_5_short_token_edge_cases(self):
        """5) Short-token edge tests (the AI family)"""
        print(f"\n{'='*70}")
        print("5) SHORT-TOKEN EDGE CASES")
        print(f"{'='*70}")
        
        print("Testing short keyword patterns that mirror extraction logic:")
        print(f"{'Status':<6} {'Test Name':<25} | {'Time':>8} | {'Hit Rate':>6} | {'Plan':>8}")
        print("-" * 70)
        
        # Plain 'ai' word boundary
        self.run_query_test(
            "ai_word_boundary",
            "SELECT COUNT(*) FROM job_postings WHERE content_norm ~* '\\\\yai\\\\y'",
            expect_no_seq_scan=True,
            max_time_ms=1500
        )
        
        # Spaced letters (from A.I. / A-I). Should match and use trigram.
        self.run_query_test(
            "ai_spaced_letters",
            "SELECT COUNT(*) FROM job_postings WHERE content_norm ~* '\\\\ya i\\\\y'", 
            expect_no_seq_scan=True,
            max_time_ms=1500
        )
        
        # Test 'ml' (machine learning abbreviation)
        self.run_query_test(
            "ml_word_boundary",
            "SELECT COUNT(*) FROM job_postings WHERE content_norm ~* '\\\\yml\\\\y'",
            expect_no_seq_scan=True,
            max_time_ms=1500
        )
        
        # Combined short keywords with date batch (realistic Stage 0 query)
        self.run_query_test(
            "multi_short_keywords",
            """SELECT COUNT(*) FROM job_postings 
               WHERE tst_created >= DATE '2024-01-01' AND tst_created < DATE '2024-04-01'
               AND (content_norm ~* '\\\\yai\\\\y' OR content_norm ~* '\\\\ya i\\\\y' OR content_norm ~* '\\\\yml\\\\y')""",
            expect_no_seq_scan=True,
            expect_bitmap_and=True,
            max_time_ms=2000
        )
    
    def test_6_optimization_recommendations(self):
        """6) Database optimization recommendations"""
        print(f"\n{'='*70}")
        print("6) DATABASE OPTIMIZATION RECOMMENDATIONS") 
        print(f"{'='*70}")
        
        cur = self.conn.cursor()
        
        # Check for BRIN vs BTREE on date column
        cur.execute("""
            SELECT indexname, indexdef 
            FROM pg_indexes 
            WHERE tablename = 'job_postings' AND indexdef LIKE '%tst_created%'
        """)
        
        date_indexes = cur.fetchall()
        has_btree_date = any('btree' in idx[1] for idx in date_indexes)
        has_brin_date = any('brin' in idx[1] for idx in date_indexes)
        
        print("Date column indexing strategy:")
        if has_btree_date and not has_brin_date:
            cur.execute("""
                SELECT pg_size_pretty(pg_relation_size(indexrelid))
                FROM pg_stat_user_indexes 
                WHERE relname = 'job_postings' AND indexrelid::regclass::text LIKE '%tst_created%'
                  AND indexrelid IN (
                    SELECT indexrelid FROM pg_index WHERE 
                    pg_get_indexdef(indexrelid) LIKE '%btree%'
                  )
            """)
            
            btree_size = cur.fetchone()
            if btree_size:
                print(f"  ✅ BTREE on tst_created: {btree_size[0]}")
                print(f"  💡 SUGGESTION: For large date range scans, consider BRIN index:")
                print(f"     CREATE INDEX CONCURRENTLY idx_job_postings_tst_created_brin")
                print(f"     ON job_postings USING brin (tst_created) WITH (pages_per_range = 64);")
                print(f"     (Would be ~few MB vs {btree_size[0]} BTREE)")
        
        # Check maintenance settings 
        cur.execute("SHOW maintenance_work_mem;")
        maint_mem = cur.fetchone()[0]
        print(f"\nMaintenance settings:")
        print(f"  maintenance_work_mem: {maint_mem}")
        
        # Convert to MB for comparison
        if 'GB' in maint_mem:
            maint_mb = float(maint_mem.replace('GB', '')) * 1024
        elif 'MB' in maint_mem:
            maint_mb = float(maint_mem.replace('MB', ''))
        else:
            maint_mb = float(maint_mem.replace('kB', '')) / 1024
            
        if maint_mb < 1024:  # Less than 1GB
            print(f"  💡 SUGGESTION: Consider raising maintenance_work_mem to 1-2GB for index builds")
            print(f"     SET maintenance_work_mem = '2GB';")
        
        cur.execute("SHOW gin_pending_list_limit;")  
        gin_limit = cur.fetchone()[0]
        print(f"  gin_pending_list_limit: {gin_limit}")
        
        # Check autovacuum stats
        cur.execute("""
            SELECT last_vacuum, last_autovacuum, last_analyze, last_autoanalyze,
                   n_dead_tup, n_live_tup
            FROM pg_stat_user_tables 
            WHERE relname = 'job_postings'
        """)
        
        vacuum_stats = cur.fetchone()
        if vacuum_stats:
            last_vacuum, last_autovac, last_analyze, last_autoanalyze, dead_tup, live_tup = vacuum_stats
            print(f"\nVacuum/analyze status:")
            print(f"  Last manual vacuum: {last_vacuum}")
            print(f"  Last autovacuum: {last_autovac}")  
            print(f"  Last analyze: {last_analyze or last_autoanalyze}")
            
            if dead_tup and live_tup and dead_tup > live_tup * 0.1:
                print(f"  ⚠️  High dead tuple ratio: {dead_tup:,} dead / {live_tup:,} live")
                print(f"     Consider manual VACUUM if autovacuum is not keeping up")
        
        cur.close()
    
    def generate_results_report(self):
        """Generate final pass/fail report and save CSV."""
        print(f"\n{'='*70}")
        print("FINAL RESULTS REPORT")
        print(f"{'='*70}")
        
        total_tests = len(self.results)
        failed_tests = len(self.failures)
        passed_tests = total_tests - failed_tests
        
        print(f"Tests run: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        
        if self.failures:
            print(f"\nFailure details:")
            for failure in self.failures:
                print(f"  ❌ {failure}")
        
        # Save results to CSV
        if self.results:
            csv_path = "tests/stage0_index_performance_results.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['query_id', 'exec_time_ms', 'buffer_hit_pct', 'has_seq_scan', 'has_bitmap_index', 'has_bitmap_and']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
            print(f"\n📊 Detailed results saved to: {csv_path}")
        
        # Final verdict
        if failed_tests == 0:
            print(f"\n🎉 ALL TESTS PASSED!")
            print("Stage 0 indexes are optimized and working correctly.")
            print("Expected performance targets:")
            print("  - Content-only queries: < 1.5s (warm cache)")
            print("  - Date-batched queries: < 800ms (selective), < 2s (common)")
            print("  - All queries use indexes (no sequential scans)")
            print("  - BitmapAnd combining date + content indexes")
            return True
        else:
            print(f"\n❌ {failed_tests} TESTS FAILED!")
            print("Stage 0 indexes need attention - see failure details above.")
            return False


def run_all_tests():
    """Main test runner with proper error handling."""
    test_suite = TestStage0IndexPerformance()
    
    try:
        print("="*70)
        print("STAGE 0 INDEX PERFORMANCE VERIFICATION")
        print("="*70)
        print("Tight, repeatable verification of Stage 0 database optimization")
        print("Testing ~14M row job_postings table for index usage and performance")
        
        # Get table size
        cur = test_suite.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM job_postings;")
        total_rows = cur.fetchone()[0]
        print(f"Table size: {total_rows:,} rows")
        cur.close()
        
        # Run all test phases
        if not test_suite.test_1_pre_flight_checks():
            print("❌ Critical indexes missing - aborting performance tests")
            return False
            
        test_suite.test_2_query_plans()
        test_suite.test_3_performance_with_targets()  
        test_suite.test_4_bitmap_and_benefit()
        test_suite.test_5_short_token_edge_cases()
        
        # Add database optimization recommendations test
        test_suite.test_6_optimization_recommendations()
        
        # Generate final report
        success = test_suite.generate_results_report()
        return success
        
    except Exception as e:
        print(f"\n❌ PERFORMANCE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        test_suite.close()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)