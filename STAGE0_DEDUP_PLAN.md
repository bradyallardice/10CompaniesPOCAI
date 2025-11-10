# Stage 0 Database Deduplication Implementation Plan

## Overview
Add functionality to Stage 0 to create a deduplicated version of the `job_postings_unified` table (7M records) using a two-step deduplication process similar to Stage 2's methodology.

## Current Context
- **Source Table**: `job_postings_unified` (7M records in PostgreSQL)
- **Target**: Create new table `job_postings_unified_dedup`
- **Deduplication Logic**:
  1. Exact duplicates on (content_norm, company_id, title) - keep earliest tst_created
  2. 0.99 similarity on content_norm within same (company_id, title) - keep earliest tst_created

## Implementation Strategy

### Phase 1: Database-Level Exact Deduplication
**Approach**: Use pure SQL for exact duplicate removal to leverage database indexing and avoid memory issues.

**SQL Strategy**:
```sql
-- Step 1: Create deduplicated table with exact matches removed
CREATE TABLE job_postings_unified_dedup AS
SELECT DISTINCT ON (content_norm, company_id, title) *
FROM job_postings_unified
ORDER BY content_norm, company_id, title, tst_created ASC;
```

**Benefits**:
- Leverages PostgreSQL's DISTINCT ON for efficient deduplication
- Memory efficient - database handles the sorting/deduplication
- Preserves earliest tst_created automatically via ORDER BY

### Phase 2: Similarity-Based Deduplication
**Approach**: Batch processing with TF-IDF similarity within company+title groups.

**Process**:
1. **Extract company+title groups** with >1 record from Phase 1 result
2. **Process groups in batches** (500-1000 groups per batch)
3. **Apply TF-IDF similarity** (0.99 threshold) within each group
4. **Update master table** by removing similar duplicates

**Memory Management**:
- Process groups sequentially to avoid loading entire dataset
- Use temporary tables for intermediate results
- Clear memory between batches

### Phase 3: Integration with Stage 0

**New Function**: `create_deduplicated_table()`
- Add command line flag: `--create-dedup-table`
- Integrate with existing database connection patterns
- Add progress logging and batch reporting

## Technical Implementation Details

### Database Schema
```sql
-- Create deduplicated table with same structure as original
CREATE TABLE job_postings_unified_dedup (LIKE job_postings_unified);

-- Add indexes for performance
CREATE INDEX idx_dedup_content_company_title
ON job_postings_unified_dedup (content_norm, company_id, title);

CREATE INDEX idx_dedup_tst_created
ON job_postings_unified_dedup (tst_created);
```

### Batch Processing Algorithm
```python
def create_deduplicated_table():
    # Phase 1: Exact deduplication via SQL
    create_exact_dedup_table()

    # Phase 2: Similarity deduplication
    company_title_groups = get_groups_with_multiple_records()

    for batch in batch_groups(company_title_groups, batch_size=500):
        similar_pairs = find_similar_within_batch(batch)
        remove_similar_duplicates(similar_pairs)

    # Phase 3: Create final indexes and validate
    create_final_indexes()
    validate_deduplication_results()
```

### Similarity Processing
- **Reuse Stage 2's TF-IDF logic** from `deduplicate_similar_content()`
- **Group-by-group processing** to avoid memory issues
- **Vectorization within groups** only (not across entire dataset)
- **PostgreSQL temp tables** for intermediate similarity results

## Time and Storage Estimates (Updated for 7M Records)

### Time Estimates

**Phase 1 (Exact Deduplication)**:
- Database query execution: 15-25 minutes
- Depends on existing indexes and disk I/O performance

**Phase 2 (Similarity Deduplication)**:
- Groups to process: ~250K-500K company+title combinations
- TF-IDF processing: ~1-2 hours (depends on group sizes)
- Assuming 10-50 records per group on average

**Total Estimated Time**: **1.5-3 hours** (reduced from 3-5 hours for 14.7M)

### Storage Requirements

**Original Table Size**: ~7M records
- Estimated row size: ~2-5KB (text content + metadata)
- Current storage: ~15-35GB

**Expected Deduplication Reduction**:
- Exact duplicates: 15-25% reduction (typical for job postings)
- Similarity duplicates: 5-10% additional reduction
- **Total reduction**: 20-35%

**Storage Needs**:
- **Temporary storage**: 15-35GB (for intermediate tables)
- **Final deduplicated table**: 10-28GB (65-80% of original)
- **Peak storage**: 30-70GB (original + temp + final)

### Performance Optimizations

1. **Index Strategy**:
   - Create indexes on deduplication keys before processing
   - Drop unnecessary indexes during processing, rebuild after

2. **Batch Size Tuning**:
   - Start with 500 company+title groups per batch
   - Adjust based on memory usage and processing time

3. **Parallel Processing**:
   - Process independent company+title groups in parallel
   - Use PostgreSQL's parallel query capabilities

4. **Memory Management**:
   - Clear TF-IDF vectors between batches
   - Use generators for large result sets
   - Monitor memory usage and adjust batch sizes

## Implementation Steps

### Step 1: Add Command Line Interface
- Add `--create-dedup-table` flag to Stage 0
- Add configuration parameters (batch_size, similarity_threshold)
- Example: `python stage_0_get_job_ads.py --create-dedup-table --batch-size 500`

### Step 2: Implement Phase 1 (Exact Deduplication)
- Create SQL-based exact deduplication function
- Add progress logging and row count validation
- Handle edge cases (NULL values, empty content_norm)

### Step 3: Implement Phase 2 (Similarity Deduplication)
- Port Stage 2's TF-IDF similarity logic from `stage_2_deduplicate_translate.py`
- Implement batch processing for company+title groups
- Add memory management and progress tracking
- Reuse `create_content_hash()` and similarity calculation methods

### Step 4: Integration and Testing
- Test with smaller subsets first (100K, 1M records)
- Validate deduplication results against expected patterns
- Performance testing and optimization
- Compare results with Stage 2 deduplication on overlapping data

### Step 5: Production Deployment
- Run full deduplication on 7M dataset
- Monitor resource usage and performance
- Create backup and rollback procedures

## Code Integration Points

### Existing Functions to Leverage
From `stage_2_deduplicate_translate.py`:
- `create_content_hash()` - for exact duplicate detection
- `deduplicate_similar_content()` - for TF-IDF similarity logic
- TF-IDF vectorization and cosine similarity calculations

### New Functions to Add to Stage 0
```python
def create_deduplicated_table(batch_size=500, similarity_threshold=0.99):
    """Create deduplicated version of job_postings_unified table"""

def create_exact_dedup_table():
    """Phase 1: Remove exact duplicates using SQL"""

def process_similarity_batches(batch_size=500):
    """Phase 2: Process similarity-based deduplication in batches"""

def validate_deduplication_results():
    """Validate final results and generate statistics"""
```

## Risk Mitigation

1. **Memory Issues**:
   - Batch processing with configurable batch sizes
   - Temp table cleanup after each batch
   - Memory monitoring and automatic batch size adjustment

2. **Long Runtime**:
   - Checkpointing after each major phase
   - Resumable processing from last checkpoint
   - Progress logging with time estimates

3. **Data Loss**:
   - Comprehensive validation comparing original vs deduplicated counts
   - Backup procedures before starting deduplication
   - Rollback capability to restore original table

4. **Disk Space**:
   - Pre-flight storage space validation
   - Cleanup procedures for temporary tables
   - Configurable storage thresholds

## Expected Benefits

1. **Reduced Storage**: 20-35% reduction in table size (~5-6M final records)
2. **Improved Performance**: Faster queries on downstream processing (Stages 1-6)
3. **Better Data Quality**: Eliminates redundant job postings
4. **Consistent Methodology**: Aligns with Stage 2's proven deduplication approach
5. **Cost Savings**: Reduced storage and compute costs for downstream processing

## Usage Examples

```bash
# Create deduplicated table with default settings
python stage_0_get_job_ads.py --create-dedup-table

# Custom batch size and similarity threshold
python stage_0_get_job_ads.py --create-dedup-table --batch-size 1000 --similarity-threshold 0.95

# Test mode on subset of data
python stage_0_get_job_ads.py --create-dedup-table --test-mode --limit-rows 100000
```

## Monitoring and Validation

### Progress Metrics
- Records processed per phase
- Deduplication rates (exact vs similarity)
- Processing time per batch
- Memory usage tracking
- Storage space utilization

### Validation Checks
- Row count consistency
- Earliest tst_created preservation validation
- Content integrity checks
- Index creation and performance validation
- Comparison with Stage 2 results on overlapping datasets

---

**Status**: Planning Phase
**Priority**: Medium
**Estimated Effort**: 2-3 days development + 1.5-3 hours execution time
**Dependencies**: Access to PostgreSQL database, sufficient disk space (30-70GB peak)