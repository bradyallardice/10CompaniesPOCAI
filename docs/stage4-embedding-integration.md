# Stage 4 Integration with Embedding Manager

## Overview

This document outlines how Stage 4 (`stage_4_onet_similarity.py`) will integrate with the `embedding_manager.py` utility for managing large embedding files.

---

## Integration Points

### 1. Initialization

**Add import at top of file:**
```python
from Code.utilities.embedding_manager import EmbeddingManager
```

**In `ONETSimilarityMatcher.__init__()` method:**
```python
def __init__(self, ...):
    # Existing initialization code
    self.embeddings_dir = embeddings_dir

    # NEW: Initialize embedding manager
    self.embedding_manager = EmbeddingManager(embeddings_dir=embeddings_dir)
    logger.info("Embedding manager initialized")
```

---

### 2. Loading Cached Embeddings

**Current code in `_load_embeddings()` method (line ~253):**
```python
if not os.path.exists(cache_path):
    return None
```

**With embedding manager:**
```python
# Try to get embedding path (handles Dropbox sync verification)
try:
    # Extract filename pattern from cache_path
    cache_filename = os.path.basename(cache_path)
    actual_path = self.embedding_manager.get_embedding_path(cache_filename)
    cache_path = str(actual_path)
except FileNotFoundError as e:
    logger.debug(f"Embedding not found in cache: {e}")
    return None
```

---

### 3. Loading OpenAI Embeddings

**Current code in `_load_openai_embeddings()` method (line ~290):**
```python
pattern = f"openai_{model_name}_{text_type}_*.pkl"
cache_files = glob.glob(os.path.join(self.embeddings_dir, pattern))

if not cache_files:
    raise FileNotFoundError(...)
```

**With embedding manager:**
```python
pattern = f"openai_{model_name}_{text_type}_*.pkl"

try:
    cache_path = self.embedding_manager.get_embedding_path(pattern)
    cache_files = [str(cache_path)]
except FileNotFoundError:
    raise FileNotFoundError(
        f"No OpenAI embeddings found matching: {pattern}\n"
        f"Please ensure Dropbox has synced embeddings or regenerate with:\n"
        f"python3 generate_openai_embeddings.py"
    )
```

---

### 4. Loading Checkpoint Files

**Current code in checkpoint loading (multiple locations):**
```python
checkpoint_pattern = os.path.join(checkpoint_dir, "similarities_*.parquet")
existing_checkpoints = glob.glob(checkpoint_pattern)
```

**With embedding manager:**
```python
# Get checkpoint files (verifies Dropbox sync)
try:
    checkpoint_pattern = "similarities_*.parquet"
    existing_checkpoints = self.embedding_manager.get_checkpoint_files(checkpoint_pattern)
    # Convert Path objects to strings for compatibility
    existing_checkpoints = [str(p) for p in existing_checkpoints]
except Exception as e:
    logger.warning(f"No checkpoints found: {e}")
    existing_checkpoints = []
```

---

### 5. Cleanup After Processing (Optional)

**Add command-line argument in `main()` function:**
```python
parser.add_argument("--cleanup-embeddings", action="store_true",
                   help="Cleanup local embeddings after processing (saves ~59GB)")
parser.add_argument("--cleanup-checkpoints-only", action="store_true",
                   help="Cleanup only checkpoint files (saves ~30-40GB)")
```

**At end of `main()` function, before final return:**
```python
# Optional: Cleanup embeddings to save disk space
if args.cleanup_embeddings:
    logger.info("Cleaning up embeddings (files remain in Dropbox cloud)...")
    matcher.embedding_manager.cleanup()
    logger.info("Cleanup complete - saved ~59GB of disk space")

if args.cleanup_checkpoints_only:
    logger.info("Cleaning up checkpoint files only...")
    matcher.embedding_manager.cleanup_checkpoints_only()
    logger.info("Cleanup complete - saved ~30-40GB of disk space")
```

---

## Complete Integration Example

Here's how a typical Stage 4 run would work with embedding manager:

```python
def main():
    # Parse arguments
    args = parser.parse_args()

    # Initialize matcher (includes embedding manager)
    matcher = ONETSimilarityMatcher(
        embeddings_dir=args.embeddings_dir,
        # ... other args
    )

    # Load data
    df_apps = pd.read_csv(args.input)
    df_onet = pd.read_csv(args.onet_tasks)

    # Deduplicate applications
    apps_dedup = matcher.deduplicate_applications(df_apps)

    # Embed applications (uses embedding manager for caching)
    # If embeddings exist in Dropbox but not local, they'll be downloaded
    app_embeddings = matcher.embed_texts(apps_dedup, description="AI applications")

    # Embed O*NET tasks (uses embedding manager for caching)
    onet_embeddings = matcher.embed_texts(df_onet['task'], description="O*NET tasks")

    # Calculate similarities (uses embedding manager for checkpoints)
    similarities = matcher.calculate_similarities(
        app_embeddings,
        onet_embeddings,
        apps_dedup,
        df_onet
    )

    # Save results
    output_file = args.output
    similarities.to_csv(output_file, index=False)

    # Optional cleanup
    if args.cleanup_embeddings:
        logger.info("Cleaning up embeddings...")
        matcher.embedding_manager.cleanup()

    logger.info("Stage 4 complete!")
```

---

## Error Handling

### Scenario 1: Embeddings Not Downloaded

```python
# User runs Stage 4 but embeddings are in Dropbox cloud (not local)
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv

# Embedding manager detects file not available
FileNotFoundError: No embedding file found matching: apps_BAAI_bge-large-en-v1.5_*.pkl
Search path: Data/embeddings/apps_BAAI_bge-large-en-v1.5_*.pkl
Please ensure Dropbox has synced this file or regenerate embeddings

# Solution: Make embeddings available in Dropbox
# Right-click Data/embeddings/ → "Make available offline"
# Re-run Stage 4
```

### Scenario 2: Dropbox Placeholder Detected

```python
# File exists but is a Dropbox placeholder (not fully downloaded)
FileNotFoundError: File appears to be Dropbox placeholder: Data/embeddings/apps_BAAI_bge-large-en-v1.5_abc123.pkl
Size: 123 bytes (suspiciously small)
Solution: Right-click in Finder → 'Make available offline'

# Solution: Wait for Dropbox to finish downloading
# Re-run Stage 4
```

### Scenario 3: Pattern Matches Multiple Files

```python
# Multiple versions of same embedding exist
ValueError: Multiple embedding files match pattern: apps_BAAI_bge-large-en-v1.5_*.pkl
Matches: [
    'Data/embeddings/apps_BAAI_bge-large-en-v1.5_abc123.pkl',
    'Data/embeddings/apps_BAAI_bge-large-en-v1.5_def456.pkl'
]
Please use a more specific pattern

# Solution 1: Use full filename
# Solution 2: Cleanup old versions: python3 embedding_manager.py --cleanup
```

---

## Benefits of Integration

### Before Integration
- ❌ Stage 4 fails silently if embeddings not downloaded
- ❌ No verification of Dropbox placeholders
- ❌ Manual cleanup required
- ❌ No centralized tracking of accessed files

### After Integration
- ✅ **Fail-fast**: Clear errors if embeddings not available
- ✅ **Automatic verification**: Detects Dropbox placeholders
- ✅ **Optional cleanup**: Save disk space after processing
- ✅ **Centralized management**: Single utility for all embedding operations
- ✅ **Better logging**: Track which embeddings are accessed

---

## Testing Plan

### 1. Test with Embeddings Available
```bash
# Ensure embeddings downloaded
python3 embedding_manager.py --list

# Run Stage 4
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv

# Expected: Normal operation, embeddings loaded from cache
```

### 2. Test with Embeddings Not Downloaded
```bash
# Remove local embeddings
python3 embedding_manager.py --cleanup

# Run Stage 4
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv

# Expected: Clear error message with instructions
```

### 3. Test Cleanup After Processing
```bash
# Run Stage 4 with cleanup
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv --cleanup-embeddings

# Expected: Processing completes, then embeddings removed
# Check disk space freed: du -sh Data/embeddings/
```

---

## Migration Path

### Phase 1: Add Embedding Manager (Current)
- ✅ Create `embedding_manager.py`
- ✅ Add to `.gitignore`
- ✅ Document usage

### Phase 2: Integrate with Stage 4 (Next)
- [ ] Add import to `stage_4_onet_similarity.py`
- [ ] Initialize embedding manager in `__init__()`
- [ ] Update `_load_embeddings()` to use manager
- [ ] Update `_load_openai_embeddings()` to use manager
- [ ] Update checkpoint loading to use manager
- [ ] Add cleanup arguments to CLI

### Phase 3: Test and Validate
- [ ] Test on small dataset
- [ ] Test cleanup functionality
- [ ] Test Dropbox sync scenarios
- [ ] Update documentation

### Phase 4: Production Use
- [ ] Run on full dataset
- [ ] Monitor disk space savings
- [ ] Document best practices

---

## Backward Compatibility

The integration is designed to be **backward compatible**:

- **Existing embeddings**: Will still work (just verified by manager)
- **Existing code**: Can still access embeddings directly if needed
- **Existing workflows**: No breaking changes to Stage 4 CLI

**Optional features**:
- Cleanup is opt-in via `--cleanup-embeddings` flag
- Embedding manager can be disabled by not importing it

---

## Next Steps

To implement this integration:

1. **Review this document** to ensure approach makes sense
2. **Test embedding_manager.py** standalone to verify functionality
3. **Update Stage 4** with integration points outlined above
4. **Test on small dataset** before full production run
5. **Document workflows** for team members

Would you like me to proceed with the Stage 4 integration?
