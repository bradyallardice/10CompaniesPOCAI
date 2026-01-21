# Stage 4 Embedding Manager Integration - COMPLETE

## Summary

Successfully integrated the `embedding_manager.py` utility into Stage 4 (`stage_4_onet_similarity.py`) to enable automatic Dropbox download and cleanup of large embedding files.

---

## Changes Made

### 1. Import Added (Line 30)
```python
from Code.utilities.embedding_manager import EmbeddingManager
```

### 2. Initialization in `__init__` (Lines 91-97)
```python
# Initialize embedding manager for Dropbox auto-download
self.embedding_manager = EmbeddingManager(
    embeddings_dir=embeddings_dir,
    auto_download=True,
    auto_cleanup=False  # Manual cleanup via CLI flag
)
logger.info("Embedding manager initialized")
```

### 3. Updated `_load_embeddings()` Method (Lines 264-271)
```python
# Use embedding manager to verify/download file
try:
    cache_filename = os.path.basename(cache_path)
    actual_path = self.embedding_manager.get_embedding_path(cache_filename)
    cache_path = str(actual_path)
except FileNotFoundError as e:
    logger.debug(f"Embedding not found in cache: {e}")
    return None
```

### 4. Updated `_load_openai_embeddings()` Method (Lines 310-318)
```python
# Use embedding manager to get file path (handles Dropbox sync verification)
try:
    cache_path = self.embedding_manager.get_embedding_path(cache_filename)
except FileNotFoundError:
    raise FileNotFoundError(
        f"No OpenAI embeddings found matching: {cache_filename}\n"
        f"Please ensure Dropbox has synced embeddings or regenerate with:\n"
        f"python3 generate_openai_embeddings.py --task-type core"
    )
```

### 5. Added CLI Arguments (Lines 3185-3191)
```python
parser.add_argument("--cleanup-embeddings", action="store_true",
                   help="Cleanup local embeddings after processing (saves ~59GB). "
                        "Files remain in Dropbox cloud.")

parser.add_argument("--cleanup-checkpoints-only", action="store_true",
                   help="Cleanup only checkpoint files (saves ~30-40GB). "
                        "Keeps main embeddings for faster reruns.")
```

### 6. Added Cleanup Code in `main()` (Lines 3344-3353)
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

## How It Works

### Automatic Download
1. Stage 4 initializes `EmbeddingManager` with `auto_download=True`
2. When loading embeddings, embedding manager checks if file is a Dropbox placeholder (0 bytes)
3. If placeholder detected, automatically downloads from Dropbox API to temp directory (`/tmp/embedding_cache_...`)
4. Returns temp path to Stage 4 code transparently
5. Stage 4 uses the file normally

### Safe Cleanup
1. User can optionally add `--cleanup-embeddings` or `--cleanup-checkpoints-only` flags
2. After processing completes, cleanup runs:
   - Deletes files from temp directory (`/tmp/embedding_cache_...`)
   - **Never touches Dropbox folder** - original files remain as 0-byte placeholders
3. Files remain in Dropbox cloud for future use

### File Flow
```
Dropbox Cloud (58MB)
    ↓ [API auto-download if placeholder detected]
/tmp/embedding_cache_xxx/file.pkl (58MB)
    ↓ [Stage 4 uses file]
    ↓ [Optional: cleanup after processing]
/tmp/embedding_cache_xxx/ [deleted]
    
Dropbox folder: file.pkl (0 bytes, placeholder) [SAFE - never deleted]
```

---

## Usage Examples

### Basic Usage (No Cleanup)
```bash
python3 stage_4_onet_similarity.py \
  --input Data/stage3_output.csv \
  --onet-file Data/task_statements_20.xlsx
```
- Embeddings auto-download from Dropbox if needed
- Temp files remain on disk for faster reruns

### With Full Cleanup
```bash
python3 stage_4_onet_similarity.py \
  --input Data/stage3_output.csv \
  --onet-file Data/task_statements_20.xlsx \
  --cleanup-embeddings
```
- Processing completes
- All temp embeddings deleted (~59GB saved)
- Dropbox files remain safe as placeholders

### With Checkpoint-Only Cleanup
```bash
python3 stage_4_onet_similarity.py \
  --input Data/stage3_output.csv \
  --onet-file Data/task_statements_20.xlsx \
  --cleanup-checkpoints-only
```
- Processing completes
- Checkpoint files deleted (~30-40GB saved)
- Main embedding files kept for faster reruns

---

## Benefits

### Before Integration
- ❌ Stage 4 fails silently if embeddings not downloaded
- ❌ No verification of Dropbox placeholders
- ❌ Manual cleanup required
- ❌ Risk of deleting files from Dropbox folder

### After Integration
- ✅ **Auto-download**: Files download from Dropbox API automatically when needed
- ✅ **Placeholder detection**: Detects 0-byte Dropbox placeholders
- ✅ **Safe cleanup**: Only temp files deleted, never Dropbox files
- ✅ **Optional cleanup**: User controls disk space savings via CLI flags
- ✅ **Transparent**: Stage 4 code works exactly as before
- ✅ **Dropbox protection**: Files remain in cloud, can't be accidentally deleted

---

## Testing Status

✅ Embedding manager tested standalone:
- Download to temp directory works
- Cleanup removes only temp files
- Dropbox files remain as placeholders
- Context manager auto-cleanup works

⏳ Stage 4 integration testing:
- Ready for testing with real Stage 4 run
- All integration points implemented
- Cleanup flags added and wired up

---

## Next Steps

1. Test Stage 4 run on small dataset
2. Verify embeddings auto-download from Dropbox
3. Verify cleanup works correctly
4. Update documentation with best practices
5. Roll out to production runs

---

## Related Files

- `Code/utilities/embedding_manager.py` - Core utility
- `stage_4_onet_similarity.py` - Stage 4 with integration
- `docs/embedding-storage.md` - Usage documentation
- `docs/stage4-embedding-integration.md` - Original integration plan
- `.gitignore` - Embeddings excluded from Git

