# Embedding Storage and Management

## Overview

The pipeline generates ~59GB of embedding files that are managed through a combination of:
- **Git ignore**: Excludes embeddings from version control
- **Dropbox storage**: Files remain in Dropbox cloud for backup/sync
- **Local download**: Files downloaded on-demand when needed
- **Cleanup utility**: Remove local copies after use to save disk space

---

## File Structure

```
Data/embeddings/
├── *.pkl                              # Main embedding files (~10-20 files, 10-30 GB)
├── checkpoints/                       # Similarity checkpoint files
│   └── *.parquet                      # (~100+ files, 20-30 GB)
└── cross_encoder_checkpoints/         # Cross-encoder refinement
    └── *.parquet                      # (~50+ files, 5-10 GB)
```

**Total: ~48GB** (as of January 2026)

---

## Git Configuration

Embeddings are excluded from Git via `.gitignore`:

```gitignore
# Embeddings and similarity matrices (large files - 59GB total)
# These are stored in Dropbox but excluded from Git
# Use embedding_manager.py to download/cleanup as needed
Data/embeddings/*.pkl
Data/embeddings/checkpoints/
Data/embeddings/cross_encoder_checkpoints/
Data/embeddings/openai_text-embedding-3-large_*.pkl
```

**Result:**
- ✅ Git repo stays small (won't commit 59GB)
- ✅ Files remain in Dropbox for backup
- ✅ Can be synced across machines via Dropbox

---

## Dropbox Selective Sync (Recommended)

To save local disk space on machines that don't need embeddings:

### Setup
1. **Right-click** `Data/embeddings/` folder in Finder
2. Select **"Make available → Online only"**
3. Files remain in Dropbox cloud but not downloaded locally

### When Needed
- Files download automatically when accessed by Stage 4
- Or manually: Right-click → "Make available offline"

### Benefits
- **Saves 59GB** of local disk space
- **Automatic download** when needed
- **Still backed up** in Dropbox cloud

---

## Embedding Manager Utility

The `embedding_manager.py` utility provides programmatic control over embeddings.

### CLI Usage

**List all embeddings and sizes:**
```bash
python3 Code/utilities/embedding_manager.py --list
```

**Cleanup all local embeddings:**
```bash
python3 embedding_manager.py --cleanup
# Prompts for confirmation before deleting
```

**Cleanup only checkpoints (keep main embeddings):**
```bash
python3 embedding_manager.py --cleanup-checkpoints
# Saves ~30-40GB while preserving main embedding files
```

### Python API

**Basic usage (no auto-cleanup):**
```python
from Code.utilities.embedding_manager import EmbeddingManager

em = EmbeddingManager()

# Get path to specific embedding
embedding_path = em.get_embedding_path('apps_BAAI_bge-large-en-v1.5_*.pkl')

# Load embedding
import pickle
with open(embedding_path, 'rb') as f:
    embeddings = pickle.load(f)

# Manual cleanup when done
em.cleanup()  # Removes only accessed files
```

**Context manager (auto-cleanup on exit):**
```python
from Code.utilities.embedding_manager import EmbeddingManager

with EmbeddingManager(auto_cleanup=True) as em:
    # Get embedding path
    embedding_path = em.get_embedding_path('apps_BAAI_bge-large-en-v1.5_*.pkl')

    # Use embedding
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)

    # Process data...

# Auto-cleanup happens here (removes accessed files)
```

**Force cleanup (delete everything):**
```python
from Code.utilities.embedding_manager import EmbeddingManager

em = EmbeddingManager()
em.cleanup(force=True)  # Deletes ALL embeddings
```

---

## Integration with Stage 4

Stage 4 is the primary consumer of embeddings. The integration points are:

### 1. Loading Embeddings

**Before (direct file access):**
```python
embedding_path = os.path.join(self.embeddings_dir, f"apps_{model_name}_{hash}.pkl")
with open(embedding_path, 'rb') as f:
    embeddings = pickle.load(f)
```

**After (with EmbeddingManager):**
```python
from Code.utilities.embedding_manager import EmbeddingManager

# At start of stage
em = EmbeddingManager(embeddings_dir=self.embeddings_dir)

# When loading embedding
embedding_path = em.get_embedding_path(f"apps_{model_name}_*.pkl")
with open(embedding_path, 'rb') as f:
    embeddings = pickle.load(f)
```

### 2. Loading Checkpoints

**Before:**
```python
checkpoint_files = glob.glob(f"{checkpoint_dir}/*.parquet")
```

**After:**
```python
checkpoint_files = em.get_checkpoint_files("similarities_*.parquet")
```

### 3. Cleanup After Processing

**Optional cleanup at end of Stage 4:**
```python
# After all processing complete
if args.cleanup_embeddings:
    logger.info("Cleaning up embeddings...")
    em.cleanup()  # Removes accessed files
```

---

## Workflow Examples

### Scenario 1: First-Time Stage 4 Run

```bash
# Embeddings not downloaded yet
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv

# Output:
# ERROR: No embedding file found matching: apps_BAAI_bge-large-en-v1.5_*.pkl
# Please ensure Dropbox has synced this file or regenerate embeddings

# Solution: Make embeddings available offline in Dropbox
# Right-click Data/embeddings/ → "Make available offline"

# Re-run Stage 4
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv
# Now works - embeddings downloaded from Dropbox
```

### Scenario 2: Cleanup After Processing

```bash
# Run Stage 4
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv

# Cleanup embeddings to save space (optional)
python3 embedding_manager.py --cleanup-checkpoints
# Removes ~30-40GB of checkpoint files, keeps main embeddings
```

### Scenario 3: Moving Between Machines

```bash
# Machine A: Process and cleanup
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv
python3 embedding_manager.py --cleanup
# Local copy removed, still in Dropbox cloud

# Machine B: Download and process
# Set Data/embeddings/ to "Make available offline"
python3 stage_4_onet_similarity.py --input Data/stage3_output.csv
# Downloads from Dropbox automatically
```

---

## Benefits of This Approach

### For Git
- ✅ **Small repo size**: No 59GB binary files in version control
- ✅ **Fast clones**: Clone only code, not embeddings
- ✅ **Clean history**: No large binary diffs

### For Dropbox
- ✅ **Centralized backup**: All embeddings backed up in cloud
- ✅ **Selective sync**: Download only when needed
- ✅ **Cross-machine sharing**: Same files available everywhere

### For Development
- ✅ **Disk space control**: Remove when not needed
- ✅ **Automatic verification**: Checks for Dropbox placeholders
- ✅ **Fail-fast errors**: Clear messages if files not available

---

## Troubleshooting

### Error: "File appears to be Dropbox placeholder"

**Cause:** File is in Dropbox cloud but not downloaded locally

**Solution:**
1. Right-click `Data/embeddings/` in Finder
2. Select "Make available offline"
3. Wait for download to complete
4. Re-run script

### Error: "No embedding file found matching pattern"

**Cause:** Embedding not generated yet or wrong pattern

**Solution:**
1. Check available embeddings: `python3 Code/utilities/embedding_manager.py --list`
2. Generate if missing: `python3 stage_4_onet_similarity.py --regenerate`
3. Verify pattern matches actual filenames

### Error: "Multiple embedding files match pattern"

**Cause:** Multiple versions of same embedding exist

**Solution:**
1. List embeddings: `python3 Code/utilities/embedding_manager.py --list`
2. Use more specific pattern (include full hash)
3. Or cleanup old versions: `python3 embedding_manager.py --cleanup`

---

## Best Practices

1. **Set Dropbox to "Online only" by default**
   - Saves disk space on all machines
   - Downloads automatically when needed

2. **Cleanup checkpoints regularly**
   - Checkpoint files (~30-40GB) are recreatable
   - Keep main embeddings (~20GB) for faster reruns

3. **Don't commit embeddings to Git**
   - `.gitignore` already configured
   - Verify with `git status` before commits

4. **Regenerate if lost**
   - All embeddings can be regenerated from Stage 3 output
   - Takes time but ensures reproducibility

5. **Monitor disk space**
   - Use `python3 Code/utilities/embedding_manager.py --list` to check sizes
   - Cleanup when not actively using Stage 4+

---

## Related Documentation

- [Pipeline Specifications](pipeline-specs.md) - Stage 4 details
- [Code Practices](.claude/rules/code-practices.md) - File management rules
- [Cost Optimization](cost-optimization.md) - Embedding generation costs
