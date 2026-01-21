#!/usr/bin/env python3
"""
Embedding Manager - Manage large embedding files with Dropbox sync

This utility helps manage the 59GB of embedding files by:
1. Checking if embeddings exist locally
2. Auto-downloading from Dropbox if they're "Online only" placeholders
3. Cleaning up local copies after use to save disk space

Usage:
    from Code.utilities.embedding_manager import EmbeddingManager

    with EmbeddingManager(auto_cleanup=True) as em:
        # Embeddings auto-download if needed and cleanup on exit
        embedding_path = em.get_embedding_path('apps_BAAI_bge-large-en-v1.5_*.pkl')
        # ... use embedding ...
    # Auto-cleanup happens here

Dropbox API Setup (optional but recommended):
    1. Install: pip install dropbox
    2. Get access token: https://www.dropbox.com/developers/apps
    3. Set environment variable: export DROPBOX_ACCESS_TOKEN="your_token"
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Optional
import shutil

# Optional Dropbox API support
try:
    import dropbox
    from dropbox.files import WriteMode
    from dropbox.exceptions import ApiError
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Context manager for embedding files with automatic cleanup

    Attributes:
        embeddings_dir: Path to Data/embeddings/ directory
        auto_cleanup: Whether to cleanup on exit (default: False, user controlled)
        accessed_files: Track which files were accessed for cleanup
    """

    def __init__(self, embeddings_dir: Optional[str] = None, auto_cleanup: bool = False,
                 auto_download: bool = True, dropbox_token: Optional[str] = None):
        """
        Initialize embedding manager

        Args:
            embeddings_dir: Path to embeddings directory (default: Data/embeddings)
            auto_cleanup: If True, cleanup embeddings on exit (default: False)
            auto_download: If True, auto-download placeholders from Dropbox (default: True)
            dropbox_token: Dropbox API token (reads from config.env if None)
        """
        if embeddings_dir is None:
            # Default to Data/embeddings relative to project root
            # This script is in Code/utilities/, so go up two levels to project root
            project_root = Path(__file__).parent.parent.parent
            embeddings_dir = project_root / "Data" / "embeddings"
            self.project_root = project_root
        else:
            self.project_root = Path(embeddings_dir).parent.parent

        self.embeddings_dir = Path(embeddings_dir)
        self.auto_cleanup = auto_cleanup
        self.auto_download = auto_download
        self.accessed_files: List[Path] = []

        # Create temp directory for downloads (outside Dropbox)
        import tempfile
        self.temp_dir = Path(tempfile.mkdtemp(prefix="embedding_cache_"))
        logger.debug(f"Created temp directory: {self.temp_dir}")

        # Initialize Dropbox client if auto_download enabled
        self.dbx = None
        if auto_download and DROPBOX_AVAILABLE:
            token = dropbox_token or self._load_dropbox_token()
            if token:
                try:
                    self.dbx = dropbox.Dropbox(token)
                    # Test connection
                    self.dbx.users_get_current_account()
                    logger.info("Dropbox API connected successfully")
                except Exception as e:
                    logger.warning(f"Failed to connect to Dropbox API: {e}")
                    logger.warning("Auto-download disabled - will use manual mode")
                    self.dbx = None
            else:
                logger.warning("No Dropbox token found - auto-download disabled")
        elif auto_download and not DROPBOX_AVAILABLE:
            logger.warning("Dropbox SDK not installed - auto-download disabled")
            logger.warning("Install with: pip install dropbox")

        # Ensure directory exists
        if not self.embeddings_dir.exists():
            raise FileNotFoundError(
                f"Embeddings directory not found: {self.embeddings_dir}\n"
                f"Expected to be synced via Dropbox"
            )

    def _load_dropbox_token(self) -> Optional[str]:
        """Load Dropbox API token from config.env or environment variable"""
        # Try environment variable first
        token = os.environ.get('DROPBOX_API_KEY') or os.environ.get('DROPBOX_ACCESS_TOKEN')
        if token:
            return token

        # Try config.env file
        config_path = self.project_root / "config.env"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('DROPBOX_API_KEY=') or line.startswith('DROPBOX_ACCESS_TOKEN='):
                            token = line.split('=', 1)[1].strip().strip('"').strip("'")
                            return token
            except Exception as e:
                logger.warning(f"Failed to read config.env: {e}")

        return None

    def __enter__(self):
        """Enter context manager"""
        logger.info(f"Embedding manager active: {self.embeddings_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager with optional cleanup"""
        if self.auto_cleanup and self.accessed_files:
            logger.info("Auto-cleanup enabled - removing temp embeddings")
            self.cleanup()

        # Always cleanup temp directory
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.debug(f"Removed temp directory: {self.temp_dir}")

        return False

    def get_embedding_path(self, pattern: str) -> Path:
        """
        Get path to embedding file matching pattern

        Args:
            pattern: Glob pattern (e.g., 'apps_BAAI_bge-large-en-v1.5_*.pkl')

        Returns:
            Path to embedding file

        Raises:
            FileNotFoundError: If no matching file found or multiple matches
        """
        search_pattern = str(self.embeddings_dir / pattern)
        matches = glob.glob(search_pattern)

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No embedding file found matching: {pattern}\n"
                f"Search path: {search_pattern}\n"
                f"Please ensure Dropbox has synced this file or regenerate embeddings"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple embedding files match pattern: {pattern}\n"
                f"Matches: {matches}\n"
                f"Please use a more specific pattern"
            )

        embedding_path = Path(matches[0]).resolve()

        # Verify file is actually downloaded (not a Dropbox placeholder)
        self._verify_file_downloaded(embedding_path)

        # Check if file was downloaded to temp - return temp path instead
        # Note: _verify_file_downloaded adds to _temp_path_map if downloaded
        if hasattr(self, '_temp_path_map') and embedding_path in self._temp_path_map:
            temp_path = self._temp_path_map[embedding_path]
            logger.info(f"Returning temp path: {temp_path}")
            # Track temp path for cleanup
            if temp_path not in self.accessed_files:
                self.accessed_files.append(temp_path)
            return temp_path

        # Track for cleanup (original Dropbox path if not downloaded)
        if embedding_path not in self.accessed_files:
            self.accessed_files.append(embedding_path)

        logger.info(f"Returning original path: {embedding_path}")
        return embedding_path

    def get_checkpoint_files(self, pattern: str = "*.parquet") -> List[Path]:
        """
        Get all checkpoint files matching pattern

        Args:
            pattern: Glob pattern for checkpoint files

        Returns:
            List of checkpoint file paths
        """
        checkpoint_dir = self.embeddings_dir / "checkpoints"
        if not checkpoint_dir.exists():
            checkpoint_dir = self.embeddings_dir / "cross_encoder_checkpoints"

        if not checkpoint_dir.exists():
            logger.warning("No checkpoint directory found")
            return []

        search_pattern = str(checkpoint_dir / pattern)
        matches = [Path(f) for f in glob.glob(search_pattern)]

        # Verify all are downloaded
        for checkpoint in matches:
            self._verify_file_downloaded(checkpoint)

        # Return temp paths if files were auto-downloaded
        result_paths = []
        for checkpoint in matches:
            # Check if downloaded to temp
            if hasattr(self, '_temp_path_map') and checkpoint in self._temp_path_map:
                temp_path = self._temp_path_map[checkpoint]
                result_paths.append(temp_path)
                # Track temp path for cleanup
                if temp_path not in self.accessed_files:
                    self.accessed_files.append(temp_path)
            else:
                result_paths.append(checkpoint)
                # Track original path for cleanup
                if checkpoint not in self.accessed_files:
                    self.accessed_files.append(checkpoint)

        logger.info(f"Found {len(result_paths)} checkpoint files")
        return result_paths

    def _verify_file_downloaded(self, file_path: Path):
        """
        Verify file is fully downloaded from Dropbox.
        If file is a placeholder and auto_download is enabled, download it via Dropbox API.

        Args:
            file_path: Path to file to verify

        Raises:
            FileNotFoundError: If file appears to be Dropbox placeholder and auto-download fails
        """
        # Resolve to absolute path for consistent handling
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if file is suspiciously small (might be placeholder)
        file_size = file_path.stat().st_size
        if file_size < 1000:  # Less than 1KB is suspicious for embeddings
            logger.warning(f"Detected Dropbox placeholder: {file_path.name} ({file_size} bytes)")

            # Try to download if auto_download enabled
            if self.auto_download and self.dbx:
                logger.info(f"Auto-downloading from Dropbox: {file_path.name}")
                try:
                    temp_path = self._download_from_dropbox(file_path)
                    # Replace file_path reference with temp_path for caller
                    # Store mapping so get_embedding_path can return temp path
                    if not hasattr(self, '_temp_path_map'):
                        self._temp_path_map = {}
                    self._temp_path_map[file_path] = temp_path

                    # Verify download succeeded
                    new_size = temp_path.stat().st_size
                    if new_size < 1000:
                        raise FileNotFoundError(
                            f"Download failed - file still appears to be placeholder\n"
                            f"Size after download: {new_size} bytes"
                        )
                    logger.info(f"Successfully downloaded: {file_path.name} ({new_size / 1e6:.1f} MB)")
                    return temp_path  # Return temp path instead
                except Exception as e:
                    logger.error(f"Failed to auto-download {file_path.name}: {e}")
                    raise FileNotFoundError(
                        f"File appears to be Dropbox placeholder and auto-download failed:\n"
                        f"  File: {file_path}\n"
                        f"  Size: {file_size} bytes (suspiciously small)\n"
                        f"  Error: {e}\n"
                        f"Manual solution: Right-click in Finder → 'Make available offline'"
                    )
            else:
                # Auto-download not available
                reason = ""
                if not self.auto_download:
                    reason = "auto_download=False"
                elif not DROPBOX_AVAILABLE:
                    reason = "Dropbox SDK not installed"
                elif not self.dbx:
                    reason = "Dropbox API not connected"

                raise FileNotFoundError(
                    f"File appears to be Dropbox placeholder: {file_path}\n"
                    f"Size: {file_size} bytes (suspiciously small)\n"
                    f"Auto-download disabled: {reason}\n"
                    f"Manual solution: Right-click in Finder → 'Make available offline'"
                )

        logger.debug(f"Verified file downloaded: {file_path.name} ({file_size / 1e6:.1f} MB)")

    def _download_from_dropbox(self, file_path: Path) -> Path:
        """
        Download a file from Dropbox API to temp directory (NOT the Dropbox folder)

        Args:
            file_path: Original Dropbox path (used to determine what to download)

        Returns:
            Path to the downloaded temp file

        Raises:
            Exception: If download fails

        Note:
            Downloads to temp directory to avoid affecting Dropbox sync.
            The original file in Dropbox remains untouched (stays as placeholder).
        """
        # Convert local path to Dropbox path
        # First resolve to absolute path
        file_path = Path(file_path).resolve()
        path_str = str(file_path)

        # Find the Dropbox folder root by looking for /Dropbox/ in the path
        if "/Dropbox/" not in path_str:
            raise ValueError(f"File is not in Dropbox folder: {file_path}")

        # Extract path relative to Dropbox root
        dropbox_relative = path_str.split("/Dropbox/", 1)[1]
        dropbox_path = "/" + dropbox_relative

        logger.debug(f"Downloading from Dropbox path: {dropbox_path}")

        # Download to TEMP directory (OUTSIDE Dropbox folder!)
        temp_file_path = self.temp_dir / file_path.name

        try:
            metadata, response = self.dbx.files_download(dropbox_path)

            # Write to TEMP file ONLY (do NOT touch Dropbox folder at all!)
            with open(temp_file_path, 'wb') as f:
                f.write(response.content)

            # Validate download completed successfully
            actual_size = temp_file_path.stat().st_size
            expected_size = metadata.size

            if actual_size != expected_size:
                logger.error(f"Download size mismatch: expected {expected_size}, got {actual_size}")
                temp_file_path.unlink()  # Delete corrupted file
                raise IOError(
                    f"Download corrupted: size mismatch\n"
                    f"Expected: {expected_size} bytes\n"
                    f"Got: {actual_size} bytes"
                )

            # Verify pickle file can be loaded (catch corruption early)
            if temp_file_path.suffix == '.pkl':
                logger.info(f"Validating pickle file integrity...")
                try:
                    import pickle
                    with open(temp_file_path, 'rb') as f:
                        _ = pickle.load(f)
                    logger.info(f"✓ Pickle file validated successfully")
                except Exception as e:
                    logger.error(f"✗ Downloaded pickle file is CORRUPTED: {e}")
                    temp_file_path.unlink()  # Delete corrupted file
                    raise IOError(
                        f"Downloaded file appears corrupted (pickle load failed):\n"
                        f"Error: {e}\n"
                        f"This may indicate the file in Dropbox is corrupted."
                    )

            logger.debug(f"Downloaded {metadata.size} bytes to temp: {temp_file_path}")

            # Return temp path (do NOT create symlink or modify Dropbox folder)
            return temp_file_path

        except ApiError as e:
            if e.error.is_path() and e.error.get_path().is_not_found():
                raise FileNotFoundError(f"File not found in Dropbox: {dropbox_path}")
            else:
                raise

    def cleanup(self, force: bool = False):
        """
        Remove local copies of embeddings to save disk space

        Args:
            force: If True, delete all embeddings. If False, only delete accessed files.

        Note:
            Files remain in Dropbox cloud and can be re-synced when needed
        """
        if force:
            # Delete all embeddings
            logger.warning("Force cleanup: removing ALL embeddings")
            files_to_delete = list(self.embeddings_dir.glob("*.pkl"))
            files_to_delete += list((self.embeddings_dir / "checkpoints").glob("*.parquet")) if (self.embeddings_dir / "checkpoints").exists() else []
            files_to_delete += list((self.embeddings_dir / "cross_encoder_checkpoints").glob("*.parquet")) if (self.embeddings_dir / "cross_encoder_checkpoints").exists() else []
        else:
            # Only delete accessed files
            files_to_delete = self.accessed_files

        if not files_to_delete:
            logger.info("No files to cleanup")
            return

        # Convert to Path objects if needed
        files_to_delete = [Path(f) if isinstance(f, str) else f for f in files_to_delete]

        total_size = sum(f.stat().st_size for f in files_to_delete if f.exists())
        logger.info(f"Cleaning up {len(files_to_delete)} files ({total_size / 1e9:.2f} GB)")

        for file_path in files_to_delete:
            if file_path.exists() or file_path.is_symlink():
                try:
                    # Remove symlink (doesn't affect the target file in temp or Dropbox)
                    file_path.unlink()
                    logger.debug(f"Removed symlink: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {file_path}: {e}")

        logger.info("Cleanup complete - symlinks removed, files safe in Dropbox and temp cleaned by context exit")

    def cleanup_checkpoints_only(self):
        """Remove checkpoint files but keep main embeddings"""
        checkpoint_dirs = [
            self.embeddings_dir / "checkpoints",
            self.embeddings_dir / "cross_encoder_checkpoints"
        ]

        total_size = 0
        total_files = 0

        for checkpoint_dir in checkpoint_dirs:
            if not checkpoint_dir.exists():
                continue

            for checkpoint_file in checkpoint_dir.glob("*.parquet"):
                if checkpoint_file.exists():
                    size = checkpoint_file.stat().st_size
                    total_size += size
                    total_files += 1
                    checkpoint_file.unlink()
                    logger.debug(f"Deleted checkpoint: {checkpoint_file.name}")

        logger.info(
            f"Deleted {total_files} checkpoint files ({total_size / 1e9:.2f} GB)\n"
            f"Main embeddings preserved"
        )

    def list_embeddings(self) -> dict:
        """
        List all embeddings and their sizes

        Returns:
            Dictionary with categories and file info
        """
        embeddings = {
            "main_embeddings": [],
            "checkpoints": [],
            "cross_encoder_checkpoints": []
        }

        # Main embeddings
        for pkl_file in self.embeddings_dir.glob("*.pkl"):
            if pkl_file.exists():
                size_mb = pkl_file.stat().st_size / 1e6
                embeddings["main_embeddings"].append({
                    "name": pkl_file.name,
                    "size_mb": size_mb,
                    "path": str(pkl_file)
                })

        # Checkpoints
        checkpoint_dir = self.embeddings_dir / "checkpoints"
        if checkpoint_dir.exists():
            for parquet_file in checkpoint_dir.glob("*.parquet"):
                if parquet_file.exists():
                    size_mb = parquet_file.stat().st_size / 1e6
                    embeddings["checkpoints"].append({
                        "name": parquet_file.name,
                        "size_mb": size_mb,
                        "path": str(parquet_file)
                    })

        # Cross encoder checkpoints
        ce_checkpoint_dir = self.embeddings_dir / "cross_encoder_checkpoints"
        if ce_checkpoint_dir.exists():
            for parquet_file in ce_checkpoint_dir.glob("*.parquet"):
                if parquet_file.exists():
                    size_mb = parquet_file.stat().st_size / 1e6
                    embeddings["cross_encoder_checkpoints"].append({
                        "name": parquet_file.name,
                        "size_mb": size_mb,
                        "path": str(parquet_file)
                    })

        return embeddings

    def print_summary(self):
        """Print summary of embeddings and sizes"""
        embeddings = self.list_embeddings()

        print("\n" + "="*80)
        print("EMBEDDING STORAGE SUMMARY")
        print("="*80)

        for category, files in embeddings.items():
            if not files:
                continue

            total_size = sum(f["size_mb"] for f in files)
            print(f"\n{category.replace('_', ' ').title()}: {len(files)} files ({total_size / 1e3:.2f} GB)")

            for file_info in sorted(files, key=lambda x: x["size_mb"], reverse=True)[:5]:
                print(f"  - {file_info['name']}: {file_info['size_mb']:.1f} MB")

            if len(files) > 5:
                print(f"  ... and {len(files) - 5} more files")

        total_all = sum(
            sum(f["size_mb"] for f in files)
            for files in embeddings.values()
        )
        print(f"\nTotal storage: {total_all / 1e3:.2f} GB")
        print("="*80 + "\n")


def main():
    """CLI interface for embedding manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Manage embedding files")
    parser.add_argument("--list", action="store_true", help="List all embeddings")
    parser.add_argument("--cleanup", action="store_true", help="Remove all local embeddings")
    parser.add_argument("--cleanup-checkpoints", action="store_true", help="Remove only checkpoint files")
    parser.add_argument("--embeddings-dir", help="Path to embeddings directory")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    em = EmbeddingManager(embeddings_dir=args.embeddings_dir)

    if args.list:
        em.print_summary()
    elif args.cleanup:
        confirm = input("Delete all local embeddings? (yes/no): ")
        if confirm.lower() == "yes":
            em.cleanup(force=True)
        else:
            print("Cleanup cancelled")
    elif args.cleanup_checkpoints:
        confirm = input("Delete checkpoint files only? (yes/no): ")
        if confirm.lower() == "yes":
            em.cleanup_checkpoints_only()
        else:
            print("Cleanup cancelled")
    else:
        em.print_summary()


if __name__ == "__main__":
    main()
