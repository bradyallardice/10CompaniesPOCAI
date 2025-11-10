#!/usr/bin/env python3
"""
Cache system tests for Stage 6
Tests the parquet caching functionality for job list data
"""

import pandas as pd
import pytest
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch, MagicMock
sys.path.append('..')

from stage_6_link_exposure_to_jobs import build_jobs_from_db, parse_x28_occupations

class TestCacheSystem:
    """Test the parquet caching system for job data"""

    @pytest.fixture
    def sample_jobs_data(self):
        """Sample jobs data for cache testing"""
        return pd.DataFrame({
            'company_id': ['1', '1', '2', '2'],
            'company_name': ['Alpha AG', 'Alpha AG', 'Beta SA', 'Beta SA'],
            'year': [2020, 2021, 2020, 2021],
            'title': ['Data Scientist', 'ML Engineer', 'BI Analyst', 'Data Engineer'],
            'x28_occupations': [
                "['11001282', '11001335']",
                "['22002001']",
                "['33003001']",
                "['11001282']"
            ]
        })

    def test_cache_saves_and_loads_correctly(self, sample_jobs_data, tmp_path):
        """Test that cache saves raw data and loads to identical results"""

        # Mock database connection and query
        mock_conn = MagicMock()

        # Mock the SQL query to return our sample data
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = sample_jobs_data

            with patch('stage_6_link_exposure_to_jobs.log') as mock_log:
                # Change working directory to tmp_path for cache files
                original_cwd = Path.cwd()

                try:
                    import os
                    os.chdir(tmp_path)

                    # Create Data directory in tmp_path
                    data_dir = tmp_path / "Data"
                    data_dir.mkdir(exist_ok=True)

                    # First call - should query database and create cache
                    result1 = build_jobs_from_db(mock_conn, 2025, fail_on_warn=False)

                    # Verify cache file was created
                    cache_file = data_dir / "stage6_job_cache_max2025.parquet"
                    assert cache_file.exists(), "Cache file should be created"

                    # Second call - should load from cache
                    result2 = build_jobs_from_db(mock_conn, 2025, fail_on_warn=False)

                    # Verify both results are identical
                    pd.testing.assert_frame_equal(
                        result1.sort_values(['company_id', 'year', 'title']).reset_index(drop=True),
                        result2.sort_values(['company_id', 'year', 'title']).reset_index(drop=True)
                    )

                    # Verify database was only queried once (second call used cache)
                    assert mock_read_sql.call_count == 1, "Database should only be queried once"

                finally:
                    os.chdir(original_cwd)

    def test_cache_different_year_max_creates_separate_files(self, sample_jobs_data, tmp_path):
        """Test that different year_max values create separate cache files"""

        mock_conn = MagicMock()

        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = sample_jobs_data

            with patch('stage_6_link_exposure_to_jobs.log'):
                original_cwd = Path.cwd()

                try:
                    import os
                    os.chdir(tmp_path)

                    data_dir = tmp_path / "Data"
                    data_dir.mkdir(exist_ok=True)

                    # Call with year_max=2025
                    build_jobs_from_db(mock_conn, 2025, fail_on_warn=False)
                    cache_2025 = data_dir / "stage6_job_cache_max2025.parquet"
                    assert cache_2025.exists(), "Cache for 2025 should be created"

                    # Call with year_max=2024
                    build_jobs_from_db(mock_conn, 2024, fail_on_warn=False)
                    cache_2024 = data_dir / "stage6_job_cache_max2024.parquet"
                    assert cache_2024.exists(), "Cache for 2024 should be created"

                    # Both files should exist and be different
                    assert cache_2025.stat().st_size > 0
                    assert cache_2024.stat().st_size > 0

                finally:
                    os.chdir(original_cwd)

    def test_cache_corruption_fallback_to_database(self, sample_jobs_data, tmp_path):
        """Test graceful fallback to database when cache is corrupted"""

        mock_conn = MagicMock()

        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = sample_jobs_data

            with patch('stage_6_link_exposure_to_jobs.log') as mock_log:
                original_cwd = Path.cwd()

                try:
                    import os
                    os.chdir(tmp_path)

                    data_dir = tmp_path / "Data"
                    data_dir.mkdir(exist_ok=True)

                    # Create corrupted cache file
                    cache_file = data_dir / "stage6_job_cache_max2025.parquet"
                    cache_file.write_text("corrupted data")

                    # Should detect corruption and fall back to database
                    result = build_jobs_from_db(mock_conn, 2025, fail_on_warn=False)

                    # Verify it worked despite corrupted cache
                    assert len(result) > 0, "Should return data despite cache corruption"
                    assert 'x28_codes' in result.columns, "Should have parsed X28 codes"

                    # Verify fallback message was logged
                    mock_log.assert_any_call("🔄 Falling back to database query...")

                finally:
                    os.chdir(original_cwd)

    def test_cache_preserves_raw_data_format(self, sample_jobs_data, tmp_path):
        """Test that cache preserves raw x28_occupations without parsed x28_codes"""

        mock_conn = MagicMock()

        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = sample_jobs_data

            with patch('stage_6_link_exposure_to_jobs.log'):
                original_cwd = Path.cwd()

                try:
                    import os
                    os.chdir(tmp_path)

                    data_dir = tmp_path / "Data"
                    data_dir.mkdir(exist_ok=True)

                    # Generate cache
                    result = build_jobs_from_db(mock_conn, 2025, fail_on_warn=False)

                    # Read cache file directly
                    cache_file = data_dir / "stage6_job_cache_max2025.parquet"
                    cached_data = pd.read_parquet(cache_file)

                    # Cache should have raw x28_occupations but not parsed x28_codes
                    assert 'x28_occupations' in cached_data.columns, "Cache should preserve raw data"
                    assert 'x28_codes' not in cached_data.columns, "Cache should not include parsed data"

                    # Result should have parsed data
                    assert 'x28_codes' in result.columns, "Result should include parsed data"

                    # Raw data in cache should match original
                    original_x28 = sample_jobs_data['x28_occupations'].iloc[0]
                    cached_x28 = cached_data['x28_occupations'].iloc[0]
                    assert original_x28 == cached_x28, "Raw data should be preserved exactly"

                finally:
                    os.chdir(original_cwd)

    def test_cache_handles_large_datasets_efficiently(self, tmp_path):
        """Test cache efficiency with larger datasets (memory usage)"""

        # Create larger sample dataset
        large_data = pd.DataFrame({
            'company_id': [str(i // 1000) for i in range(10000)],
            'company_name': [f'Company {i // 1000}' for i in range(10000)],
            'year': [2020 + (i % 5) for i in range(10000)],
            'title': [f'Job {i}' for i in range(10000)],
            'x28_occupations': [f"['{11000000 + i}']" for i in range(10000)]
        })

        mock_conn = MagicMock()

        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = large_data

            with patch('stage_6_link_exposure_to_jobs.log'):
                original_cwd = Path.cwd()

                try:
                    import os
                    os.chdir(tmp_path)

                    data_dir = tmp_path / "Data"
                    data_dir.mkdir(exist_ok=True)

                    # Process large dataset
                    result = build_jobs_from_db(mock_conn, 2025, fail_on_warn=False)

                    # Verify it completed successfully
                    assert len(result) == 10000, "Should process all records"
                    assert 'x28_codes' in result.columns, "Should parse X28 codes"

                    # Verify cache file was created and has reasonable size
                    cache_file = data_dir / "stage6_job_cache_max2025.parquet"
                    assert cache_file.exists(), "Cache should be created"

                    # Parquet should compress reasonably
                    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                    assert file_size_mb < 10, f"Cache file too large: {file_size_mb:.1f} MB"

                finally:
                    os.chdir(original_cwd)

    def test_cache_x28_parsing_consistency(self, tmp_path):
        """Test that X28 parsing is consistent between cached and fresh data"""

        test_data = pd.DataFrame({
            'company_id': ['1', '2', '3'],
            'company_name': ['Co A', 'Co B', 'Co C'],
            'year': [2020, 2021, 2022],
            'title': ['Job A', 'Job B', 'Job C'],
            'x28_occupations': [
                "['11001282', '11001335']",  # Multiple codes
                "[]",  # Empty array
                "['22002001']"  # Single code
            ]
        })

        mock_conn = MagicMock()

        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = test_data

            with patch('stage_6_link_exposure_to_jobs.log'):
                original_cwd = Path.cwd()

                try:
                    import os
                    os.chdir(tmp_path)

                    data_dir = tmp_path / "Data"
                    data_dir.mkdir(exist_ok=True)

                    # First call - database
                    result1 = build_jobs_from_db(mock_conn, 2025, fail_on_warn=False)

                    # Second call - cache
                    result2 = build_jobs_from_db(mock_conn, 2025, fail_on_warn=False)

                    # X28 parsing should be identical
                    for i in range(len(result1)):
                        codes1 = result1.iloc[i]['x28_codes']
                        codes2 = result2.iloc[i]['x28_codes']
                        assert codes1 == codes2, f"X28 parsing mismatch at row {i}: {codes1} vs {codes2}"

                finally:
                    os.chdir(original_cwd)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])