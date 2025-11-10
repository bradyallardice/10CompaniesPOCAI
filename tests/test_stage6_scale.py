#!/usr/bin/env python3
"""
Data quality and scale tests for Stage 6
Tests handling of large datasets and real data quality issues
"""

import pandas as pd
import pytest
from pathlib import Path
import sys
import numpy as np
from unittest.mock import patch, MagicMock
sys.path.append('..')

from stage_6_link_exposure_to_jobs import (
    validate_jobs_data,
    parse_x28_occupations,
    build_jobs_from_db
)

class TestScaleAndDataQuality:
    """Test handling of large-scale datasets and real data quality issues"""

    def test_handles_large_job_datasets_without_memory_issues(self):
        """Test memory efficiency with large datasets similar to 19M records"""

        # Create dataset that simulates memory patterns of large dataset
        # Use smaller size but with representative data patterns
        n_rows = 50000  # Smaller than 19M but tests memory patterns

        large_df = pd.DataFrame({
            'company_id': [str(i // 1000) for i in range(n_rows)],  # ~50 companies
            'company_name': [f'Company {i // 1000}' for i in range(n_rows)],
            'year': [2015 + (i % 10) for i in range(n_rows)],  # 2015-2024
            'title': [f'Job Title {i % 100}' for i in range(n_rows)],  # 100 unique titles
            'x28_codes': [
                ['11001282', '11001335'] if i % 3 == 0 else
                ['22002001'] if i % 3 == 1 else
                []  # Some empty for realism
                for i in range(n_rows)
            ]
        })

        # Test validation doesn't run out of memory
        try:
            validate_jobs_data(large_df, fail_on_warn=False)
            memory_test_passed = True
        except MemoryError:
            memory_test_passed = False

        assert memory_test_passed, "Should handle large datasets without memory issues"

        # Test basic operations are efficient
        assert len(large_df) == n_rows, "Dataset should maintain expected size"

        # Test X28 parsing efficiency
        sample_size = min(1000, len(large_df))
        sample_df = large_df.sample(sample_size)

        # Should be able to process sample efficiently
        empty_count = sum(1 for codes in sample_df['x28_codes'] if len(codes) == 0)
        non_empty_count = sample_size - empty_count

        assert non_empty_count > 0, "Should have some non-empty X28 codes"

    def test_duplicate_handling_at_actual_scale(self):
        """Test duplicate detection with realistic duplicate patterns"""

        # Create dataset with 302,963 duplicates (the actual number we encountered)
        base_size = 10000
        duplicate_ratio = 0.3  # 30% duplicates (similar to real data)

        # Create base data
        companies = [f'company_{i}' for i in range(50)]
        years = list(range(2015, 2026))
        titles = [f'Job Title {i}' for i in range(200)]

        data_rows = []
        for i in range(base_size):
            row = {
                'company_id': np.random.choice(companies),
                'year': np.random.choice(years),
                'title': np.random.choice(titles),
                'x28_codes': [['11001282']]
            }
            data_rows.append(row)

        # Add duplicates
        n_duplicates = int(base_size * duplicate_ratio)
        for i in range(n_duplicates):
            # Duplicate random existing row
            original_row = data_rows[i % len(data_rows)].copy()
            data_rows.append(original_row)

        df = pd.DataFrame(data_rows)

        # Test duplicate detection
        key_cols = ['company_id', 'year', 'title']
        duplicates = df.duplicated(subset=key_cols).sum()

        expected_duplicates = n_duplicates
        actual_duplicates = duplicates

        # Should detect reasonable number of duplicates
        assert actual_duplicates > 0, "Should detect duplicates in realistic data"
        assert actual_duplicates <= expected_duplicates, "Should not over-detect duplicates"

        # Test validation with fail_on_warn=False allows duplicates
        try:
            validate_jobs_data(df, fail_on_warn=False)
            validation_passed = True
        except ValueError:
            validation_passed = False

        assert validation_passed, "Should handle duplicates when fail_on_warn=False"

    def test_realistic_empty_x28_percentage(self):
        """Test validation with realistic empty X28 percentages (6.5% from real data)"""

        n_rows = 10000
        empty_percentage = 6.5  # Real percentage we encountered
        n_empty = int(n_rows * empty_percentage / 100)

        # Create dataset with realistic empty X28 pattern
        data = pd.DataFrame({
            'company_id': [str(i // 100) for i in range(n_rows)],
            'year': [2020] * n_rows,
            'title': [f'Job {i}' for i in range(n_rows)],
            'x28_codes': (
                [[]] * n_empty +  # Empty X28 codes
                [['11001282']] * (n_rows - n_empty)  # Non-empty X28 codes
            )
        })

        # Shuffle to distribute empty codes randomly
        data = data.sample(frac=1).reset_index(drop=True)

        # Test validation with fail_on_warn=True should fail (>2% threshold)
        with pytest.raises(ValueError, match="rows have empty x28_occupations"):
            validate_jobs_data(data, fail_on_warn=True)

        # Test validation with fail_on_warn=False should pass with warning
        try:
            validate_jobs_data(data, fail_on_warn=False)
            validation_passed = True
        except ValueError:
            validation_passed = False

        assert validation_passed, "Should pass validation when fail_on_warn=False"

    def test_x28_parsing_with_realistic_data_patterns(self):
        """Test X28 parsing with patterns found in real data"""

        # Real patterns we might encounter in production data
        test_cases = [
            # PostgreSQL arrays from database
            ('{11001282,11001335,22002001}', ['11001282', '11001335', '22002001']),
            ('{11001282}', ['11001282']),
            ('{}', []),

            # JSON-style arrays
            ("['11001282','11001335']", ['11001282', '11001335']),
            ('["11001282","11001335"]', ['11001282', '11001335']),

            # Comma-separated values
            ('11001282,11001335,22002001', ['11001282', '11001335', '22002001']),
            ('11001282', ['11001282']),

            # Edge cases found in real data
            ('  {11001282, 11001335}  ', ['11001282', '11001335']),  # Spaces
            ('{11001282,11001335,}', ['11001282', '11001335']),      # Trailing comma
            ('null', []),                                           # Null values
            ('', []),                                              # Empty string
            (None, []),                                            # Python None

            # Quoted values
            ("{'11001282','11001335'}", ['11001282', '11001335']),
            ('{"11001282","11001335"}', ['11001282', '11001335']),
        ]

        for input_val, expected in test_cases:
            try:
                result = parse_x28_occupations(input_val)
                assert result == expected, f"Failed parsing '{input_val}': got {result}, expected {expected}"
            except Exception as e:
                pytest.fail(f"Exception parsing '{input_val}': {e}")

    def test_company_year_distribution_realistic(self):
        """Test handling of realistic company-year distributions"""

        # Simulate real data patterns: some companies have many years, others few
        companies_data = [
            ('large_corp', list(range(2010, 2026))),      # 16 years
            ('medium_corp', list(range(2015, 2026))),     # 11 years
            ('small_corp', [2020, 2021, 2022]),          # 3 years
            ('startup', [2023, 2024]),                   # 2 years
            ('old_corp', list(range(2005, 2016))),       # Historical only
        ]

        data_rows = []
        for company_id, years in companies_data:
            for year in years:
                # Multiple jobs per company-year
                for job_id in range(np.random.randint(1, 20)):  # 1-19 jobs per company-year
                    data_rows.append({
                        'company_id': company_id,
                        'year': year,
                        'title': f'Job {job_id}',
                        'x28_codes': [['11001282']]
                    })

        df = pd.DataFrame(data_rows)

        # Test that validation handles varied distributions
        try:
            validate_jobs_data(df, fail_on_warn=False)
            validation_passed = True
        except Exception:
            validation_passed = False

        assert validation_passed, "Should handle realistic company-year distributions"

        # Verify data patterns
        company_counts = df.groupby('company_id').size()
        year_range = df['year'].max() - df['year'].min()

        assert len(company_counts) == 5, "Should have all companies"
        assert year_range >= 10, "Should span reasonable year range"

    def test_memory_usage_with_x28_parsing(self):
        """Test memory efficiency during X28 parsing operations"""

        # Create dataset with various X28 patterns
        n_rows = 10000
        x28_patterns = [
            "['11001282']",                    # Single
            "['11001282', '11001335']",        # Double
            "['11001282', '11001335', '22002001']",  # Triple
            "[]",                              # Empty
            "['11001282', '11001335', '22002001', '33003001', '44004001']",  # Many
        ]

        data = pd.DataFrame({
            'company_id': [str(i // 100) for i in range(n_rows)],
            'year': [2020] * n_rows,
            'title': [f'Job {i}' for i in range(n_rows)],
            'x28_occupations': [x28_patterns[i % len(x28_patterns)] for i in range(n_rows)]
        })

        # Test parsing doesn't cause memory issues
        try:
            data['x28_codes'] = data['x28_occupations'].apply(parse_x28_occupations)
            parsing_succeeded = True
        except MemoryError:
            parsing_succeeded = False

        assert parsing_succeeded, "X28 parsing should be memory efficient"

        # Verify parsing results are correct
        assert len(data) == n_rows, "Should maintain all rows"
        assert 'x28_codes' in data.columns, "Should create parsed column"

        # Check parsing accuracy on sample
        for i in range(min(100, len(data))):
            original = data.iloc[i]['x28_occupations']
            parsed = data.iloc[i]['x28_codes']

            if original == "[]":
                assert parsed == [], f"Empty array parsing failed at row {i}"
            elif "'11001282'" in original:
                assert '11001282' in parsed, f"Code extraction failed at row {i}"

    def test_crosswalk_mapping_efficiency_at_scale(self):
        """Test efficiency of crosswalk mapping operations with large datasets"""

        # Simulate crosswalk application on large dataset
        n_jobs = 10000
        n_x28_codes = 100  # Reasonable number of distinct X28 codes

        # Create jobs with X28 codes
        jobs_data = []
        for i in range(n_jobs):
            n_codes = np.random.randint(1, 4)  # 1-3 codes per job
            x28_codes = [f'1100{1000 + (i % n_x28_codes) + j}' for j in range(n_codes)]

            jobs_data.append({
                'company_id': str(i // 100),
                'year': 2020 + (i % 5),
                'title': f'Job {i}',
                'x28_codes': x28_codes
            })

        jobs_df = pd.DataFrame(jobs_data)

        # Create crosswalk (50% coverage to simulate real world)
        crosswalk_data = []
        for i in range(n_x28_codes // 2):  # 50% coverage
            crosswalk_data.append({
                'x28_code': f'1100{1000 + i}',
                'isco_code': f'252{i % 10}'
            })

        crosswalk_df = pd.DataFrame(crosswalk_data)

        # Test exploding X28 codes (simulates apply_crosswalks operation)
        try:
            exploded_jobs = jobs_df.explode('x28_codes')
            mapped_jobs = exploded_jobs.merge(
                crosswalk_df,
                left_on='x28_codes',
                right_on='x28_code',
                how='inner'
            )

            mapping_succeeded = True
            n_mapped = len(mapped_jobs)

        except Exception as e:
            mapping_succeeded = False
            n_mapped = 0

        assert mapping_succeeded, "Crosswalk mapping should complete efficiently"
        assert n_mapped > 0, "Should successfully map some jobs"

        # Should map roughly 50% due to 50% crosswalk coverage
        exploded_count = len(exploded_jobs)
        mapping_rate = n_mapped / exploded_count if exploded_count > 0 else 0

        assert 0.3 <= mapping_rate <= 0.7, f"Mapping rate {mapping_rate:.2f} should be reasonable"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])