#!/usr/bin/env python3
"""
Real file integration tests for Stage 6
Tests that actual data files load correctly with proper column detection and parsing
"""

import pandas as pd
import pytest
from pathlib import Path
import sys
sys.path.append('..')

from stage_6_link_exposure_to_jobs import (
    load_crosswalk_x28_to_isco,
    load_crosswalk_isco_to_onet,
    load_stage5_exposures,
    parse_x28_occupations
)

class TestRealFileIntegration:
    """Test integration with actual data files in the Data/ directory"""

    @pytest.fixture
    def data_dir(self):
        """Path to actual data directory"""
        return Path(__file__).parent.parent / "Data"

    def test_x28_isco_crosswalk_loads_correctly(self, data_dir):
        """Test that actual X28→ISCO crosswalk file loads with correct format"""
        crosswalk_file = data_dir / "240711_occupation_to_ch_isco_19.csv"

        # Skip if file doesn't exist (optional for CI)
        if not crosswalk_file.exists():
            pytest.skip(f"Crosswalk file not found: {crosswalk_file}")

        # Load crosswalk
        df = load_crosswalk_x28_to_isco(str(crosswalk_file))

        # Verify structure
        assert len(df) > 0, "Crosswalk should not be empty"
        assert 'x28_code' in df.columns, "Should have x28_code column"
        assert 'isco_code' in df.columns, "Should have isco_code column"

        # Verify data types
        assert df['x28_code'].dtype == 'object', "X28 codes should be strings"
        assert df['isco_code'].dtype == 'object', "ISCO codes should be strings"

        # Verify no null values in key columns
        assert df['x28_code'].notna().all(), "No null X28 codes allowed"
        assert df['isco_code'].notna().all(), "No null ISCO codes allowed"

        # Verify reasonable data ranges
        assert len(df) > 1000, "Should have substantial number of mappings"
        assert len(df) < 10000, "Sanity check - shouldn't be too large"

        # Check sample data format
        sample_x28 = df['x28_code'].iloc[0]
        sample_isco = df['isco_code'].iloc[0]

        assert len(str(sample_x28)) >= 4, "X28 codes should be reasonable length"
        assert len(str(sample_isco)) >= 3, "ISCO codes should be reasonable length"

    def test_isco_onet_crosswalk_loads_correctly(self, data_dir):
        """Test that actual ISCO→ONET Excel crosswalk loads with correct format"""
        crosswalk_file = data_dir / "ESCO_to_ONET-SOC.xlsx"

        if not crosswalk_file.exists():
            pytest.skip(f"ESCO crosswalk file not found: {crosswalk_file}")

        # Load crosswalk
        df = load_crosswalk_isco_to_onet(str(crosswalk_file))

        # Verify structure
        assert len(df) > 0, "Crosswalk should not be empty"
        assert 'isco_code' in df.columns, "Should have isco_code column"
        assert 'onet_soc' in df.columns, "Should have onet_soc column"

        # Verify data quality
        assert df['isco_code'].notna().all(), "No null ISCO codes allowed"
        assert df['onet_soc'].notna().all(), "No null ONET codes allowed"

        # Verify reasonable data ranges
        assert len(df) > 1000, "Should have substantial number of mappings"

        # Check ISCO format (should be like 2521.10 or similar)
        sample_isco = str(df['isco_code'].iloc[0])
        assert '.' in sample_isco or len(sample_isco) >= 4, "ISCO codes should be detailed"

        # Check ONET format (should be like 15-1111.00)
        sample_onet = str(df['onet_soc'].iloc[0])
        assert '-' in sample_onet or len(sample_onet) >= 6, "ONET codes should be standard format"

    def test_stage5_exposure_files_load_correctly(self, data_dir):
        """Test that actual Stage 5 exposure files load with expected format"""
        # Test both time-variant and time-invariant files if they exist
        exposure_files = [
            "isco_firm_year_ai_exposure_time_variant.csv",
            "onet_firm_year_ai_exposure_time_variant.csv",
            "isco_firm_year_ai_exposure_time_invariant.csv",
            "onet_firm_year_ai_exposure_time_invariant.csv"
        ]

        found_files = []
        for filename in exposure_files:
            filepath = data_dir / filename
            if filepath.exists():
                found_files.append((filepath, 'isco' if 'isco' in filename else 'onet'))

        if not found_files:
            pytest.skip("No Stage 5 exposure files found")

        for filepath, occ_type in found_files:
            # Load exposure data
            df = load_stage5_exposures(str(filepath), occ_type)

            # Verify required columns
            expected_cols = ['company_id', 'year', f'{occ_type}_code', 'ai_exposure_hampole']
            for col in expected_cols:
                assert col in df.columns, f"Missing required column: {col}"

            # Verify data types
            assert df['company_id'].dtype == 'object', "Company IDs should be strings"
            assert pd.api.types.is_integer_dtype(df['year']), "Years should be integers"
            assert pd.api.types.is_numeric_dtype(df['ai_exposure_hampole']), "Exposure should be numeric"

            # Verify data ranges
            assert len(df) > 0, "Exposure data should not be empty"
            assert df['year'].min() >= 2010, "Years should be reasonable"
            assert df['year'].max() <= 2030, "Years should be reasonable"
            assert df['ai_exposure_hampole'].min() >= 0, "Exposure should be non-negative"
            assert df['ai_exposure_hampole'].max() <= 10, "Exposure should be reasonable range"

    def test_postgresql_array_parsing_formats(self):
        """Test X28 parsing with actual PostgreSQL array formats"""
        # Test various PostgreSQL array formats we might encounter
        test_cases = [
            # Standard PostgreSQL array format
            ('{11001282,11001335}', ['11001282', '11001335']),
            # Single element array
            ('{11001282}', ['11001282']),
            # Empty array
            ('{}', []),
            # Array with quotes (sometimes happens)
            ('{"11001282","11001335"}', ['11001282', '11001335']),
            # Nested or complex formats
            ("{11001282, 11001335}", ['11001282', '11001335']),
        ]

        for input_val, expected in test_cases:
            result = parse_x28_occupations(input_val)
            assert result == expected, f"Failed parsing {input_val}"

    def test_crosswalk_coverage_analysis(self, data_dir):
        """Test coverage of X28 codes in crosswalk vs typical job data"""
        x28_crosswalk_file = data_dir / "240711_occupation_to_ch_isco_19.csv"

        if not x28_crosswalk_file.exists():
            pytest.skip("X28 crosswalk file not available")

        # Load crosswalk
        crosswalk = load_crosswalk_x28_to_isco(str(x28_crosswalk_file))
        available_x28_codes = set(crosswalk['x28_code'])

        # Test with some common X28 codes that should be mappable
        common_x28_codes = ['11001282', '11001335', '11000441']  # Example codes

        coverage = len(set(common_x28_codes) & available_x28_codes)
        total = len(common_x28_codes)
        coverage_pct = (coverage / total) * 100

        # We expect reasonable coverage but not necessarily 100%
        assert coverage_pct >= 50, f"X28 coverage too low: {coverage_pct:.1f}%"

        print(f"X28 crosswalk coverage: {coverage_pct:.1f}% ({coverage}/{total})")
        print(f"Total X28 mappings available: {len(available_x28_codes):,}")

    def test_file_format_robustness(self, data_dir):
        """Test robustness with various file format edge cases"""
        x28_file = data_dir / "240711_occupation_to_ch_isco_19.csv"

        if not x28_file.exists():
            pytest.skip("X28 crosswalk file not available")

        # Test that we can handle the actual semicolon-separated format
        with open(x28_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        # Should detect semicolon separator
        assert ';' in first_line, "File should use semicolon separators"

        # Should have expected header structure
        headers = first_line.split(';')
        assert len(headers) >= 3, "Should have multiple columns"

        # Should be able to load despite format challenges
        df = load_crosswalk_x28_to_isco(str(x28_file))
        assert len(df) > 0, "Should successfully load despite format complexity"

    def test_excel_header_complexity(self, data_dir):
        """Test handling of complex Excel headers in ESCO file"""
        excel_file = data_dir / "ESCO_to_ONET-SOC.xlsx"

        if not excel_file.exists():
            pytest.skip("ESCO Excel file not available")

        # Should handle complex headers and return clean data
        df = load_crosswalk_isco_to_onet(str(excel_file))

        # Verify it parsed the headers correctly
        assert 'isco_code' in df.columns, "Should extract ISCO codes correctly"
        assert 'onet_soc' in df.columns, "Should extract ONET codes correctly"

        # Verify data looks reasonable despite complex headers
        assert len(df) > 0, "Should extract data despite header complexity"

        # Check first few values are not header remnants
        first_isco = str(df['isco_code'].iloc[0])
        first_onet = str(df['onet_soc'].iloc[0])

        assert not any(word in first_isco.lower() for word in ['code', 'title', 'esco', 'isco']), \
               "ISCO values should be codes, not headers"
        assert not any(word in first_onet.lower() for word in ['code', 'title', 'onet', 'soc']), \
               "ONET values should be codes, not headers"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])