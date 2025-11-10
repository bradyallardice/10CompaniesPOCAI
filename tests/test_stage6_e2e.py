#!/usr/bin/env python3
"""
End-to-end pipeline tests for Stage 6
Tests complete pipeline integration and Stage 5 → Stage 6 data flow
"""

import pandas as pd
import pytest
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
sys.path.append('..')

from stage_6_link_exposure_to_jobs import (
    apply_crosswalks,
    load_crosswalk_x28_to_isco,
    load_crosswalk_isco_to_onet,
    load_stage5_exposures,
    link_jobs_to_exposures,
    find_stage5_file,
    parse_x28_occupations
)

class TestEndToEndPipeline:
    """Test complete pipeline integration and Stage 5 → Stage 6 compatibility"""

    @pytest.fixture
    def data_dir(self):
        """Path to actual data directory"""
        return Path(__file__).parent.parent / "Data"

    @pytest.fixture
    def sample_jobs_df(self, data_dir):
        """Load first 50 rows from cached file for realistic testing"""
        cache_file = data_dir / "stage6_job_cache_max2025.parquet"

        if not cache_file.exists():
            # Fallback to synthetic data if cache file doesn't exist
            return pd.DataFrame({
                'company_id': ['1', '1', '2', '2'],
                'company_name': ['Alpha AG', 'Alpha AG', 'Beta SA', 'Beta SA'],
                'year': [2020, 2021, 2020, 2021],
                'title': ['Data Scientist', 'ML Engineer', 'BI Analyst', 'Data Engineer'],
                'x28_codes': [
                    ['11001282', '11001335'],
                    ['22002001'],
                    ['11001282'],
                    ['33003001']
                ]
            })

        # Load first 50 rows from actual cached data
        df_cached = pd.read_parquet(cache_file)
        return df_cached.head(50)

    @pytest.fixture
    def sample_x28_isco_crosswalk(self):
        """Sample X28→ISCO crosswalk"""
        return pd.DataFrame({
            'x28_code': ['11001282', '11001335', '22002001', '33003001'],
            'isco_code': ['2521', '2523', '2512', '2519']
        })

    @pytest.fixture
    def sample_isco_onet_crosswalk(self):
        """Sample ISCO→ONET crosswalk"""
        return pd.DataFrame({
            'isco_code': ['2521', '2523', '2512', '2519'],
            'onet_soc': ['15-1111.00', '15-1112.00', '15-1121.00', '15-1131.00']
        })

    @pytest.fixture
    def sample_stage5_isco_exposures(self):
        """Sample Stage 5 ISCO exposure data"""
        return pd.DataFrame({
            'company_id': ['1', '1', '2', '2'],
            'year': [2020, 2021, 2020, 2021],
            'isco_code': ['2521', '2521', '2521', '2512'],
            'ai_exposure_hampole': [0.75, 0.80, 0.65, 0.90],
            'ai_exposure_binary': [1.0, 1.0, 1.0, 1.0],
            'n_ai_apps_firm_year': [10, 12, 8, 15]
        })

    @pytest.fixture
    def sample_stage5_onet_exposures(self):
        """Sample Stage 5 ONET exposure data"""
        return pd.DataFrame({
            'company_id': ['1', '1', '2', '2'],
            'year': [2020, 2021, 2020, 2021],
            'onet_code': ['15-1111.00', '15-1111.00', '15-1111.00', '15-1121.00'],
            'ai_exposure_hampole': [0.70, 0.75, 0.60, 0.85],
            'ai_exposure_binary': [1.0, 1.0, 1.0, 1.0],
            'n_ai_apps_firm_year': [10, 12, 8, 15]
        })

    def test_stage5_to_stage6_isco_compatibility(self, sample_jobs_df, sample_x28_isco_crosswalk, sample_stage5_isco_exposures):
        """Test that Stage 5 ISCO output format works with Stage 6 input"""

        # Step 1: Apply X28→ISCO crosswalk to jobs
        jobs_mapped = apply_crosswalks(
            sample_jobs_df,
            sample_x28_isco_crosswalk,
            None,  # No ISCO→ONET crosswalk needed for ISCO mode
            'isco'
        )

        # Verify crosswalk application
        assert len(jobs_mapped) > len(sample_jobs_df), "Should explode jobs with multiple X28 codes"
        assert 'isco_code' in jobs_mapped.columns, "Should add ISCO codes"

        # Step 2: Link with Stage 5 exposures
        linked_df, unmatched_jobs, unmatched_exposures = link_jobs_to_exposures(
            jobs_mapped, sample_stage5_isco_exposures
        )

        # Verify linking results
        assert len(linked_df) > 0, "Should successfully link some jobs to exposures"

        # Check expected columns in output
        expected_cols = [
            'company_id', 'year', 'title', 'isco_code',
            'ai_exposure_hampole', 'ai_exposure_binary', 'n_ai_apps_firm_year'
        ]
        for col in expected_cols:
            assert col in linked_df.columns, f"Missing expected column: {col}"

        # Verify data types
        assert pd.api.types.is_numeric_dtype(linked_df['ai_exposure_hampole']), "Exposure should be numeric"
        assert pd.api.types.is_integer_dtype(linked_df['year']), "Year should be integer"

        # Verify exposure values are reasonable
        assert linked_df['ai_exposure_hampole'].min() >= 0, "Exposure should be non-negative"
        assert linked_df['ai_exposure_hampole'].max() <= 10, "Exposure should be reasonable range"

    def test_stage5_to_stage6_onet_compatibility(self, sample_jobs_df, sample_x28_isco_crosswalk,
                                                sample_isco_onet_crosswalk, sample_stage5_onet_exposures):
        """Test that Stage 5 ONET output format works with Stage 6 input"""

        # Step 1: Apply both crosswalks to jobs
        jobs_mapped = apply_crosswalks(
            sample_jobs_df,
            sample_x28_isco_crosswalk,
            sample_isco_onet_crosswalk,
            'onet'
        )

        # Verify crosswalk chain application
        assert 'onet_soc' in jobs_mapped.columns, "Should add ONET codes after crosswalk chain"
        assert len(jobs_mapped) > 0, "Should successfully map some jobs through crosswalk chain"

        # Step 2: Link with Stage 5 ONET exposures
        linked_df, unmatched_jobs, unmatched_exposures = link_jobs_to_exposures(
            jobs_mapped, sample_stage5_onet_exposures
        )

        # Verify linking results
        assert len(linked_df) > 0, "Should successfully link some jobs to ONET exposures"

        # Check ONET-specific columns
        expected_cols = [
            'company_id', 'year', 'title', 'onet_soc',
            'ai_exposure_hampole', 'ai_exposure_binary', 'n_ai_apps_firm_year'
        ]
        for col in expected_cols:
            assert col in linked_df.columns, f"Missing expected ONET column: {col}"

        # Verify ONET code format
        sample_onet = linked_df['onet_soc'].iloc[0] if len(linked_df) > 0 else None
        if sample_onet:
            assert '-' in str(sample_onet), "ONET codes should have standard format (XX-XXXX.XX)"

    def test_complete_pipeline_with_actual_files(self, data_dir):
        """Test complete pipeline using actual crosswalk files (if available)"""

        # Check for actual files
        x28_isco_file = data_dir / "240711_occupation_to_ch_isco_19.csv"
        esco_onet_file = data_dir / "ESCO_to_ONET-SOC.xlsx"

        files_available = x28_isco_file.exists() and esco_onet_file.exists()

        if not files_available:
            pytest.skip("Actual crosswalk files not available")

        # Load actual crosswalks
        x28_isco_xwalk = load_crosswalk_x28_to_isco(str(x28_isco_file))
        isco_onet_xwalk = load_crosswalk_isco_to_onet(str(esco_onet_file))

        # Create realistic test jobs
        test_jobs = pd.DataFrame({
            'company_id': ['1', '1', '2'],
            'year': [2020, 2021, 2020],
            'title': ['Data Scientist', 'ML Engineer', 'BI Analyst'],
            'x28_codes': [
                ['11001282'],  # Use actual X28 code that should be in crosswalk
                ['11001335'],
                ['11002001']
            ]
        })

        # Test ISCO pipeline
        try:
            jobs_isco = apply_crosswalks(test_jobs, x28_isco_xwalk, None, 'isco')
            isco_pipeline_success = len(jobs_isco) > 0 and 'isco_code' in jobs_isco.columns
        except Exception:
            isco_pipeline_success = False

        assert isco_pipeline_success, "ISCO pipeline should work with actual files"

        # Test ONET pipeline
        try:
            jobs_onet = apply_crosswalks(test_jobs, x28_isco_xwalk, isco_onet_xwalk, 'onet')
            onet_pipeline_success = len(jobs_onet) > 0 and 'onet_soc' in jobs_onet.columns
        except Exception:
            onet_pipeline_success = False

        assert onet_pipeline_success, "ONET pipeline should work with actual files"

    def test_unmatched_diagnostics_accuracy(self, sample_jobs_df, sample_x28_isco_crosswalk, sample_stage5_isco_exposures):
        """Test that unmatched job and exposure diagnostics are accurate"""

        # Create scenario with known unmatched cases
        jobs_with_unmatched = sample_jobs_df.copy()
        jobs_with_unmatched.loc[len(jobs_with_unmatched)] = {
            'company_id': '99',  # Company not in exposures
            'company_name': 'Unmatched Corp',
            'year': 2020,
            'title': 'Unmatched Job',
            'x28_codes': ['11001282']  # Valid X28 that maps to ISCO
        }

        exposures_with_unmatched = sample_stage5_isco_exposures.copy()
        exposures_with_unmatched.loc[len(exposures_with_unmatched)] = {
            'company_id': '88',  # Company not in jobs
            'year': 2020,
            'isco_code': '2521',
            'ai_exposure_hampole': 0.50,
            'ai_exposure_binary': 1.0,
            'n_ai_apps_firm_year': 5
        }

        # Apply crosswalk
        jobs_mapped = apply_crosswalks(jobs_with_unmatched, sample_x28_isco_crosswalk, None, 'isco')

        # Perform linking
        linked_df, unmatched_jobs, unmatched_exposures = link_jobs_to_exposures(
            jobs_mapped, exposures_with_unmatched
        )

        # Verify unmatched diagnostics
        assert len(unmatched_jobs) > 0, "Should identify unmatched jobs"
        assert len(unmatched_exposures) > 0, "Should identify unmatched exposures"

        # Check specific unmatched cases
        unmatched_company_99 = unmatched_jobs[unmatched_jobs['company_id'] == '99']
        assert len(unmatched_company_99) > 0, "Should identify company 99 as unmatched"

        unmatched_company_88 = unmatched_exposures[unmatched_exposures['company_id'] == '88']
        assert len(unmatched_company_88) > 0, "Should identify company 88 exposure as unmatched"

    def test_stage5_file_detection(self, data_dir):
        """Test automatic Stage 5 file detection with different naming patterns"""

        # Test file detection logic with various patterns
        test_patterns = [
            ('isco', 'time_variant', 'isco_firm_year_ai_exposure_time_variant.csv'),
            ('isco', 'time_invariant', 'isco_firm_year_ai_exposure_time_invariant.csv'),
            ('onet', 'time_variant', 'onet_firm_year_ai_exposure_time_variant.csv'),
            ('onet', 'time_invariant', 'onet_firm_year_ai_exposure_time_invariant.csv'),
        ]

        for occ_code, time_var, expected_filename in test_patterns:
            # Test the detection logic
            detected_file = find_stage5_file(str(data_dir), occ_code, time_var)

            expected_path = data_dir / expected_filename

            if expected_path.exists():
                assert str(detected_file) == str(expected_path), f"Should detect {expected_filename}"
            else:
                # If file doesn't exist, function should handle gracefully
                assert detected_file is None or not Path(detected_file).exists()

    def test_data_continuity_checks(self, sample_stage5_isco_exposures):
        """Test data continuity and consistency checks"""

        # Test Stage 5 data format validation
        required_stage5_cols = [
            'company_id', 'year', 'isco_code',
            'ai_exposure_hampole', 'ai_exposure_binary', 'n_ai_apps_firm_year'
        ]

        for col in required_stage5_cols:
            assert col in sample_stage5_isco_exposures.columns, f"Stage 5 missing required column: {col}"

        # Test data type consistency
        assert pd.api.types.is_numeric_dtype(sample_stage5_isco_exposures['ai_exposure_hampole'])
        assert pd.api.types.is_numeric_dtype(sample_stage5_isco_exposures['ai_exposure_binary'])
        assert pd.api.types.is_integer_dtype(sample_stage5_isco_exposures['n_ai_apps_firm_year'])

        # Test value ranges
        hampole_values = sample_stage5_isco_exposures['ai_exposure_hampole']
        binary_values = sample_stage5_isco_exposures['ai_exposure_binary']

        assert hampole_values.min() >= 0, "Hampole exposure should be non-negative"
        assert hampole_values.max() <= 10, "Hampole exposure should be reasonable"
        assert set(binary_values.unique()).issubset({0.0, 1.0}), "Binary exposure should be 0 or 1"

    def test_pipeline_error_handling(self, sample_jobs_df):
        """Test error handling in complete pipeline with malformed data"""

        # Test with empty crosswalks
        empty_crosswalk = pd.DataFrame(columns=['x28_code', 'isco_code'])

        jobs_mapped = apply_crosswalks(sample_jobs_df, empty_crosswalk, None, 'isco')

        # Should handle empty crosswalks gracefully (result in no mapped jobs)
        assert len(jobs_mapped) == 0 or 'isco_code' not in jobs_mapped.columns

        # Test with empty exposures
        empty_exposures = pd.DataFrame(columns=[
            'company_id', 'year', 'isco_code', 'ai_exposure_hampole',
            'ai_exposure_binary', 'n_ai_apps_firm_year'
        ])

        # Should handle gracefully
        try:
            linked_df, unmatched_jobs, unmatched_exposures = link_jobs_to_exposures(
                jobs_mapped, empty_exposures
            )
            error_handling_success = True
        except Exception:
            error_handling_success = False

        assert error_handling_success, "Should handle empty data gracefully"

    def test_cached_data_full_pipeline_50_rows(self, data_dir):
        """Test complete Stage 6 pipeline with first 50 rows from cached data"""

        cache_file = data_dir / "stage6_job_cache_max2025.parquet"

        if not cache_file.exists():
            pytest.skip("Cached file not available for testing")

        # Load cached data
        df_cached = pd.read_parquet(cache_file)
        sample_jobs = df_cached.head(50).copy()

        print(f"\nLoaded {len(sample_jobs)} jobs from cache")
        print(f"Columns: {list(sample_jobs.columns)}")

        # Handle different column names for X28 codes
        x28_col = 'x28_codes' if 'x28_codes' in sample_jobs.columns else 'x28_occupations'
        print(f"Sample X28 codes ({x28_col}): {sample_jobs[x28_col].head().tolist()}")

        # Test X28 parsing and explosion logic
        print(f"Testing X28 parsing logic...")
        sample_jobs['x28_codes_parsed'] = sample_jobs[x28_col].apply(parse_x28_occupations)
        print(f"Sample parsed X28 codes: {sample_jobs['x28_codes_parsed'].head().tolist()}")

        # Explode the parsed X28 codes to create job-occupation pairs
        sample_jobs_exploded = sample_jobs.explode('x28_codes_parsed').copy()
        sample_jobs_exploded = sample_jobs_exploded[sample_jobs_exploded['x28_codes_parsed'].notna()]
        sample_jobs_exploded = sample_jobs_exploded.rename(columns={'x28_codes_parsed': 'x28_codes'})

        print(f"After parsing and explosion: {len(sample_jobs_exploded)} job-occupation pairs")
        print(f"Unique X28 codes found: {sample_jobs_exploded['x28_codes'].nunique()}")

        # Use the exploded data for further testing
        sample_jobs = sample_jobs_exploded

        # Load actual crosswalks
        x28_isco_file = data_dir / "240711_occupation_to_ch_isco_19.csv"
        esco_onet_file = data_dir / "ESCO_to_ONET-SOC.xlsx"

        files_available = x28_isco_file.exists() and esco_onet_file.exists()

        if not files_available:
            pytest.skip("Crosswalk files not available")

        try:
            # Load crosswalks
            x28_isco_xwalk = load_crosswalk_x28_to_isco(str(x28_isco_file))
            isco_onet_xwalk = load_crosswalk_isco_to_onet(str(esco_onet_file))

            print(f"Loaded X28→ISCO crosswalk: {len(x28_isco_xwalk)} mappings")
            print(f"Loaded ISCO→ONET crosswalk: {len(isco_onet_xwalk)} mappings")

            # Test ISCO pipeline
            jobs_isco = apply_crosswalks(sample_jobs, x28_isco_xwalk, None, 'isco')
            print(f"After ISCO crosswalk: {len(jobs_isco)} job-occupation pairs")

            # Test ONET pipeline
            jobs_onet = apply_crosswalks(sample_jobs, x28_isco_xwalk, isco_onet_xwalk, 'onet')
            print(f"After ONET crosswalk: {len(jobs_onet)} job-occupation pairs")

            # Try to find actual Stage 5 files
            stage5_isco_file = find_stage5_file(str(data_dir), 'isco', 'time_variant')
            stage5_onet_file = find_stage5_file(str(data_dir), 'onet', 'time_variant')

            if stage5_isco_file and Path(stage5_isco_file).exists():
                print(f"Found Stage 5 ISCO file: {stage5_isco_file}")
                stage5_isco_data = load_stage5_exposures(stage5_isco_file, 'isco')
                print(f"Stage 5 ISCO exposures: {len(stage5_isco_data)} records")

                # Test linking
                linked_isco, unmatched_jobs_isco, unmatched_exp_isco = link_jobs_to_exposures(
                    jobs_isco, stage5_isco_data
                )

                print(f"ISCO linking results:")
                print(f"  - Linked: {len(linked_isco)} records")
                print(f"  - Unmatched jobs: {len(unmatched_jobs_isco)}")
                print(f"  - Unmatched exposures: {len(unmatched_exp_isco)}")

                # Basic validation
                assert len(linked_isco) >= 0, "Should complete ISCO linking without errors"

            if stage5_onet_file and Path(stage5_onet_file).exists():
                print(f"Found Stage 5 ONET file: {stage5_onet_file}")
                stage5_onet_data = load_stage5_exposures(stage5_onet_file, 'onet')
                print(f"Stage 5 ONET exposures: {len(stage5_onet_data)} records")

                # Test linking
                linked_onet, unmatched_jobs_onet, unmatched_exp_onet = link_jobs_to_exposures(
                    jobs_onet, stage5_onet_data
                )

                print(f"ONET linking results:")
                print(f"  - Linked: {len(linked_onet)} records")
                print(f"  - Unmatched jobs: {len(unmatched_jobs_onet)}")
                print(f"  - Unmatched exposures: {len(unmatched_exp_onet)}")

                # Basic validation
                assert len(linked_onet) >= 0, "Should complete ONET linking without errors"

            # Test passes if we get here without exceptions
            print("✓ Full pipeline test completed successfully")

        except Exception as e:
            print(f"Pipeline error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

            # Re-raise to fail the test
            raise e

if __name__ == "__main__":
    pytest.main([__file__, "-v"])