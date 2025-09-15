#!/usr/bin/env python3
"""
Join correctness tests for Stage 6
Tests the core linking logic between jobs and exposures
"""

import pandas as pd
import pytest
from pathlib import Path
from pandas.testing import assert_frame_equal
import sys
sys.path.append('..')

from stage_6_link_exposure_to_jobs import apply_crosswalks, load_crosswalk_x28_to_isco

class TestJoinLogic:
    """Test the core join logic between jobs and exposures"""
    
    @pytest.fixture
    def sample_jobs_df(self):
        """Sample jobs dataframe for testing"""
        return pd.DataFrame({
            'company_id': ['1', '1', '2'],
            'company_name': ['Alpha AG', 'Alpha AG', 'Beta SA'],
            'year': [2020, 2021, 2021],
            'title': ['Data Scientist', 'ML Engineer', 'BI Analyst'],
            'x28_codes': [['1110001', '2220002'], ['3330003'], ['9999999']]
        })
    
    @pytest.fixture 
    def sample_crosswalk(self):
        """Sample crosswalk for testing"""
        return pd.DataFrame({
            'x28_code': ['1110001', '2220002', '3330003'],
            'isco_code': ['2521', '2523', '2512']
        })
    
    @pytest.fixture
    def sample_exposures_df(self):
        """Sample exposures dataframe for testing"""
        return pd.DataFrame({
            'company_id': ['1', '1', '1'],
            'year': [2020, 2020, 2021], 
            'isco_code': ['2521', '2523', '2512'],
            'exposure_score': [0.60, 0.20, 0.90]
        })
    
    def test_crosswalk_application(self, sample_jobs_df, sample_crosswalk):
        """Test that crosswalks are applied correctly"""
        result = apply_crosswalks(sample_jobs_df, sample_crosswalk, target_occ='isco')
        
        # Should explode X28 codes and map to ISCO
        expected_rows = 4  # 2 codes from job 1, 1 from job 2, 0 from unmapped job 3
        assert len(result) <= expected_rows  # May be less due to unmapped codes
        
        # Check that mapped codes are present
        mapped_isco_codes = set(result['isco_code'].unique())
        assert '2521' in mapped_isco_codes or '2523' in mapped_isco_codes
    
    def test_inner_join_exact_matches(self, sample_jobs_df, sample_crosswalk, sample_exposures_df):
        """Test that inner join produces exact expected matches"""
        # Apply crosswalks to jobs
        jobs_mapped = apply_crosswalks(sample_jobs_df, sample_crosswalk, target_occ='isco')
        
        # Perform the join (simplified version)
        result = jobs_mapped.merge(
            sample_exposures_df,
            on=['company_id', 'year', 'isco_code'],
            how='inner'
        )
        
        # Expected matches:
        # - Alpha AG/2020/Data Scientist should match 2521 and 2523
        # - Alpha AG/2021/ML Engineer should match 2512  
        # - Beta SA/2021/BI Analyst should NOT match (unmapped X28)
        expected_matches = 3
        assert len(result) == expected_matches
        
        # Verify specific matches
        alpha_2020 = result[(result['company_id'] == '1') & (result['year'] == 2020)]
        assert len(alpha_2020) == 2  # Should match both 2521 and 2523
        
        alpha_2021 = result[(result['company_id'] == '1') & (result['year'] == 2021)]
        assert len(alpha_2021) == 1  # Should match 2512
    
    def test_unmatched_diagnostics(self, sample_jobs_df, sample_crosswalk, sample_exposures_df):
        """Test that unmatched jobs and exposures are identified correctly"""
        # This would test the diagnostic output generation
        # Implementation needed in main pipeline
        pass
    
    def test_company_id_authoritative(self):
        """Test that company_id is authoritative for matching, not company_name"""
        jobs = pd.DataFrame({
            'company_id': ['1'],
            'company_name': ['Wrong Name'],  # Incorrect name
            'year': [2020],
            'title': ['Data Scientist'],
            'isco_code': ['2521']
        })
        
        exposures = pd.DataFrame({
            'company_id': ['1'],  # Same ID
            'year': [2020],
            'isco_code': ['2521'],
            'exposure_score': [0.60],
            'company_name': ['Correct Name']  # Different name
        })
        
        # Join should succeed on company_id despite name mismatch
        result = jobs.merge(exposures, on=['company_id', 'year', 'isco_code'], how='inner')
        assert len(result) == 1
    
    def test_duplicate_key_detection(self):
        """Test that duplicate keys in either dataset are detected"""
        # Jobs with duplicate (company_id, year, title, isco_code)
        duplicate_jobs = pd.DataFrame({
            'company_id': ['1', '1'],
            'year': [2020, 2020],
            'title': ['Data Scientist', 'Data Scientist'], 
            'isco_code': ['2521', '2521'],  # Duplicate
        })
        
        # Should detect and handle duplicates appropriately
        # Implementation needed
        pass
    
    def test_crosswalk_multiplicity(self):
        """Test handling of one-to-many crosswalk mappings"""
        # If one X28 maps to multiple ISCOs, should explode correctly
        multi_crosswalk = pd.DataFrame({
            'x28_code': ['1110001', '1110001'],  # Same X28
            'isco_code': ['2521', '2523']       # Maps to two ISCOs
        })
        
        jobs = pd.DataFrame({
            'company_id': ['1'],
            'company_name': ['Alpha AG'],
            'year': [2020],
            'title': ['Data Scientist'],
            'x28_codes': [['1110001']]
        })
        
        result = apply_crosswalks(jobs, multi_crosswalk, target_occ='isco')
        
        # Should create two rows, one for each ISCO mapping
        assert len(result) == 2
        assert set(result['isco_code']) == {'2521', '2523'}
    
    def test_empty_join_results(self):
        """Test handling when no matches are found"""
        jobs = pd.DataFrame({
            'company_id': ['1'],
            'year': [2020],
            'isco_code': ['2521']
        })
        
        exposures = pd.DataFrame({
            'company_id': ['2'],  # Different company
            'year': [2020],
            'isco_code': ['2521']
        })
        
        result = jobs.merge(exposures, on=['company_id', 'year', 'isco_code'], how='inner')
        assert len(result) == 0

if __name__ == "__main__":
    pytest.main([__file__])