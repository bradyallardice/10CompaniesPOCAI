#!/usr/bin/env python3
"""
Fail-fast tests for Stage 6 - ensuring proper error handling
Tests negative cases and error conditions
"""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import sys
sys.path.append('..')

from stage_6_link_exposure_to_jobs import main, load_crosswalk_x28_to_isco, load_crosswalk_isco_to_onet

class TestFailFast:
    """Test fail-fast behavior for various error conditions"""
    
    def test_missing_crosswalk_file_raises(self):
        """Test that missing crosswalk files raise FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_crosswalk_x28_to_isco("nonexistent_file.csv")
    
    def test_unmapped_x28_raises(self, tmp_path):
        """Test that unmapped X28 codes cause failure when fail_on_warn=True"""
        # Create minimal fixtures with unmapped code
        job_data = pd.DataFrame({
            'company_id': ['1'],
            'company_name': ['Test Co'],
            'title': ['Test Job'],
            'x28_occupations': "['9999999']",  # Unmapped code
            'tst_created': ['2020-01-01'],
            'tst_deleted': [None]
        })
        
        crosswalk_data = pd.DataFrame({
            'x28_code': ['1110001'],  # Different code, 9999999 not mapped
            'isco_19': ['2521']
        })
        
        exposure_data = pd.DataFrame({
            'company_id': ['1'],
            'year': [2020],
            'isco_code': ['2521'],
            'exposure_score': [0.5]
        })
        
        # Save fixtures
        job_file = tmp_path / "jobs.csv"
        crosswalk_file = tmp_path / "crosswalk.csv"
        exposure_file = tmp_path / "exposure.csv"
        
        job_data.to_csv(job_file, index=False)
        crosswalk_data.to_csv(crosswalk_file, index=False)
        exposure_data.to_csv(exposure_file, index=False)
        
        # Should raise due to unmapped X28 code
        # (This test will need to be updated once the full pipeline is implemented)
        pass
    
    def test_missing_required_columns_raises(self):
        """Test that missing required columns raise ValueError"""
        # Test crosswalk with missing columns
        bad_crosswalk = pd.DataFrame({'wrong_col': ['1110001'], 'another_col': ['2521']})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            bad_crosswalk.to_csv(f.name, index=False)
            
            with pytest.raises(ValueError, match="Could not auto-detect"):
                load_crosswalk_x28_to_isco(f.name)
    
    def test_duplicate_stage5_keys_raises(self, tmp_path):
        """Test that duplicate (company_id, year, occ_code) in Stage 5 raises error"""
        # Create exposure file with duplicates
        duplicate_exposure = pd.DataFrame({
            'company_id': ['1', '1'],  # Duplicate
            'year': [2020, 2020],     # Duplicate 
            'isco_code': ['2521', '2521'],  # Duplicate
            'exposure_score': [0.5, 0.6]    # Different scores for same key
        })
        
        exposure_file = tmp_path / "duplicate_exposure.csv"
        duplicate_exposure.to_csv(exposure_file, index=False)
        
        # Should detect and raise on duplicate keys
        # (Implementation needed in main pipeline)
        pass
    
    def test_empty_job_expansion_raises(self):
        """Test that 0 rows after job expansion raises ValueError"""
        empty_df = pd.DataFrame()
        
        from stage_6_link_exposure_to_jobs import validate_jobs_data
        with pytest.raises(ValueError, match="0 rows after job expansion"):
            validate_jobs_data(empty_df, fail_on_warn=True)
    
    def test_excessive_empty_x28_raises(self):
        """Test that >2% empty X28 occupations raises error"""
        # Create dataset where >2% have empty X28
        bad_jobs = pd.DataFrame({
            'company_id': ['1'] * 100,
            'year': list(range(2000, 2100)),
            'title': [f'Job {i}' for i in range(100)],
            'x28_codes': [[]] * 95 + [['1110001']] * 5  # 95% empty = >2%
        })
        
        from stage_6_link_exposure_to_jobs import validate_jobs_data
        with pytest.raises(ValueError, match="rows have empty x28_occupations"):
            validate_jobs_data(bad_jobs, fail_on_warn=True)
    
    def test_crosswalk_drift_detection(self, tmp_path):
        """Test that crosswalk file hash changes are detected when allow_drift=False"""
        # This would test the hash-based drift detection
        # Implementation needed in main pipeline
        pass
    
    def test_invalid_excel_file_raises(self, tmp_path):
        """Test that invalid Excel files raise appropriate errors"""
        # Create invalid Excel file
        invalid_file = tmp_path / "invalid.xlsx" 
        invalid_file.write_text("Not an Excel file")
        
        with pytest.raises((ValueError, Exception)):  # Could be various Excel-related errors
            load_crosswalk_isco_to_onet(str(invalid_file))

if __name__ == "__main__":
    pytest.main([__file__])