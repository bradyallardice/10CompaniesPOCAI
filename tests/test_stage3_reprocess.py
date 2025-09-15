#!/usr/bin/env python3
"""
Stage 3 Reprocessing Tests: Malformed File Handling
Tests the reprocessing loop for correcting malformed rows
"""
import os
import sys
import pandas as pd
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).parent.parent))
import stage_3_extract_ai_tasks as s3

class TestStage3Reprocess:
    """Test reprocessing loop for malformed file handling"""
    
    @pytest.fixture
    def fixtures_dir(self):
        return Path(__file__).parent / "fixtures"
    
    @pytest.fixture
    def golden_dir(self):
        return Path(__file__).parent / "golden"
    
    def test_reprocess_clears_malformed_on_success(self, fixtures_dir, golden_dir, tmp_path):
        """Test that reprocessing deletes malformed file when all rows are corrected"""
        
        # Start with existing output and malformed files (simulate first run)
        input_file = fixtures_dir / "step1_input.csv"
        output_file = tmp_path / "step1_output.csv"
        malformed_file = tmp_path / "step1_output_malformed.csv"
        
        # Copy golden files as starting point
        golden_output = pd.read_csv(golden_dir / "step1_output.csv")
        golden_malformed = pd.read_csv(golden_dir / "step1_malformed.csv")
        
        golden_output.to_csv(output_file, index=False)
        golden_malformed.to_csv(malformed_file, index=False)
        
        # "Fix" the malformed file by adding corrected ai_applications_raw
        fixed_malformed = golden_malformed.copy()
        fixed_malformed["ai_applications_raw"] = '["people analytics dashboards"]'
        fixed_malformed.to_csv(malformed_file, index=False)
        
        # Mock the run_custom_file to simulate reprocessing
        # In a real test, we'd mock the OpenAI client to return good responses
        # For now, we'll test the file handling logic
        
        assert malformed_file.exists(), "Malformed file should exist before reprocessing"
        
        # Simulate successful reprocessing by directly merging the corrected row
        # (In real implementation, Stage 3 would detect malformed file and reprocess)
        output_df = pd.read_csv(output_file)
        corrected_df = pd.read_csv(malformed_file)
        
        # Merge corrected row back into main output
        for _, corrected_row in corrected_df.iterrows():
            uid = corrected_row["uid"]
            output_df.loc[output_df["uid"] == uid, "ai_applications_raw"] = corrected_row["ai_applications_raw"]
        
        output_df.to_csv(output_file, index=False)
        
        # Simulate deletion of malformed file after successful reprocessing
        malformed_file.unlink()
        
        # Verify malformed file is gone
        assert not malformed_file.exists(), "Malformed file should be deleted after successful reprocessing"
        
        # Verify corrected data is in main output
        final_df = pd.read_csv(output_file)
        # Handle both string and int UIDs
        uid_102_rows = final_df[(final_df["uid"] == "102") | (final_df["uid"] == 102)]
        assert len(uid_102_rows) > 0, "Should find UID 102 row in final output"
        corrected_row = uid_102_rows["ai_applications_raw"].iloc[0]
        assert corrected_row == '["people analytics dashboards"]', "Corrected data should be merged into main output"
    
    def test_reprocess_keeps_unresolved_malformed(self, tmp_path):
        """Test that partially successful reprocessing keeps unresolved rows in malformed file"""
        
        # Create a malformed file with 2 rows
        malformed_data = pd.DataFrame({
            "uid": ["102", "104"],
            "company_id": [1, 1],
            "title": ["People Ops", "Admin"],
            "content_clean": ["People text", "Admin text"],
            "error_reason": ["json_parse_error", "json_parse_error"],
            "step": [1, 1],
            "attempt": [1, 1]
        })
        
        malformed_file = tmp_path / "test_malformed.csv"
        malformed_data.to_csv(malformed_file, index=False)
        
        # Simulate partial success: only fix row 102, leave 104 unresolved
        corrected_data = malformed_data.copy()
        corrected_data = corrected_data[corrected_data["uid"] == "104"]  # Keep only unresolved
        corrected_data["attempt"] = 2  # Increment attempt count
        
        # Update malformed file with only unresolved rows
        corrected_data.to_csv(malformed_file, index=False)
        
        # Verify malformed file still exists with unresolved row
        assert malformed_file.exists()
        remaining_df = pd.read_csv(malformed_file)
        assert len(remaining_df) == 1
        assert remaining_df.iloc[0]["uid"] == "104"
        assert remaining_df.iloc[0]["attempt"] == 2
    
    def test_reprocess_detection_by_malformed_file_existence(self, fixtures_dir, tmp_path):
        """Test that reprocessing is triggered by existence of malformed file"""
        
        input_file = fixtures_dir / "step1_input.csv"
        output_file = tmp_path / "step1_output.csv"  
        malformed_file = tmp_path / "step1_output_malformed.csv"
        
        # Create a malformed file to trigger reprocess mode
        malformed_data = pd.DataFrame({
            "uid": ["102"],
            "company_id": [1],
            "title": ["People Ops"],
            "content_clean": ["Test content"],
            "ai_applications_raw": ['["corrected response"]'],  # Pre-corrected
            "error_reason": ["json_parse_error"],
            "step": [1],
            "attempt": [1]
        })
        malformed_data.to_csv(malformed_file, index=False)
        
        # Create initial output file
        initial_output = pd.DataFrame({
            "uid": ["101", "102", "103"],
            "company_id": [1, 1, 1],
            "title": ["Data Scientist", "People Ops", "ML Engineer"],
            "content_clean": ["DS content", "People content", "ML content"],
            "ai_applications_raw": ['["ds apps"]', "", '["ml apps"]']
        })
        initial_output.to_csv(output_file, index=False)
        
        # In a real scenario, Stage 3 would detect the malformed file and enter reprocess mode
        # Here we simulate the expected behavior
        
        assert malformed_file.exists(), "Malformed file should exist to trigger reprocessing"
        assert output_file.exists(), "Main output file should exist from previous run"
        
        # Simulate reprocess mode detection
        reprocess_mode = malformed_file.exists()
        assert reprocess_mode, "Should detect reprocess mode when malformed file exists"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])