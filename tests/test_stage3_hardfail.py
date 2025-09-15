#!/usr/bin/env python3
"""
Stage 3 Hard Failure Tests: Fundamental Error Handling  
Tests scenarios that should cause immediate failure with non-zero exit
"""
import os
import sys
import pandas as pd
from pathlib import Path
import pytest
from unittest.mock import patch, Mock

sys.path.append(str(Path(__file__).parent.parent))
import stage_3_extract_ai_tasks as s3

class TestStage3HardFailures:
    """Test hard failures that should cause immediate exit with error code"""
    
    def test_missing_input_file_hard_error(self):
        """Test hard error when input file doesn't exist"""
        
        nonexistent_file = "/path/that/does/not/exist.csv"
        
        with pytest.raises((SystemExit, FileNotFoundError, ValueError)):
            s3.run_custom_file(
                nonexistent_file,
                "content_clean", 
                "", "o3",
                False, False, "medium", "medium", 1
            )
    
    def test_missing_required_column_hard_error(self, tmp_path):
        """Test hard error when required column is missing from input"""
        
        # Create input file missing the content column
        bad_input = tmp_path / "missing_column.csv"
        pd.DataFrame({
            "uid": [1, 2],
            "title": ["Test 1", "Test 2"]
            # Missing content_clean column
        }).to_csv(bad_input, index=False)
        
        with pytest.raises((SystemExit, ValueError, KeyError)):
            s3.run_custom_file(
                str(bad_input),
                "content_clean",  # Column doesn't exist
                "", "o3",
                False, False, "medium", "medium", 1
            )
    
    def test_invalid_step_number_hard_error(self, tmp_path):
        """Test hard error when invalid step number is provided"""
        
        # Create valid input file
        good_input = tmp_path / "valid.csv"
        pd.DataFrame({
            "uid": [1],
            "content_clean": ["Test content"]
        }).to_csv(good_input, index=False)
        
        with pytest.raises((SystemExit, ValueError)):
            s3.run_custom_file(
                str(good_input),
                "content_clean",
                "", "o3",
                False, False, "medium", "medium",
                99  # Invalid step number
            )
    
    def test_invalid_model_name_hard_error(self, tmp_path):
        """Test hard error when invalid model name is provided"""
        
        good_input = tmp_path / "valid.csv"
        pd.DataFrame({
            "uid": [1],
            "content_clean": ["Test content"]
        }).to_csv(good_input, index=False)
        
        with pytest.raises((SystemExit, ValueError)):
            s3.run_custom_file(
                str(good_input),
                "content_clean", 
                "", 
                "invalid-model-name",  # Invalid model
                False, False, "medium", "medium", 1
            )
    
    def test_unreadable_csv_hard_error(self, tmp_path):
        """Test hard error when CSV is corrupted/unreadable"""
        
        # Create corrupted CSV file
        bad_csv = tmp_path / "corrupted.csv"
        with open(bad_csv, 'w') as f:
            f.write("invalid,csv,structure\nwith\nunmatched\nquotes\"and,malformed")
        
        with pytest.raises((SystemExit, pd.errors.ParserError, ValueError)):
            s3.run_custom_file(
                str(bad_csv),
                "content_clean",
                "", "o3",
                False, False, "medium", "medium", 1
            )
    
    def test_api_authentication_hard_error(self, tmp_path, monkeypatch):
        """Test hard error when API authentication fails completely"""
        
        # Mock OpenAI client to raise authentication error
        def mock_openai_client(*args, **kwargs):
            raise Exception("Authentication failed: Invalid API key")
        
        monkeypatch.setattr(s3, "openai", Mock())
        monkeypatch.setattr(s3.openai, "OpenAI", mock_openai_client)
        
        good_input = tmp_path / "valid.csv"
        pd.DataFrame({
            "uid": [1],
            "content_clean": ["Test content"]
        }).to_csv(good_input, index=False)
        
        with pytest.raises((SystemExit, Exception)):
            s3.run_custom_file(
                str(good_input),
                "content_clean",
                "", "o3", 
                False, False, "medium", "medium", 1
            )
    
    def test_output_directory_not_writable_hard_error(self, tmp_path, monkeypatch):
        """Test hard error when output directory is not writable"""
        
        good_input = tmp_path / "valid.csv"
        pd.DataFrame({
            "uid": [1],
            "content_clean": ["Test content"]
        }).to_csv(good_input, index=False)
        
        # Mock Path.write_text to raise permission error
        original_write = Path.write_text
        def mock_write_error(self, *args, **kwargs):
            if "output" in str(self):
                raise PermissionError("Permission denied")
            return original_write(self, *args, **kwargs)
        
        monkeypatch.setattr(Path, "write_text", mock_write_error)
        
        # Also need to mock pandas.to_csv for the same error
        original_to_csv = pd.DataFrame.to_csv
        def mock_to_csv_error(self, path, *args, **kwargs):
            if isinstance(path, (str, Path)) and "output" in str(path):
                raise PermissionError("Permission denied")
            return original_to_csv(self, path, *args, **kwargs)
        
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_to_csv_error)
        
        # Should fail when trying to write output
        with pytest.raises((SystemExit, PermissionError, OSError)):
            s3.run_custom_file(
                str(good_input),
                "content_clean",
                "", "o3",
                False, False, "medium", "medium", 1
            )
    
    def test_empty_input_file_hard_error(self, tmp_path):
        """Test hard error when input file is completely empty"""
        
        empty_input = tmp_path / "empty.csv"
        empty_input.touch()  # Create empty file
        
        with pytest.raises((SystemExit, ValueError, pd.errors.EmptyDataError)):
            s3.run_custom_file(
                str(empty_input),
                "content_clean",
                "", "o3",
                False, False, "medium", "medium", 1
            )
    
    def test_zero_rows_after_filtering_hard_error(self, tmp_path):
        """Test hard error when input has headers but zero data rows"""
        
        headers_only = tmp_path / "headers_only.csv"
        pd.DataFrame(columns=["uid", "content_clean"]).to_csv(headers_only, index=False)
        
        with pytest.raises((SystemExit, ValueError)):
            s3.run_custom_file(
                str(headers_only),
                "content_clean",
                "", "o3",
                False, False, "medium", "medium", 1
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])