#!/usr/bin/env python3
"""
Unit tests for Stage 6 individual functions
Tests pure functions like year expansion, X28 parsing, etc.
"""

import pandas as pd
import pytest
from datetime import datetime
import sys
sys.path.append('..')

from stage_6_link_exposure_to_jobs import parse_x28_occupations, build_jobs_from_db, validate_jobs_data

def test_parse_x28_occupations_json_array():
    """Test parsing JSON-style arrays"""
    assert parse_x28_occupations("['1110001','2220002']") == ['1110001', '2220002']
    assert parse_x28_occupations('["3330003"]') == ['3330003']
    assert parse_x28_occupations("[]") == []

def test_parse_x28_occupations_comma_separated():
    """Test parsing comma-separated values"""
    assert parse_x28_occupations("1110001,2220002") == ['1110001', '2220002']
    assert parse_x28_occupations("1110001, 2220002") == ['1110001', '2220002']
    assert parse_x28_occupations("'1110001','2220002'") == ['1110001', '2220002']

def test_parse_x28_occupations_empty_cases():
    """Test handling of empty/null cases"""
    assert parse_x28_occupations("") == []
    assert parse_x28_occupations(None) == []
    assert parse_x28_occupations("null") == []
    assert parse_x28_occupations("None") == []

def test_parse_x28_occupations_whitespace():
    """Test whitespace handling"""
    assert parse_x28_occupations(" [ '1110001' , '2220002' ] ") == ['1110001', '2220002']
    assert parse_x28_occupations("  1110001  ,  2220002  ") == ['1110001', '2220002']

def test_validate_jobs_data_success():
    """Test successful validation"""
    df = pd.DataFrame({
        'company_id': ['1', '2'],
        'year': [2020, 2021],
        'title': ['Data Scientist', 'ML Engineer'],
        'x28_codes': [['1110001'], ['2220002']]
    })
    # Should not raise
    validate_jobs_data(df, fail_on_warn=True)

def test_validate_jobs_data_empty_dataframe():
    """Test validation fails on empty dataframe"""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="0 rows after job expansion"):
        validate_jobs_data(df, fail_on_warn=True)

def test_validate_jobs_data_too_many_empty_x28():
    """Test validation fails when >2% have empty X28"""
    df = pd.DataFrame({
        'company_id': ['1', '2', '3', '4', '5'],
        'year': [2020, 2021, 2022, 2023, 2024],
        'title': ['A', 'B', 'C', 'D', 'E'],
        'x28_codes': [['1110001'], [], [], [], []]  # 80% empty
    })
    with pytest.raises(ValueError, match="rows have empty x28_occupations"):
        validate_jobs_data(df, fail_on_warn=True)

def test_validate_jobs_data_duplicates():
    """Test validation fails on duplicate keys"""
    df = pd.DataFrame({
        'company_id': ['1', '1'],
        'year': [2020, 2020],
        'title': ['Data Scientist', 'Data Scientist'],  # Duplicate
        'x28_codes': [['1110001'], ['2220002']]
    })
    with pytest.raises(ValueError, match="duplicate.*combinations"):
        validate_jobs_data(df, fail_on_warn=True)

def test_active_years_earliest_only():
    """Test that active years start from earliest tst_created, ignoring deletions"""
    # This would require implementing the year expansion logic
    # For now, this is a placeholder showing the expected behavior
    pass

def test_x28_normalization():
    """Test X28 code normalization and cleaning"""
    # Test various formats normalize to consistent form
    assert parse_x28_occupations("1110001") == ['1110001']
    assert parse_x28_occupations("  1110001  ") == ['1110001']
    assert parse_x28_occupations("'1110001'") == ['1110001']
    assert parse_x28_occupations('"1110001"') == ['1110001']

if __name__ == "__main__":
    pytest.main([__file__])