#!/usr/bin/env python3
"""
Comprehensive test suite for Stage 5: AI Task Exposure Calculation and ISCO Crosswalk

This test suite validates the core functionality of the Stage 5 pipeline with controlled,
minimal test data. It covers:

1. UID normalization and deduplication handling
2. Firm-level task exposure calculation (Hampole method)
3. Weighted occupation exposure aggregation
4. AI intensity adjustment
5. Full end-to-end pipeline integration

Test Design:
- Uses minimal synthetic data to make calculations verifiable by hand
- Tests the critical deduplication reintegration logic (u2 -> u1 mapping)
- Validates mathematical correctness of exposure calculations
- Ensures proper data flow between pipeline steps
"""

import math
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# Add the project root to the Python path so we can import the stage modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stage_5_onet_to_isco_exposure import TaskFirmExposurePipeline


@pytest.fixture
def pipeline(tmp_path):
    """Create a TaskFirmExposurePipeline instance for testing."""
    # Use a temp data_dir so nothing in your real Data/ is touched
    return TaskFirmExposurePipeline(
        aggregation_method="mean",
        time_invariant=False,
        occupation_exposure="none",
        data_dir=str(tmp_path)  # not used by these tests, but set anyway
    )


@pytest.fixture
def stage2_mapping_df():
    """
    Stage 2 deduplication mapping: u2 was removed and deduplicated into u1 (kept).
    
    This tests the critical scenario where Stage 4 references a removed UID (u2)
    that needs to be mapped back to the kept UID (u1) for proper job attribution.
    """
    return pd.DataFrame({
        "kept_uid":   ["u1"],
        "removed_uid": ["u2"],
    })


@pytest.fixture
def stage2_mapping_csv(tmp_path, stage2_mapping_df):
    """Save the Stage 2 mapping to a CSV file for testing file-based loading."""
    p = tmp_path / "similar_duplicates_removed.csv"
    stage2_mapping_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def original_jobs_df():
    """
    Original job postings dataset with minimal required columns.
    
    Contains only the canonical kept UID (u1) that corresponds to FirmX.
    This represents the deduplicated dataset that Stage 5 needs to work with.
    """
    return pd.DataFrame({
        "job_uid": ["u1"],
        "company_name": ["FirmX"],
        "x28_occupations": [""],
        "x28_industries": [""],
        "tst_created": pd.to_datetime(["2020-05-20"]),
        "year": [2020],
    })


@pytest.fixture
def job_app_mapping_df():
    """
    Job-to-application mapping from Stage 4 pipeline.
    
    Critical test case:
    - App A was attached to u2 (a removed UID) - tests deduplication handling
    - App B attached to u1 (kept UID) - standard case
    
    Both should end up attributed to FirmX through proper UID normalization.
    """
    return pd.DataFrame({
        "app_text": ["App A", "App B"],
        "job_uids": ["u2",    "u1"],
        "first_occurrence_tst_created": ["2020-01-01", "2020-01-15"],
    })


@pytest.fixture
def task_app_matches_df():
    """
    Task-to-application matches (like a tiny top_5_matches.csv from Stage 4).
    
    Test scenario:
    - Task T1 matches both App A and App B (should get full exposure)
    - Task T2 matches only App A (should get partial exposure)
    
    This enables testing of the Hampole share calculation logic.
    """
    return pd.DataFrame({
        "app_text":     ["App A", "App B", "App A"],
        "onet_task_id": ["T1",    "T1",    "T2"],
    })


@pytest.fixture
def task_statements_df():
    """
    Minimal O*NET Task Statements mapping tasks to occupations.
    
    Both tasks T1 and T2 belong to occupation 11-1011.00 (Chief Executives).
    This allows testing of occupation-level aggregation with multiple tasks.
    """
    return pd.DataFrame({
        "Task ID": ["T1", "T2"],
        "O*NET-SOC Code": ["11-1011.00", "11-1011.00"],
        "Task": ["Do thing 1", "Do thing 2"],
        "Task Type": ["Core", "Core"],
        "Title": ["Chief Executives", "Chief Executives"],
    })


@pytest.fixture
def task_ratings_df():
    """
    O*NET Task Ratings with importance weights.
    
    Test scenario:
    - Task T1 has importance weight 2.0 (high importance)
    - Task T2 has importance weight 1.0 (lower importance)
    
    This 2:1 ratio makes the weighted average calculation easy to verify by hand.
    """
    return pd.DataFrame({
        "Scale ID": ["IM", "IM"],
        "Scale Name": ["Importance", "Importance"],
        "O*NET-SOC Code": ["11-1011.00", "11-1011.00"],
        "Task ID": ["T1", "T2"],
        # Give T1 twice the weight of T2 for easy manual verification
        "Data Value": [2.0, 1.0],
    })


def test_uid_normalization_and_provenance(pipeline, job_app_mapping_df, original_jobs_df, stage2_mapping_df):
    """
    Test the critical UID normalization and provenance tracking logic.
    
    This tests the core deduplication reintegration functionality:
    1. Normalize removed UIDs to kept UIDs (u2 -> u1)
    2. Join only on kept UIDs to match with job data
    3. Preserve provenance showing what Stage 4 originally listed
    
    Expected behavior:
    - Both App A (originally u2) and App B (originally u1) should be attributed to FirmX
    - App A should retain provenance showing it came from u2
    - Only kept UIDs (u1) should appear in the final joined dataset
    """
    expanded = pipeline.expand_job_app_mapping_to_full_dataset(
        job_app_mapping=job_app_mapping_df,
        original_jobs=original_jobs_df,
        stage2_mapping=stage2_mapping_df
    )

    # 1) We should only see KEPT UIDs in the join output
    assert set(expanded["job_uid"].unique()) == {"u1"}

    # 2) We should get both apps attached to FirmX 
    # (because both App A and App B end up attributed to u1 through normalization)
    assert set(expanded["app_text"].unique()) == {"App A", "App B"}
    assert set(expanded["company_name"].unique()) == {"FirmX"}

    # 3) Provenance tracking: preserve what Stage 4 originally listed
    # App A was mapped to "u2" in Stage 4 → provenance should show "u2"
    prov_a = expanded.loc[expanded["app_text"] == "App A", "provenance_job_uids"].iloc[0]
    assert prov_a == "u2"
    
    # App B was mapped to "u1" in Stage 4 → provenance should show "u1"
    prov_b = expanded.loc[expanded["app_text"] == "App B", "provenance_job_uids"].iloc[0]
    assert prov_b == "u1"


def test_step2_exposure_hampole_share(pipeline,
                                      task_app_matches_df,
                                      job_app_mapping_df,
                                      original_jobs_df,
                                      stage2_mapping_csv):
    """
    Test Step 2: Firm-level task exposure calculation using Hampole share method.
    
    Test scenario:
    - FirmX in 2020 has 2 AI applications: App A and App B
    - Task T1 is matched by both App A and App B → share = 2/2 = 1.0
    - Task T2 is matched by App A only → share = 1/2 = 0.5
    
    This validates the core exposure calculation: share of firm's apps that match each task.
    """
    task_firm = pipeline.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches_df,
        job_app_mapping=job_app_mapping_df,
        original_jobs=original_jobs_df,
        stage2_mapping_file=stage2_mapping_csv
    )

    # Filter to FirmX-2020 data
    firm_filter = (task_firm["company_name"] == "FirmX") & (task_firm["year"] == 2020)
    firm_data = task_firm.loc[firm_filter].copy()

    # Sanity check: firm-year should report 2 AI applications total
    assert set(firm_data["n_ai_apps_firm_year"].unique()) == {2}

    # Extract exposure values for each task
    t1_exposure = firm_data.loc[firm_data["onet_task_id"] == "T1", "hampole_task_exposure"].iloc[0]
    t2_exposure = firm_data.loc[firm_data["onet_task_id"] == "T2", "hampole_task_exposure"].iloc[0]

    # Verify calculated exposures match expected values
    assert t1_exposure == pytest.approx(1.0, rel=1e-9)  # Both apps match T1: 2/2 = 1.0
    assert t2_exposure == pytest.approx(0.5, rel=1e-9)  # Only App A matches T2: 1/2 = 0.5


def test_step3_and_step4_weighting_and_intensity(pipeline,
                                                 task_app_matches_df,
                                                 job_app_mapping_df,
                                                 original_jobs_df,
                                                 stage2_mapping_csv,
                                                 task_statements_df,
                                                 task_ratings_df):
    """
    Test Step 3 (weighted occupation exposure) and Step 4 (AI intensity adjustment).
    
    Mathematical verification:
    
    Step 3 - Weighted occupation exposure:
    - T1: weight=2.0, exposure=1.0 → contribution = 2.0 * 1.0 = 2.0
    - T2: weight=1.0, exposure=0.5 → contribution = 1.0 * 0.5 = 0.5
    - Weighted average = (2.0 + 0.5) / (2.0 + 1.0) = 2.5 / 3.0 ≈ 0.8333333
    
    Step 4 - AI intensity adjustment:
    - N_apps = 2, so intensity factor = log(1 + 2) = log(3) ≈ 1.0986
    - Final exposure = 0.8333333 * log(3) ≈ 0.9155
    """
    # Step 2: Calculate firm-level task exposure
    task_firm = pipeline.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches_df,
        job_app_mapping=job_app_mapping_df,
        original_jobs=original_jobs_df,
        stage2_mapping_file=stage2_mapping_csv
    )

    # Step 3: Calculate weighted occupation-level exposure
    occ_firm = pipeline.step3_calculate_occupation_firm_exposure(
        task_firm_exposure=task_firm,
        task_statements=task_statements_df,
        task_ratings=task_ratings_df
    )

    # Extract the FirmX-2020 row (should be exactly one occupation: Chief Executives)
    firm_row = occ_firm[(occ_firm["company_name"] == "FirmX") & (occ_firm["year"] == 2020)].iloc[0]

    # Verify Step 3 calculation: weighted average of task exposures
    expected_weighted = (1.0 * 2.0 + 0.5 * 1.0) / (2.0 + 1.0)  # (2.0 + 0.5) / 3.0 = 0.8333333
    assert firm_row["hampole_occupation_exposure"] == pytest.approx(expected_weighted, rel=1e-9)
    assert firm_row["n_ai_apps_firm_year"] == 2

    # Step 4: Apply AI intensity adjustment
    final_exposure = pipeline.step4_apply_ai_intensity_adjustment(occ_firm)
    final_row = final_exposure[(final_exposure["company_name"] == "FirmX") & (final_exposure["year"] == 2020)].iloc[0]

    # Verify Step 4 calculation: weighted exposure * log(1 + N_apps)
    expected_final = expected_weighted * math.log(1 + 2)  # 0.8333333 * log(3)
    assert final_row["hampole_ai_exposure_avg"] == pytest.approx(expected_final, rel=1e-9)


def test_binary_exposure_method(pipeline,
                                task_app_matches_df,
                                job_app_mapping_df,
                                original_jobs_df,
                                stage2_mapping_csv):
    """
    Test the binary exposure calculation method as an alternative to Hampole shares.
    
    Binary method: 1 if firm uses ANY application that matches the task, 0 otherwise.
    
    Expected results for FirmX:
    - Task T1: Binary = 1 (matched by both App A and App B)
    - Task T2: Binary = 1 (matched by App A)
    
    Both tasks should have exposure = 1 since at least one app matches each.
    """
    task_firm = pipeline.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches_df,
        job_app_mapping=job_app_mapping_df,
        original_jobs=original_jobs_df,
        stage2_mapping_file=stage2_mapping_csv
    )

    # Filter to FirmX-2020
    firm_filter = (task_firm["company_name"] == "FirmX") & (task_firm["year"] == 2020)
    firm_data = task_firm.loc[firm_filter].copy()

    # Extract binary exposure values
    t1_binary = firm_data.loc[firm_data["onet_task_id"] == "T1", "binary_task_exposure"].iloc[0]
    t2_binary = firm_data.loc[firm_data["onet_task_id"] == "T2", "binary_task_exposure"].iloc[0]

    # Both tasks should have binary exposure = 1 (at least one matching app each)
    assert t1_binary == 1.0
    assert t2_binary == 1.0


def test_edge_case_no_task_matches(pipeline):
    """
    Test edge case where a task has no matching applications.
    
    This ensures the pipeline handles sparse data gracefully and doesn't crash
    when some tasks in the O*NET data have no corresponding AI applications.
    """
    # Task T3 has no matches in the applications
    sparse_task_matches = pd.DataFrame({
        "app_text":     ["App A"],
        "onet_task_id": ["T1"],
    })
    
    # Job mapping and original jobs (minimal)
    job_mapping = pd.DataFrame({
        "app_text": ["App A"],
        "job_uids": ["u1"],
        "first_occurrence_tst_created": ["2020-01-01"],
    })
    
    original_jobs = pd.DataFrame({
        "job_uid": ["u1"],
        "company_name": ["FirmX"],
        "x28_occupations": [""],
        "x28_industries": [""],
        "tst_created": pd.to_datetime(["2020-05-20"]),
        "year": [2020],
    })
    
    # Task statements including unmatched task T3
    task_statements = pd.DataFrame({
        "Task ID": ["T1", "T3"],
        "O*NET-SOC Code": ["11-1011.00", "11-1011.00"],
        "Task": ["Do thing 1", "Do thing 3"],
        "Task Type": ["Core", "Core"],
        "Title": ["Chief Executives", "Chief Executives"],
    })
    
    # This should not crash, even though T3 has no matches
    task_firm = pipeline.step2_calculate_firm_task_exposure(
        task_app_matches=sparse_task_matches,
        job_app_mapping=job_mapping,
        original_jobs=original_jobs,
        stage2_mapping_file=None  # No deduplication mapping needed
    )
    
    # Should only contain data for T1 (T3 won't appear since it has no matches)
    task_ids = set(task_firm["onet_task_id"].unique())
    assert "T1" in task_ids
    assert "T3" not in task_ids  # No matches = no exposure data generated


def test_multiple_firms_same_occupation(pipeline):
    """
    Test scenario with multiple firms in the same occupation.
    
    This validates that the pipeline correctly calculates separate exposure scores
    for different firms even when they're in the same occupation, demonstrating
    the key innovation of firm-specific exposure measurement.
    """
    # Two firms with different AI application portfolios
    job_mapping = pd.DataFrame({
        "app_text": ["App A", "App B", "App C"],
        "job_uids": ["u1", "u2", "u3"],
        "first_occurrence_tst_created": ["2020-01-01", "2020-01-02", "2020-01-03"],
    })
    
    original_jobs = pd.DataFrame({
        "job_uid": ["u1", "u2", "u3"],
        "company_name": ["FirmX", "FirmX", "FirmY"],  # FirmX has 2 apps, FirmY has 1
        "x28_occupations": ["", "", ""],
        "x28_industries": ["", "", ""],
        "tst_created": pd.to_datetime(["2020-05-20", "2020-05-21", "2020-05-22"]),
        "year": [2020, 2020, 2020],
    })
    
    # Task T1 matches all apps, T2 matches only App A and App C
    task_matches = pd.DataFrame({
        "app_text":     ["App A", "App B", "App C", "App A", "App C"],
        "onet_task_id": ["T1",    "T1",    "T1",    "T2",    "T2"],
    })
    
    task_firm = pipeline.step2_calculate_firm_task_exposure(
        task_app_matches=task_matches,
        job_app_mapping=job_mapping,
        original_jobs=original_jobs,
        stage2_mapping_file=None
    )
    
    # FirmX has 2 apps total
    firmx_data = task_firm[task_firm["company_name"] == "FirmX"]
    assert set(firmx_data["n_ai_apps_firm_year"].unique()) == {2}
    
    # FirmY has 1 app total  
    firmy_data = task_firm[task_firm["company_name"] == "FirmY"]
    assert set(firmy_data["n_ai_apps_firm_year"].unique()) == {1}
    
    # FirmX T1 exposure: 2/2 = 1.0 (both apps match)
    firmx_t1 = firmx_data[firmx_data["onet_task_id"] == "T1"]["hampole_task_exposure"].iloc[0]
    assert firmx_t1 == pytest.approx(1.0, rel=1e-9)
    
    # FirmX T2 exposure: 1/2 = 0.5 (only App A matches)
    firmx_t2 = firmx_data[firmx_data["onet_task_id"] == "T2"]["hampole_task_exposure"].iloc[0]
    assert firmx_t2 == pytest.approx(0.5, rel=1e-9)
    
    # FirmY T1 exposure: 1/1 = 1.0 (App C matches)
    firmy_t1 = firmy_data[firmy_data["onet_task_id"] == "T1"]["hampole_task_exposure"].iloc[0]
    assert firmy_t1 == pytest.approx(1.0, rel=1e-9)
    
    # FirmY T2 exposure: 1/1 = 1.0 (App C matches)
    firmy_t2 = firmy_data[firmy_data["onet_task_id"] == "T2"]["hampole_task_exposure"].iloc[0]
    assert firmy_t2 == pytest.approx(1.0, rel=1e-9)


def test_temporal_variation(pipeline):
    """
    Test temporal variation in AI exposure across years.
    
    This validates that the pipeline correctly handles time-varying exposure
    when firms adopt different AI applications in different years.
    """
    # Same firm, different years, different apps
    job_mapping = pd.DataFrame({
        "app_text": ["App A", "App B"],
        "job_uids": ["u1", "u2"],
        "first_occurrence_tst_created": ["2020-01-01", "2021-01-01"],
    })
    
    original_jobs = pd.DataFrame({
        "job_uid": ["u1", "u2"],
        "company_name": ["FirmX", "FirmX"],
        "x28_occupations": ["", ""],
        "x28_industries": ["", ""],
        "tst_created": pd.to_datetime(["2020-05-20", "2021-05-20"]),
        "year": [2020, 2021],
    })
    
    # Task T1 matches both apps across years
    task_matches = pd.DataFrame({
        "app_text":     ["App A", "App B"],
        "onet_task_id": ["T1",    "T1"],
    })
    
    task_firm = pipeline.step2_calculate_firm_task_exposure(
        task_app_matches=task_matches,
        job_app_mapping=job_mapping,
        original_jobs=original_jobs,
        stage2_mapping_file=None
    )
    
    # Should have separate entries for 2020 and 2021
    years = set(task_firm["year"].unique())
    assert years == {2020, 2021}
    
    # Each year should show 1 app for this firm-year
    year_2020 = task_firm[task_firm["year"] == 2020]
    year_2021 = task_firm[task_firm["year"] == 2021]
    
    assert set(year_2020["n_ai_apps_firm_year"].unique()) == {1}
    assert set(year_2021["n_ai_apps_firm_year"].unique()) == {1}
    
    # Both years should have full exposure (1/1 = 1.0) since the single app matches
    assert year_2020["hampole_task_exposure"].iloc[0] == pytest.approx(1.0, rel=1e-9)
    assert year_2021["hampole_task_exposure"].iloc[0] == pytest.approx(1.0, rel=1e-9)


if __name__ == "__main__":
    # Allow running tests directly with: python test_stage5.py
    pytest.main([__file__, "-v"])