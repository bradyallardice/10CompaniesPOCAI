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


def test_mathematical_verification_multi_occupation_all_six_types():
    """
    COMPREHENSIVE MATHEMATICAL VERIFICATION - ALL 6 EXPOSURE TYPES WITH MULTIPLE OCCUPATIONS

    This test verifies end-to-end calculation correctness for all 6 exposure types
    using multiple occupations to properly test cross-occupation aggregation:

    1. firm × year (time-variant firm-level)
    2. firm (time-invariant firm-level)
    3. firm × occupation × year (firm-level with occupation breakdown)
    4. firm × occupation (firm-level time-invariant by occupation)
    5. occupation × year (occupation-level varying by year)
    6. occupation (occupation-level time-invariant)

    ENHANCED TEST SCENARIO:
    - 2 companies (CompanyA, CompanyB)
    - 2 years (2020, 2021)
    - 3 AI applications (App1: 2020, App2: 2020, App3: 2021)
    - 2 occupations with different task structures
    - 4 tasks with varying importance weights

    Occupations & Tasks:
      11-1011.00 (Chief Executives):
        - Task1 (importance: 4.0) - matched by App1, App2
        - Task2 (importance: 3.0) - matched by App3

      15-1211.00 (Computer Systems Analysts):
        - Task3 (importance: 5.0) - matched by App1
        - Task4 (importance: 2.0) - matched by App2, App3

    Company usage:
    - CompanyA: App1(2020), App2(2020), App3(2021) - has BOTH occupations
    - CompanyB: App1(2020) only - has ONLY 11-1011.00 occupation

    This tests cross-occupation aggregation and firm-occupation variation.
    """

    # ========================================================================
    # GOLDEN INPUT DATA WITH MULTIPLE OCCUPATIONS
    # ========================================================================

    # Job-application mapping (CompanyA has all apps, CompanyB has App1 only)
    job_app_mapping = pd.DataFrame({
        "app_text": ["App1", "App2", "App3", "App1"],
        "job_uids": ["u1", "u2", "u3", "u4"],
        "first_occurrence_tst_created": ["2020-01-01", "2020-01-15", "2021-01-01", "2020-01-01"],
    })

    # Original jobs with occupation assignments
    original_jobs = pd.DataFrame({
        "job_uid": ["u1", "u2", "u3", "u4"],
        "company_name": ["CompanyA", "CompanyA", "CompanyA", "CompanyB"],
        "x28_occupations": ["", "", "", ""],
        "x28_industries": ["", "", "", ""],
        "tst_created": pd.to_datetime(["2020-05-20", "2020-05-25", "2021-05-20", "2020-06-01"]),
        "year": [2020, 2020, 2021, 2020],
    })

    # Task-application matches across multiple occupations
    task_app_matches = pd.DataFrame({
        "app_text": ["App1", "App2", "App3", "App1", "App2", "App3"],
        "onet_task_id": ["Task1", "Task1", "Task2", "Task3", "Task4", "Task4"],
    })

    # Task statements for TWO occupations
    task_statements = pd.DataFrame({
        "Task ID": ["Task1", "Task2", "Task3", "Task4"],
        "O*NET-SOC Code": ["11-1011.00", "11-1011.00", "15-1211.00", "15-1211.00"],
        "Task": ["Execute strategic planning", "Implement AI solutions", "Analyze system requirements", "Design data architectures"],
        "Task Type": ["Core", "Core", "Core", "Core"],
        "Title": ["Chief Executives", "Chief Executives", "Computer Systems Analysts", "Computer Systems Analysts"],
    })

    # Task ratings with different importance weights per occupation
    task_ratings = pd.DataFrame({
        "O*NET-SOC Code": ["11-1011.00", "11-1011.00", "15-1211.00", "15-1211.00"],
        "Task ID": ["Task1", "Task2", "Task3", "Task4"],
        "Scale ID": ["IM", "IM", "IM", "IM"],
        "Scale Name": ["Importance", "Importance", "Importance", "Importance"],
        "Data Value": [4.0, 3.0, 5.0, 2.0],  # Different weights: CEO(4,3), Analyst(5,2)
    })

    # ========================================================================
    # EXPECTED MATHEMATICAL CALCULATIONS FOR VERIFICATION
    # ========================================================================

    # FIXED CALCULATIONS WITH "FROM FIRST APPEARANCE ONWARDS" BEHAVIOR:

    # OCCUPATION 11-1011.00 (CEO): Task1(4.0), Task2(3.0), total_weight=7.0
    # CompanyA-2020: Task1=2/2=1.0, Task2=0/2=0.0, weighted=(1.0×4.0+0.0×3.0)/7.0=0.571, n_apps=2
    # CompanyA-2021: Task1=2/3=0.667, Task2=1/3=0.333, weighted=(0.667×4.0+0.333×3.0)/7.0=0.524, n_apps=3
    # CompanyB-2020: Task1=1/1=1.0, Task2=0/1=0.0, weighted=(1.0×4.0+0.0×3.0)/7.0=0.571, n_apps=1
    # CompanyB-2021: Task1=1/1=1.0, Task2=0/1=0.0, weighted=(1.0×4.0+0.0×3.0)/7.0=0.571, n_apps=1

    # OCCUPATION 15-1211.00 (Analyst): Task3(5.0), Task4(2.0), total_weight=7.0
    # CompanyA-2020: Task3=1/2=0.5, Task4=1/2=0.5, weighted=(0.5×5.0+0.5×2.0)/7.0=0.5, n_apps=2
    # CompanyA-2021: Task3=1/3=0.333, Task4=2/3=0.667, weighted=(0.333×5.0+0.667×2.0)/7.0=0.4286, n_apps=3
    # CompanyB-2020: Task3=1/1=1.0, Task4=0/1=0.0, weighted=(1.0×5.0+0.0×2.0)/7.0=0.714, n_apps=1
    # CompanyB-2021: Task3=1/1=1.0, Task4=0/1=0.0, weighted=(1.0×5.0+0.0×2.0)/7.0=0.714, n_apps=1

    # ========================================================================
    # TYPE 1: FIRM × YEAR (occupation_exposure='none', time_invariant=False)
    # ========================================================================

    pipeline_1 = TaskFirmExposurePipeline(
        aggregation_method="mean", time_invariant=False, occupation_exposure="none", data_dir=""
    )

    task_firm_1 = pipeline_1.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches, job_app_mapping=job_app_mapping,
        original_jobs=original_jobs, stage2_mapping_file=None
    )
    occupation_firm_1 = pipeline_1.step3_calculate_occupation_firm_exposure(
        task_firm_exposure=task_firm_1, task_statements=task_statements, task_ratings=task_ratings
    )
    final_1 = pipeline_1.step4_apply_ai_intensity_adjustment(occupation_firm_1)

    # ========================================================================
    # TYPE 3: FIRM × OCCUPATION × YEAR - Verify detailed occupation breakdown
    # ========================================================================

    # CompanyA should have both occupations, CompanyB should have only CEO occupation
    company_a_ceo_2020 = final_1[(final_1['company_name'] == 'CompanyA') &
                                  (final_1['onet_code'] == '11-1011.00') &
                                  (final_1['year'] == 2020)]
    company_a_analyst_2020 = final_1[(final_1['company_name'] == 'CompanyA') &
                                      (final_1['onet_code'] == '15-1211.00') &
                                      (final_1['year'] == 2020)]
    company_b_ceo_2020 = final_1[(final_1['company_name'] == 'CompanyB') &
                                  (final_1['onet_code'] == '11-1011.00') &
                                  (final_1['year'] == 2020)]

    # Verify AI app counts
    assert company_a_ceo_2020['n_ai_apps_firm_year'].iloc[0] == 2
    assert company_a_analyst_2020['n_ai_apps_firm_year'].iloc[0] == 2
    assert company_b_ceo_2020['n_ai_apps_firm_year'].iloc[0] == 1

    # Verify occupation-specific exposure calculations
    # CEO occupation: (1.0×4.0+0.0×3.0)/7.0=0.571
    assert company_a_ceo_2020['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.571, rel=1e-3)
    assert company_b_ceo_2020['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.571, rel=1e-3)

    # Analyst occupation: (0.5×5.0+0.5×2.0)/7.0=0.5
    assert company_a_analyst_2020['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.5, rel=1e-3)

    # CompanyA should ALSO have analyst occupation exposure in 2021 via App3
    company_a_analyst_2021 = final_1[(final_1['company_name'] == 'CompanyA') &
                                      (final_1['onet_code'] == '15-1211.00') &
                                      (final_1['year'] == 2021)]
    assert len(company_a_analyst_2021) == 1

    # CompanyA analyst 2021: Fixed behavior - has App1, App2, App3 available (from first appearance onwards)
    # Task3=1/3=0.333 (App1 matches), Task4=2/3=0.667 (App2, App3 match)
    # Weighted: (0.333×5.0 + 0.667×2.0) / 7.0 = (1.667 + 1.333) / 7.0 = 3.0/7.0 = 0.4286
    assert company_a_analyst_2021['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.4286, rel=1e-3)

    # CompanyB should ALSO have analyst occupation exposure (Stage 5 creates all possibilities)
    # because it uses App1 which matches Task3 (analyst task)
    company_b_analyst_2020 = final_1[(final_1['company_name'] == 'CompanyB') &
                                      (final_1['onet_code'] == '15-1211.00') &
                                      (final_1['year'] == 2020)]
    assert len(company_b_analyst_2020) == 1

    # CompanyB analyst: Task3=1/1=1.0, Task4=0/1=0.0, weighted=(1.0×5.0+0.0×2.0)/7.0=0.714
    assert company_b_analyst_2020['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.714, rel=1e-3)

    # CompanyB should also have analyst exposure in 2021 (same as 2020 since only App1)
    company_b_analyst_2021 = final_1[(final_1['company_name'] == 'CompanyB') &
                                      (final_1['onet_code'] == '15-1211.00') &
                                      (final_1['year'] == 2021)]
    assert len(company_b_analyst_2021) == 1
    assert company_b_analyst_2021['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.714, rel=1e-3)

    # ========================================================================
    # TYPE 2: FIRM (occupation_exposure='none', time_invariant=True)
    # ========================================================================

    pipeline_2 = TaskFirmExposurePipeline(
        aggregation_method="mean", time_invariant=True, occupation_exposure="none", data_dir=""
    )

    task_firm_2 = pipeline_2.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches, job_app_mapping=job_app_mapping,
        original_jobs=original_jobs, stage2_mapping_file=None
    )
    occupation_firm_2 = pipeline_2.step3_calculate_occupation_firm_exposure(
        task_firm_exposure=task_firm_2, task_statements=task_statements, task_ratings=task_ratings
    )
    final_2 = pipeline_2.step4_apply_ai_intensity_adjustment(occupation_firm_2)

    # ========================================================================
    # TYPE 4: FIRM × OCCUPATION - Verify time-invariant by occupation
    # ========================================================================

    # Both companies should have 2 occupation records (Stage 5 creates all possibilities)
    company_a_occupations = final_2[final_2['company_name'] == 'CompanyA']['onet_code'].unique()
    company_b_occupations = final_2[final_2['company_name'] == 'CompanyB']['onet_code'].unique()

    assert set(company_a_occupations) == {'11-1011.00', '15-1211.00'}
    assert set(company_b_occupations) == {'11-1011.00', '15-1211.00'}  # CompanyB gets both because App1 affects both

    # Verify time-invariant calculations (all apps ever used)
    company_a_ceo_ti = final_2[(final_2['company_name'] == 'CompanyA') &
                               (final_2['onet_code'] == '11-1011.00')]
    company_a_analyst_ti = final_2[(final_2['company_name'] == 'CompanyA') &
                                   (final_2['onet_code'] == '15-1211.00')]
    company_b_ceo_ti = final_2[(final_2['company_name'] == 'CompanyB') &
                               (final_2['onet_code'] == '11-1011.00')]
    company_b_analyst_ti = final_2[(final_2['company_name'] == 'CompanyB') &
                                   (final_2['onet_code'] == '15-1211.00')]

    assert company_a_ceo_ti['n_ai_apps_firm_year'].iloc[0] == 3  # All 3 apps
    assert company_a_analyst_ti['n_ai_apps_firm_year'].iloc[0] == 3  # All 3 apps
    assert company_b_ceo_ti['n_ai_apps_firm_year'].iloc[0] == 1  # Only App1
    assert company_b_analyst_ti['n_ai_apps_firm_year'].iloc[0] == 1  # Only App1

    # CEO: Task1=2/3, Task2=1/3, weighted=(0.667×4.0+0.333×3.0)/7.0=0.524
    assert company_a_ceo_ti['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.524, rel=1e-3)

    # Analyst: Task3=1/3, Task4=2/3, weighted=(0.333×5.0+0.667×2.0)/7.0=0.429
    assert company_a_analyst_ti['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.429, rel=1e-3)

    # CompanyB CEO: Task1=1/1, Task2=0/1, weighted=(1.0×4.0+0.0×3.0)/7.0=0.571
    assert company_b_ceo_ti['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.571, rel=1e-3)

    # CompanyB Analyst: Task3=1/1, Task4=0/1, weighted=(1.0×5.0+0.0×2.0)/7.0=0.714
    assert company_b_analyst_ti['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.714, rel=1e-3)

    # ========================================================================
    # TYPE 5: OCCUPATION × YEAR (occupation_exposure='time-variant')
    # ========================================================================

    pipeline_5 = TaskFirmExposurePipeline(
        aggregation_method="mean", time_invariant=False, occupation_exposure="time-variant", data_dir=""
    )

    task_firm_5 = pipeline_5.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches, job_app_mapping=job_app_mapping,
        original_jobs=original_jobs, stage2_mapping_file=None
    )
    occupation_firm_5 = pipeline_5.step3_calculate_occupation_firm_exposure(
        task_firm_exposure=task_firm_5, task_statements=task_statements, task_ratings=task_ratings
    )
    final_5 = pipeline_5.step4_apply_ai_intensity_adjustment(occupation_firm_5)

    # In occupation-level mode, all companies should have same exposure by year
    # but different occupations should have different exposure patterns

    results_2020_ceo = final_5[(final_5['year'] == 2020) & (final_5['onet_code'] == '11-1011.00')]
    results_2020_analyst = final_5[(final_5['year'] == 2020) & (final_5['onet_code'] == '15-1211.00')]

    # FIXED: Companies now maintain firm-specific app counts (not identical)
    company_a_ceo_2020 = results_2020_ceo[results_2020_ceo['company_name'] == 'CompanyA']
    company_b_ceo_2020 = results_2020_ceo[results_2020_ceo['company_name'] == 'CompanyB']
    company_a_analyst_2020 = results_2020_analyst[results_2020_analyst['company_name'] == 'CompanyA']
    company_b_analyst_2020 = results_2020_analyst[results_2020_analyst['company_name'] == 'CompanyB']

    # Verify firm-specific app counts are maintained in occupation mode
    assert company_a_ceo_2020['n_ai_apps_firm_year'].iloc[0] == 2  # CompanyA: 2 apps
    assert company_b_ceo_2020['n_ai_apps_firm_year'].iloc[0] == 1  # CompanyB: 1 app
    assert company_a_analyst_2020['n_ai_apps_firm_year'].iloc[0] == 2  # CompanyA: 2 apps
    assert company_b_analyst_2020['n_ai_apps_firm_year'].iloc[0] == 1  # CompanyB: 1 app

    # Verify firm-specific exposure calculations
    # CompanyA CEO: Task1=2/2=1.0, Task2=0/2=0.0, weighted=(1.0×4.0+0.0×3.0)/7.0=0.571
    assert company_a_ceo_2020['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.571, rel=1e-3)
    # CompanyB CEO: Task1=1/1=1.0, Task2=0/1=0.0, weighted=(1.0×4.0+0.0×3.0)/7.0=0.571
    assert company_b_ceo_2020['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.571, rel=1e-3)
    # CompanyA Analyst: Task3=1/2=0.5, Task4=1/2=0.5, weighted=(0.5×5.0+0.5×2.0)/7.0=0.5
    assert company_a_analyst_2020['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.5, rel=1e-3)
    # CompanyB Analyst: Task3=1/1=1.0, Task4=0/1=0.0, weighted=(1.0×5.0+0.0×2.0)/7.0=0.714
    assert company_b_analyst_2020['hampole_occupation_exposure'].iloc[0] == pytest.approx(0.714, rel=1e-3)

    # ========================================================================
    # TYPE 6: OCCUPATION (occupation_exposure='time-invariant')
    # ========================================================================

    pipeline_6 = TaskFirmExposurePipeline(
        aggregation_method="mean", time_invariant=False, occupation_exposure="time-invariant", data_dir=""
    )

    task_firm_6 = pipeline_6.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches, job_app_mapping=job_app_mapping,
        original_jobs=original_jobs, stage2_mapping_file=None
    )
    occupation_firm_6 = pipeline_6.step3_calculate_occupation_firm_exposure(
        task_firm_exposure=task_firm_6, task_statements=task_statements, task_ratings=task_ratings
    )
    final_6 = pipeline_6.step4_apply_ai_intensity_adjustment(occupation_firm_6)

    # FIXED: Records now have firm-specific app counts, not identical
    company_a_records_6 = final_6[final_6['company_name'] == 'CompanyA']
    company_b_records_6 = final_6[final_6['company_name'] == 'CompanyB']

    # CompanyA should have 3 apps, CompanyB should have 1 app (firm-specific)
    assert all(company_a_records_6['n_ai_apps_firm_year'] == 3)
    assert all(company_b_records_6['n_ai_apps_firm_year'] == 1)

    # Split by occupation and company
    company_a_ceo_6 = final_6[(final_6['onet_code'] == '11-1011.00') & (final_6['company_name'] == 'CompanyA')]
    company_a_analyst_6 = final_6[(final_6['onet_code'] == '15-1211.00') & (final_6['company_name'] == 'CompanyA')]
    company_b_ceo_6 = final_6[(final_6['onet_code'] == '11-1011.00') & (final_6['company_name'] == 'CompanyB')]
    company_b_analyst_6 = final_6[(final_6['onet_code'] == '15-1211.00') & (final_6['company_name'] == 'CompanyB')]

    # CompanyA (3 apps): CEO=(2/3×4.0+1/3×3.0)/7.0=0.524, Analyst=(1/3×5.0+2/3×2.0)/7.0=0.429
    assert all(company_a_ceo_6['hampole_occupation_exposure'].round(4) == 0.5238)
    assert all(company_a_analyst_6['hampole_occupation_exposure'].round(4) == 0.4286)

    # CompanyB (1 app): CEO=(1/1×4.0+0/1×3.0)/7.0=0.571, Analyst=(1/1×5.0+0/1×2.0)/7.0=0.714
    assert all(company_b_ceo_6['hampole_occupation_exposure'].round(4) == 0.5714)
    assert all(company_b_analyst_6['hampole_occupation_exposure'].round(4) == 0.7143)

    # ========================================================================
    # COMPREHENSIVE VERIFICATION SUMMARY
    # ========================================================================

    print("✅ ALL 6 EXPOSURE TYPES WITH MULTIPLE OCCUPATIONS VERIFIED!")
    print(f"   Type 1 (firm×year): {len(final_1)} records across 2 occupations")
    print(f"   Type 2 (firm): {len(final_2)} records, CompanyA=2 occs, CompanyB=1 occ")
    print(f"   Type 3 (firm×occ×year): Verified occupation-specific breakdowns")
    print(f"   Type 4 (firm×occ): Verified time-invariant by occupation")
    print(f"   Type 5 (occ×year): {len(final_5)} records, occupation-specific patterns")
    print(f"   Type 6 (occ): {len(final_6)} records, cross-occupation verification")
    print("   ✓ Multi-occupation task portfolios correctly aggregated")
    print("   ✓ Firm-occupation variation properly captured")
    print("   ✓ Cross-occupation weighted averaging verified")
    print("   ✓ All 6 exposure types mathematically consistent")

    # Final assertion: ensure we tested meaningful cross-occupation variation
    assert len(set(final_1['onet_code'])) == 2  # 2 occupations tested
    # Both companies should have exposure records for both occupations and both years (4 records each)
    assert len(final_1[final_1['company_name'] == 'CompanyA']) == 4  # CompanyA: 2 occupations × 2 years
    assert len(final_1[final_1['company_name'] == 'CompanyB']) == 4  # CompanyB: 2 occupations × 2 years (with fixed behavior)


def test_fixed_occupation_exposure_firm_specific_vs_pure():
    """
    Test the FIXED occupation exposure logic that supports both:
    1. Firm-specific portfolios (each firm has different AI app counts)
    2. Pure occupation exposure (all firms have identical AI app counts)

    This addresses the bug where occupation-level exposure gave all firms
    the same AI app count instead of respecting firm-specific portfolios.
    """

    # Test data with CLEAR firm differences
    job_app_mapping = pd.DataFrame({
        "app_text": ["App1", "App2", "App3"],
        "job_uids": ["u1", "u2", "u3"],
        "first_occurrence_tst_created": ["2020-01-01", "2020-01-15", "2020-01-01"],
    })

    # CompanyA has 2 apps, CompanyB has 1 app - CLEAR difference
    original_jobs = pd.DataFrame({
        "job_uid": ["u1", "u2", "u3"],
        "company_name": ["CompanyA", "CompanyA", "CompanyB"],  # CompanyA=2 apps, CompanyB=1 app
        "x28_occupations": ["", "", ""],
        "x28_industries": ["", "", ""],
        "tst_created": pd.to_datetime(["2020-05-20", "2020-05-25", "2020-06-01"]),
        "year": [2020, 2020, 2020],
    })

    task_app_matches = pd.DataFrame({
        "app_text": ["App1", "App2", "App3"],
        "onet_task_id": ["Task1", "Task1", "Task2"],
    })

    task_statements = pd.DataFrame({
        "Task ID": ["Task1", "Task2"],
        "O*NET-SOC Code": ["11-1011.00", "11-1011.00"],
        "Task": ["Execute strategy", "Implement AI"],
        "Task Type": ["Core", "Core"],
        "Title": ["Chief Executives", "Chief Executives"],
    })

    task_ratings = pd.DataFrame({
        "O*NET-SOC Code": ["11-1011.00", "11-1011.00"],
        "Task ID": ["Task1", "Task2"],
        "Scale ID": ["IM", "IM"],
        "Scale Name": ["Importance", "Importance"],
        "Data Value": [4.0, 3.0],
    })

    # ========================================================================
    # TEST 1: FIRM-SPECIFIC OCCUPATION EXPOSURE (should detect firm differences)
    # ========================================================================

    pipeline_firm_specific = TaskFirmExposurePipeline(
        aggregation_method="mean",
        time_invariant=False,
        occupation_exposure="time-variant",
        data_dir=""
    )

    task_firm = pipeline_firm_specific.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches,
        job_app_mapping=job_app_mapping,
        original_jobs=original_jobs,
        stage2_mapping_file=None
    )

    # Verify firm-specific AI app counts
    company_a_records = task_firm[task_firm['company_name'] == 'CompanyA']
    company_b_records = task_firm[task_firm['company_name'] == 'CompanyB']

    # CompanyA should have 2 apps, CompanyB should have 1 app
    assert all(company_a_records['n_ai_apps_firm_year'] == 2), f"CompanyA should have 2 apps, got: {company_a_records['n_ai_apps_firm_year'].unique()}"
    assert all(company_b_records['n_ai_apps_firm_year'] == 1), f"CompanyB should have 1 app, got: {company_b_records['n_ai_apps_firm_year'].unique()}"

    # CompanyA exposure: Task1=2/2=1.0, Task2=0/2=0.0 (App3 is CompanyB's)
    company_a_task1 = company_a_records[company_a_records['onet_task_id'] == 'Task1']['hampole_task_exposure'].iloc[0]
    assert company_a_task1 == pytest.approx(1.0, rel=1e-3), f"CompanyA Task1 should be 1.0, got {company_a_task1}"

    # CompanyB exposure: Task2=1/1=1.0 (only App3), Task1 should not appear
    company_b_tasks = company_b_records['onet_task_id'].unique()
    assert 'Task2' in company_b_tasks, "CompanyB should have Task2 exposure"
    company_b_task2 = company_b_records[company_b_records['onet_task_id'] == 'Task2']['hampole_task_exposure'].iloc[0]
    assert company_b_task2 == pytest.approx(1.0, rel=1e-3), f"CompanyB Task2 should be 1.0, got {company_b_task2}"

    print("✅ FIXED: Firm-specific occupation exposure maintains different AI app counts per firm")
    print(f"   CompanyA: {company_a_records['n_ai_apps_firm_year'].iloc[0]} apps")
    print(f"   CompanyB: {company_b_records['n_ai_apps_firm_year'].iloc[0]} apps")
    print("   ✓ Firms maintain their own AI portfolios within occupation framework")

    # ========================================================================
    # TEST 2: PURE OCCUPATION EXPOSURE (artificial scenario for testing)
    # ========================================================================

    # Create artificial scenario where all firms have identical app counts
    # This would trigger the "pure occupation" mode in the fixed logic

    # Modify data so all firms have same apps (for pure occupation testing)
    job_app_mapping_identical = pd.DataFrame({
        "app_text": ["App1", "App1", "App1"],  # All firms get App1
        "job_uids": ["u1", "u2", "u3"],
        "first_occurrence_tst_created": ["2020-01-01", "2020-01-01", "2020-01-01"],
    })

    pipeline_pure = TaskFirmExposurePipeline(
        aggregation_method="mean",
        time_invariant=False,
        occupation_exposure="time-variant",
        data_dir=""
    )

    task_firm_pure = pipeline_pure.step2_calculate_firm_task_exposure(
        task_app_matches=task_app_matches,
        job_app_mapping=job_app_mapping_identical,
        original_jobs=original_jobs,
        stage2_mapping_file=None
    )

    # In pure mode, all firms should have identical app counts
    all_app_counts = task_firm_pure['n_ai_apps_firm_year'].unique()
    assert len(all_app_counts) == 1, f"Pure occupation mode should give all firms same app count, got: {all_app_counts}"

    print("✅ FIXED: Pure occupation exposure gives all firms identical patterns")
    print(f"   All firms: {all_app_counts[0]} apps (identical)")
    print("   ✓ Supports both firm-specific and pure occupation modes")


if __name__ == "__main__":
    # Allow running tests directly with: python test_stage5.py
    pytest.main([__file__, "-v"])