#!/usr/bin/env python3
"""
Sample 500 random companies from the database and extract their AI jobs.

This script:
1. Connects to the PostgreSQL database
2. Retrieves all unique companies (including those with no AI jobs)
3. Randomly samples 500 companies with seed=42 for reproducibility
4. Filters ai_development_deduplicated_custom.csv to extract jobs from those companies
5. Saves two output files for analysis

Reproducibility: Uses random_state=42, change the RANDOM_SEED constant to modify
"""

import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os

# Configuration
RANDOM_SEED = 123
N_COMPANIES = 1000
DATA_DIR = 'Data'
AI_JOBS_INPUT = f'{DATA_DIR}/ai_development_deduplicated_custom.csv'
COMPANIES_OUTPUT = f'{DATA_DIR}/1000_random_companies_sampled.csv'
AI_JOBS_OUTPUT = f'{DATA_DIR}/ai_jobs_from_1000_random_companies.csv'

def main():
    # Load environment variables
    load_dotenv('config.env')

    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    print(f"Random seed set to {RANDOM_SEED} for reproducibility")

    # Step 1: Connect to database and get all unique companies
    print("\n[1/5] Connecting to database and retrieving all companies...")
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )

    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT company_id, company_name
            FROM job_postings_unified
            ORDER BY company_id
        """)
        all_companies = pd.DataFrame(cur.fetchall(), columns=['company_id', 'company_name'])

    conn.close()
    print(f"✓ Found {len(all_companies)} unique companies in database")

    # Step 2: Randomly sample 500 companies
    print(f"\n[2/5] Randomly sampling {N_COMPANIES} companies...")
    n_sample = min(N_COMPANIES, len(all_companies))
    sampled_companies = all_companies.sample(n=n_sample, random_state=RANDOM_SEED)
    # Convert company_id to string to match CSV format
    sampled_companies['company_id'] = sampled_companies['company_id'].astype(str)
    print(f"✓ Sampled {len(sampled_companies)} companies")
    print(f"  Sample company IDs: {sorted(sampled_companies['company_id'].tolist())[:10]}... (showing first 10)")

    # Step 3: Load AI jobs file
    print(f"\n[3/5] Loading AI jobs from {AI_JOBS_INPUT}...")
    ai_jobs = pd.read_csv(AI_JOBS_INPUT)
    # Convert company_id to string for matching
    ai_jobs['company_id'] = ai_jobs['company_id'].astype(str)
    print(f"✓ Loaded {len(ai_jobs)} total AI jobs from file")

    # Step 4: Filter AI jobs to those from sampled companies
    print(f"\n[4/5] Filtering AI jobs from sampled companies...")
    ai_jobs_from_sample = ai_jobs[ai_jobs['company_id'].isin(sampled_companies['company_id'])]
    print(f"✓ Found {len(ai_jobs_from_sample)} AI jobs from the {len(sampled_companies)} sampled companies")

    # Step 5: Save outputs
    print(f"\n[5/5] Saving output files...")
    sampled_companies.to_csv(COMPANIES_OUTPUT, index=False)
    print(f"✓ Saved companies list to {COMPANIES_OUTPUT}")

    ai_jobs_from_sample.to_csv(AI_JOBS_OUTPUT, index=False)
    print(f"✓ Saved AI jobs to {AI_JOBS_OUTPUT}")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Companies sampled: {len(sampled_companies)}")
    print(f"Total AI jobs from sampled companies: {len(ai_jobs_from_sample)}")
    print(f"Average AI jobs per company: {len(ai_jobs_from_sample) / len(sampled_companies):.2f}")
    print(f"Companies with AI jobs: {ai_jobs_from_sample['company_id'].nunique()}")
    print(f"Companies with NO AI jobs: {len(sampled_companies) - ai_jobs_from_sample['company_id'].nunique()}")
    print(f"\nOutput files:")
    print(f"  - {COMPANIES_OUTPUT}")
    print(f"  - {AI_JOBS_OUTPUT}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
