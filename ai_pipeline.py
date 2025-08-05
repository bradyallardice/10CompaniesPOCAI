#!/usr/bin/env python3
"""
AI Tool Usage Extraction Pipeline for 10 Swiss Companies
Follows the Hampole et al. methodology for extracting AI tool usage from job postings.
"""

import os
import sys
import pandas as pd
import regex as re
import html
import unicodedata
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Add the Code/JobAds directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Code', 'JobAds'))
from Code.JobAds.get_ads_from_50_companies import load_random_company_ads

class AIExtractionPipeline:
    """Pipeline for extracting AI tool usage from job postings"""
    
    def __init__(self, config_path="config.env"):
        """Initialize the pipeline with configuration"""
        load_dotenv(config_path)
        self.data_dir = Path("data")
        self.keywords_dir = self.data_dir / "keywords"
        self.raw_ads_dir = self.data_dir / "raw_ads"
        self.candidates_dir = self.data_dir / "candidates"
        self.llm_output_dir = self.data_dir / "llm_output"
        
        # Load multilingual keywords
        self.keywords = self._load_keywords()
        self.keyword_pattern = self._compile_keyword_pattern()
        
    def _load_keywords(self):
        """Load keywords from all language files"""
        keywords = []
        for lang_file in self.keywords_dir.glob("*.txt"):
            with open(lang_file, 'r', encoding='utf-8') as f:
                keywords.extend([line.strip() for line in f if line.strip()])
        return keywords
    
    def _compile_keyword_pattern(self):
        """Compile regex pattern for keyword matching"""
        escaped_keywords = [re.escape(kw) for kw in self.keywords]
        pattern = r"\b(?:" + "|".join(escaped_keywords) + r")\b"
        return re.compile(pattern, re.IGNORECASE)
    
    def step1_pull_ads(self, num_companies=10):
        """Step 1: Pull job ads from random companies"""
        print(f"Step 1: Pulling ads from {num_companies} random companies...")
        
        ads_df = load_random_company_ads(
            num_companies=num_companies, 
            output_path=str(self.raw_ads_dir) + "/"
        )
        
        print(f"Retrieved {len(ads_df):,} ads from {num_companies} companies")
        return ads_df
    
    def step2_clean_and_tag(self, ads_df):
        """Step 2: Clean text and add language tags"""
        print("Step 2: Cleaning text and adding language tags...")
        
        def clean_text(text):
            """Clean HTML and normalize text"""
            if pd.isna(text):
                return ""
            text = html.unescape(str(text))
            text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
            text = unicodedata.normalize('NFKC', text).strip()
            text = re.sub(r'\s+', ' ', text.lower())
            return text
        
        # Clean text
        ads_df["clean_text"] = ads_df["description"].apply(clean_text)
        
        # Simple language detection based on existing language_code column
        # or fallback to basic heuristics
        if "language_code" in ads_df.columns:
            ads_df["lang"] = ads_df["language_code"]
        else:
            # Simple heuristic language detection
            ads_df["lang"] = "en"  # Default to English
        
        # Save cleaned data
        output_file = self.raw_ads_dir / "ads_clean.parquet"
        ads_df.to_parquet(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
        
        return ads_df
    
    def step3_keyword_filter(self, ads_df):
        """Step 3: Filter ads using multilingual keyword matching"""
        print("Step 3: Filtering ads using keyword matching...")
        
        # Apply keyword filter
        candidates = ads_df[
            ads_df["clean_text"].str.contains(self.keyword_pattern, na=False)
        ]
        
        retention_rate = len(candidates) / len(ads_df) * 100
        print(f"Retained {len(candidates):,} ads ({retention_rate:.1f}% of total)")
        
        # Save candidates
        output_file = self.candidates_dir / "ai_candidates.parquet"
        candidates.to_parquet(output_file, index=False)
        print(f"Candidate ads saved to {output_file}")
        
        return candidates
    
    def run_data_preparation(self, num_companies=10):
        """Run the complete data preparation pipeline (steps 1-3)"""
        print("=" * 60)
        print("AI Tool Usage Extraction Pipeline - Data Preparation")
        print("=" * 60)
        
        # Step 1: Pull ads
        ads_df = self.step1_pull_ads(num_companies)
        
        # Step 2: Clean and tag
        ads_df = self.step2_clean_and_tag(ads_df)
        
        # Step 3: Keyword filter
        candidates = self.step3_keyword_filter(ads_df)
        
        print("\nData preparation complete!")
        print(f"- Total ads: {len(ads_df):,}")
        print(f"- Candidate ads: {len(candidates):,}")
        print(f"- Ready for LLM processing")
        
        return candidates

def main():
    """Main function to run the pipeline"""
    pipeline = AIExtractionPipeline()
    
    # Run data preparation only (no LLM processing)
    candidates = pipeline.run_data_preparation(num_companies=10)
    
    print("\nSetup complete! Next steps:")
    print("1. Update config.env with your database credentials and API keys")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the pipeline: python ai_pipeline.py")

if __name__ == "__main__":
    main()