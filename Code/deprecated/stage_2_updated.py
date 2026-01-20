#!/usr/bin/env python3
"""
Deduplicate and Translate AI Development Data - UPDATED VERSION
- NEW: Occurrence-level AI/KI/IA classification system
- Deduplicates job postings based on content similarity
- Translates full raw text to English for analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from dotenv import load_dotenv
import openai
from datetime import datetime
import time
import psycopg2
from sqlalchemy import create_engine
from langdetect import detect, DetectorFactory
import warnings

# Set seed for reproducible language detection
DetectorFactory.seed = 0

class DeduplicateAndTranslate:
    """Deduplicate and translate AI development job postings"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "Data"
        
        # Load environment variables
        load_dotenv('config.env')
        
        # Set up OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def detect_ai_false_positives_robust(self, df):
        """
        UPDATED: Detect and filter AI false positives using occurrence-level classification.
        Filters jobs where ALL ai/ki/ia occurrences are false positives.
        """
        from langdetect import detect
        
        print("Filtering AI false positives with occurrence-level classification...")
        
        # Load AI context terms from CSV
        ai_context_path = self.data_dir / 'ai_context.csv'
        if not ai_context_path.exists():
            print(f"WARNING: {ai_context_path} not found. Using basic tech terms.")
            ai_context_terms = {
                'en': {'model', 'models', 'machine', 'learning', 'neural', 'network', 'algorithm', 
                       'algorithms', 'dataset', 'feature', 'features', 'ml', 'nlp', 'vision',
                       'deep', 'artificial', 'intelligence', 'data', 'science', 'analytics'},
                'fr': {'apprentissage', 'réseau', 'neurones', 'algorithme', 'algorithmes',
                       'données', 'intelligence', 'artificielle'},
                'it': {'modello', 'modelli', 'apprendimento', 'rete', 'neurale', 'algoritmi',
                       'dati', 'intelligenza', 'artificiale'},
                'de': {'model', 'models', 'machine', 'learning', 'neural', 'network', 'algorithm'}
            }
        else:
            ai_context_df = pd.read_csv(ai_context_path)
            
            # Build context dictionaries by language
            ai_context_terms = {
                'en': set(),
                'fr': set(), 
                'it': set(),
                'de': set()  # Will use English terms for German
            }
            
            for _, row in ai_context_df.iterrows():
                # English terms
                if pd.notna(row['en_primary']):
                    ai_context_terms['en'].add(row['en_primary'].lower())
                    ai_context_terms['de'].add(row['en_primary'].lower())  # German uses English AI terms
                if pd.notna(row['en_variants']):
                    for variant in row['en_variants'].split(';'):
                        ai_context_terms['en'].add(variant.strip().lower())
                        ai_context_terms['de'].add(variant.strip().lower())
                
                # French terms
                if pd.notna(row['fr_primary']):
                    ai_context_terms['fr'].add(row['fr_primary'].lower())
                if pd.notna(row['fr_variants']):
                    for variant in row['fr_variants'].split(';'):
                        ai_context_terms['fr'].add(variant.strip().lower())
                
                # Italian terms
                if pd.notna(row['it_primary']):
                    ai_context_terms['it'].add(row['it_primary'].lower())
                if pd.notna(row['it_variants']):
                    for variant in row['it_variants'].split(';'):
                        ai_context_terms['it'].add(variant.strip().lower())
        
        def detect_language_robust(text):
            """Robust language detection with fallback"""
            try:
                clean_text = re.sub(r'[^\w\s]', ' ', str(text)[:1000])
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                if len(clean_text) < 10:
                    return 'unknown'
                
                detected = detect(clean_text)
                return detected
            except Exception:
                return 'unknown'
        
        def classify_ai_occurrence(content, match_pos, match_text, detected_lang):
            """
            Classify individual ai/ki/ia occurrence as legitimate or false positive.
            Returns: 'legitimate', 'false_positive'
            """
            
            # Rule 1: All caps AI/KI/IA that doesn't match Swiss canton pattern = legitimate
            if match_text.isupper():
                if not is_swiss_canton_pattern(content, match_pos, match_text):
                    return 'legitimate'
                else:
                    # Swiss canton pattern - check for tech context to override
                    if has_tech_context_in_window(content, match_pos, detected_lang):
                        return 'legitimate'
                    else:
                        return 'false_positive'
            
            # Rule 2: Language-specific lowercase rules
            match_lower = match_text.lower()
            
            if detected_lang == 'de':
                # German: ki and ai are OK
                if match_lower in ['ki', 'ai']:
                    return 'legitimate'
                elif match_lower == 'ia':
                    return 'false_positive'
            
            elif detected_lang == 'en':
                # English: ai is OK
                if match_lower == 'ai':
                    return 'legitimate'
                elif match_lower in ['ia', 'ki']:
                    return 'false_positive'
            
            elif detected_lang in ['it', 'fr']:
                # Italian/French: ia is OK
                if match_lower == 'ia':
                    return 'legitimate'
                elif match_lower in ['ai', 'ki']:
                    # Need to check context for ai/ki in Italian/French
                    return classify_romance_ai_ki(content, match_pos, detected_lang)
            
            else:
                # Unknown language: default to checking context
                if match_lower in ['ai', 'ki', 'ia']:
                    return classify_unknown_language(content, match_pos, match_text)
            
            return 'legitimate'  # Default to keeping
        
        def is_swiss_canton_pattern(content, ai_pos, match_text):
            """Check if AI/KI/IA refers to Swiss canton in typical formatting"""
            if match_text.upper() != 'AI':  # Only check for AI canton
                return False
            
            swiss_cantons = {'AG', 'AI', 'AR', 'BE', 'BL', 'BS', 'FR', 'GE', 'GL', 'GR', 
                           'JU', 'LU', 'NE', 'NW', 'OW', 'SG', 'SH', 'SO', 'SZ', 'TG', 
                           'TI', 'UR', 'VD', 'VS', 'ZG', 'ZH'}
            
            # Extract surrounding context (±50 chars)
            start = max(0, ai_pos - 50)
            end = min(len(content), ai_pos + 50)
            context = content[start:end]
            
            # Check various Swiss canton patterns
            patterns = [
                r'\([^)]*AI[^)]*\)',  # Parentheses
                r'[A-Z]{2}(?:/[A-Z]{2})*/?AI/?(?:[A-Z]{2}/)*',  # Slash-separated
                r'[A-Z]{2}(?:\|[A-Z]{2})*\|?AI\|?(?:[A-Z]{2}\|?)*',  # Pipe-separated
                r'[A-Z]{2}(?:,\s*[A-Z]{2})*,?\s*AI\s*,?(?:\s*[A-Z]{2},?)*'  # Comma-separated
            ]
            
            for pattern in patterns:
                if re.search(pattern, context):
                    # Extract canton codes from the match
                    codes = re.findall(r'[A-Z]{2}', context)
                    swiss_codes = [code for code in codes if code in swiss_cantons]
                    if len(swiss_codes) >= 3 and 'AI' in swiss_codes:
                        return True
            
            return False
        
        def has_tech_context_in_window(content, pos, language):
            """Check for AI context terms within ±6 token window"""
            # Get language-specific context terms
            if language in ai_context_terms:
                context_terms = ai_context_terms[language]
            else:
                # Fallback to English terms
                context_terms = ai_context_terms['en']
            
            # Tokenize and find position
            text_lower = content.lower()
            tokens = re.findall(r'\b\w+\b', text_lower)
            
            # Find token position of the occurrence
            token_pos = None
            current_pos = 0
            for i, token in enumerate(tokens):
                token_start = text_lower.find(token, current_pos)
                token_end = token_start + len(token)
                if token_start <= pos < token_end:
                    token_pos = i
                    break
                current_pos = token_end
            
            if token_pos is None:
                return False
            
            # Check ±6 token window
            window_start = max(0, token_pos - 6)
            window_end = min(len(tokens), token_pos + 7)
            window_tokens = tokens[window_start:window_end]
            
            # Check if any context term is in the window
            return any(token in context_terms for token in window_tokens)
        
        def classify_romance_ai_ki(content, pos, language):
            """Classify ai/ki in Italian/French using context window"""
            if has_tech_context_in_window(content, pos, language):
                return 'legitimate'
            else:
                return 'false_positive'
        
        def classify_unknown_language(content, pos, match_text):
            """Classify ai/ki/ia for unknown language - default to tech context check"""
            if has_tech_context_in_window(content, pos, 'en'):  # Use English terms as fallback
                return 'legitimate'
            else:
                return 'false_positive'
        
        def should_filter_ai_job(row):
            """Main filtering logic: filter only if ALL ai/ki/ia occurrences are false positives"""
            keywords = str(row['matched_keywords']).strip().lower()
            if keywords != 'ai':
                return False  # Keep jobs with multiple keywords
            
            content = str(row['content_clean'])
            
            # Find all ai/ki/ia positions (case insensitive)
            ai_occurrences = []
            for pattern in [r'\bai\b', r'\bki\b', r'\bia\b']:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    ai_occurrences.append({
                        'pos': match.start(),
                        'text': match.group(),
                        'pattern': pattern
                    })
            
            if not ai_occurrences:
                return False  # Shouldn't happen, but safety check
            
            # Detect language once for the entire document
            detected_lang = detect_language_robust(content)
            
            # Classify each occurrence
            classifications = []
            for occurrence in ai_occurrences:
                classification = classify_ai_occurrence(
                    content, 
                    occurrence['pos'], 
                    occurrence['text'], 
                    detected_lang
                )
                classifications.append(classification)
            
            # Count legitimate vs false positive occurrences
            legitimate_count = sum(1 for c in classifications if c == 'legitimate')
            false_positive_count = sum(1 for c in classifications if c == 'false_positive')
            
            # Filter job only if ALL occurrences are false positives
            if legitimate_count > 0:
                return False  # Keep job - has at least one legitimate AI occurrence
            else:
                return True   # Filter job - all occurrences are false positives
        
        def should_filter_ml_job(row):
            """Filter ML jobs where ML directly follows a number"""
            keywords = str(row['matched_keywords']).strip().lower()
            if keywords != 'ml':
                return False  # Keep jobs with multiple keywords
            
            content = str(row['content_clean'])
            
            # Find all 'ml' positions (case insensitive)
            ml_occurrences = []
            for match in re.finditer(r'\bml\b', content, re.IGNORECASE):
                ml_occurrences.append(match.start())
            
            if not ml_occurrences:
                return False
            
            # Check if any ML occurrence is legitimate (not following a number)
            for ml_pos in ml_occurrences:
                # Check if ML is preceded by a number
                start = max(0, ml_pos - 10)  # Check 10 chars before
                before_ml = content[start:ml_pos]
                
                # Look for number pattern right before ML
                if not re.search(r'\d\s*$', before_ml):
                    return False  # Found ML not following number - keep job
            
            # All ML occurrences follow numbers - filter job
            return True
        
        # Apply robust filtering
        initial_count = len(df)
        
        # Filter AI false positives
        ai_false_positives_mask = df.apply(should_filter_ai_job, axis=1)
        
        # Filter ML false positives  
        ml_false_positives_mask = df.apply(should_filter_ml_job, axis=1)
        
        # Combine filters
        combined_false_positives_mask = ai_false_positives_mask | ml_false_positives_mask
        
        # Count and categorize removed jobs
        removed_jobs = df[combined_false_positives_mask]
        df_filtered = df[~combined_false_positives_mask]
        
        # Analyze what was removed
        if len(removed_jobs) > 0:
            print(f"Analyzing {len(removed_jobs)} jobs marked for removal:")
            removed_by_lang = {}
            ai_removed = sum(ai_false_positives_mask)
            ml_removed = sum(ml_false_positives_mask)
            print(f"  AI false positives: {ai_removed}")
            print(f"  ML false positives: {ml_removed}")
            
            for _, job in removed_jobs.iterrows():
                try:
                    lang = detect_language_robust(job['content_clean'])
                    removed_by_lang[lang] = removed_by_lang.get(lang, 0) + 1
                except:
                    removed_by_lang['unknown'] = removed_by_lang.get('unknown', 0) + 1
            
            for lang, count in removed_by_lang.items():
                print(f"  {lang}: {count} jobs")
        
        removed_count = initial_count - len(df_filtered)
        
        print(f"Total false positives removed: {removed_count:,} ({removed_count/initial_count*100:.1f}%)")
        print(f"Remaining after occurrence-level filter: {len(df_filtered):,} jobs")
        
        return df_filtered
    
    # Keep all other methods from original file...
    # (This is just the updated function for demonstration)

# Test the new approach
if __name__ == "__main__":
    processor = DeduplicateAndTranslate()
    # Load a small sample for testing
    df_sample = pd.read_csv(processor.data_dir / 'batched_ai_jobs_20250829_084432.csv').head(1000)
    result = processor.detect_ai_false_positives_robust(df_sample)
    print(f"Sample test: {len(df_sample)} -> {len(result)} jobs")