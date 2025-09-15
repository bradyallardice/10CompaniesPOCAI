#!/usr/bin/env python3
"""
Deduplicate and Translate AI Development Data
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
        
    def load_ai_development_data(self, dataset_type="train", custom_file=None):
        """Load AI development detection results for specified dataset"""
        if custom_file:
            # Use custom file path
            ai_file = self.data_dir / custom_file
            print(f"Using custom AI detection file: {ai_file.name}")
        elif dataset_type == "train":
            # Find the latest enhanced AI detection file (original training set)
            enhanced_files = list(self.data_dir.glob("enhanced_ai_detection_*.csv"))
            if enhanced_files:
                ai_file = max(enhanced_files, key=lambda x: x.stat().st_mtime)
                print(f"Using latest enhanced AI detection file: {ai_file.name}")
            else:
                ai_file = self.data_dir / "ai_development_detection.csv"
        elif dataset_type == "test":
            # Find the latest test set AI detection file
            test_files = list(self.data_dir.glob("enhanced_ai_detection_test_*.csv"))
            if not test_files:
                print(f"No test set AI detection files found! Need to run step 1 on test set first.")
                return None
            ai_file = max(test_files, key=lambda x: x.stat().st_mtime)
            print(f"Using latest test set AI detection file: {ai_file.name}")
        else:
            raise ValueError(f"dataset_type must be 'train' or 'test', got '{dataset_type}'")
        
        if not ai_file.exists():
            print(f"AI development detection file not found: {ai_file}")
            return None
        
        df = pd.read_csv(ai_file)
        
# No need for ai_confidence column
        
        print(f"Loaded {len(df)} AI-related job postings from {dataset_type} set")
        return df
    
    def create_content_hash(self, text):
        """Create hash for content-based deduplication"""
        if pd.isna(text):
            return ""
        
        # Normalize text for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def detect_ai_false_positives_robust(self, df):
        """UPDATED: Occurrence-level AI/KI/IA classification system"""
        print("Filtering AI false positives with occurrence-level classification...")
        
        # Load AI context terms from CSV
        ai_context_path = self.data_dir / 'ai_context.csv'
        if not ai_context_path.exists():
            raise FileNotFoundError(f"AI context file not found: {ai_context_path}")
        
        ai_context_df = pd.read_csv(ai_context_path)
        
        # Build context dictionaries by language
        ai_context_terms = {
            'en': set(),
            'fr': set(), 
            'it': set(),
            'de': set()  # Will use English terms for German
        }
        
        for _, row in ai_context_df.iterrows():
            # English terms (also used for German)
            if pd.notna(row['en_primary']):
                ai_context_terms['en'].add(row['en_primary'].lower())
                ai_context_terms['de'].add(row['en_primary'].lower())
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
            """Detect language using langdetect with fallback"""
            if pd.isna(text) or not text.strip():
                return 'unknown'
            
            try:
                # Clean text for better detection
                clean_text = re.sub(r'[^\w\s]', ' ', text)
                clean_text = re.sub(r'\s+', ' ', clean_text.strip())
                
                if len(clean_text.split()) < 10:  # Too short for reliable detection
                    return 'unknown'
                
                detected = detect(clean_text)
                return detected
            except:
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
                elif match_lower in ['ia']:
                    return 'false_positive'
            
            elif detected_lang in ['it', 'fr']:
                # Italian/French: ia is OK
                if match_lower == 'ia':
                    return 'legitimate'
                elif match_lower in ['ai']:
                    # Need to check context for ai/ki in Italian/French
                    if has_tech_context_in_window(content, match_pos, detected_lang):
                        return 'legitimate'
                    else:
                        return 'false_positive'
            
            else:
                # Unknown language: default to checking context
                if match_lower in ['ai', 'ki', 'ia']:
                    if has_tech_context_in_window(content, match_pos, 'en'):  # Use English terms as fallback
                        return 'legitimate'
                    else:
                        return 'false_positive'
            
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
            
            # Filter job only if ALL occurrences are false positives
            return legitimate_count == 0
        
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
                
                # Look for number pattern right before ML (allowing punctuation after number)
                if not re.search(r'\d\s*[-\s]*$', before_ml):
                    return False  # Found ML not following number - keep job
            
            # All ML occurrences follow numbers - filter job
            return True
        
        def should_filter_aa_job(row):
            """Filter AA jobs unless text is in Italian or French (where AA might be AI-related)"""
            keywords = str(row['matched_keywords']).strip().lower()
            if keywords != 'aa':
                return False  # Keep jobs with multiple keywords
            
            content = str(row['content_clean'])
            
            # Detect language
            detected_lang = detect_language_robust(content)
            
            # Keep AA jobs only if in Italian or French
            if detected_lang in ['it', 'fr']:
                return False  # Keep job - AA might be legitimate in IT/FR
            
            # Filter job if in other languages (AA likely random characters)
            return True
        
        
        # Apply occurrence-level filtering
        initial_count = len(df)
        
        # Filter AI false positives
        ai_false_positives_mask = df.apply(should_filter_ai_job, axis=1)
        
        # Filter ML false positives  
        ml_false_positives_mask = df.apply(should_filter_ml_job, axis=1)
        
        # Filter AA false positives
        aa_false_positives_mask = df.apply(should_filter_aa_job, axis=1)
        
        # Combine filters
        combined_false_positives_mask = ai_false_positives_mask | ml_false_positives_mask | aa_false_positives_mask
        
        # Count and categorize removed jobs
        removed_jobs = df[combined_false_positives_mask]
        df_filtered = df[~combined_false_positives_mask]
        
        # Analyze what was removed
        if len(removed_jobs) > 0:
            print(f"Analyzing {len(removed_jobs)} jobs marked for removal:")
            ai_removed = sum(ai_false_positives_mask)
            ml_removed = sum(ml_false_positives_mask)
            aa_removed = sum(aa_false_positives_mask)
            print(f"  AI false positives: {ai_removed}")
            print(f"  ML false positives: {ml_removed}")
            print(f"  AA false positives: {aa_removed}")
            
            removed_by_lang = {}
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
        
        # Save false positives for analysis
        if len(removed_jobs) > 0:
            fp_output_path = os.path.join(self.data_dir, 'false_positives_removed.csv')
            removed_jobs.to_csv(fp_output_path, index=False)
            print(f"📄 False positives exported to: {fp_output_path}")
        
        return df_filtered
    
    def deduplicate_exact_matches(self, df):
        """Remove exact duplicate job postings"""
        print("Removing exact duplicates...")
        
        # Create content hash for exact matching
        df['content_hash'] = df['content_clean'].apply(self.create_content_hash)
        
        # Remove exact duplicates
        initial_count = len(df)
        # Sort by tst_created to keep earliest posting when deduplicating
        df_sorted = df.sort_values('tst_created', ascending=True)
        df_dedup = df_sorted.drop_duplicates(subset=['content_hash'], keep='first')
        
        print(f"Removed {initial_count - len(df_dedup)} exact duplicates")
        print(f"Remaining: {len(df_dedup)} job postings")
        
        return df_dedup
    
    def deduplicate_similar_content(self, df, similarity_threshold=0.99):
        """Smart deduplication: exact matches first, then TF-IDF within company+title groups"""
        print(f"Smart deduplication process (threshold: {similarity_threshold})...")
        initial_count = len(df)
        
        # Step 1: Remove exact duplicates first (company + title + content_clean)
        print("Step 1: Removing exact duplicates (company + title + content_clean)...")
        # Sort by tst_created to keep earliest posting when deduplicating
        df_sorted = df.sort_values('tst_created', ascending=True)
        df_step1 = df_sorted.drop_duplicates(subset=['company_name', 'title', 'content_clean'], keep='first')
        exact_removed = initial_count - len(df_step1)
        print(f"Removed {exact_removed:,} exact duplicates")
        print(f"Remaining: {len(df_step1):,} jobs")
        
        if len(df_step1) == 0:
            return df_step1
        
        # Step 2: TF-IDF similarity within company+title groups only
        print("Step 2: TF-IDF similarity within company+title groups...")
        
        # Group by company + title
        df_step1['company_title_key'] = (df_step1['company_name'].fillna('') + '|||' + 
                                        df_step1['title'].fillna('')).str.lower().str.strip()
        
        groups = df_step1.groupby('company_title_key')
        print(f"Found {len(groups)} unique company+title combinations")
        
        # Only process groups with multiple entries
        groups_to_process = [(name, group) for name, group in groups if len(group) > 1]
        print(f"Processing {len(groups_to_process)} groups with multiple entries")
        
        total_removed = 0
        all_similar_pairs = []
        indices_to_remove = set()
        
        for group_name, group_df in tqdm(groups_to_process, desc="Processing groups"):
            if len(group_df) < 2:
                continue
            
            # Ensure group is sorted by tst_created (earliest first) to prioritize earliest job
            group_df = group_df.sort_values('tst_created', ascending=True)
                
            # Extract texts for this group
            group_texts = group_df['content_clean'].fillna('').tolist()
            group_indices = group_df.index.tolist()
            
            if len(set(group_texts)) < 2:  # All texts are identical (shouldn't happen after step 1)
                continue
            
            try:
                # Create TF-IDF vectors for this group
                vectorizer = TfidfVectorizer(
                    max_features=min(5000, len(group_texts) * 100),
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1  # Lower min_df for small groups
                )
                
                group_tfidf = vectorizer.fit_transform(group_texts)
                
                if group_tfidf.shape[1] == 0:  # No features extracted
                    continue
                
                # Calculate similarity matrix for this group
                group_similarity = cosine_similarity(group_tfidf)
                
                # Find similar pairs within this group
                for i in range(len(group_similarity)):
                    global_i = group_indices[i]
                    if global_i in indices_to_remove:
                        continue
                    
                    for j in range(i + 1, len(group_similarity)):
                        global_j = group_indices[j]
                        if global_j in indices_to_remove:
                            continue
                        
                        if group_similarity[i][j] >= similarity_threshold:
                            # Record the similar pair
                            all_similar_pairs.append({
                                'similarity_score': group_similarity[i][j],
                                'kept_uid': df_step1.loc[global_i]['uid'],
                                'kept_title': df_step1.loc[global_i]['title'],
                                'kept_company': df_step1.loc[global_i]['company_name'],
                                'removed_uid': df_step1.loc[global_j]['uid'],
                                'removed_title': df_step1.loc[global_j]['title'], 
                                'removed_company': df_step1.loc[global_j]['company_name']
                            })
                            
                            # Mark for removal (keep first, remove second)
                            indices_to_remove.add(global_j)
                            total_removed += 1
            
            except Exception as e:
                print(f"Warning: Skipping group {group_name[:50]}... due to error: {e}")
                continue
        
        # Step 3: Remove similar duplicates
        df_final = df_step1.drop(indices_to_remove)
        df_final = df_final.drop(columns=['company_title_key'])  # Clean up temp column
        
        print(f"Removed {total_removed:,} similar duplicates within groups")
        print(f"Final count: {len(df_final):,} jobs")
        print(f"Total removed: {initial_count - len(df_final):,} ({(initial_count - len(df_final))/initial_count*100:.1f}%)")
        
        # Print and export similar pairs (limit display for large datasets)
        if all_similar_pairs:
            max_display = min(20, len(all_similar_pairs))
            print(f"\n📋 Details of first {max_display}/{len(all_similar_pairs)} similar pairs removed:")
            print("="*80)
            for i, pair in enumerate(all_similar_pairs[:max_display], 1):
                print(f"\nPair {i} (Similarity: {pair['similarity_score']:.3f})")
                print(f"KEPT:    {pair['kept_title'][:60]}...")
                print(f"         Company: {pair['kept_company']}")
                print(f"REMOVED: {pair['removed_title'][:60]}...")
                print(f"         Company: {pair['removed_company']}")
                print("-" * 40)
            
            if len(all_similar_pairs) > max_display:
                print(f"\n... and {len(all_similar_pairs) - max_display} more pairs")
            
            # Export similar pairs
            self.export_similar_pairs(all_similar_pairs, df_step1)
        
        return df_final
    
    def export_similar_pairs(self, similar_pairs, original_df):
        """Export similar duplicate pairs to CSV for review"""
        pairs_data = []
        
        for pair in similar_pairs:
            # Get full content for kept job
            kept_row = original_df[original_df['uid'] == pair['kept_uid']].iloc[0]
            removed_row = original_df[original_df['uid'] == pair['removed_uid']].iloc[0]
            
            pairs_data.append({
                'similarity_score': pair['similarity_score'],
                'kept_uid': pair['kept_uid'],
                'kept_title': pair['kept_title'],
                'kept_company': pair['kept_company'],
                'kept_content': kept_row['content_clean'],
                'removed_uid': pair['removed_uid'],
                'removed_title': pair['removed_title'],
                'removed_company': pair['removed_company'],
                'removed_content': removed_row['content_clean']
            })
        
        # Create DataFrame and export
        pairs_df = pd.DataFrame(pairs_data)
        output_path = self.data_dir / f"similar_duplicates_removed.csv"
        pairs_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n📄 Similar duplicate pairs exported to: {output_path}")
        return output_path
    
    def detect_language_simple(self, text):
        """Simple language detection"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        # German indicators
        german_words = ['der', 'die', 'das', 'und', 'mit', 'für', 'von', 'ist', 'sind', 'haben']
        german_score = sum(1 for word in german_words if f' {word} ' in f' {text_lower} ')
        
        # French indicators
        french_words = ['le', 'la', 'les', 'et', 'de', 'avec', 'pour', 'est', 'sont', 'ont']
        french_score = sum(1 for word in french_words if f' {word} ' in f' {text_lower} ')
        
        # Italian indicators
        italian_words = ['il', 'la', 'le', 'gli', 'con', 'per', 'di', 'e', 'sono', 'hanno']
        italian_score = sum(1 for word in italian_words if f' {word} ' in f' {text_lower} ')
        
        # English indicators
        english_words = ['the', 'and', 'with', 'for', 'is', 'are', 'have', 'has', 'this', 'that']
        english_score = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
        
        scores = {'de': german_score, 'fr': french_score, 'it': italian_score, 'en': english_score}
        max_lang = max(scores.keys(), key=lambda k: scores[k])
        
        return max_lang if scores[max_lang] > 0 else 'unknown'
    
    def translate_text(self, text, target_lang='en'):
        """Translate text using OpenAI API"""
        if not text or pd.isna(text):
            return ""
        
        # Detect source language
        source_lang = self.detect_language_simple(text)
        
        # Skip if already English
        if source_lang == 'en':
            return text
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": f"Translate the following job posting text from {source_lang} to English. Maintain the original structure and meaning."},
                    {"role": "user", "content": text}
                ],
                max_tokens=2000,
                temperature=0
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original if translation fails
    
    def translate_batch(self, df, batch_size=50):
        """Translate job postings in batches"""
        print("Translating job postings to English...")
        
        # Add columns for translation
        df['original_language'] = df['content_clean'].apply(self.detect_language_simple)
        df['translated_text'] = ""
        
        # Process in batches
        non_english = df[df['original_language'] != 'en']
        print(f"Found {len(non_english)} non-English job postings to translate")
        
        for i in tqdm(range(0, len(non_english), batch_size), desc="Translating batches"):
            batch = non_english.iloc[i:i+batch_size]
            
            for idx, row in batch.iterrows():
                df.loc[idx, 'translated_text'] = self.translate_text(row['content_clean'])
                time.sleep(0.1)  # Rate limiting
        
        # For English jobs, use original text
        df.loc[df['original_language'] == 'en', 'translated_text'] = df.loc[df['original_language'] == 'en', 'content_clean']
        
        return df
    
    def export_results(self, df, dataset_type="train"):
        """Export deduplicated and translated results"""
        
        # Define columns for export - only include columns that exist
        base_columns = [
            'uid', 'title', 'company_name', 'company_size', 'location_raw',
            'content_clean', 'tst_created', 'url'
        ]
        
        optional_columns = [
            'detected_language', 'matching_sentences', 'matched_keywords', 
            'company_id', 'company_is_recruiter', 'tst_deleted', 
            'x28_industries', 'x28_occupations', 'cantons', 'duplicate_group'
        ]
        
        export_columns = []
        for col in base_columns + optional_columns:
            if col in df.columns:
                export_columns.append(col)
        
        # Export file with dataset type in filename
        output_path = self.data_dir / f"ai_development_deduplicated_{dataset_type}.csv"
        df[export_columns].to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n{'='*60}")
        print(f"EXPORT COMPLETE - {dataset_type.upper()} SET")
        print(f"{'='*60}")
        print(f"📄 Deduplicated AI jobs: {output_path}")
        print(f"   - Total jobs: {len(df)}")
        if 'matched_keywords' in df.columns:
            all_keywords = []
            for keywords in df['matched_keywords'].dropna():
                all_keywords.extend([k.strip() for k in str(keywords).split(',')])
            from collections import Counter
            top_keywords = Counter(all_keywords).most_common(5)
            print(f"   - Top 5 keywords: {dict(top_keywords)}")
        
        return output_path
    
    def run_deduplication_and_translation(self, dataset_type="train", custom_file=None, skip_translation=False):
        """Run complete deduplication process"""
        process_type = "DEDUPLICATION" if skip_translation else "DEDUPLICATION AND TRANSLATION"
        print("="*60)
        print(f"AI DEVELOPMENT DATA {process_type} - {dataset_type.upper()} SET")
        print("="*60)
        
        # Load data
        df = self.load_ai_development_data(dataset_type, custom_file)
        if df is None:
            return None
        
        # Deduplication first
        df_dedup1 = self.deduplicate_exact_matches(df)
        # Robust AI false positive filter (after deduplication)
        df_filtered = self.detect_ai_false_positives_robust(df_dedup1)
        df_dedup2 = self.deduplicate_similar_content(df_filtered)
        
        
        # Translation (only if not skipped)
        if not skip_translation:
            df_final = self.translate_batch(df_dedup2)
        else:
            df_final = df_dedup2
            print("⏭️ Skipping translation as requested")
        
        # Export results
        output_path = self.export_results(df_final, dataset_type)
        
        action = "Deduplication" if skip_translation else "Deduplication and translation"
        print(f"\n🎯 {action} complete for {dataset_type} set!")
        print(f"Results saved to: {output_path}")
        
        return output_path
    
    def run_both_datasets(self):
        """Run deduplication and translation on both training and test sets"""
        print("🚀 Starting deduplication and translation for both train and test sets...")
        
        train_path = self.run_deduplication_and_translation("train")
        test_path = self.run_deduplication_and_translation("test")
        
        print(f"\n{'='*60}")
        print("BOTH DATASETS PROCESSED")
        print(f"{'='*60}")
        print(f"✅ Training set: {train_path}")
        print(f"✅ Test set: {test_path}")
        
        return train_path, test_path

def create_complete_company_test_set(input_filename, target_jobs=1000, min_companies=100, random_seed=42):
    """Create test set with complete company coverage by randomly selecting companies"""
    
    input_file = f"Data/{input_filename}"
    
    print("🎯 Creating complete company coverage test set")
    print("=" * 50)
    
    # Load the dataset
    print(f"📖 Loading: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df):,} jobs")
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_file}")
        return
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return
    
    # Get company job counts
    company_counts = df['company_name'].value_counts()
    print(f"Total companies: {len(company_counts)}")
    print(f"\\nCompany size distribution:")
    print(f"  - Companies with 1 job: {(company_counts == 1).sum()}")
    print(f"  - Companies with 2-5 jobs: {((company_counts >= 2) & (company_counts <= 5)).sum()}")
    print(f"  - Companies with 6-10 jobs: {((company_counts >= 6) & (company_counts <= 10)).sum()}")
    print(f"  - Companies with >10 jobs: {(company_counts > 10).sum()}")
    
    # Set random seed for reproducible selection
    np.random.seed(random_seed)
    
    # Strategy: Start with smaller companies to maximize company diversity
    companies_by_size = company_counts.sort_values()  # Smallest first
    
    selected_companies = []
    total_jobs = 0
    
    print(f"\\n🎯 Selecting companies (target: ~{target_jobs} jobs, min: {min_companies} companies):")
    print("=" * 80)
    
    # First pass: Add companies with 1-3 jobs to maximize diversity
    small_companies = companies_by_size[companies_by_size <= 3]
    small_company_list = small_companies.index.values.copy()
    np.random.shuffle(small_company_list)
    
    for company in small_company_list:
        job_count = companies_by_size[company]
        if total_jobs + job_count <= target_jobs:
            selected_companies.append(company)
            total_jobs += job_count
            
            if len(selected_companies) % 50 == 0:
                print(f"  Selected {len(selected_companies)} companies, {total_jobs} jobs")
            
            if total_jobs >= target_jobs and len(selected_companies) >= min_companies:
                break
    
    print(f"After small companies: {len(selected_companies)} companies, {total_jobs} jobs")
    
    # Second pass: Add medium companies if needed
    if total_jobs < target_jobs or len(selected_companies) < min_companies:
        medium_companies = companies_by_size[(companies_by_size > 3) & (companies_by_size <= 10)]
        medium_company_list = medium_companies.index.values.copy()
        np.random.shuffle(medium_company_list)
        
        for company in medium_company_list:
            if company in selected_companies:
                continue
            job_count = companies_by_size[company]
            if total_jobs + job_count <= target_jobs + 50:  # Allow slight overage
                selected_companies.append(company)
                total_jobs += job_count
                
                if total_jobs >= target_jobs and len(selected_companies) >= min_companies:
                    break
    
    print(f"After medium companies: {len(selected_companies)} companies, {total_jobs} jobs")
    
    # Third pass: Add larger companies if still significantly under target
    if total_jobs < target_jobs * 0.9:
        large_companies = companies_by_size[companies_by_size > 10]
        large_company_list = large_companies.index.values.copy()
        np.random.shuffle(large_company_list)
        
        for company in large_company_list:
            if company in selected_companies:
                continue
            job_count = companies_by_size[company]
            if total_jobs + job_count <= target_jobs + 100:  # More flexible for larger companies
                selected_companies.append(company)
                total_jobs += job_count
                
                if total_jobs >= target_jobs:
                    break
    
    print(f"\\n✅ Final selection: {len(selected_companies)} companies, {total_jobs} jobs")
    
    # Create the test set with ALL jobs from selected companies
    test_df = df[df['company_name'].isin(selected_companies)].copy()
    
    # Verify we got all jobs
    print(f"✓ Verification: {len(test_df)} jobs extracted")
    
    # Shuffle the final test set
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Create output filename
    base_name = input_filename.replace('.csv', '')
    output_file = f"Data/{base_name}_test_complete_companies_{len(test_df)}_jobs_{len(selected_companies)}_companies.csv"
    
    # Save test set
    test_df.to_csv(output_file, index=False)
    
    print(f"\\n📄 Test set saved to: {output_file}")
    print(f"\\n📊 Test set statistics:")
    print(f"  - Total jobs: {len(test_df):,}")
    print(f"  - Companies: {len(selected_companies):,}")
    print(f"  - Complete company coverage: ✓")
    
    # Show company size distribution in test set
    test_company_counts = test_df['company_name'].value_counts()
    print(f"\\n🏢 Company size distribution in test set:")
    print(f"  - Companies with 1 job: {(test_company_counts == 1).sum()}")
    print(f"  - Companies with 2-5 jobs: {((test_company_counts >= 2) & (test_company_counts <= 5)).sum()}")
    print(f"  - Companies with 6-10 jobs: {((test_company_counts >= 6) & (test_company_counts <= 10)).sum()}")
    print(f"  - Companies with >10 jobs: {(test_company_counts > 10).sum()}")
    
    # Show top 10 companies by job count in test set
    print(f"\\n🔝 Top 10 companies in test set:")
    for i, (company, count) in enumerate(test_company_counts.head(10).items(), 1):
        print(f"  {i:2d}. {company}: {count} jobs")
    
    # Show keyword statistics
    if 'matched_keywords' in test_df.columns:
        all_keywords = []
        for keywords in test_df['matched_keywords'].dropna():
            all_keywords.extend([k.strip().lower() for k in str(keywords).split(',')])
        
        from collections import Counter
        top_keywords = Counter(all_keywords).most_common(10)
        print(f"\\n🔍 Top 10 AI keywords in test set:")
        for i, (keyword, count) in enumerate(top_keywords, 1):
            print(f"  {i:2d}. {keyword}: {count}")
    
    return output_file

def simple_content_dedupe(filename):
    """Simple deduplication with Italian false positive filtering"""
    data_dir = Path(__file__).parent / "Data"
    input_file = data_dir / filename
    
    print("="*60)
    print(f"SIMPLE CONTENT DEDUPLICATION WITH ITALIAN FILTERING")
    print("="*60)
    
    # Load data
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Initial count: {initial_count:,} jobs")
    
    # Apply Italian false positive filter
    processor = DeduplicateAndTranslate()
    df_filtered = processor.detect_italian_french_and_filter_ai_false_positives(df)
    
    # Remove exact duplicates based on company + title + content_clean
    print("Removing exact duplicates (company + title + content_clean)...")
    # Sort by tst_created to keep earliest posting when deduplicating
    df_sorted = df_filtered.sort_values('tst_created', ascending=True)
    df_dedup = df_sorted.drop_duplicates(subset=['company_name', 'title', 'content_clean'], keep='first')
    
    final_count = len(df_dedup)
    removed_count = initial_count - final_count
    
    print(f"Total removed: {removed_count:,} jobs ({removed_count/initial_count*100:.1f}%)")
    print(f"Final remaining: {final_count:,} jobs")
    
    # Export results
    base_name = filename.replace('.csv', '')
    output_file = data_dir / f"{base_name}_simple_dedup_filtered.csv"
    
    df_dedup.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ Simple deduplication with Italian filtering complete!")
    print(f"📄 Output: {output_file}")
    
    return output_file

def run_custom_file(filename, skip_translation=False):
    """Run deduplication and translation on a custom file"""
    processor = DeduplicateAndTranslate()
    output_path = processor.run_deduplication_and_translation("custom", filename, skip_translation)
    
    if output_path:
        print(f"\n📋 Next Steps:")
        if not skip_translation:
            print(f"1. Review translated content for accuracy")
        else:
            print(f"1. Review deduplicated content")
        print(f"2. Validate AI-related classifications")
        print(f"3. Use deduplicated data for further analysis")

def extract_training_companies_from_deduplicated(deduplicated_file, training_companies=None):
    """Extract jobs from specific training companies from deduplicated dataset"""
    
    # Default training companies list from stage 0
    if training_companies is None:
        training_companies = [
            "A. Köhler AG", "Akeret Baumanagement AG", "Aktiengesellschaft Cilander", 
            "Aletsch Bahnen AG", "Alters- und Gesundheitszentrum (AGZ)", "Appenzeller Verlag AG",
            "Arthouse Commercio Movie AG", "Assura Holding SA", "Au Bonheur des Animaux Myriam Lienhard",
            "Berner Bildungszentrum Pflege AG", "Bison Schweiz AG", "Bourgeois Avocats SA",
            "Bourquin SA", "Bundesamt für Zivilluftfahrt (BAZL)", "Bäckerei Konditorei Zeller AG",
            "CHEMODEX AG", "CKW AG", "Cecchettin SA", "Commune d'Arzier-Le Muids",
            "Confiserie Eichenberger AG", "Cybersystems GmbH", "Eidgenössisches Justiz- und Polizeidepartement (EJPD)",
            "Energie Wasser Bern", "FNX Sàrl", "Franke Industrie AG", "Fun Planet Loisirs Brig-Glis AG",
            "Gate Gourmet Switzerland GmbH", "Gebr. Marthaler AG", "Gebro Pharma AG", "Gemeinde Grabs",
            "Gemeinde Matten", "Gemeinde Zollikon (ZH)", "Genossenschaft Lindenmühle", "Gloor Metallbau GmbH",
            "Grünig-Interscreen AG", "Hammer Metall AG", "Häfliger und Partner AG", "IM Architektur AG",
            "ISP Electro Solutions AG", "Imoberdorf AG", "Innovation Process Technology AG", "Ivoclar Vivadent AG",
            "Jaisli-Xamax AG", "John Schwab S.A.", "KV Zürich Business School", "KiK Kultur im Kammgarn",
            "Kita Kiddi 2", "Knuchel Farben AG", "Küng Rechtsanwälte & Notare AG", "Lichtensteiger AG Bäckerei",
            "Manufacture Jaeger-LeCoultre, Branch of Richemont International SA", "Metzgerei Kast GmbH",
            "Mivelaz Bois SA", "Mändli Handels- und Montage AG", "Narcocare AG", "North Thin Ply Technology Sàrl",
            "Obergericht des Kantons Zürich", "Optiprint AG", "PPCmetrics AG", "Perlen Packaging AG, Perlen",
            "Police Nyon Région", "Qualipet AG", "R.I.C. Risk & Insurance Consulting AG", "Region Oberaargau",
            "Restaurant Bauernhof AG", "Resto-Lounge Sàrl", "Rudolf Wirz Strassen- und Tiefbau AG",
            "SCHMID WETLI AG", "SIB Schweiz. Institut für Betriebsökonomie AG", "STB Engineering AG",
            "Schneeberger Décolletages S.A.", "Schweizerische Agentur für Innovationsförderung (Innosuisse)",
            "Schweizerisches Rotes Kreuz Kanton Schaffhausen", "Sedelec SA Lausanne", "Selmoni Ingenieur AG",
            "Senevita Mülibach AG", "Seniorenzentrum Zwyden", "Sepp Knüsel AG", "SmartLiberty SA",
            "Société coopérative Générations", "Spirig HealthCare AG", 
            "Spitex für Stadt und Land AG, Zweigniederlassung Schwyz", "Spühler Partner Architekten AG",
            "Stiftung AR SUNNSYTE Wohnen begleiten pflegen", "Stiftung Besuchsdienst Innerschweiz BDI",
            "Tamedia Espace AG", "Th. Willy AG Auto-Zentrum", "Triag AG", "Trisa Accessoires AG",
            "Union des Associations Européennes de Football (UEFA)", "Verein Spitex Heitersberg",
            "Verein pflegimuri", "Walter Lüthi Holzbau AG", "Windredli GmbH", "Zürcher Kunstgesellschaft",
            "shelterschweiz"
        ]
    
    data_dir = Path(__file__).parent / "Data"
    input_file = data_dir / deduplicated_file
    
    print("=" * 60)
    print("EXTRACTING TRAINING COMPANIES FROM DEDUPLICATED DATA")
    print("=" * 60)
    print(f"Input file: {deduplicated_file}")
    print(f"Training companies: {len(training_companies)}")
    
    # Load deduplicated dataset
    df = pd.read_csv(input_file)
    initial_count = len(df)
    print(f"Total jobs in deduplicated dataset: {initial_count:,}")
    
    # Show all companies in the dataset
    all_companies = set(df['company_name'].unique())
    print(f"Total companies in dataset: {len(all_companies):,}")
    
    # Find which training companies are present
    present_training_companies = [c for c in training_companies if c in all_companies]
    missing_training_companies = [c for c in training_companies if c not in all_companies]
    
    print(f"\n📊 Training companies analysis:")
    print(f"Present in dataset: {len(present_training_companies)}")
    print(f"Missing from dataset: {len(missing_training_companies)}")
    
    if missing_training_companies:
        print(f"\n❌ Missing training companies:")
        for company in missing_training_companies[:10]:  # Show first 10
            print(f"  - {company}")
        if len(missing_training_companies) > 10:
            print(f"  ... and {len(missing_training_companies) - 10} more")
    
    # Extract jobs from present training companies
    training_jobs = df[df['company_name'].isin(present_training_companies)].copy()
    
    print(f"\n✅ Extracted jobs from training companies:")
    print(f"Total training jobs: {len(training_jobs):,}")
    
    # Show breakdown by company
    company_counts = training_jobs['company_name'].value_counts()
    print(f"\nTop 10 companies by job count:")
    for company, count in company_counts.head(10).items():
        print(f"  {company}: {count:,} jobs")
    
    # Flag potential false positives for review
    potential_false_positives = []
    for company in present_training_companies:
        company_jobs = training_jobs[training_jobs['company_name'] == company]
        if len(company_jobs) > 0:
            # Check for companies with only "ai" matches (potential false positives)
            ai_only_jobs = 0
            for _, job in company_jobs.iterrows():
                keywords = str(job.get('matched_keywords', '')).lower().strip()
                if keywords == 'ai':
                    ai_only_jobs += 1
            
            false_positive_rate = ai_only_jobs / len(company_jobs)
            if false_positive_rate > 0.5:  # More than 50% are "ai" only
                potential_false_positives.append((company, ai_only_jobs, len(company_jobs), false_positive_rate))
    
    if potential_false_positives:
        print(f"\n⚠️  Potential false positives detected:")
        for company, ai_only, total, rate in potential_false_positives:
            print(f"  {company}: {ai_only}/{total} jobs ({rate:.1%}) are 'ai' keyword only")
    
    # Save training set
    output_file = data_dir / f"ai_development_training_companies_{len(training_jobs)}_jobs.csv"
    training_jobs.to_csv(output_file, index=False)
    
    print(f"\n📄 Training set saved to: {output_file.name}")
    print(f"Ready for Stage 3 AI task extraction!")
    
    return output_file

def main():
    """Main function - process training set by default"""
    processor = DeduplicateAndTranslate()
    output_path = processor.run_deduplication_and_translation("train")
    
    if output_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Review translated content for accuracy")
        print(f"2. Validate AI-related classifications")
        print(f"3. Use translated text for further analysis")

def run_train_only():
    """Run deduplication and translation on training set only"""
    processor = DeduplicateAndTranslate()
    output_path = processor.run_deduplication_and_translation("train")
    
    if output_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Review translated content for accuracy")
        print(f"2. Run on test set for validation")
        print(f"3. Compare train vs test characteristics")

def run_test_only():
    """Run deduplication and translation on test set only"""
    processor = DeduplicateAndTranslate()
    output_path = processor.run_deduplication_and_translation("test")
    
    if output_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Review translated content for accuracy") 
        print(f"2. Compare with training set results")
        print(f"3. Proceed with test set analysis")

def run_both_sets():
    """Run deduplication and translation on both training and test sets"""
    processor = DeduplicateAndTranslate()
    train_path, test_path = processor.run_both_datasets()
    
    if train_path and test_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Compare deduplication rates between sets")
        print(f"2. Analyze language distribution differences")
        print(f"3. Proceed with AI task extraction on both sets")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--train":
            run_train_only()
        elif sys.argv[1] == "--test":
            run_test_only()
        elif sys.argv[1] == "--both":
            run_both_sets()
        elif sys.argv[1] == "--custom" and len(sys.argv) > 2:
            run_custom_file(sys.argv[2])
        elif sys.argv[1] == "--dedup-only" and len(sys.argv) > 2:
            run_custom_file(sys.argv[2], skip_translation=True)
        elif sys.argv[1] == "--simple-dedup" and len(sys.argv) > 2:
            simple_content_dedupe(sys.argv[2])
        elif sys.argv[1] == "--create-test-set" and len(sys.argv) > 2:
            if len(sys.argv) == 3:
                create_complete_company_test_set(sys.argv[2])  # Use defaults
            elif len(sys.argv) == 5:
                create_complete_company_test_set(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
            else:
                print("Usage for test set creation:")
                print("  --create-test-set <file>                              # Use defaults (1000 jobs, 100 companies)")
                print("  --create-test-set <file> <target_jobs> <min_companies> # Custom parameters")
        elif sys.argv[1] == "--extract-training-companies" and len(sys.argv) > 2:
            extract_training_companies_from_deduplicated(sys.argv[2])
        else:
            print("Usage:")
            print("  python3 stage_2_deduplicate_translate.py --train                           # Process training set only")
            print("  python3 stage_2_deduplicate_translate.py --test                            # Process test set only") 
            print("  python3 stage_2_deduplicate_translate.py --both                            # Process both sets")
            print("  python3 stage_2_deduplicate_translate.py --custom <file>                   # Process custom file (dedup + translate)")
            print("  python3 stage_2_deduplicate_translate.py --dedup-only <file>               # Process custom file (dedup only)")
            print("  python3 stage_2_deduplicate_translate.py --simple-dedup <file>             # Simple exact content dedup only")
            print("  python3 stage_2_deduplicate_translate.py --create-test-set <file>          # Create test set (default: ~1000 jobs, 100+ companies)")
            print("  python3 stage_2_deduplicate_translate.py --create-test-set <file> 500 50   # Create custom test set")
            print("  python3 stage_2_deduplicate_translate.py --extract-training-companies <file> # Extract training companies from deduplicated dataset")
    else:
        # Default behavior - run training set only
        run_train_only()