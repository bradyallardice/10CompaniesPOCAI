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

class DeduplicateAndTranslate:
    """Deduplicate and translate AI development job postings"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "Data"
        
        # Load environment variables
        load_dotenv('config.env')
        
        # Set up OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def load_ai_development_data(self, dataset_type="train"):
        """Load AI development detection results for specified dataset"""
        if dataset_type == "train":
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
        print(f"Loaded {len(df)} AI-related job postings from {dataset_type} set")
        return df
    
    def create_content_hash(self, text):
        """Create hash for content-based deduplication"""
        if pd.isna(text):
            return ""
        
        # Normalize text for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def deduplicate_exact_matches(self, df):
        """Remove exact duplicate job postings"""
        print("Removing exact duplicates...")
        
        # Create content hash for exact matching
        df['content_hash'] = df['content_clean'].apply(self.create_content_hash)
        
        # Remove exact duplicates
        initial_count = len(df)
        df_dedup = df.drop_duplicates(subset=['content_hash'], keep='first')
        
        print(f"Removed {initial_count - len(df_dedup)} exact duplicates")
        print(f"Remaining: {len(df_dedup)} job postings")
        
        return df_dedup
    
    def deduplicate_similar_content(self, df, similarity_threshold=0.99):
        """Remove similar job postings using TF-IDF similarity"""
        print(f"Removing similar content (threshold: {similarity_threshold})...")
        
        # Extract text for similarity calculation
        texts = df['content_clean'].fillna('').tolist()
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find similar pairs
        to_remove = set()
        similar_pairs = []
        for i in range(len(similarity_matrix)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(similarity_matrix)):
                if j in to_remove:
                    continue
                if similarity_matrix[i][j] >= similarity_threshold:
                    # Record the similar pair
                    kept_idx = i if df.iloc[i]['ai_confidence'] >= df.iloc[j]['ai_confidence'] else j
                    removed_idx = j if df.iloc[i]['ai_confidence'] >= df.iloc[j]['ai_confidence'] else i
                    
                    similar_pairs.append({
                        'similarity_score': similarity_matrix[i][j],
                        'kept_uid': df.iloc[kept_idx]['uid'],
                        'kept_title': df.iloc[kept_idx]['title'],
                        'kept_company': df.iloc[kept_idx]['company_name'],
                        'kept_confidence': df.iloc[kept_idx]['ai_confidence'],
                        'removed_uid': df.iloc[removed_idx]['uid'],
                        'removed_title': df.iloc[removed_idx]['title'], 
                        'removed_company': df.iloc[removed_idx]['company_name'],
                        'removed_confidence': df.iloc[removed_idx]['ai_confidence']
                    })
                    
                    # Keep the one with higher AI confidence
                    if df.iloc[i]['ai_confidence'] >= df.iloc[j]['ai_confidence']:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        # Remove similar duplicates
        df_dedup = df.drop(df.index[list(to_remove)])
        
        print(f"Removed {len(to_remove)} similar duplicates")
        print(f"Remaining: {len(df_dedup)} job postings")
        
        # Print details of similar pairs
        if similar_pairs:
            print(f"\n📋 Details of {len(similar_pairs)} similar duplicate pairs removed:")
            print("="*80)
            for i, pair in enumerate(similar_pairs, 1):
                print(f"\nPair {i} (Similarity: {pair['similarity_score']:.3f})")
                print(f"KEPT:    {pair['kept_title'][:60]}...")
                print(f"         Company: {pair['kept_company']}, Confidence: {pair['kept_confidence']}")
                print(f"         UID: {pair['kept_uid']}")
                print(f"REMOVED: {pair['removed_title'][:60]}...")
                print(f"         Company: {pair['removed_company']}, Confidence: {pair['removed_confidence']}")
                print(f"         UID: {pair['removed_uid']}")
                print("-" * 80)
            
            # Export similar pairs to CSV
            self.export_similar_pairs(similar_pairs, df)
        
        return df_dedup
    
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
                'kept_confidence': pair['kept_confidence'],
                'kept_content': kept_row['content_clean'],
                'removed_uid': pair['removed_uid'],
                'removed_title': pair['removed_title'],
                'removed_company': pair['removed_company'],
                'removed_confidence': pair['removed_confidence'],
                'removed_content': removed_row['content_clean']
            })
        
        # Create DataFrame and export
        pairs_df = pd.DataFrame(pairs_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"similar_duplicates_removed_{timestamp}.csv"
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define columns for export
        export_columns = [
            'uid',
            'title',
            'company_name',
            'company_size',
            'location_raw',
            'detected_language',
            'original_language',
            'content_clean',  # This is the raw text in original language
            'translated_text',  # This is the English translation
            'matching_sentences',
            'is_ai_related',
            'ai_confidence',
            'ai_keywords_str',
            'review_priority',
            'tst_created',
            'url'
        ]
        
        # Export file with dataset type in filename
        output_path = self.data_dir / f"ai_development_deduplicated_translated_{dataset_type}_{timestamp}.csv"
        df[export_columns].to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n{'='*60}")
        print(f"EXPORT COMPLETE - {dataset_type.upper()} SET")
        print(f"{'='*60}")
        print(f"📄 Deduplicated and translated AI jobs: {output_path}")
        print(f"   - Total jobs: {len(df)}")
        print(f"   - Language breakdown: {df['original_language'].value_counts().to_dict()}")
        print(f"   - Priority breakdown: {df['review_priority'].value_counts().to_dict()}")
        
        return output_path
    
    def run_deduplication_and_translation(self, dataset_type="train"):
        """Run complete deduplication and translation process"""
        print("="*60)
        print(f"AI DEVELOPMENT DATA DEDUPLICATION AND TRANSLATION - {dataset_type.upper()} SET")
        print("="*60)
        
        # Load data
        df = self.load_ai_development_data(dataset_type)
        if df is None:
            return None
        
        # Deduplication
        df_dedup1 = self.deduplicate_exact_matches(df)
        df_dedup2 = self.deduplicate_similar_content(df_dedup1)
        
        # Translation
        df_translated = self.translate_batch(df_dedup2)
        
        # Export results
        output_path = self.export_results(df_translated, dataset_type)
        
        print(f"\n🎯 Deduplication and translation complete for {dataset_type} set!")
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
        else:
            print("Usage:")
            print("  python3 step_2_deduplicate_translate.py --train  # Process training set only")
            print("  python3 step_2_deduplicate_translate.py --test   # Process test set only") 
            print("  python3 step_2_deduplicate_translate.py --both   # Process both sets")
    else:
        # Default behavior - run training set only
        run_train_only()