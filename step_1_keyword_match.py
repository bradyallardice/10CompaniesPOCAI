#!/usr/bin/env python3
"""
Enhanced AI Development Detection Export
Improved version with comprehensive text preprocessing and two-pass matching
Handles spacing issues, accents, and compound words for Swiss job postings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import unicodedata
import psycopg2
import os
import html
from dotenv import load_dotenv
from typing import List, Dict, Set, Tuple

class EnhancedAIDetection:
    """Enhanced AI development keyword detection with comprehensive preprocessing"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "Data"
        
        # Initialize keyword structures
        self._setup_keyword_structures()
        self._compile_patterns()
        
    def _setup_keyword_structures(self):
        """Setup restructured keyword lists for two-pass matching"""
        
        ai_development_keywords = [
        # ------------------------------------------------------------------
        # 0. UNIVERSAL ABBREVIATIONS & PROPER NAMES
        # ------------------------------------------------------------------
        "ai", "ml", "dl", "rl", "nlp", "nlu", "llm", "mlops",
        # libraries & clouds
        "pytorch", "tensorflow", "keras", "mxnet", "theano", "torch",
        "torch7", "caffe", "caffe2", "cntk", "deeplearning4j", "lasagne", "chainer",
        "scikit-learn", "sklearn", "weka", "mahout", "spark mllib", "h2o",
        "sagemaker", "vertex ai", "amazon machine learning", "google cloud ml",
        "azure ml", "ibm watson", "huggingface", "hugging face",
        # big-data / infra
        "hadoop", "spark", "mapreduce", "kafka", "storm", "flink",
        "mlflow", "kubeflow",
        "onnx", "onnxruntime",
        "gpu", "cuda", "cudnn", "opencl", "fpga",

        # ------------------------------------------------------------------
        # 1. ENGLISH CORE CONCEPTS, MODELS & ROLES
        # ------------------------------------------------------------------
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "neural networks", "convolutional neural network",
        "large language model", "language model",
        "computer vision", "natural language processing",
        "generative ai", "transformer", "transformers",
        "gpt", "bert", "clip",
        "cnn", "rnn", "lstm", "gru",
        "gan", "generative adversarial network", "vae", "autoencoder",
        "diffusion model", "stable diffusion", "Next Best Action",
        # roles & titles
        "ai engineer", "ai developer", "ai programmer",
        "machine learning engineer", "ml engineer", "deep learning engineer",
        "data scientist", "ml scientist", "research scientist", "applied scientist",
        "data mining engineer", "predictive modeler", "statistical modeler",
        "computer vision engineer", "nlp engineer", "mlops engineer",
        "model ops engineer", "cognitive computing engineer", "watson engineer",

        # ------------------------------------------------------------------
        # 2. LEARNING PARADIGMS
        # ------------------------------------------------------------------
        "supervised learning", "unsupervised learning", "semi-supervised learning",
        "self-supervised learning", "reinforcement learning",
        "few-shot learning", "zero-shot learning",
        "transfer learning", "meta learning",
        "active learning", "online learning",

        # ------------------------------------------------------------------
        # 3. CLASSICAL ML ALGORITHMS
        #  (English first – then German / IT / FR forms that recruiters use)
        # ------------------------------------------------------------------
        "logistic regression", "logistische regression", "regressione logistica",
        "régression logistique",
        "linear regression", "lineare regression", "regressione lineare",
        "régression linéaire",
        "decision tree", "entscheidungsbaum", "albero decisionale",
        "arbre de décision",
        "random forest", "zufallswald", "forêt aléatoire",
        "gradient boosting",
        "gboost", "xgboost", "lightgbm", "catboost", "adaboost", "bagging",
        "support vector machine", "svm", "machine à vecteurs de support",
        "naive bayes", "naiver bayes", "naïve bayes",
        "knn", "k-nearest neighbors", "k-nächste nachbarn",
        "k-means", "clustering",
        "principal component analysis", "pca",
        "hauptkomponentenanalyse", "analisi delle componenti principali",
        "analyse en composantes principales",
        "hidden markov model", "hmm",
        "verstecktes markov-modell", "modello di markov nascosto",
        "modèle de markov caché",
        "latent dirichlet allocation", "lda",
        "genetic algorithm", 
        "genetischer algorithmus", "algoritmo genetico", "algorithme génétique",

        # ------------------------------------------------------------------
        # 4. EARLY DEEP-LEARNING BUZZ (2010-2016)
        # ------------------------------------------------------------------
        "deep belief network", "dbn",
        "restricted boltzmann machine", "rbm",
        "self-organizing map", "som", "selbstorganisierende karte",
        "mappa auto-organizzante", "carte auto-organisatrice",

        # ------------------------------------------------------------------
        # 5. DEPLOYMENT / MLOps KEYWORDS
        # ------------------------------------------------------------------
        "model deployment", "modellbereitstellung",
        "deploy del modello", "déploiement de modèle",
        "model serving", "serving del modello", "serving de modèle",
        "model monitoring", "modellüberwachung",
        "monitoraggio del modello", "surveillance du modèle",

        # ------------------------------------------------------------------
        # 6. DOMAIN-SPECIFIC TASKS
        # ------------------------------------------------------------------
        "speech recognition", "spracherkennung",
        "riconoscimento vocale", "reconnaissance vocale",
        "asr",
        "audio processing", "audioverarbeitung",
        "elaborazione audio", "traitement audio",
        "text mining", "fouille de texte",
        "sentiment analysis", "sentiment-analyse",
        "analisi del sentiment", "analyse de sentiment",
        "information extraction", "informationsextraktion",
        "estrazione di informazioni", "extraction d'information",
        "recommendation system", "recommender",
        "empfehlungssystem", "sistema di raccomandazione",
        "système de recommandation",
        "predictive analytics", "prädiktive analytik",
        "analisi predittiva", "analytique prédictive",
        "object detection", "objekterkennung",
        "rilevamento oggetti", "détection d'objets",
        "image segmentation", "bildsegmentierung",
        "segmentazione delle immagini", "segmentation d'image",
        "hyperparameter tuning", "hyperparameteroptimierung",
        "ottimizzazione degli iperparametri",
        "optimisation des hyperparamètres",

        # ------------------------------------------------------------------
        # 7. RESPONSIBLE & TRUSTWORTHY AI 
        # ------------------------------------------------------------------
        "explainable ai", "xai", "interpretability",
        "responsible ai", "ethical ai", "fairness", "model governance",
        "erklärbare ki", "erklärbare künstliche intelligenz",
        "verantwortungsvolle ki", "ki-ethik",
        "intelligenza artificiale spiegabile", "ia spiegabile",
        "ia responsabile", "etica ia", "governance dei modelli",
        "intelligence artificielle explicable", "ia explicable",
        "ia responsable", "éthique de l'ia", "gouvernance des modèles",

        # ------------------------------------------------------------------
        # 8. LEGACY / MARKETING BUZZWORDS
        # ------------------------------------------------------------------
        "cognitive computing", "kognitives computing",
        "computing cognitivo", "informatique cognitive",
        "expert system", "expertensystem", "sistema esperto", "système expert",
        "knowledge engineering", "wissensengineering",
        "ingegneria della conoscenza", "ingénierie des connaissances",
        "predictive modeling", "prädiktive modellierung",
        "modellazione predittiva", "modélisation prédictive",
        "pattern recognition", "mustererkennung",
        "riconoscimento di pattern", "reconnaissance de formes"
    ]

        
        # Separate keywords by length for different matching strategies
        self.short_keywords = []
        self.dotted_abbreviations = []
        self.long_keywords_spaced = []
        
        for keyword in ai_development_keywords:
            if '.' in keyword and len(keyword) <= 4:
                self.dotted_abbreviations.append(keyword)
            elif len(keyword) <= 5:
                self.short_keywords.append(keyword)
            else:
                self.long_keywords_spaced.append(keyword)
        
        # Generate space-removed variants for long keywords
        self.long_keywords_no_spaces = []
        for keyword in self.long_keywords_spaced:
            # Create multiple variants
            variants = self._create_keyword_variants(keyword)
            self.long_keywords_no_spaces.extend(variants)
        
        # Remove duplicates
        self.long_keywords_no_spaces = list(set(self.long_keywords_no_spaces))
        
        # Italian exclusions
        self.italian_excluded_terms = ['ai']
        
        print(f"Keywords initialized:")
        print(f"  - Short keywords: {len(self.short_keywords)}")
        print(f"  - Dotted abbreviations: {len(self.dotted_abbreviations)}")
        print(f"  - Long spaced keywords: {len(self.long_keywords_spaced)}")
        print(f"  - Long no-space variants: {len(self.long_keywords_no_spaces)}")
    
    def _create_keyword_variants(self, keyword: str) -> List[str]:
        """Create multiple variants of a keyword for better matching"""
        variants = []
        
        # Original keyword
        variants.append(keyword)
        
        # Space-removed version
        no_spaces = re.sub(r'[\s\-]', '', keyword)
        variants.append(no_spaces)
        
        # Accent-normalized versions
        normalized = self._normalize_accents(keyword)
        if normalized != keyword:
            variants.append(normalized)
            variants.append(re.sub(r'[\s\-]', '', normalized))
        
        # Swiss German substitutions
        swiss_variant = self._apply_swiss_substitutions(keyword)
        if swiss_variant != keyword:
            variants.append(swiss_variant)
            variants.append(re.sub(r'[\s\-]', '', swiss_variant))
        
        # Hyphen variations
        if ' ' in keyword:
            variants.append(keyword.replace(' ', '-'))
            variants.append(keyword.replace(' ', '/'))
        
        return list(set(variants))
    
    def _normalize_accents(self, text: str) -> str:
        """Remove accents and diacritics from text"""
        if not text:
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFD', text)
        # Remove accent marks
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        return text
    
    def _apply_swiss_substitutions(self, text: str) -> str:
        """Apply Swiss German character substitutions"""
        substitutions = {
            'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'à': 'a', 'â': 'a', 'ç': 'c', 'î': 'i',
            'ô': 'o', 'û': 'u', 'ù': 'u', 'ñ': 'n'
        }
        
        result = text
        for accented, replacement in substitutions.items():
            result = result.replace(accented, replacement)
        
        return result
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance"""
        
        # Short keywords pattern (boundary-aware)
        short_pattern_parts = []
        for keyword in self.short_keywords:
            if keyword != 'ai':  # Handle 'ai' specially
                short_pattern_parts.append(rf'\b{re.escape(keyword)}\b')
        
        self.short_pattern = re.compile('|'.join(short_pattern_parts), re.IGNORECASE) if short_pattern_parts else None
        
        # Special pattern for 'ai' (more restrictive)
        self.ai_pattern = re.compile(r'\b(ai)\b', re.IGNORECASE)
        
        # Dotted abbreviations pattern
        dotted_pattern_parts = []
        for keyword in self.dotted_abbreviations:
            escaped = re.escape(keyword)
            dotted_pattern_parts.append(rf'\b{escaped}\b')
        
        self.dotted_pattern = re.compile('|'.join(dotted_pattern_parts), re.IGNORECASE) if dotted_pattern_parts else None
        
        # Long keywords pattern (no word boundaries needed)
        long_pattern_parts = [re.escape(keyword) for keyword in self.long_keywords_no_spaces]
        self.long_pattern = re.compile('|'.join(long_pattern_parts), re.IGNORECASE) if long_pattern_parts else None
    
    def comprehensive_text_cleaning(self, text: str) -> str:
        """Multi-stage text cleaning for Swiss job postings"""
        if not text or pd.isna(text):
            return ""
        
        # Stage 1: HTML/Encoding cleanup
        text = html.unescape(text)  # Convert &eacute; → é
        text = unicodedata.normalize('NFC', text)  # Normalize Unicode
        
        # Stage 2: Remove formatting artifacts
        text = re.sub(r'[\r\n\t]+', ' ', text)  # Normalize line breaks/tabs
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\/]', ' ', text)  # Remove special chars except basic punctuation
        
        # Stage 3: Normalize punctuation spacing
        text = re.sub(r'([a-zA-ZäöüÄÖÜàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿ])([.,:;!?])', r'\1 \2', text)
        text = re.sub(r'([.,:;!?])([a-zA-ZäöüÄÖÜàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿ])', r'\1 \2', text)
        
        # Stage 4: Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def two_pass_keyword_detection(self, text: str, language: str = None) -> List[str]:
        """Two-pass keyword detection: space-removed + boundary-aware"""
        if not text or pd.isna(text):
            return []
        
        # Clean text
        cleaned_text = self.comprehensive_text_cleaning(text)
        matches = []
        
        # PASS 1: Space-removed matching for long keywords
        if self.long_pattern:
            text_no_spaces = re.sub(r'\s+', '', cleaned_text.lower())
            long_matches = self.long_pattern.findall(text_no_spaces)
            matches.extend(long_matches)
        
        # PASS 2: Boundary-aware matching for short keywords
        if self.short_pattern:
            short_matches = self.short_pattern.findall(cleaned_text)
            matches.extend(short_matches)
        
        # Special handling for 'ai' keyword
        if language != 'it' and self.ai_pattern:  # Exclude from Italian
            ai_matches = self.ai_pattern.findall(cleaned_text)
            matches.extend(ai_matches)
        
        # Dotted abbreviations
        if self.dotted_pattern:
            dotted_matches = self.dotted_pattern.findall(cleaned_text)
            matches.extend(dotted_matches)
        
        # Remove duplicates and empty matches
        matches = [match for match in set(matches) if match.strip()]
        
        # Quality validation
        validated_matches = self._validate_matches(cleaned_text, matches)
        
        return validated_matches
    
    def _validate_matches(self, text: str, matches: List[str]) -> List[str]:
        """Quality validation for matched keywords"""
        validated = []
        
        for match in matches:
            # Skip very short matches
            if len(match) < 2:
                continue
            
            # Check for false positive contexts
            if self._is_false_positive_context(text, match):
                continue
            
            validated.append(match)
        
        return validated
    
    def _is_false_positive_context(self, text: str, match: str) -> bool:
        """Check if match appears in known false positive contexts"""
        
        # Company name patterns
        company_patterns = [
            rf'(AG|GmbH|SA|Ltd|Inc|Corp)\s*[^\w]*{re.escape(match)}[^\w]',
            rf'{re.escape(match)}[^\w]*(AG|GmbH|SA|Ltd|Inc|Corp)',
            rf'www\..*{re.escape(match)}.*\.(ch|com|de|fr|it)',
        ]
        
        for pattern in company_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Email patterns
        if re.search(rf'{re.escape(match)}@|@.*{re.escape(match)}', text, re.IGNORECASE):
            return True
        
        # Negation patterns
        negation_patterns = [
            rf'(not|nicht|pas|non|no)\s+.*{re.escape(match)}',
            rf'{re.escape(match)}\s+.*(not|nicht|pas|non|no)\s+(required|necessary|needed)',
        ]
        
        for pattern in negation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def detect_language_improved(self, text: str) -> str:
        """Fast language detection for Swiss job postings"""
        if not text or pd.isna(text):
            return 'unknown'
        
        # Basic text cleaning - much faster
        text_lower = str(text).lower()
        
        # Use simple regex counting for key indicators
        de_score = len(re.findall(r'\b(der|die|das|und|mit|für|von|ist|sind|haben|hat)\b', text_lower))
        fr_score = len(re.findall(r'\b(le|la|les|et|de|avec|pour|est|sont|ont|sur)\b', text_lower))
        it_score = len(re.findall(r'\b(il|la|le|con|per|di|e|sono|hanno|su|in)\b', text_lower))
        en_score = len(re.findall(r'\b(the|and|with|for|is|are|have|has|on|in)\b', text_lower))
        
        scores = {'de': de_score, 'fr': fr_score, 'it': it_score, 'en': en_score}
        
        # Return language with highest score
        if max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def extract_matching_sentences(self, text: str, keywords: List[str]) -> str:
        """Extract sentences containing the matched keywords"""
        if not text or pd.isna(text) or not keywords:
            return ""
        
        # Clean text
        cleaned_text = self.comprehensive_text_cleaning(text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+\s*', cleaned_text)
        
        matching_sentences = []
        used_keywords = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Check if sentence contains any of the keywords
            sentence_lower = sentence.lower()
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # Check both original and space-removed versions
                if (keyword_lower in sentence_lower or 
                    re.sub(r'\s+', '', keyword_lower) in re.sub(r'\s+', '', sentence_lower)):
                    
                    if keyword not in used_keywords:
                        matching_sentences.append(sentence)
                        used_keywords.add(keyword)
                        break
        
        return ' | '.join(matching_sentences[:5])  # Limit to 5 sentences
    
    def calculate_confidence(self, matches: List[str]) -> float:
        """Calculate confidence score based on matches"""
        if not matches:
            return 0.0
        
        # Base score for each match
        base_score = len(matches) * 3
        
        # Bonus for high-value AI terms
        high_value_terms = [
            'artificial intelligence', 'machine learning', 'deep learning', 
            'neural network', 'natural language processing', 'computer vision',
            'large language model', 'künstliche intelligenz', 'apprentissage automatique', 
            'intelligenza artificiale', 'maschinelles lernen', 'apprentissage profond'
        ]
        
        bonus_score = 0
        for match in matches:
            # Check if match is a substring of any high-value term
            for hv_term in high_value_terms:
                if match.lower() in hv_term.lower() or re.sub(r'\s+', '', match.lower()) in re.sub(r'\s+', '', hv_term.lower()):
                    bonus_score += 5
                    break
        
        return base_score + bonus_score
    
    def load_job_ads_from_sample(self, dataset_type="train") -> pd.DataFrame:
        """Load job ads from sample file for specified dataset"""
        
        if dataset_type == "train":
            # Use the 100-company sample
            ads_file = self.data_dir / "raw_ads" / "random_sample_100_companies_ads.csv"
        elif dataset_type == "test":
            # Use the test set
            test_files = list(self.data_dir.glob("test_set_*_companies_*.csv"))
            if not test_files:
                print(f"Error: No test set files found")
                return None
            ads_file = max(test_files, key=lambda x: x.stat().st_mtime)
        else:
            raise ValueError(f"dataset_type must be 'train' or 'test', got '{dataset_type}'")
        
        if not ads_file.exists():
            print(f"Error: Sample file not found at {ads_file}")
            return None
        
        # Load the CSV file
        df = pd.read_csv(ads_file)
        
        print(f"Loaded {len(df)} job ads from {dataset_type} set ({ads_file.name})")
        print(f"Unique companies: {df['company_name'].nunique()}")
        
        # Ensure we have the content_clean column
        if 'content_clean' not in df.columns:
            print("Error: content_clean column not found in sample data")
            return None
        
        return df
    
    def process_job_ads(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process job ads with enhanced AI detection"""
        print("="*60)
        print("ENHANCED AI DEVELOPMENT DETECTION")
        print("="*60)
        
        # Language detection
        print("Detecting languages...")
        tqdm.pandas(desc="Language detection")
        df['detected_language'] = df['content_clean'].progress_apply(self.detect_language_improved)
        
        # AI keyword detection
        print("Detecting AI keywords...")
        tqdm.pandas(desc="AI keyword detection")
        df['ai_keywords'] = df.progress_apply(
            lambda row: self.two_pass_keyword_detection(row['content_clean'], row['detected_language']), 
            axis=1
        )
        
        # Calculate confidence and AI relation
        df['ai_confidence'] = df['ai_keywords'].apply(self.calculate_confidence)
        df['is_ai_related'] = df['ai_confidence'] >= 5.0
        df['ai_keywords_str'] = df['ai_keywords'].apply(lambda x: ', '.join(x) if x else '')
        
        # Extract matching sentences
        print("Extracting matching sentences...")
        tqdm.pandas(desc="Sentence extraction")
        df['matching_sentences'] = df.progress_apply(
            lambda row: self.extract_matching_sentences(row['content_clean'], row['ai_keywords']), 
            axis=1
        )
        
        # Add review priority
        df['review_priority'] = df['ai_confidence'].apply(
            lambda x: 'HIGH' if x >= 15 else 'MEDIUM' if x >= 8 else 'LOW'
        )
        
        # Filter AI-related jobs
        ai_jobs = df[df['is_ai_related']].copy()
        
        print(f"Found {len(ai_jobs)} jobs with AI development keywords")
        print(f"Retention rate: {len(ai_jobs)/len(df)*100:.1f}%")
        
        # Language breakdown
        print(f"Language breakdown: {ai_jobs['detected_language'].value_counts().to_dict()}")
        
        # Top keywords
        if len(ai_jobs) > 0:
            print("Top keywords:")
            all_keywords = []
            for keywords in ai_jobs['ai_keywords']:
                all_keywords.extend(keywords)
            
            keyword_counts = pd.Series(all_keywords).value_counts().head(10)
            for keyword, count in keyword_counts.items():
                print(f"  {keyword}: {count}")
        
        return ai_jobs
    
    def export_results(self, ai_jobs: pd.DataFrame, dataset_type="train") -> str:
        """Export results to CSV file"""
        
        # Define columns for export
        export_columns = [
            'uid', 'title', 'company_name', 'company_size', 'location_raw',
            'detected_language', 'content_clean', 'matching_sentences',
            'is_ai_related', 'ai_confidence', 'ai_keywords_str', 'review_priority',
            'tst_created', 'url'
        ]
        
        # Create timestamped filename
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.data_dir / f"enhanced_ai_detection_{dataset_type}_{timestamp}.csv"
        
        # Export
        ai_jobs[export_columns].to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n📄 Results exported to: {output_path}")
        print(f"   - Jobs found: {len(ai_jobs)}")
        print(f"   - Priority breakdown: HIGH={len(ai_jobs[ai_jobs['review_priority']=='HIGH'])}, "
              f"MEDIUM={len(ai_jobs[ai_jobs['review_priority']=='MEDIUM'])}, "
              f"LOW={len(ai_jobs[ai_jobs['review_priority']=='LOW'])}")
        
        return str(output_path)
    
    def run_detection(self, dataset_type="train") -> str:
        """Run the complete enhanced AI detection pipeline"""
        
        # Load data from specified dataset
        df = self.load_job_ads_from_sample(dataset_type)
        
        if df is None:
            return None
        
        # Process with enhanced detection
        ai_jobs = self.process_job_ads(df)
        
        # Export results
        output_path = self.export_results(ai_jobs, dataset_type)
        
        return output_path


def main():
    """Main function to run enhanced AI detection"""
    
    print("Enhanced AI Development Detection")
    print("=" * 50)
    
    # Initialize detector
    detector = EnhancedAIDetection()
    
    # Run detection on training set by default
    output_path = detector.run_detection("train")
    
    print(f"\n🎯 Detection complete!")
    print(f"Results saved to: {output_path}")
    print(f"\n📋 Review the results and validate the enhanced matching approach.")

def run_test_only():
    """Run AI detection on test set only"""
    detector = EnhancedAIDetection()
    output_path = detector.run_detection("test")
    
    print(f"\n🎯 AI detection complete for test set!")
    print(f"Results saved to: {output_path}")

def run_both_sets():
    """Run AI detection on both training and test sets"""
    detector = EnhancedAIDetection()
    
    print("🚀 Starting AI detection for both train and test sets...")
    
    train_path = detector.run_detection("train")
    test_path = detector.run_detection("test")
    
    print(f"\n{'='*60}")
    print("BOTH DATASETS PROCESSED")
    print(f"{'='*60}")
    print(f"✅ Training set: {train_path}")
    print(f"✅ Test set: {test_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            run_test_only()
        elif sys.argv[1] == "--both":
            run_both_sets()
        else:
            print("Usage:")
            print("  python3 step_1_keyword_match.py         # Process training set only")
            print("  python3 step_1_keyword_match.py --test  # Process test set only") 
            print("  python3 step_1_keyword_match.py --both  # Process both sets")
    else:
        # Default behavior - run training set only
        main()