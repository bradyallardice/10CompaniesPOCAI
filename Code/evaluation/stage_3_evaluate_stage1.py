#!/usr/bin/env python3
"""
Step 3 Evaluation: Semantic Evaluation of AI Capability Extraction (Step 1)

This module evaluates how well the LLM prompt in step 1 extracts AI capabilities
from job advertisements using semantic similarity matching against ground truth labels.

Based on the Semantic Evaluation Playbook specifications.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import hashlib
import re
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import argparse
import time

# LLM libraries
try:
    import openai
    from anthropic import Anthropic
except ImportError:
    print("WARNING: OpenAI and/or Anthropic not installed.")
    print("Install with: pip install openai anthropic")
    print("Continuing without LLM capabilities...")

# ML/NLP libraries
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from scipy.optimize import linear_sum_assignment
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Please install with: pip install sentence-transformers torch scipy")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Step1PromptExecutor:
    """
    Executes the exact step 1 AI capability extraction prompt from step_3_extract_ai_tasks.py
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", max_retries: int = 3, reasoning_effort: str = "low"):
        """
        Initialize the prompt executor with OpenAI API.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use
            max_retries: Maximum number of retries for failed calls
            reasoning_effort: Reasoning effort for o-series models (low, medium, high)
        """
        self.model = model
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        
        # Initialize AITaskExtractor for token tracking
        self.extractor = None
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
            logger.info(f"Initialized OpenAI client with model {self.model}")
        except ImportError:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI client: {e}")
    
    def get_step1_extractor(self):
        """
        Get an instance of the AITaskExtractor to use the improved v2 method
        Reuses the same instance to accumulate token usage across all calls.
        
        Returns:
            AITaskExtractor instance
        """
        if self.extractor is None:
            try:
                from stage_3_extract_ai_tasks import AITaskExtractor
                self.extractor = AITaskExtractor(model_name=self.model, reasoning_effort=self.reasoning_effort)
                logger.info(f"Initialized AITaskExtractor with model {self.model} and reasoning effort {self.reasoning_effort}")
            except ImportError as e:
                logger.error(f"Failed to import AITaskExtractor: {e}")
                raise ImportError("Could not import AITaskExtractor from stage_3_extract_ai_tasks.py")
        return self.extractor
    
    def get_token_usage_summary(self) -> dict:
        """
        Get token usage summary from the AITaskExtractor.
        
        Returns:
            Dictionary with token usage data
        """
        if self.extractor is None:
            return {
                'total_tokens': 0,
                'total_cost': 0.0,
                'api_calls': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0
            }
        
        return {
            'total_tokens': self.extractor.total_tokens,
            'total_cost': self.extractor.total_cost,
            'api_calls': self.extractor.api_calls,
            'prompt_tokens': self.extractor.total_prompt_tokens,
            'completion_tokens': self.extractor.total_completion_tokens
        }
    
    def get_token_usage_delta(self, previous_usage: dict) -> dict:
        """
        Get delta in token usage since previous snapshot.
        
        Args:
            previous_usage: Previous token usage snapshot
            
        Returns:
            Dictionary with token usage delta
        """
        current_usage = self.get_token_usage_summary()
        
        return {
            'total_tokens': current_usage['total_tokens'] - previous_usage.get('total_tokens', 0),
            'total_cost': current_usage['total_cost'] - previous_usage.get('total_cost', 0.0),
            'api_calls': current_usage['api_calls'] - previous_usage.get('api_calls', 0),
            'prompt_tokens': current_usage['prompt_tokens'] - previous_usage.get('prompt_tokens', 0),
            'completion_tokens': current_usage['completion_tokens'] - previous_usage.get('completion_tokens', 0)
        }
    
    def get_prompt_text(self) -> str:
        """
        Get the full prompt text from the AITaskExtractor for logging purposes
        
        Returns:
            The system prompt text used in step_1_extract_ai_applications_v2
        """
        try:
            extractor = self.get_step1_extractor()
            # Get the prompt text by inspecting the method source or calling a getter
            import inspect
            source = inspect.getsource(extractor.step_1_extract_ai_applications_v2)
            
            # Extract the system_prompt variable content
            lines = source.split('\n')
            prompt_lines = []
            in_prompt = False
            for line in lines:
                if 'system_prompt = """' in line:
                    in_prompt = True
                    # Get content after the triple quotes
                    after_quotes = line.split('system_prompt = """')[1]
                    if after_quotes:
                        prompt_lines.append(after_quotes)
                elif in_prompt:
                    if '"""' in line and not line.strip().startswith('"""'):
                        # End of prompt, get content before triple quotes
                        before_quotes = line.split('"""')[0]
                        if before_quotes:
                            prompt_lines.append(before_quotes)
                        break
                    else:
                        prompt_lines.append(line)
            
            return '\n'.join(prompt_lines).strip()
        except Exception as e:
            logger.warning(f"Could not extract prompt text: {e}")
            return "[Could not extract prompt text]"
    
    def execute_step1_prompt(self, job_content: str, use_ensemble: bool = False, 
                             num_models: int = 5, temperature: float = 0.8) -> Dict[str, List[str]]:
        """
        Execute step 1 prompt on a job advertisement using the exact implementation.
        
        Args:
            job_content: The job advertisement content
            use_ensemble: Whether to use ensemble method with majority voting
            num_models: Number of models in ensemble (only if use_ensemble=True)
            temperature: Temperature for ensemble diversity (only if use_ensemble=True)
            
        Returns:
            Dictionary with 'ai_capabilities' key containing list of extracted capabilities
        """
        try:
            extractor = self.get_step1_extractor()
            
            if use_ensemble:
                logger.info(f"Using ensemble method with {num_models} models at temperature {temperature}")
                response_content = extractor.step_1_extract_ai_applications_ensemble(
                    job_content, num_models=num_models, temperature=temperature
                )
            else:
                # Use the improved v2 method from AITaskExtractor
                response_content = extractor.step_1_extract_ai_applications_v2(job_content)
            
            if not response_content:
                logger.warning("No response from step 1 extraction")
                return {"ai_capabilities": []}
            
            # Parse the response and convert to evaluation format
            parsed_response = self._convert_step1_to_evaluation_format(response_content)
            
            # Add small delay to respect rate limits (smaller for ensemble since it has its own delays)
            time.sleep(0.1 if use_ensemble else 0.5)
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Step 1 extraction failed: {e}")
            return {"ai_capabilities": []}
    
    def _convert_step1_to_evaluation_format(self, step1_response: str) -> Dict[str, List[str]]:
        """
        Convert step 1 response format to evaluation format.
        
        Step 1 returns: {"ai_application_tasks": [{"ai_capability": "...", ...}, ...]}
        Evaluation expects: {"ai_capabilities": ["...", "...", ...]}
        
        Args:
            step1_response: JSON response from step 1
            
        Returns:
            Dictionary with ai_capabilities list for evaluation
        """
        try:
            # Parse the JSON response
            parsed = json.loads(step1_response)
            
            # Handle both old and new format
            # Old format: {"ai_application_tasks": [...]}
            # New format: {"ai_capabilities": [...]}
            ai_tasks = parsed.get('ai_application_tasks', parsed.get('ai_capabilities', []))
            capabilities = []
            
            for task in ai_tasks:
                if isinstance(task, dict) and 'ai_capability' in task:
                    capability = task['ai_capability']
                    if capability and isinstance(capability, str):
                        capabilities.append(capability.strip())
            
            return {"ai_capabilities": capabilities}
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse step 1 JSON response: {e}")
            logger.error(f"Response was: {step1_response}")
            return {"ai_capabilities": []}
        except Exception as e:
            logger.error(f"Error converting step 1 response: {e}")
            return {"ai_capabilities": []}
    
    def process_dataset(self, df: pd.DataFrame, content_column: str = 'content_clean',
                       use_ensemble: bool = False, num_models: int = 5, temperature: float = 0.8) -> pd.DataFrame:
        """
        Process entire dataset with step 1 prompt.
        
        Args:
            df: DataFrame with job advertisements
            content_column: Column containing job content
            use_ensemble: Whether to use ensemble method with majority voting
            num_models: Number of models in ensemble (only if use_ensemble=True)
            temperature: Temperature for ensemble diversity (only if use_ensemble=True)
            
        Returns:
            DataFrame with added predicted_json column
        """
        method_desc = f"ensemble ({num_models} models, temp={temperature})" if use_ensemble else "single model"
        logger.info(f"Processing {len(df)} job advertisements with step 1 prompt using {method_desc}")
        
        predictions = []
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Processing job {idx + 1}/{len(df)}")
            
            # Get job content
            job_content = row.get(content_column, '')
            if pd.isna(job_content) or not job_content:
                job_content = row.get('translated_text', '')
            if pd.isna(job_content) or not job_content:
                job_content = row.get('content', '')
            
            if pd.isna(job_content) or not job_content:
                logger.warning(f"No content found for row {idx}")
                predictions.append(json.dumps({"ai_capabilities": []}))
                continue
            
            # Execute step 1 prompt (with ensemble if specified)
            prediction = self.execute_step1_prompt(
                str(job_content),
                use_ensemble=use_ensemble,
                num_models=num_models,
                temperature=temperature
            )
            
            # Clean Unicode characters in predictions before saving
            cleaned_prediction = self._clean_unicode_in_prediction(prediction)
            predictions.append(json.dumps(cleaned_prediction, ensure_ascii=False))
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['predicted_json'] = predictions
        
        logger.info(f"Completed processing {len(df)} job advertisements using {method_desc}")
        return result_df
    
    def _clean_unicode_in_prediction(self, prediction_dict: dict) -> dict:
        """
        Clean Unicode characters in prediction dictionary.
        
        Args:
            prediction_dict: Dictionary containing AI capabilities
            
        Returns:
            Cleaned dictionary with normalized Unicode
        """
        if not isinstance(prediction_dict, dict) or 'ai_capabilities' not in prediction_dict:
            return prediction_dict
        
        capabilities = prediction_dict['ai_capabilities']
        if not isinstance(capabilities, list):
            return prediction_dict
        
        # Clean each capability string
        cleaned_capabilities = []
        for cap in capabilities:
            if isinstance(cap, str):
                cleaned_cap = self._normalize_unicode_string(cap)
                cleaned_capabilities.append(cleaned_cap)
            else:
                cleaned_capabilities.append(cap)
        
        # Return cleaned dict
        cleaned_dict = prediction_dict.copy()
        cleaned_dict['ai_capabilities'] = cleaned_capabilities
        return cleaned_dict
    
    def _normalize_unicode_string(self, text: str) -> str:
        """
        Normalize Unicode characters in a string.
        
        Args:
            text: Input text string
            
        Returns:
            Text with normalized Unicode characters
        """
        import unicodedata
        
        # Normalize Unicode characters (NFKC normalization)
        text = unicodedata.normalize('NFKC', text)
        
        # Replace various types of hyphens and dashes with regular hyphen
        text = re.sub(r'[‐‑–—−]', '-', text)  # hyphen, non-breaking hyphen, en-dash, em-dash, minus
        
        # Replace various quotes with regular quotes
        text = re.sub(r'[""''`]', '"', text)
        
        return text


class SemanticEvaluator:
    """
    Semantic evaluator for AI capability extraction using BGE embeddings and similarity matching.
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 similarity_threshold: float = 0.7,
                 embeddings_dir: str = "Data/embeddings"):
        """
        Initialize the semantic evaluator.
        
        Args:
            model_name: Embedding model name (fixed per playbook)
            similarity_threshold: Similarity threshold for TP matching
            embeddings_dir: Directory for cached embeddings
        """
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.embeddings_dir = embeddings_dir
        
        # Create embeddings directory
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Initialize model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        
        # Batch size
        self.batch_size = 64 if self.device == "cpu" else 256
        
        # Initialize logs
        self.bad_rows_ground_truth = []
        self.bad_rows_predictions = []
    
    def _normalize_unicode_string(self, text: str) -> str:
        """
        Normalize Unicode characters in a string (shared between ground truth and predictions).
        
        Args:
            text: Input text string
            
        Returns:
            Text with normalized Unicode characters
        """
        import unicodedata
        
        # Normalize Unicode characters (NFKC normalization)
        text = unicodedata.normalize('NFKC', text)
        
        # Replace various types of hyphens and dashes with regular hyphen
        text = re.sub(r'[‐‑–—−]', '-', text)  # hyphen, non-breaking hyphen, en-dash, em-dash, minus
        
        # Replace various quotes with regular quotes
        text = re.sub(r'[""''`]', '"', text)
        
        return text
    
    def normalize_sentence(self, sentence: str) -> str:
        """
        Normalize a sentence following the playbook specifications.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Normalized sentence
        """
        if pd.isna(sentence) or not sentence:
            return ""
        
        # Convert to string and lowercase
        sentence = str(sentence).lower().strip()
        
        # Normalize Unicode characters (fix the \u2011 issue)
        import unicodedata
        sentence = unicodedata.normalize('NFKC', sentence)
        
        # Replace various types of hyphens and dashes with regular hyphen
        sentence = re.sub(r'[‐‑–—−]', '-', sentence)  # hyphen, non-breaking hyphen, en-dash, em-dash, minus
        
        # Replace various quotes with regular quotes
        sentence = re.sub(r'[""''`]', '"', sentence)
        
        # Remove boilerplate tokens
        boilerplate_tokens = ['ai', 'ml', 'model', 'algorithm', 'machine learning', 
                            'artificial intelligence', 'deep learning']
        for token in boilerplate_tokens:
            sentence = re.sub(rf'\b{re.escape(token)}\b', '', sentence, flags=re.IGNORECASE)
        
        # Remove punctuation and squeeze whitespace
        sentence = re.sub(r'[^\w\s-]', ' ', sentence)  # Keep regular hyphens
        sentence = re.sub(r'\s+', ' ', sentence)
        
        return sentence.strip()
    
    def deduplicate_predictions(self, predictions: List[str]) -> List[str]:
        """
        Deduplicate predictions using semantic similarity (≥ 0.90 threshold).
        
        Args:
            predictions: List of prediction strings
            
        Returns:
            Deduplicated list of predictions
        """
        if len(predictions) <= 1:
            return predictions
        
        # Embed all predictions
        embeddings = self.model.encode(predictions, 
                                     batch_size=self.batch_size,
                                     normalize_embeddings=True,
                                     convert_to_numpy=True)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Find duplicates (similarity ≥ 0.90)
        keep_indices = []
        used_indices = set()
        
        for i in range(len(predictions)):
            if i in used_indices:
                continue
            
            keep_indices.append(i)
            used_indices.add(i)
            
            # Mark similar predictions as used
            for j in range(i + 1, len(predictions)):
                if j not in used_indices and similarity_matrix[i, j] >= 0.90:
                    used_indices.add(j)
        
        return [predictions[i] for i in keep_indices]
    
    def extract_ground_truth(self, final_json_str: str, row_idx: int, company_id: str) -> List[str]:
        """
        Extract ground truth capabilities from the Final column JSON.
        
        Args:
            final_json_str: JSON string from Final column
            row_idx: Row index for logging
            company_id: Company ID for logging
            
        Returns:
            List of ground truth capability strings
        """
        try:
            if pd.isna(final_json_str) or not final_json_str:
                return []
            
            # Parse JSON
            final_data = json.loads(final_json_str)
            
            # Extract ai_application_tasks array
            ai_tasks = final_data.get('ai_application_tasks', [])
            if not isinstance(ai_tasks, list):
                return []
            
            # Collect ai_capability values with Unicode normalization
            capabilities = []
            for task in ai_tasks:
                if isinstance(task, dict) and 'ai_capability' in task:
                    cap = task['ai_capability']
                    if cap and isinstance(cap, str):
                        # Apply same Unicode normalization as predictions
                        normalized_cap = self._normalize_unicode_string(cap.strip())
                        capabilities.append(normalized_cap)
            
            return capabilities
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.bad_rows_ground_truth.append({
                'row_index': row_idx,
                'company_id': company_id,
                'error': str(e)
            })
            logger.warning(f"Failed to parse ground truth JSON at row {row_idx}: {e}")
            return []
    
    def extract_predictions(self, predicted_json_str: str, row_idx: int, company_id: str) -> List[str]:
        """
        Extract predictions from the predicted_json column.
        
        Args:
            predicted_json_str: JSON string from prediction column
            row_idx: Row index for logging
            company_id: Company ID for logging
            
        Returns:
            List of predicted capability strings
        """
        try:
            if pd.isna(predicted_json_str) or not predicted_json_str:
                return []
            
            # Parse JSON
            pred_data = json.loads(predicted_json_str)
            
            # Extract ai_capabilities key
            if 'ai_capabilities' not in pred_data:
                self.bad_rows_predictions.append({
                    'row_index': row_idx,
                    'company_id': company_id,
                    'error': 'Missing ai_capabilities key'
                })
                return []
            
            capabilities = pred_data['ai_capabilities']
            
            if not isinstance(capabilities, list):
                self.bad_rows_predictions.append({
                    'row_index': row_idx,
                    'company_id': company_id,
                    'error': 'ai_capabilities is not a list'
                })
                return []
            
            # Filter to valid strings with Unicode normalization
            valid_caps = []
            for cap in capabilities:
                if isinstance(cap, str) and cap.strip():
                    # Apply same Unicode normalization as ground truth
                    normalized_cap = self._normalize_unicode_string(cap.strip())
                    valid_caps.append(normalized_cap)
            
            return valid_caps
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.bad_rows_predictions.append({
                'row_index': row_idx,
                'company_id': company_id,
                'error': str(e)
            })
            logger.warning(f"Failed to parse predictions JSON at row {row_idx}: {e}")
            return []
    
    def compute_similarity_matrix(self, gold_caps: List[str], pred_caps: List[str]) -> np.ndarray:
        """
        Compute similarity matrix between gold and predicted capabilities.
        
        Args:
            gold_caps: Ground truth capabilities
            pred_caps: Predicted capabilities
            
        Returns:
            m x n similarity matrix where cell (i,j) is similarity between pred_i and gold_j
        """
        if not gold_caps or not pred_caps:
            return np.array([]).reshape(len(pred_caps), len(gold_caps))
        
        # Embed both lists
        gold_embeddings = self.model.encode(gold_caps,
                                          batch_size=self.batch_size,
                                          normalize_embeddings=True,
                                          convert_to_numpy=True)
        
        pred_embeddings = self.model.encode(pred_caps,
                                          batch_size=self.batch_size,
                                          normalize_embeddings=True,
                                          convert_to_numpy=True)
        
        # Compute cosine similarity matrix (m x n)
        similarity_matrix = np.dot(pred_embeddings, gold_embeddings.T)
        
        return similarity_matrix
    
    def greedy_matching(self, similarity_matrix: np.ndarray) -> Tuple[int, int, int]:
        """
        Perform greedy one-to-one matching to compute TP, FP, FN.
        
        Args:
            similarity_matrix: m x n matrix of similarities
            
        Returns:
            Tuple of (TP, FP, FN)
        """
        if similarity_matrix.size == 0:
            m, n = similarity_matrix.shape
            return 0, m, n  # No matches possible
        
        m, n = similarity_matrix.shape  # m = predictions, n = gold
        
        # Greedy matching
        used_preds = set()
        used_golds = set()
        tp = 0
        
        # Create list of all (pred_idx, gold_idx, similarity) tuples above threshold
        candidates = []
        for i in range(m):
            for j in range(n):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    candidates.append((i, j, similarity_matrix[i, j]))
        
        # Sort by similarity (descending) for greedy selection
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Greedy matching
        for pred_idx, gold_idx, sim in candidates:
            if pred_idx not in used_preds and gold_idx not in used_golds:
                tp += 1
                used_preds.add(pred_idx)
                used_golds.add(gold_idx)
        
        # Compute FP and FN
        fp = m - len(used_preds)  # Unmatched predictions
        fn = n - len(used_golds)  # Unmatched gold standards
        
        return tp, fp, fn
    
    def get_detailed_matching(self, gold_caps: List[str], pred_caps: List[str], similarity_matrix: np.ndarray) -> List[Dict]:
        """
        Get detailed matching information for export to CSV.
        
        Args:
            gold_caps: Ground truth capabilities (normalized)
            pred_caps: Predicted capabilities (normalized and deduplicated)
            similarity_matrix: m x n similarity matrix
            
        Returns:
            List of dictionaries with matching details
        """
        matches = []
        
        if similarity_matrix.size == 0:
            # Handle empty cases
            for i, gold_cap in enumerate(gold_caps):
                matches.append({
                    'ground_truth': gold_cap,
                    'prediction': None,
                    'similarity_score': 0.0,
                    'match_type': 'FN',
                    'above_threshold': False
                })
            for i, pred_cap in enumerate(pred_caps):
                matches.append({
                    'ground_truth': None,
                    'prediction': pred_cap,
                    'similarity_score': 0.0,
                    'match_type': 'FP',
                    'above_threshold': False
                })
            return matches
        
        m, n = similarity_matrix.shape  # m = predictions, n = gold
        
        # Greedy matching to find actual matches
        used_preds = set()
        used_golds = set()
        
        # Create list of all (pred_idx, gold_idx, similarity) tuples above threshold
        candidates = []
        for i in range(m):
            for j in range(n):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    candidates.append((i, j, similarity_matrix[i, j]))
        
        # Sort by similarity (descending) for greedy selection
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Apply greedy matching
        for pred_idx, gold_idx, sim in candidates:
            if pred_idx not in used_preds and gold_idx not in used_golds:
                matches.append({
                    'ground_truth': gold_caps[gold_idx],
                    'prediction': pred_caps[pred_idx],
                    'similarity_score': float(sim),
                    'match_type': 'TP',
                    'above_threshold': True
                })
                used_preds.add(pred_idx)
                used_golds.add(gold_idx)
        
        # Add unmatched predictions (FP)
        for i in range(m):
            if i not in used_preds:
                # Find best similarity to any gold standard
                best_sim = 0.0
                best_gold = None
                for j in range(n):
                    if similarity_matrix[i, j] > best_sim:
                        best_sim = similarity_matrix[i, j]
                        best_gold = gold_caps[j]
                
                matches.append({
                    'ground_truth': best_gold,
                    'prediction': pred_caps[i],
                    'similarity_score': float(best_sim),
                    'match_type': 'FP',
                    'above_threshold': best_sim >= self.similarity_threshold
                })
        
        # Add unmatched gold standards (FN)
        for j in range(n):
            if j not in used_golds:
                # Find best similarity to any prediction
                best_sim = 0.0
                best_pred = None
                for i in range(m):
                    if similarity_matrix[i, j] > best_sim:
                        best_sim = similarity_matrix[i, j]
                        best_pred = pred_caps[i]
                
                matches.append({
                    'ground_truth': gold_caps[j],
                    'prediction': best_pred,
                    'similarity_score': float(best_sim),
                    'match_type': 'FN',
                    'above_threshold': best_sim >= self.similarity_threshold
                })
        
        return matches
    
    def evaluate_single_advert(self, gold_caps: List[str], pred_caps: List[str]) -> Dict[str, float]:
        """
        Evaluate a single advertisement.
        
        Args:
            gold_caps: Ground truth capabilities
            pred_caps: Predicted capabilities
            
        Returns:
            Dictionary with precision, recall, f1 scores
        """
        # Normalize both lists
        gold_normalized = [self.normalize_sentence(cap) for cap in gold_caps]
        pred_normalized = [self.normalize_sentence(cap) for cap in pred_caps]
        
        # Remove empty strings
        gold_normalized = [cap for cap in gold_normalized if cap]
        pred_normalized = [cap for cap in pred_normalized if cap]
        
        # Deduplicate predictions
        pred_deduped = self.deduplicate_predictions(pred_normalized)
        
        # Handle empty cases
        if not gold_normalized and not pred_deduped:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}  # Perfect empty case
        
        if not pred_deduped:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}  # Model missed all
        
        if not gold_normalized:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}  # Pure hallucination
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(gold_normalized, pred_deduped)
        
        # Greedy matching
        tp, fp, fn = self.greedy_matching(similarity_matrix)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def evaluate_single_advert_with_details(self, gold_caps: List[str], pred_caps: List[str], 
                                          job_id: str = None, company_id: str = None) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate a single advertisement and return both metrics and detailed matching information.
        
        Args:
            gold_caps: Ground truth capabilities
            pred_caps: Predicted capabilities
            job_id: Job identifier for tracking
            company_id: Company identifier for tracking
            
        Returns:
            Tuple of (metrics_dict, detailed_matches_list)
        """
        # Store original capabilities for export
        original_gold = gold_caps.copy()
        original_pred = pred_caps.copy()
        
        # Normalize both lists
        gold_normalized = [self.normalize_sentence(cap) for cap in gold_caps]
        pred_normalized = [self.normalize_sentence(cap) for cap in pred_caps]
        
        # Remove empty strings
        gold_normalized = [cap for cap in gold_normalized if cap]
        pred_normalized = [cap for cap in pred_normalized if cap]
        
        # Deduplicate predictions
        pred_deduped = self.deduplicate_predictions(pred_normalized)
        
        # Handle empty cases
        if not gold_normalized and not pred_deduped:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}, []
        
        if not pred_deduped:
            # All ground truth unmatched
            detailed_matches = []
            for gold_cap in original_gold:
                detailed_matches.append({
                    'job_id': job_id,
                    'company_id': company_id,
                    'ground_truth_original': gold_cap,
                    'ground_truth_normalized': self.normalize_sentence(gold_cap),
                    'prediction_original': None,
                    'prediction_normalized': None,
                    'similarity_score': 0.0,
                    'match_type': 'FN',
                    'above_threshold': False
                })
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, detailed_matches
        
        if not gold_normalized:
            # All predictions unmatched
            detailed_matches = []
            for pred_cap in original_pred:
                detailed_matches.append({
                    'job_id': job_id,
                    'company_id': company_id,
                    'ground_truth_original': None,
                    'ground_truth_normalized': None,
                    'prediction_original': pred_cap,
                    'prediction_normalized': self.normalize_sentence(pred_cap),
                    'similarity_score': 0.0,
                    'match_type': 'FP',
                    'above_threshold': False
                })
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, detailed_matches
        
        # Compute similarity matrix
        similarity_matrix = self.compute_similarity_matrix(gold_normalized, pred_deduped)
        
        # Get detailed matching information
        basic_matches = self.get_detailed_matching(gold_normalized, pred_deduped, similarity_matrix)
        
        # Enhance with original text and identifiers
        detailed_matches = []
        for match in basic_matches:
            # Find original texts
            gold_orig = None
            pred_orig = None
            
            if match['ground_truth']:
                # Find original ground truth text
                for i, norm_gold in enumerate(gold_normalized):
                    if norm_gold == match['ground_truth']:
                        if i < len(original_gold):
                            gold_orig = original_gold[i]
                        break
            
            if match['prediction']:
                # Find original prediction text
                for i, norm_pred in enumerate(pred_deduped):
                    if norm_pred == match['prediction']:
                        # Find corresponding original
                        for j, orig_norm in enumerate([self.normalize_sentence(p) for p in original_pred]):
                            if orig_norm == norm_pred:
                                pred_orig = original_pred[j]
                                break
                        break
            
            detailed_matches.append({
                'job_id': job_id,
                'company_id': company_id,
                'ground_truth_original': gold_orig,
                'ground_truth_normalized': match['ground_truth'],
                'prediction_original': pred_orig,
                'prediction_normalized': match['prediction'],
                'similarity_score': match['similarity_score'],
                'match_type': match['match_type'],
                'above_threshold': match['above_threshold']
            })
        
        # Greedy matching for metrics
        tp, fp, fn = self.greedy_matching(similarity_matrix)
        
        # Compute metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        
        return metrics, detailed_matches
    
    def evaluate_dataset(self, df: pd.DataFrame, prediction_column: str = 'predicted_json') -> Dict[str, float]:
        """
        Evaluate an entire dataset.
        
        Args:
            df: DataFrame with Final and prediction columns
            prediction_column: Name of column containing predictions
            
        Returns:
            Dictionary with macro-averaged metrics
        """
        logger.info(f"Evaluating {len(df)} advertisements")
        
        # Reset error logs
        self.bad_rows_ground_truth = []
        self.bad_rows_predictions = []
        
        advert_scores = []
        total_ground_truth_caps = 0
        total_predicted_caps = 0
        
        for idx, row in df.iterrows():
            # Extract ground truth and predictions
            company_id = row.get('company_name', f'row_{idx}')
            
            gold_caps = self.extract_ground_truth(row['Final'], idx, company_id)
            pred_caps = self.extract_predictions(row[prediction_column], idx, company_id)
            
            # Count capabilities
            total_ground_truth_caps += len(gold_caps)
            total_predicted_caps += len(pred_caps)
            
            # Evaluate single advert
            scores = self.evaluate_single_advert(gold_caps, pred_caps)
            advert_scores.append(scores)
        
        # Macro-average across all adverts
        macro_precision = np.mean([s['precision'] for s in advert_scores])
        macro_recall = np.mean([s['recall'] for s in advert_scores])
        macro_f1 = np.mean([s['f1'] for s in advert_scores])
        
        logger.info(f"Evaluation complete: P={macro_precision:.4f}, R={macro_recall:.4f}, F1={macro_f1:.4f}")
        logger.info(f"Malformed ground truth rows: {len(self.bad_rows_ground_truth)}")
        logger.info(f"Malformed prediction rows: {len(self.bad_rows_predictions)}")
        logger.info(f"Total ground truth capabilities: {total_ground_truth_caps}")
        logger.info(f"Total predicted capabilities: {total_predicted_caps}")
        
        return {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'malformed_ground_truth': len(self.bad_rows_ground_truth),
            'malformed_predictions': len(self.bad_rows_predictions),
            'total_adverts': len(df),
            'total_ground_truth_caps': total_ground_truth_caps,
            'total_predicted_caps': total_predicted_caps
        }
    
    def evaluate_dataset_with_details(self, df: pd.DataFrame, prediction_column: str = 'predicted_json') -> Tuple[Dict[str, float], List[Dict]]:
        """
        Evaluate an entire dataset and return both metrics and detailed matching information.
        
        Args:
            df: DataFrame with Final and prediction columns
            prediction_column: Name of column containing predictions
            
        Returns:
            Tuple of (macro_metrics_dict, detailed_matches_list)
        """
        logger.info(f"Evaluating {len(df)} advertisements with detailed matching")
        
        # Reset error logs
        self.bad_rows_ground_truth = []
        self.bad_rows_predictions = []
        
        advert_scores = []
        all_detailed_matches = []
        total_ground_truth_caps = 0
        total_predicted_caps = 0
        
        for idx, row in df.iterrows():
            # Extract identifiers
            company_id = row.get('company_name', f'row_{idx}')
            job_id = row.get('uid', f'job_{idx}')
            
            # Extract ground truth and predictions
            gold_caps = self.extract_ground_truth(row['Final'], idx, company_id)
            pred_caps = self.extract_predictions(row[prediction_column], idx, company_id)
            
            # Count capabilities
            total_ground_truth_caps += len(gold_caps)
            total_predicted_caps += len(pred_caps)
            
            # Evaluate single advert with details
            scores, detailed_matches = self.evaluate_single_advert_with_details(gold_caps, pred_caps, job_id, company_id)
            advert_scores.append(scores)
            all_detailed_matches.extend(detailed_matches)
        
        # Macro-average across all adverts
        macro_precision = np.mean([s['precision'] for s in advert_scores])
        macro_recall = np.mean([s['recall'] for s in advert_scores])
        macro_f1 = np.mean([s['f1'] for s in advert_scores])
        
        logger.info(f"Evaluation complete: P={macro_precision:.4f}, R={macro_recall:.4f}, F1={macro_f1:.4f}")
        logger.info(f"Malformed ground truth rows: {len(self.bad_rows_ground_truth)}")
        logger.info(f"Malformed prediction rows: {len(self.bad_rows_predictions)}")
        logger.info(f"Total detailed matches: {len(all_detailed_matches)}")
        logger.info(f"Total ground truth capabilities: {total_ground_truth_caps}")
        logger.info(f"Total predicted capabilities: {total_predicted_caps}")
        
        metrics = {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'malformed_ground_truth': len(self.bad_rows_ground_truth),
            'malformed_predictions': len(self.bad_rows_predictions),
            'total_adverts': len(df),
            'total_ground_truth_caps': total_ground_truth_caps,
            'total_predicted_caps': total_predicted_caps
        }
        
        return metrics, all_detailed_matches, advert_scores
    
    def write_error_logs(self, output_dir: str = "Data"):
        """
        Write error logs to files.
        
        Args:
            output_dir: Directory to write log files
        """
        # Write ground truth errors
        if self.bad_rows_ground_truth:
            gt_log_file = os.path.join(output_dir, "bad_rows_ground_truth.log")
            with open(gt_log_file, 'w') as f:
                f.write("Row Index,Company ID,Error\n")
                for error in self.bad_rows_ground_truth:
                    f.write(f"{error['row_index']},{error['company_id']},\"{error['error']}\"\n")
            logger.info(f"Wrote ground truth error log: {gt_log_file}")
        
        # Write prediction errors
        if self.bad_rows_predictions:
            pred_log_file = os.path.join(output_dir, "bad_rows_predictions.log")
            with open(pred_log_file, 'w') as f:
                f.write("Row Index,Company ID,Error\n")
                for error in self.bad_rows_predictions:
                    f.write(f"{error['row_index']},{error['company_id']},\"{error['error']}\"\n")
            logger.info(f"Wrote predictions error log: {pred_log_file}")


class ThresholdTuner:
    """
    Tunes similarity threshold on dev set following playbook specifications.
    """
    
    def __init__(self, evaluator: SemanticEvaluator):
        self.evaluator = evaluator
    
    def tune_threshold(self, dev_df: pd.DataFrame, 
                      prediction_column: str = 'predicted_json',
                      threshold_range: Tuple[float, float] = (0.70, 0.95),
                      step: float = 0.01) -> float:
        """
        Tune similarity threshold on dev set.
        
        Args:
            dev_df: Dev set DataFrame
            prediction_column: Name of prediction column
            threshold_range: (min, max) threshold range
            step: Step size for grid search
            
        Returns:
            Best threshold value
        """
        logger.info("Tuning similarity threshold on dev set")
        
        min_thresh, max_thresh = threshold_range
        thresholds = np.arange(min_thresh, max_thresh + step, step)
        
        best_threshold = min_thresh
        best_f1 = 0.0
        results = []
        
        for threshold in thresholds:
            # Set threshold
            self.evaluator.similarity_threshold = threshold
            
            # Evaluate
            metrics = self.evaluator.evaluate_dataset(dev_df, prediction_column)
            f1 = metrics['macro_f1']
            
            results.append({
                'threshold': threshold,
                'macro_f1': f1,
                'macro_precision': metrics['macro_precision'],
                'macro_recall': metrics['macro_recall']
            })
            
            logger.info(f"Threshold {threshold:.2f}: F1={f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
        
        # Save tuning results
        results_df = pd.DataFrame(results)
        results_file = os.path.join("Data", "threshold_tuning_results.csv")
        results_df.to_csv(results_file, index=False)
        logger.info(f"Saved threshold tuning results: {results_file}")
        
        return best_threshold


class PerformanceTracker:
    """
    Tracks performance history following playbook specifications.
    """
    
    def __init__(self, output_dir: str = "Data"):
        self.output_dir = output_dir
        self.runs_file = os.path.join(output_dir, "performance_history_runs.csv")
        self.prompts_file = os.path.join(output_dir, "performance_history_prompts.csv")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate existing CSV files on initialization
        self._validate_csv_files()
    
    def _validate_csv_files(self):
        """Validate existing CSV files - log issues but preserve data."""
        for csv_file in [self.runs_file, self.prompts_file]:
            if os.path.exists(csv_file):
                try:
                    pd.read_csv(csv_file)
                    logger.debug(f"CSV file {csv_file} appears valid")
                except Exception as e:
                    logger.warning(f"CSV file {csv_file} may have issues: {e}")
                    logger.warning("Will attempt to append anyway - data may be recoverable")
                    # Don't backup/rename - just log the issue and continue
                    # The append operations may still work
    
    def compute_prompt_hash(self, prompt_text: str) -> str:
        """
        Compute hash of prompt for identification.
        
        Args:
            prompt_text: The prompt text
            
        Returns:
            Short hash string
        """
        return hashlib.md5(prompt_text.encode()).hexdigest()[:8]
    
    def store_prompt_text(self, prompt_text: str):
        """
        Store the full prompt text with its hash for lookup.
        
        Args:
            prompt_text: The full prompt text to store
        """
        prompt_hash = self.compute_prompt_hash(prompt_text)
        prompt_lookup_file = os.path.join(self.output_dir, "prompt_text_lookup.csv")
        
        # Check if this hash already exists
        if os.path.exists(prompt_lookup_file):
            try:
                existing_df = pd.read_csv(prompt_lookup_file)
                if prompt_hash in existing_df['prompt_hash'].values:
                    # Hash already exists, no need to store again
                    return prompt_hash
            except Exception as e:
                logger.warning(f"Could not read existing prompt_text_lookup.csv: {e}")
                logger.warning("Attempting to append to file anyway - file may be recoverable")
                # Don't backup/rename the file - just continue and try to append
                # The append operation may still work even if reading failed
        
        # Prepare new prompt data
        prompt_data = {
            'prompt_hash': prompt_hash,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'prompt_text': prompt_text
        }
        prompt_df = pd.DataFrame([prompt_data])
        
        # Append to file with proper CSV escaping
        try:
            if not os.path.exists(prompt_lookup_file):
                prompt_df.to_csv(prompt_lookup_file, index=False, quoting=1)  # QUOTE_ALL
            else:
                prompt_df.to_csv(prompt_lookup_file, mode='a', header=False, index=False, quoting=1)  # QUOTE_ALL
            
            logger.info(f"Stored prompt text with hash {prompt_hash}")
            return prompt_hash
        except Exception as e:
            logger.error(f"Failed to write prompt text to CSV: {e}")
            logger.warning("Continuing without storing prompt text")
            return prompt_hash
    
    def log_run_result(self, prompt_id: str, run_index: int, metrics: Dict[str, float], 
                      threshold: float, dataset: str = "test", notes: str = "", prompt_text: str = "",
                      token_usage: dict = None, model_name: str = ""):
        """
        Log a single run result.
        
        Args:
            prompt_id: Identifier for the prompt
            run_index: Run number (1-5)
            metrics: Dictionary with evaluation metrics
            threshold: Similarity threshold used
            dataset: Dataset name (train/dev/test)
            notes: Optional notes
            prompt_text: Full text of the prompt used
            token_usage: Dictionary with token usage data (optional)
            model_name: Name of the model used (e.g., 'o3', 'o4-mini')
        """
        # Store prompt text and get hash
        prompt_hash = self.store_prompt_text(prompt_text) if prompt_text else ''
        
        # Prepare row data
        row_data = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'prompt_id': prompt_id,
            'dataset': dataset,
            'run_index': run_index,
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'macro_f1': metrics['macro_f1'],
            'malformed_ground_truth': metrics['malformed_ground_truth'],
            'malformed_predictions': metrics['malformed_predictions'],
            'total_ground_truth_caps': metrics.get('total_ground_truth_caps', 0),
            'total_predicted_caps': metrics.get('total_predicted_caps', 0),
            'threshold': threshold,
            'notes': notes,
            'model_name': model_name,
            'prompt_text_hash': prompt_hash
        }
        
        # Add token usage data if provided
        if token_usage:
            api_calls = token_usage.get('api_calls', 0)
            total_tokens = token_usage.get('total_tokens', 0)
            total_cost = token_usage.get('total_cost', 0.0)
            
            # Calculate per-call metrics
            cost_per_call = total_cost / api_calls if api_calls > 0 else 0.0
            tokens_per_call = total_tokens / api_calls if api_calls > 0 else 0
            
            row_data.update({
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'api_calls': api_calls,
                'prompt_tokens': token_usage.get('prompt_tokens', 0),
                'completion_tokens': token_usage.get('completion_tokens', 0),
                'cost_per_call': cost_per_call,
                'tokens_per_call': tokens_per_call
            })
        else:
            # Add empty token usage columns to maintain consistent structure
            row_data.update({
                'total_tokens': 0,
                'total_cost': 0.0,
                'api_calls': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'cost_per_call': 0.0,
                'tokens_per_call': 0
            })
        
        # Create DataFrame
        run_df = pd.DataFrame([row_data])
        
        # Append to file with error handling and proper CSV quoting
        try:
            if not os.path.exists(self.runs_file):
                run_df.to_csv(self.runs_file, index=False, quoting=1)  # QUOTE_ALL
            else:
                run_df.to_csv(self.runs_file, mode='a', header=False, index=False, quoting=1)  # QUOTE_ALL
            
            logger.info(f"Logged run {run_index} for prompt {prompt_id}")
        except Exception as e:
            logger.error(f"Failed to write run result to CSV: {e}")
            logger.error(f"Run data: {row_data}")
            logger.warning("Continuing evaluation despite logging failure")
    
    def log_prompt_summary(self, prompt_id: str, all_run_metrics: List[Dict[str, float]], 
                          threshold: float, dataset: str = "test", prompt_text: str = "",
                          token_usage: dict = None, model_name: str = ""):
        """
        Log aggregated summary for a prompt.
        
        Args:
            prompt_id: Identifier for the prompt
            all_run_metrics: List of metrics from all runs
            threshold: Similarity threshold used
            dataset: Dataset name (train/dev/test)
            prompt_text: Full text of the prompt used
            token_usage: Dictionary with token usage data (optional)
            model_name: Name of the model used (e.g., 'o3', 'o4-mini')
        """
        # Extract metrics arrays
        precisions = [m['macro_precision'] for m in all_run_metrics]
        recalls = [m['macro_recall'] for m in all_run_metrics]
        f1s = [m['macro_f1'] for m in all_run_metrics]
        
        # Extract capability counts (use first run's data since they should be consistent)
        total_ground_truth_caps = all_run_metrics[0].get('total_ground_truth_caps', 0) if all_run_metrics else 0
        total_predicted_caps = all_run_metrics[0].get('total_predicted_caps', 0) if all_run_metrics else 0
        
        # Store prompt text and get hash
        prompt_hash = self.store_prompt_text(prompt_text) if prompt_text else ''
        
        # Compute statistics
        summary_data = {
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'prompt_id': prompt_id,
            'dataset': dataset,
            'num_runs': len(all_run_metrics),
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls),
            'mean_f1': np.mean(f1s),
            'std_precision': np.std(precisions),
            'std_recall': np.std(recalls),
            'std_f1': np.std(f1s),
            'variance_precision': np.var(precisions),
            'variance_recall': np.var(recalls),
            'variance_f1': np.var(f1s),
            'total_malformed_ground_truth': sum(m['malformed_ground_truth'] for m in all_run_metrics),
            'total_malformed_predictions': sum(m['malformed_predictions'] for m in all_run_metrics),
            'total_ground_truth_caps': total_ground_truth_caps,
            'total_predicted_caps': total_predicted_caps,
            'threshold': threshold,
            'prompt_text_hash': prompt_hash
        }
        
        # Add token usage data if provided
        if token_usage:
            api_calls = token_usage.get('api_calls', 0)
            total_tokens = token_usage.get('total_tokens', 0)
            total_cost = token_usage.get('total_cost', 0.0)
            
            # Calculate per-call metrics
            cost_per_call = total_cost / api_calls if api_calls > 0 else 0.0
            tokens_per_call = total_tokens / api_calls if api_calls > 0 else 0
            
            summary_data.update({
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'api_calls': api_calls,
                'prompt_tokens': token_usage.get('prompt_tokens', 0),
                'completion_tokens': token_usage.get('completion_tokens', 0),
                'model_name': model_name
            })
        else:
            # Add empty token usage columns to maintain consistent structure
            summary_data.update({
                'total_tokens': 0,
                'total_cost': 0.0,
                'api_calls': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'model_name': model_name
            })
        
        # Create DataFrame
        summary_df = pd.DataFrame([summary_data])
        
        # Append to file with error handling and proper CSV quoting
        try:
            if not os.path.exists(self.prompts_file):
                summary_df.to_csv(self.prompts_file, index=False, quoting=1)  # QUOTE_ALL
            else:
                summary_df.to_csv(self.prompts_file, mode='a', header=False, index=False, quoting=1)  # QUOTE_ALL
            
            logger.info(f"Logged prompt summary for {prompt_id}: F1={summary_data['mean_f1']:.4f}±{summary_data['std_f1']:.4f}")
        except Exception as e:
            logger.error(f"Failed to write prompt summary to CSV: {e}")
            logger.error(f"Summary data: {summary_data}")
            logger.warning("Continuing evaluation despite logging failure")


def split_test_data(test_df: pd.DataFrame, company_column: str = 'company_name', 
                   seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split test data 50/50 at company level into dev and final test sets.
    
    Args:
        test_df: Original test DataFrame
        company_column: Name of company identifier column
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (dev_df, final_test_df)
    """
    np.random.seed(seed)
    
    # Get unique companies
    companies = test_df[company_column].unique()
    np.random.shuffle(companies)
    
    # Split companies 50/50
    mid_point = len(companies) // 2
    dev_companies = companies[:mid_point]
    test_companies = companies[mid_point:]
    
    # Create splits
    dev_df = test_df[test_df[company_column].isin(dev_companies)].copy()
    final_test_df = test_df[test_df[company_column].isin(test_companies)].copy()
    
    logger.info(f"Split {len(companies)} companies: {len(dev_companies)} dev, {len(test_companies)} test")
    logger.info(f"Dev set: {len(dev_df)} ads, Test set: {len(final_test_df)} ads")
    
    return dev_df, final_test_df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Step 3 Evaluation: Semantic Evaluation of AI Capability Extraction")
    
    parser.add_argument("--train-file", type=str, default="Data/train_data_scored_3062025_updated.csv",
                       help="Path to training data file")
    
    parser.add_argument("--test-file", type=str, default="Data/test_data_scored_08012025.csv",
                       help="Path to original test data file")
    
    parser.add_argument("--prediction-column", type=str, default="predicted_json",
                       help="Name of column containing predictions")
    
    parser.add_argument("--prompt-id", type=str, default="baseline_v1",
                       help="Identifier for the prompt being evaluated")
    
    parser.add_argument("--num-runs", type=int, default=1,
                       help="Number of evaluation runs (default: 5)")
    
    parser.add_argument("--tune-threshold", action="store_true",
                       help="Tune similarity threshold on dev set")
    
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Similarity threshold (ignored if --tune-threshold is used)")
    
    parser.add_argument("--output-dir", type=str, default="Data",
                       help="Output directory for results")
    
    parser.add_argument("--notes", type=str, default="",
                       help="Optional notes for this evaluation run")
    
    parser.add_argument("--run-step1", action="store_true",
                       help="Run step 1 prompt to generate predictions (requires OpenAI API key)")
    
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini",
                       help="OpenAI model to use for step 1 prompt")
    
    parser.add_argument("--content-column", type=str, default="content_clean",
                       help="Column name containing job advertisement content")
    
    parser.add_argument("--datasets", type=str, nargs='+', choices=["train", "dev", "test"], default=["test"],
                       help="Which dataset(s) to evaluate: train, dev, test. Can specify multiple (e.g., --datasets train dev test)")
    
    # Ensemble arguments
    parser.add_argument("--use-ensemble", action="store_true",
                       help="Use ensemble of LLMs with majority voting (requires --run-step1)")
    
    parser.add_argument("--num-models", type=int, default=5,
                       help="Number of models in ensemble (default: 5, only used with --use-ensemble)")
    
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Temperature for ensemble diversity (default: 0.8, only used with --use-ensemble)")
    
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"],
                       help="Reasoning effort for o-series models (default: low)")
    
    return parser.parse_args()


def evaluate_single_dataset(dataset_df: pd.DataFrame, dataset_name: str, args, 
                           evaluator: 'SemanticEvaluator', tracker: 'PerformanceTracker',
                           step1_executor: Optional['Step1PromptExecutor'] = None,
                           pre_dataset_usage: dict = None) -> List[Dict[str, float]]:
    """
    Evaluate a single dataset.
    
    Args:
        dataset_df: DataFrame to evaluate
        dataset_name: Name of dataset (train/dev/test)
        args: Command line arguments
        evaluator: Semantic evaluator instance
        tracker: Performance tracker instance
        step1_executor: Step 1 executor for generating predictions (optional)
        
    Returns:
        List of metrics from all runs
    """
    logger.info(f"Evaluating {dataset_name} dataset ({len(dataset_df)} ads)")
    
    # Generate predictions if needed
    current_df = dataset_df.copy()
    if args.run_step1 and step1_executor:
        logger.info(f"Running step 1 prompt on {dataset_name} dataset")
        current_df = step1_executor.process_dataset(
            current_df, 
            args.content_column,
            use_ensemble=args.use_ensemble,
            num_models=args.num_models,
            temperature=args.temperature
        )
        
        # Save dataset with predictions
        pred_file = os.path.join(args.output_dir, f"{dataset_name}_with_predictions_{args.prompt_id}.csv")
        current_df.to_csv(pred_file, index=False)
        logger.info(f"Saved {dataset_name} set with predictions: {pred_file}")
    
    elif args.prediction_column not in current_df.columns:
        logger.error(f"Prediction column '{args.prediction_column}' not found in {dataset_name} data")
        logger.error("Either use --run-step1 to generate predictions, or ensure your data has the prediction column")
        logger.error(f"Available columns: {list(current_df.columns)}")
        return []
    
    # Get prompt text for logging if step1_executor is available
    prompt_text = ""
    if step1_executor:
        prompt_text = step1_executor.get_prompt_text()
    
    # Run multiple evaluations
    all_run_metrics = []
    
    for run_idx in range(1, args.num_runs + 1):
        logger.info(f"Starting {dataset_name} evaluation run {run_idx}/{args.num_runs}")
        
        # Evaluate dataset
        if run_idx == 1:  # Only collect detailed matches on first run to avoid duplicates
            metrics, detailed_matches, advert_scores = evaluator.evaluate_dataset_with_details(current_df, args.prediction_column)
            
            # Export detailed matches to CSV
            if detailed_matches:
                details_df = pd.DataFrame(detailed_matches)
                details_filename = f"detailed_matches_{dataset_name}_{args.prompt_id}_run{run_idx}.csv"
                details_path = os.path.join(args.output_dir, details_filename)
                details_df.to_csv(details_path, index=False)
                logger.info(f"Exported {len(detailed_matches)} detailed matches to {details_path}")
            
            # Export job-level summary to CSV
            if advert_scores:
                job_summary = []
                for idx, (row_idx, row) in enumerate(current_df.iterrows()):
                    if idx < len(advert_scores):
                        score = advert_scores[idx]
                        
                        # Extract capabilities for display
                        ground_truth_caps = evaluator.extract_ground_truth(row['Final'], row_idx, row.get('company_name', f'row_{row_idx}'))
                        predicted_caps = evaluator.extract_predictions(row[args.prediction_column], row_idx, row.get('company_name', f'row_{row_idx}'))
                        
                        job_summary.append({
                            'job_id': row.get('uid', f'job_{row_idx}'),
                            'company_id': row.get('company_name', f'row_{row_idx}'),
                            'job_title': row.get('title', 'N/A'),
                            'precision': score['precision'],
                            'recall': score['recall'],
                            'f1': score['f1'],
                            'num_ground_truth': len(ground_truth_caps),
                            'num_predictions': len(predicted_caps),
                            'threshold': evaluator.similarity_threshold,
                            'content_clean': row.get('content_clean', ''),
                            'translated_text': row.get('translated_text', ''),
                            'llm_output_full': row.get(args.prediction_column, ''),
                            'ground_truth_capabilities': ' | '.join(ground_truth_caps) if ground_truth_caps else '',
                            'predicted_capabilities': ' | '.join(predicted_caps) if predicted_caps else ''
                        })
                
                if job_summary:
                    job_summary_df = pd.DataFrame(job_summary)
                    job_summary_filename = f"job_summary_{dataset_name}_{args.prompt_id}_run{run_idx}.csv"
                    job_summary_path = os.path.join(args.output_dir, job_summary_filename)
                    job_summary_df.to_csv(job_summary_path, index=False)
                    logger.info(f"Exported {len(job_summary)} job-level summaries to {job_summary_path}")
        else:
            metrics = evaluator.evaluate_dataset(current_df, args.prediction_column)
            
        all_run_metrics.append(metrics)
        
        # Log run result with dataset info
        # Get dataset-specific token usage data from step1_executor
        token_usage = step1_executor.get_token_usage_delta(pre_dataset_usage) if step1_executor and pre_dataset_usage else None
        
        tracker.log_run_result(
            prompt_id=args.prompt_id,
            run_index=run_idx,
            metrics=metrics,
            threshold=evaluator.similarity_threshold,
            dataset=dataset_name,
            notes=args.notes,
            prompt_text=prompt_text,
            token_usage=token_usage,
            model_name=args.openai_model
        )
        
        # Write error logs
        evaluator.write_error_logs(args.output_dir)
    
    # Log prompt summary with dataset info
    # Get dataset-specific token usage data from step1_executor
    final_token_usage = step1_executor.get_token_usage_delta(pre_dataset_usage) if step1_executor and pre_dataset_usage else None
    tracker.log_prompt_summary(args.prompt_id, all_run_metrics, evaluator.similarity_threshold, dataset_name, prompt_text, final_token_usage, args.openai_model)
    
    return all_run_metrics


def main():
    """Main evaluation pipeline with multi-dataset support."""
    args = parse_arguments()
    
    logger.info("Starting Step 3 Evaluation: Semantic Evaluation of AI Capability Extraction")
    logger.info(f"Evaluating datasets: {', '.join(args.datasets)}")
    
    # Load data
    logger.info(f"Loading training data: {args.train_file}")
    train_df = pd.read_csv(args.train_file)
    
    logger.info(f"Loading test data: {args.test_file}")
    test_df = pd.read_csv(args.test_file)
    
    # Check if splits already exist to avoid data leakage
    dev_file = os.path.join(args.output_dir, "dev_data_split.csv")
    final_test_file = os.path.join(args.output_dir, "final_test_data_split.csv")
    
    if os.path.exists(dev_file) and os.path.exists(final_test_file):
        logger.info("Using existing dev/test splits to avoid data leakage")
        logger.info(f"Loading existing dev split: {dev_file}")
        logger.info(f"Loading existing test split: {final_test_file}")
        dev_df = pd.read_csv(dev_file)
        final_test_df = pd.read_csv(final_test_file)
        logger.info(f"Loaded dev set: {len(dev_df)} ads, test set: {len(final_test_df)} ads")
    else:
        logger.info("Creating new dev/test splits")
        # Split test data into dev and final test
        dev_df, final_test_df = split_test_data(test_df)
        
        # Save splits for reproducibility
        dev_df.to_csv(dev_file, index=False)
        final_test_df.to_csv(final_test_file, index=False)
        logger.info(f"Saved dev split: {dev_file}")
        logger.info(f"Saved final test split: {final_test_file}")
    
    # Prepare dataset mapping
    datasets = {
        'train': train_df,
        'dev': dev_df,
        'test': final_test_df
    }
    
    # Initialize step 1 executor if needed
    step1_executor = None
    if args.run_step1:
        logger.info("Initializing step 1 prompt executor")
        try:
            step1_executor = Step1PromptExecutor(model=args.openai_model, reasoning_effort=args.reasoning_effort)
        except Exception as e:
            logger.error(f"Failed to initialize step 1 prompt executor: {e}")
            logger.error("Make sure you have set OPENAI_API_KEY in your environment or config.env")
            return
    
    # Initialize evaluator
    evaluator = SemanticEvaluator(similarity_threshold=args.threshold)
    
    # Tune threshold if requested (use dev set for tuning)
    if args.tune_threshold:
        if 'dev' in args.datasets or len(args.datasets) > 1:
            logger.info("Tuning threshold on development set")
            tuner = ThresholdTuner(evaluator)
            
            # Generate predictions for dev set if needed for threshold tuning
            dev_for_tuning = dev_df.copy()
            if args.run_step1 and step1_executor:
                dev_for_tuning = step1_executor.process_dataset(dev_for_tuning, args.content_column)
            
            best_threshold = tuner.tune_threshold(dev_for_tuning, args.prediction_column)
            evaluator.similarity_threshold = best_threshold
            logger.info(f"Using tuned threshold: {best_threshold:.3f}")
        else:
            logger.warning("Threshold tuning requested but dev set not being evaluated. Using fixed threshold.")
            logger.info(f"Using fixed threshold: {args.threshold:.3f}")
    else:
        logger.info(f"Using fixed threshold: {args.threshold:.3f}")
    
    # Initialize performance tracker
    tracker = PerformanceTracker(args.output_dir)
    
    # Evaluate each requested dataset with dataset-specific token tracking
    all_results = {}
    for dataset_name in args.datasets:
        if dataset_name not in datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            continue
            
        dataset_df = datasets[dataset_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATING {dataset_name.upper()} DATASET")
        logger.info(f"{'='*60}")
        
        # Take snapshot of token usage before processing this dataset
        pre_dataset_usage = step1_executor.get_token_usage_summary() if step1_executor else None
        
        run_metrics = evaluate_single_dataset(
            dataset_df=dataset_df,
            dataset_name=dataset_name,
            args=args,
            evaluator=evaluator,
            tracker=tracker,
            step1_executor=step1_executor,
            pre_dataset_usage=pre_dataset_usage
        )
        
        all_results[dataset_name] = run_metrics
    
    # Print final summary for all datasets
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Prompt ID: {args.prompt_id}")
    print(f"Runs: {args.num_runs}")  
    print(f"Threshold: {evaluator.similarity_threshold:.3f}")
    print("-" * 60)
    
    for dataset_name in args.datasets:
        if dataset_name in all_results and all_results[dataset_name]:
            metrics = all_results[dataset_name]
            f1s = [m['macro_f1'] for m in metrics]
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            dataset_df = datasets[dataset_name]
            
            print(f"{dataset_name.upper()} SET:")
            print(f"  Size: {len(dataset_df)} ads")
            print(f"  Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    print("="*60)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()