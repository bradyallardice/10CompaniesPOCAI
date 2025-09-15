#!/usr/bin/env python3
"""
Step 3 Evaluation: Semantic Evaluation of AI Application Filtering (Step 3)

This module evaluates how well the LLM prompt in step 3 filters AI applications
to remove AI/ML boilerplate and identify specific vs. broad applications.

Based on the same methodology as step 1 and step 2 evaluation.
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


class Step3PromptExecutor:
    """
    Executes the exact step 3 filtering prompt from step_3_extract_ai_tasks.py
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4.1-mini", max_retries: int = 3, reasoning_effort: str = "low"):
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
    
    def get_step3_extractor(self):
        """
        Get an instance of the AITaskExtractor to use the step_3_filter_applications method
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
            The system prompt text used in step_3_filter_applications
        """
        try:
            extractor = self.get_step3_extractor()
            # Get the prompt text by inspecting the method source or calling a getter
            import inspect
            source = inspect.getsource(extractor.step_3_filter_applications)
            
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
    
    def execute_step3_prompt(self, separated_task: str) -> str:
        """
        Execute step 3 prompt on a separated task.
        
        Args:
            separated_task: The separated task from step 2 to filter
            
        Returns:
            Filtered application string or "N/A"
        """
        try:
            extractor = self.get_step3_extractor()
            
            # Use the step_3_filter_applications method from AITaskExtractor
            response_content = extractor.step_3_filter_applications(separated_task)
            
            if not response_content:
                logger.warning("No response from step 3 filtering")
                return "N/A"
            
            # Parse the JSON response
            try:
                parsed = json.loads(response_content)
                # Step 3 prompt returns 'ai_task', not 'filtered_application'
                filtered_app = parsed.get('ai_task', parsed.get('filtered_application', 'N/A'))
                return filtered_app if filtered_app else "N/A"
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse step 3 JSON response: {e}")
                return "N/A"
            
        except Exception as e:
            logger.error(f"Step 3 filtering failed: {e}")
            return "N/A"
    
    def process_dataset(self, df: pd.DataFrame, task_column: str = 'separated_task') -> pd.DataFrame:
        """
        Process entire dataset with step 3 prompt.
        
        Args:
            df: DataFrame with separated tasks to filter
            task_column: Column containing separated task descriptions
            
        Returns:
            DataFrame with added filtered_applications column
        """
        logger.info(f"Processing {len(df)} separated tasks with step 3 prompt")
        
        filtered_applications = []
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Processing task {idx + 1}/{len(df)}")
            
            # Get separated task content
            separated_task = row.get(task_column, '')
            if pd.isna(separated_task) or not separated_task:
                logger.warning(f"No task found for row {idx}")
                filtered_applications.append("N/A")
                continue
            
            # Execute step 3 prompt
            filtered_app = self.execute_step3_prompt(str(separated_task))
            filtered_applications.append(filtered_app)
            
            # Add small delay to respect rate limits
            time.sleep(0.5)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['filtered_applications'] = filtered_applications
        
        logger.info(f"Completed processing {len(df)} separated tasks")
        return result_df


class Step3SemanticEvaluator:
    """
    Semantic evaluator for AI application filtering evaluation.
    
    This evaluator handles the unique aspects of step 3:
    1. Many predictions will be "N/A" (filtered out as too broad)
    2. Ground truth may also contain "N/A" values
    3. Need to evaluate both filtering decisions (keep vs N/A) and content quality
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
        
        # Handle special case for N/A
        if sentence.lower() == 'n/a':
            return 'n/a'
        
        # Normalize Unicode characters
        import unicodedata
        sentence = unicodedata.normalize('NFKC', sentence)
        
        # Replace various types of hyphens and dashes with regular hyphen
        sentence = re.sub(r'[‐‑–—−]', '-', sentence)
        
        # Replace various quotes with regular quotes
        sentence = re.sub(r'[""''`]', '"', sentence)
        
        # Remove boilerplate tokens (step 3 should have already done this, but just in case)
        boilerplate_tokens = ['ai', 'ml', 'model', 'algorithm', 'machine learning', 
                            'artificial intelligence', 'deep learning']
        for token in boilerplate_tokens:
            sentence = re.sub(rf'\b{re.escape(token)}\b', '', sentence, flags=re.IGNORECASE)
        
        # Remove punctuation and squeeze whitespace
        sentence = re.sub(r'[^\w\s-]', ' ', sentence)  # Keep regular hyphens
        sentence = re.sub(r'\s+', ' ', sentence)
        
        return sentence.strip()
    
    def extract_ground_truth(self, ground_truth_str: str, row_idx: int, task_id: str) -> str:
        """
        Extract ground truth filtered application.
        
        Args:
            ground_truth_str: Ground truth string (could be JSON or plain text)
            row_idx: Row index for logging
            task_id: Task ID for logging
            
        Returns:
            Ground truth filtered application string
        """
        try:
            if pd.isna(ground_truth_str) or not ground_truth_str:
                return "n/a"
            
            # Handle different formats
            ground_truth_str = str(ground_truth_str).strip()
            
            # Try parsing as JSON first
            try:
                ground_truth_data = json.loads(ground_truth_str)
                if isinstance(ground_truth_data, dict):
                    # Extract ai_task key (step 3 uses ai_task, not filtered_application)
                    filtered_app = ground_truth_data.get('ai_task', ground_truth_data.get('filtered_application', 'N/A'))
                    return self._normalize_unicode_string(filtered_app.strip()) if filtered_app else "n/a"
                elif isinstance(ground_truth_data, str):
                    return self._normalize_unicode_string(ground_truth_data.strip())
            except json.JSONDecodeError:
                pass
            
            # If not JSON, treat as plain text
            normalized = self._normalize_unicode_string(ground_truth_str)
            return normalized if normalized else "n/a"
            
        except Exception as e:
            self.bad_rows_ground_truth.append({
                'row_index': row_idx,
                'task_id': task_id,
                'error': str(e)
            })
            logger.warning(f"Failed to parse ground truth at row {row_idx}: {e}")
            return "n/a"
    
    def extract_predictions(self, predicted_str: str, row_idx: int, task_id: str) -> str:
        """
        Extract predictions from the filtered_applications column.
        
        Args:
            predicted_str: Prediction string from model
            row_idx: Row index for logging
            task_id: Task ID for logging
            
        Returns:
            Predicted filtered application string
        """
        try:
            if pd.isna(predicted_str) or not predicted_str:
                return "n/a"
            
            # Handle different formats
            predicted_str = str(predicted_str).strip()
            
            # Try parsing as JSON first (if step 3 returns JSON format)
            try:
                pred_data = json.loads(predicted_str)
                if isinstance(pred_data, dict):
                    filtered_app = pred_data.get('ai_task', pred_data.get('filtered_application', 'N/A'))
                    return self._normalize_unicode_string(filtered_app.strip()) if filtered_app else "n/a"
                elif isinstance(pred_data, str):
                    return self._normalize_unicode_string(pred_data.strip())
            except json.JSONDecodeError:
                pass
            
            # If not JSON, treat as plain text
            normalized = self._normalize_unicode_string(predicted_str)
            return normalized if normalized else "n/a"
            
        except Exception as e:
            self.bad_rows_predictions.append({
                'row_index': row_idx,
                'task_id': task_id,
                'error': str(e)
            })
            logger.warning(f"Failed to parse predictions at row {row_idx}: {e}")
            return "n/a"
    
    def compute_step3_similarity(self, ground_truth: str, prediction: str) -> float:
        """
        Compute similarity between ground truth and prediction for step 3.
        
        Special handling for N/A values:
        - If both are N/A: similarity = 1.0 (perfect match)
        - If one is N/A and other isn't: similarity = 0.0 (complete mismatch)
        - If neither is N/A: use semantic similarity
        
        Args:
            ground_truth: Ground truth filtered application
            prediction: Predicted filtered application
            
        Returns:
            Similarity score between 0 and 1
        """
        # Normalize both values
        gt_norm = self.normalize_sentence(ground_truth)
        pred_norm = self.normalize_sentence(prediction)
        
        # Handle N/A cases
        gt_is_na = (gt_norm.lower() in ['n/a', 'na', ''])
        pred_is_na = (pred_norm.lower() in ['n/a', 'na', ''])
        
        if gt_is_na and pred_is_na:
            return 1.0  # Both correctly identified as N/A
        elif gt_is_na or pred_is_na:
            return 0.0  # One is N/A, other isn't - complete mismatch
        
        # Neither is N/A - compute semantic similarity
        if not gt_norm or not pred_norm:
            return 0.0
        
        try:
            # Embed both texts
            embeddings = self.model.encode([gt_norm, pred_norm],
                                         batch_size=self.batch_size,
                                         normalize_embeddings=True,
                                         convert_to_numpy=True)
            
            # Compute cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            return 0.0
    
    def evaluate_single_task(self, ground_truth: str, prediction: str) -> Dict[str, float]:
        """
        Evaluate a single task filtering.
        
        Args:
            ground_truth: Ground truth filtered application
            prediction: Predicted filtered application
            
        Returns:
            Dictionary with evaluation metrics
        """
        similarity = self.compute_step3_similarity(ground_truth, prediction)
        
        # For step 3, we use a binary classification approach:
        # TP: Correctly matched (similarity >= threshold)
        # FP: Incorrectly predicted as specific when should be N/A, or wrong content
        # FN: Should have been specific but predicted as N/A, or missed correct content
        
        is_match = similarity >= self.similarity_threshold
        
        if is_match:
            # True positive
            return {
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'similarity': similarity,
                'match_type': 'TP'
            }
        else:
            # Either false positive or false negative
            gt_norm = self.normalize_sentence(ground_truth)
            pred_norm = self.normalize_sentence(prediction)
            
            gt_is_na = (gt_norm.lower() in ['n/a', 'na', ''])
            pred_is_na = (pred_norm.lower() in ['n/a', 'na', ''])
            
            if gt_is_na and not pred_is_na:
                # False positive (predicted specific when should be N/A)
                match_type = 'FP'
            elif not gt_is_na and pred_is_na:
                # False negative (predicted N/A when should be specific)
                match_type = 'FN'
            else:
                # Both are specific but don't match well enough
                match_type = 'FP'  # Consider as false positive
            
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'similarity': similarity,
                'match_type': match_type
            }
    
    def evaluate_dataset(self, df: pd.DataFrame, 
                        prediction_column: str = 'filtered_applications',
                        ground_truth_column: str = 'ground_truth_filtered_applications') -> Dict[str, float]:
        """
        Evaluate an entire dataset.
        
        Args:
            df: DataFrame with ground truth and prediction columns
            prediction_column: Name of column containing predictions
            ground_truth_column: Name of column containing ground truth
            
        Returns:
            Dictionary with macro-averaged metrics
        """
        logger.info(f"Evaluating {len(df)} filtered applications")
        
        # Reset error logs
        self.bad_rows_ground_truth = []
        self.bad_rows_predictions = []
        
        task_scores = []
        total_ground_truth_specific = 0
        total_predicted_specific = 0
        total_ground_truth_na = 0
        total_predicted_na = 0
        
        for idx, row in df.iterrows():
            # Extract identifiers
            task_id = row.get('separated_task', f'row_{idx}')
            
            ground_truth = self.extract_ground_truth(row[ground_truth_column], idx, task_id)
            prediction = self.extract_predictions(row[prediction_column], idx, task_id)
            
            # Count specific vs N/A
            gt_is_na = (ground_truth.lower() in ['n/a', 'na', ''])
            pred_is_na = (prediction.lower() in ['n/a', 'na', ''])
            
            if gt_is_na:
                total_ground_truth_na += 1
            else:
                total_ground_truth_specific += 1
                
            if pred_is_na:
                total_predicted_na += 1
            else:
                total_predicted_specific += 1
            
            # Evaluate single task
            scores = self.evaluate_single_task(ground_truth, prediction)
            task_scores.append(scores)
        
        # Macro-average across all tasks
        macro_precision = np.mean([s['precision'] for s in task_scores])
        macro_recall = np.mean([s['recall'] for s in task_scores])
        macro_f1 = np.mean([s['f1'] for s in task_scores])
        average_similarity = np.mean([s['similarity'] for s in task_scores])
        
        # Count match types
        tp_count = sum(1 for s in task_scores if s['match_type'] == 'TP')
        fp_count = sum(1 for s in task_scores if s['match_type'] == 'FP')
        fn_count = sum(1 for s in task_scores if s['match_type'] == 'FN')
        
        logger.info(f"Evaluation complete: P={macro_precision:.4f}, R={macro_recall:.4f}, F1={macro_f1:.4f}")
        logger.info(f"Average similarity: {average_similarity:.4f}")
        logger.info(f"Malformed ground truth rows: {len(self.bad_rows_ground_truth)}")
        logger.info(f"Malformed prediction rows: {len(self.bad_rows_predictions)}")
        logger.info(f"Ground truth: {total_ground_truth_specific} specific, {total_ground_truth_na} N/A")
        logger.info(f"Predictions: {total_predicted_specific} specific, {total_predicted_na} N/A")
        logger.info(f"Matches: {tp_count} TP, {fp_count} FP, {fn_count} FN")
        
        return {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'average_similarity': average_similarity,
            'malformed_ground_truth': len(self.bad_rows_ground_truth),
            'malformed_predictions': len(self.bad_rows_predictions),
            'total_tasks': len(df),
            'ground_truth_specific': total_ground_truth_specific,
            'ground_truth_na': total_ground_truth_na,
            'predicted_specific': total_predicted_specific,
            'predicted_na': total_predicted_na,
            'tp_count': tp_count,
            'fp_count': fp_count,
            'fn_count': fn_count
        }
    
    def write_error_logs(self, output_dir: str = "Data"):
        """
        Write error logs to files.
        
        Args:
            output_dir: Directory to write log files
        """
        # Write ground truth errors
        if self.bad_rows_ground_truth:
            gt_log_file = os.path.join(output_dir, "bad_rows_ground_truth_step3.log")
            with open(gt_log_file, 'w') as f:
                f.write("Row Index,Task ID,Error\n")
                for error in self.bad_rows_ground_truth:
                    f.write(f"{error['row_index']},{error['task_id']},\"{error['error']}\"\n")
            logger.info(f"Wrote ground truth error log: {gt_log_file}")
        
        # Write prediction errors
        if self.bad_rows_predictions:
            pred_log_file = os.path.join(output_dir, "bad_rows_predictions_step3.log")
            with open(pred_log_file, 'w') as f:
                f.write("Row Index,Task ID,Error\n")
                for error in self.bad_rows_predictions:
                    f.write(f"{error['row_index']},{error['task_id']},\"{error['error']}\"\n")
            logger.info(f"Wrote predictions error log: {pred_log_file}")


class Step3PerformanceTracker:
    """
    Tracks performance history for step 3 evaluation.
    """
    
    def __init__(self, output_dir: str = "Data"):
        self.output_dir = output_dir
        self.runs_file = os.path.join(output_dir, "performance_history_runs_step3.csv")
        self.prompts_file = os.path.join(output_dir, "performance_history_prompts_step3.csv")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
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
        prompt_lookup_file = os.path.join(self.output_dir, "prompt_text_lookup_step3.csv")
        
        # Check if this hash already exists
        if os.path.exists(prompt_lookup_file):
            try:
                existing_df = pd.read_csv(prompt_lookup_file)
                if prompt_hash in existing_df['prompt_hash'].values:
                    # Hash already exists, no need to store again
                    return prompt_hash
            except Exception as e:
                logger.warning(f"Could not read existing prompt_text_lookup_step3.csv: {e}")
        
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
            
            logger.info(f"Stored step 3 prompt text with hash {prompt_hash}")
            return prompt_hash
        except Exception as e:
            logger.error(f"Failed to write step 3 prompt text to CSV: {e}")
            logger.warning("Continuing without storing prompt text")
            return prompt_hash
    
    def log_run_result(self, prompt_id: str, run_index: int, metrics: Dict[str, float], 
                      threshold: float, dataset: str = "test", notes: str = "", prompt_text: str = "",
                      token_usage: dict = None, model_name: str = ""):
        """
        Log a single run result for step 3.
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
            'average_similarity': metrics['average_similarity'],
            'malformed_ground_truth': metrics['malformed_ground_truth'],
            'malformed_predictions': metrics['malformed_predictions'],
            'ground_truth_specific': metrics.get('ground_truth_specific', 0),
            'ground_truth_na': metrics.get('ground_truth_na', 0),
            'predicted_specific': metrics.get('predicted_specific', 0),
            'predicted_na': metrics.get('predicted_na', 0),
            'tp_count': metrics.get('tp_count', 0),
            'fp_count': metrics.get('fp_count', 0),
            'fn_count': metrics.get('fn_count', 0),
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
            
            logger.info(f"Logged step 3 run {run_index} for prompt {prompt_id}")
        except Exception as e:
            logger.error(f"Failed to write step 3 run result to CSV: {e}")
            logger.warning("Continuing evaluation despite logging failure")
    
    def log_prompt_summary(self, prompt_id: str, all_run_metrics: List[Dict[str, float]], 
                          threshold: float, dataset: str = "test", prompt_text: str = "",
                          token_usage: dict = None, model_name: str = ""):
        """
        Log aggregated summary for a step 3 prompt.
        """
        # Extract metrics arrays
        precisions = [m['macro_precision'] for m in all_run_metrics]
        recalls = [m['macro_recall'] for m in all_run_metrics]
        f1s = [m['macro_f1'] for m in all_run_metrics]
        similarities = [m['average_similarity'] for m in all_run_metrics]
        
        # Extract counts (use first run's data since they should be consistent)
        first_run = all_run_metrics[0] if all_run_metrics else {}
        
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
            'mean_similarity': np.mean(similarities),
            'std_precision': np.std(precisions),
            'std_recall': np.std(recalls),
            'std_f1': np.std(f1s),
            'std_similarity': np.std(similarities),
            'total_malformed_ground_truth': sum(m['malformed_ground_truth'] for m in all_run_metrics),
            'total_malformed_predictions': sum(m['malformed_predictions'] for m in all_run_metrics),
            'ground_truth_specific': first_run.get('ground_truth_specific', 0),
            'ground_truth_na': first_run.get('ground_truth_na', 0),
            'predicted_specific': first_run.get('predicted_specific', 0),
            'predicted_na': first_run.get('predicted_na', 0),
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
            
            logger.info(f"Logged step 3 prompt summary for {prompt_id}: F1={summary_data['mean_f1']:.4f}±{summary_data['std_f1']:.4f}")
        except Exception as e:
            logger.error(f"Failed to write step 3 prompt summary to CSV: {e}")
            logger.warning("Continuing evaluation despite logging failure")


def extract_separated_tasks_from_step2_results(step2_results_file: str, output_file: str):
    """
    Extract separated tasks from step 2 results to create step 3 evaluation dataset.
    
    Args:
        step2_results_file: Path to step 2 results CSV
        output_file: Path for output CSV
        
    Returns:
        DataFrame with separated tasks for step 3 evaluation
    """
    logger.info(f"Loading step 2 results: {step2_results_file}")
    
    try:
        df = pd.read_csv(step2_results_file)
        logger.info(f"Loaded {len(df)} step 2 results")
    except Exception as e:
        logger.error(f"Error loading step 2 results: {e}")
        return None
    
    # Extract separated tasks from the step 2 results
    # Expected format: each row has a 'predicted_separated_tasks' column with JSON array
    all_tasks = []
    
    for idx, row in df.iterrows():
        try:
            # Get separated tasks (JSON format)
            separated_tasks_str = row.get('predicted_separated_tasks', '[]')
            separated_tasks = json.loads(separated_tasks_str)
            
            # Create one row per separated task
            for task in separated_tasks:
                task_data = {
                    'original_ai_capability': row.get('ai_capability', ''),
                    'separated_task': task,
                    'original_uid': row.get('original_uid', ''),
                    'company_name': row.get('company_name', ''),
                    'job_title': row.get('job_title', ''),
                    'ground_truth_filtered_applications': ''  # Empty for manual filling
                }
                all_tasks.append(task_data)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse separated tasks for row {idx}: {e}")
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
    
    # Create dataset
    step3_df = pd.DataFrame(all_tasks)
    
    # Save to CSV
    step3_df.to_csv(output_file, index=False)
    logger.info(f"Created step 3 evaluation dataset: {output_file}")
    logger.info(f"Dataset contains {len(step3_df)} separated tasks")
    
    # Print some examples
    logger.info("Example separated tasks:")
    for i, task in enumerate(step3_df['separated_task'].head(5)):
        logger.info(f"{i+1}. {task}")
    
    return step3_df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Step 3 Evaluation: Semantic Evaluation of AI Application Filtering")
    
    # Mode selection
    parser.add_argument("--create-dataset", action="store_true",
                       help="Create step 3 evaluation dataset from step 2 results")
    
    parser.add_argument("--step2-results-file", type=str,
                       help="Path to step 2 results CSV file (for --create-dataset)")
    
    # Evaluation mode
    parser.add_argument("--input-file", type=str,
                       help="Path to input file with separated tasks and ground truth filtered applications")
    
    parser.add_argument("--task-column", type=str, default="separated_task",
                       help="Name of column containing separated task descriptions")
    
    parser.add_argument("--ground-truth-column", type=str, default="ground_truth_filtered_applications",
                       help="Name of column containing ground truth filtered applications")
    
    parser.add_argument("--prediction-column", type=str, default="filtered_applications",
                       help="Name of column containing predictions")
    
    parser.add_argument("--prompt-id", type=str, default="step3_baseline_v1",
                       help="Identifier for the prompt being evaluated")
    
    parser.add_argument("--num-runs", type=int, default=5,
                       help="Number of evaluation runs (default: 5)")
    
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Similarity threshold for matching")
    
    parser.add_argument("--output-dir", type=str, default="Data",
                       help="Output directory for results")
    
    parser.add_argument("--notes", type=str, default="",
                       help="Optional notes for this evaluation run")
    
    # Step 3 execution
    parser.add_argument("--run-step3", action="store_true",
                       help="Run step 3 prompt to generate predictions (requires OpenAI API key)")
    
    parser.add_argument("--openai-model", type=str, default="gpt-4.1-mini",
                       help="OpenAI model to use for step 3 prompt")
    
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"],
                       help="Reasoning effort for o-series models (default: low)")
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline for step 3."""
    args = parse_arguments()
    
    logger.info("Starting Step 3 Evaluation: Semantic Evaluation of AI Application Filtering")
    
    # Mode 1: Create step 3 evaluation dataset from step 2 results
    if args.create_dataset:
        if not args.step2_results_file:
            logger.error("--step2-results-file is required for --create-dataset")
            return
        
        output_file = os.path.join(args.output_dir, f"step3_tasks_for_eval.csv")
        
        dataset = extract_separated_tasks_from_step2_results(args.step2_results_file, output_file)
        
        if dataset is not None:
            logger.info("✅ Step 3 evaluation dataset created successfully!")
            logger.info(f"📁 Dataset: {output_file}")
            logger.info("")
            logger.info("📝 Next steps:")
            logger.info("1. Fill in the ground_truth_filtered_applications column")
            logger.info("2. Use the completed dataset for step 3 evaluation")
        else:
            logger.error("❌ Failed to create step 3 evaluation dataset")
        
        return
    
    # Mode 2: Evaluation mode (requires input file)
    if not args.input_file:
        logger.error("Either use --create-dataset to create a dataset, or provide --input-file for evaluation")
        return
    
    # Load data for evaluation
    logger.info(f"Loading input data: {args.input_file}")
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df)} separated tasks for evaluation")
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return
    
    # Initialize step 3 executor if needed
    step3_executor = None
    if args.run_step3:
        logger.info("Initializing step 3 prompt executor")
        try:
            step3_executor = Step3PromptExecutor(model=args.openai_model, reasoning_effort=args.reasoning_effort)
        except Exception as e:
            logger.error(f"Failed to initialize step 3 prompt executor: {e}")
            logger.error("Make sure you have set OPENAI_API_KEY in your environment or config.env")
            return
    
    # Generate predictions if needed
    current_df = df.copy()
    if args.run_step3 and step3_executor:
        logger.info("Running step 3 prompt on dataset")
        current_df = step3_executor.process_dataset(current_df, args.task_column)
        
        # Save dataset with predictions
        pred_file = os.path.join(args.output_dir, f"tasks_with_step3_predictions_{args.prompt_id}.csv")
        current_df.to_csv(pred_file, index=False)
        logger.info(f"Saved dataset with step 3 predictions: {pred_file}")
    
    elif args.prediction_column not in current_df.columns:
        logger.error(f"Prediction column '{args.prediction_column}' not found in data")
        logger.error("Either use --run-step3 to generate predictions, or ensure your data has the prediction column")
        logger.error(f"Available columns: {list(current_df.columns)}")
        return
    
    # Check for ground truth column
    if args.ground_truth_column not in current_df.columns:
        logger.error(f"Ground truth column '{args.ground_truth_column}' not found in data")
        logger.error("Make sure you have filled in the ground truth column")
        logger.error(f"Available columns: {list(current_df.columns)}")
        return
    
    # Initialize evaluator
    evaluator = Step3SemanticEvaluator(similarity_threshold=args.threshold)
    
    # Initialize performance tracker
    tracker = Step3PerformanceTracker(args.output_dir)
    
    # Get prompt text for logging if step3_executor is available
    prompt_text = ""
    if step3_executor:
        prompt_text = step3_executor.get_prompt_text()
    
    # Run multiple evaluations
    all_run_metrics = []
    
    for run_idx in range(1, args.num_runs + 1):
        logger.info(f"Starting evaluation run {run_idx}/{args.num_runs}")
        
        # Evaluate dataset
        metrics = evaluator.evaluate_dataset(current_df, args.prediction_column, args.ground_truth_column)
        all_run_metrics.append(metrics)
        
        # Log run result
        token_usage = step3_executor.get_token_usage_summary() if step3_executor else None
        
        tracker.log_run_result(
            prompt_id=args.prompt_id,
            run_index=run_idx,
            metrics=metrics,
            threshold=evaluator.similarity_threshold,
            dataset="test",
            notes=args.notes,
            prompt_text=prompt_text,
            token_usage=token_usage,
            model_name=args.openai_model
        )
        
        # Write error logs
        evaluator.write_error_logs(args.output_dir)
    
    # Log prompt summary
    final_token_usage = step3_executor.get_token_usage_summary() if step3_executor else None
    tracker.log_prompt_summary(args.prompt_id, all_run_metrics, evaluator.similarity_threshold, "test", prompt_text, final_token_usage, args.openai_model)
    
    # Print final summary
    print("\n" + "="*60)
    print("STEP 3 EVALUATION SUMMARY")
    print("="*60)
    print(f"Prompt ID: {args.prompt_id}")
    print(f"Runs: {args.num_runs}")  
    print(f"Threshold: {evaluator.similarity_threshold:.3f}")
    print("-" * 60)
    
    f1s = [m['macro_f1'] for m in all_run_metrics]
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    
    similarities = [m['average_similarity'] for m in all_run_metrics]
    mean_similarity = np.mean(similarities)
    
    print(f"Dataset: {len(current_df)} separated tasks")
    print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Mean Similarity: {mean_similarity:.4f}")
    
    # Show filtering statistics
    first_run = all_run_metrics[0]
    print(f"Ground Truth: {first_run['ground_truth_specific']} specific, {first_run['ground_truth_na']} N/A")
    print(f"Predictions: {first_run['predicted_specific']} specific, {first_run['predicted_na']} N/A")
    
    print("="*60)
    logger.info("Step 3 evaluation complete!")


if __name__ == "__main__":
    main()