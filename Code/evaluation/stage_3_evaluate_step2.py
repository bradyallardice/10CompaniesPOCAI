#!/usr/bin/env python3
"""
Step 3 Evaluation: Semantic Evaluation of AI Task Separation (Step 2)

This module evaluates how well the LLM prompt in step 2 separates compound AI tasks
into distinct work tasks using semantic similarity matching against ground truth labels.

Based on the same methodology as step 1 evaluation.
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


class Step2PromptExecutor:
    """
    Executes the exact step 2 task separation prompt from step_3_extract_ai_tasks.py
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", max_retries: int = 3, reasoning_effort: str = "low", use_flex: bool = False):
        """
        Initialize the prompt executor with OpenAI API.
        
        Args:
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            model: OpenAI model to use
            max_retries: Maximum number of retries for failed calls
            reasoning_effort: Reasoning effort for o-series models (low, medium, high)
            use_flex: Use flex processing for cost savings (supported models only)
        """
        self.model = model
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.use_flex = use_flex
        
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
    
    def get_step2_extractor(self):
        """
        Get an instance of the AITaskExtractor to use the step_2_separate_tasks method
        Reuses the same instance to accumulate token usage across all calls.
        
        Returns:
            AITaskExtractor instance
        """
        if self.extractor is None:
            try:
                from stage_3_extract_ai_tasks import AITaskExtractor
                self.extractor = AITaskExtractor(model_name=self.model, reasoning_effort=self.reasoning_effort, use_flex=self.use_flex)
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
            The system prompt text used in step_2_separate_tasks
        """
        try:
            extractor = self.get_step2_extractor()
            # Get the prompt text by inspecting the method source or calling a getter
            import inspect
            source = inspect.getsource(extractor.step_2_separate_tasks)
            
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
    
    def execute_step2_prompt(self, ai_capability: str) -> List[str]:
        """
        Execute step 2 prompt on an AI capability description.
        
        Args:
            ai_capability: The AI capability description to separate into tasks
            
        Returns:
            List of separated task strings
        """
        try:
            extractor = self.get_step2_extractor()
            
            # Use the step_2_separate_tasks method from AITaskExtractor
            response_content = extractor.step_2_separate_tasks(ai_capability)
            
            if not response_content:
                logger.warning("No response from step 2 separation")
                return []
            
            # Parse the response (plain text, one task per line)
            separated_tasks = response_content.strip().split('\n')
            separated_tasks = [task.strip() for task in separated_tasks if task.strip()]
            
            # Add small delay to respect rate limits
            time.sleep(0.5)
            
            return separated_tasks
            
        except Exception as e:
            logger.error(f"Step 2 separation failed: {e}")
            return []
    
    def process_dataset(self, df: pd.DataFrame, capability_column: str = 'ai_capability') -> pd.DataFrame:
        """
        Process entire dataset with step 2 prompt.
        
        Args:
            df: DataFrame with AI capabilities to separate
            capability_column: Column containing AI capability descriptions
            
        Returns:
            DataFrame with added separated_tasks column (JSON format)
        """
        logger.info(f"Processing {len(df)} AI capabilities with step 2 prompt")
        
        predictions = []
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                logger.info(f"Processing capability {idx + 1}/{len(df)}")
            
            # Get AI capability content
            ai_capability = row.get(capability_column, '')
            if pd.isna(ai_capability) or not ai_capability:
                logger.warning(f"No capability found for row {idx}")
                predictions.append(json.dumps([]))
                continue
            
            # Execute step 2 prompt
            separated_tasks = self.execute_step2_prompt(str(ai_capability))
            
            # Store as JSON
            predictions.append(json.dumps(separated_tasks, ensure_ascii=False))
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['predicted_separated_tasks'] = predictions
        
        logger.info(f"Completed processing {len(df)} AI capabilities")
        return result_df


class Step2SemanticEvaluator:
    """
    Semantic evaluator for AI task separation using BGE embeddings and similarity matching.
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
        
        # Normalize Unicode characters
        import unicodedata
        sentence = unicodedata.normalize('NFKC', sentence)
        
        # Replace various types of hyphens and dashes with regular hyphen
        sentence = re.sub(r'[‐‑–—−]', '-', sentence)
        
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
    
    def extract_ground_truth(self, ground_truth_str: str, row_idx: int, capability_id: str) -> List[str]:
        """
        Extract ground truth separated tasks from the ground truth column.
        Expected format: JSON array of task strings
        
        Args:
            ground_truth_str: JSON string or plain text from ground truth column
            row_idx: Row index for logging
            capability_id: Capability ID for logging
            
        Returns:
            List of ground truth task strings
        """
        try:
            if pd.isna(ground_truth_str) or not ground_truth_str:
                return []
            
            # Try parsing as JSON first
            try:
                ground_truth_data = json.loads(ground_truth_str)
                if isinstance(ground_truth_data, list):
                    # Direct list of tasks
                    tasks = []
                    for task in ground_truth_data:
                        if isinstance(task, str) and task.strip():
                            normalized_task = self._normalize_unicode_string(task.strip())
                            tasks.append(normalized_task)
                    return tasks
            except json.JSONDecodeError:
                pass
            
            # If not JSON, treat as plain text with tasks separated by newlines
            tasks = []
            lines = str(ground_truth_str).strip().split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    normalized_task = self._normalize_unicode_string(line)
                    tasks.append(normalized_task)
            
            return tasks
            
        except Exception as e:
            self.bad_rows_ground_truth.append({
                'row_index': row_idx,
                'capability_id': capability_id,
                'error': str(e)
            })
            logger.warning(f"Failed to parse ground truth at row {row_idx}: {e}")
            return []
    
    def extract_predictions(self, predicted_json_str: str, row_idx: int, capability_id: str) -> List[str]:
        """
        Extract predictions from the separated_tasks column.
        
        Args:
            predicted_json_str: JSON string from prediction column
            row_idx: Row index for logging
            capability_id: Capability ID for logging
            
        Returns:
            List of predicted task strings
        """
        try:
            if pd.isna(predicted_json_str) or not predicted_json_str:
                return []
            
            # Parse JSON
            pred_data = json.loads(predicted_json_str)
            
            if not isinstance(pred_data, list):
                self.bad_rows_predictions.append({
                    'row_index': row_idx,
                    'capability_id': capability_id,
                    'error': 'Prediction is not a list'
                })
                return []
            
            # Filter to valid strings with Unicode normalization
            valid_tasks = []
            for task in pred_data:
                if isinstance(task, str) and task.strip():
                    # Apply same Unicode normalization as ground truth
                    normalized_task = self._normalize_unicode_string(task.strip())
                    valid_tasks.append(normalized_task)
            
            return valid_tasks
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.bad_rows_predictions.append({
                'row_index': row_idx,
                'capability_id': capability_id,
                'error': str(e)
            })
            logger.warning(f"Failed to parse predictions JSON at row {row_idx}: {e}")
            return []
    
    def compute_similarity_matrix(self, gold_tasks: List[str], pred_tasks: List[str]) -> np.ndarray:
        """
        Compute similarity matrix between gold and predicted tasks.
        
        Args:
            gold_tasks: Ground truth tasks
            pred_tasks: Predicted tasks
            
        Returns:
            m x n similarity matrix where cell (i,j) is similarity between pred_i and gold_j
        """
        if not gold_tasks or not pred_tasks:
            return np.array([]).reshape(len(pred_tasks), len(gold_tasks))
        
        # Embed both lists
        gold_embeddings = self.model.encode(gold_tasks,
                                          batch_size=self.batch_size,
                                          normalize_embeddings=True,
                                          convert_to_numpy=True)
        
        pred_embeddings = self.model.encode(pred_tasks,
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
    
    def evaluate_single_capability(self, gold_tasks: List[str], pred_tasks: List[str]) -> Dict[str, float]:
        """
        Evaluate a single AI capability task separation.
        
        Args:
            gold_tasks: Ground truth separated tasks
            pred_tasks: Predicted separated tasks
            
        Returns:
            Dictionary with precision, recall, f1 scores
        """
        # Normalize both lists
        gold_normalized = [self.normalize_sentence(task) for task in gold_tasks]
        pred_normalized = [self.normalize_sentence(task) for task in pred_tasks]
        
        # Remove empty strings
        gold_normalized = [task for task in gold_normalized if task]
        pred_normalized = [task for task in pred_normalized if task]
        
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
    
    def evaluate_dataset(self, df: pd.DataFrame, 
                        prediction_column: str = 'separated_tasks',
                        ground_truth_column: str = 'ground_truth_separated_tasks') -> Dict[str, float]:
        """
        Evaluate an entire dataset.
        
        Args:
            df: DataFrame with ground truth and prediction columns
            prediction_column: Name of column containing predictions
            ground_truth_column: Name of column containing ground truth
            
        Returns:
            Dictionary with macro-averaged metrics
        """
        logger.info(f"Evaluating {len(df)} AI capabilities for task separation")
        
        # Reset error logs
        self.bad_rows_ground_truth = []
        self.bad_rows_predictions = []
        
        capability_scores = []
        total_ground_truth_tasks = 0
        total_predicted_tasks = 0
        
        for idx, row in df.iterrows():
            # Extract identifiers
            capability_id = row.get('ai_capability', f'row_{idx}')
            
            gold_tasks = self.extract_ground_truth(row[ground_truth_column], idx, capability_id)
            pred_tasks = self.extract_predictions(row[prediction_column], idx, capability_id)
            
            # Count tasks
            total_ground_truth_tasks += len(gold_tasks)
            total_predicted_tasks += len(pred_tasks)
            
            # Evaluate single capability
            scores = self.evaluate_single_capability(gold_tasks, pred_tasks)
            capability_scores.append(scores)
        
        # Macro-average across all capabilities
        macro_precision = np.mean([s['precision'] for s in capability_scores])
        macro_recall = np.mean([s['recall'] for s in capability_scores])
        macro_f1 = np.mean([s['f1'] for s in capability_scores])
        
        logger.info(f"Evaluation complete: P={macro_precision:.4f}, R={macro_recall:.4f}, F1={macro_f1:.4f}")
        logger.info(f"Malformed ground truth rows: {len(self.bad_rows_ground_truth)}")
        logger.info(f"Malformed prediction rows: {len(self.bad_rows_predictions)}")
        logger.info(f"Total ground truth tasks: {total_ground_truth_tasks}")
        logger.info(f"Total predicted tasks: {total_predicted_tasks}")
        
        return {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'malformed_ground_truth': len(self.bad_rows_ground_truth),
            'malformed_predictions': len(self.bad_rows_predictions),
            'total_capabilities': len(df),
            'total_ground_truth_tasks': total_ground_truth_tasks,
            'total_predicted_tasks': total_predicted_tasks
        }
    
    def write_error_logs(self, output_dir: str = "Data"):
        """
        Write error logs to files.
        
        Args:
            output_dir: Directory to write log files
        """
        # Write ground truth errors
        if self.bad_rows_ground_truth:
            gt_log_file = os.path.join(output_dir, "bad_rows_ground_truth_step2.log")
            with open(gt_log_file, 'w') as f:
                f.write("Row Index,Capability ID,Error\n")
                for error in self.bad_rows_ground_truth:
                    f.write(f"{error['row_index']},{error['capability_id']},\"{error['error']}\"\n")
            logger.info(f"Wrote ground truth error log: {gt_log_file}")
        
        # Write prediction errors
        if self.bad_rows_predictions:
            pred_log_file = os.path.join(output_dir, "bad_rows_predictions_step2.log")
            with open(pred_log_file, 'w') as f:
                f.write("Row Index,Capability ID,Error\n")
                for error in self.bad_rows_predictions:
                    f.write(f"{error['row_index']},{error['capability_id']},\"{error['error']}\"\n")
            logger.info(f"Wrote predictions error log: {pred_log_file}")


class Step2PerformanceTracker:
    """
    Tracks performance history for step 2 evaluation.
    """
    
    def __init__(self, output_dir: str = "Data"):
        self.output_dir = output_dir
        self.runs_file = os.path.join(output_dir, "performance_history_runs_step2.csv")
        self.prompts_file = os.path.join(output_dir, "performance_history_prompts_step2.csv")
        
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
        prompt_lookup_file = os.path.join(self.output_dir, "prompt_text_lookup_step2.csv")
        
        # Check if this hash already exists
        if os.path.exists(prompt_lookup_file):
            try:
                existing_df = pd.read_csv(prompt_lookup_file)
                if prompt_hash in existing_df['prompt_hash'].values:
                    # Hash already exists, no need to store again
                    return prompt_hash
            except Exception as e:
                logger.warning(f"Could not read existing prompt_text_lookup_step2.csv: {e}")
        
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
            
            logger.info(f"Stored step 2 prompt text with hash {prompt_hash}")
            return prompt_hash
        except Exception as e:
            logger.error(f"Failed to write step 2 prompt text to CSV: {e}")
            logger.warning("Continuing without storing prompt text")
            return prompt_hash
    
    def log_run_result(self, prompt_id: str, run_index: int, metrics: Dict[str, float], 
                      threshold: float, dataset: str = "test", notes: str = "", prompt_text: str = "",
                      token_usage: dict = None, model_name: str = ""):
        """
        Log a single run result for step 2.
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
            'total_ground_truth_tasks': metrics.get('total_ground_truth_tasks', 0),
            'total_predicted_tasks': metrics.get('total_predicted_tasks', 0),
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
            
            logger.info(f"Logged step 2 run {run_index} for prompt {prompt_id}")
        except Exception as e:
            logger.error(f"Failed to write step 2 run result to CSV: {e}")
            logger.warning("Continuing evaluation despite logging failure")
    
    def log_prompt_summary(self, prompt_id: str, all_run_metrics: List[Dict[str, float]], 
                          threshold: float, dataset: str = "test", prompt_text: str = "",
                          token_usage: dict = None, model_name: str = ""):
        """
        Log aggregated summary for a step 2 prompt.
        """
        # Extract metrics arrays
        precisions = [m['macro_precision'] for m in all_run_metrics]
        recalls = [m['macro_recall'] for m in all_run_metrics]
        f1s = [m['macro_f1'] for m in all_run_metrics]
        
        # Extract task counts (use first run's data since they should be consistent)
        total_ground_truth_tasks = all_run_metrics[0].get('total_ground_truth_tasks', 0) if all_run_metrics else 0
        total_predicted_tasks = all_run_metrics[0].get('total_predicted_tasks', 0) if all_run_metrics else 0
        
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
            'total_ground_truth_tasks': total_ground_truth_tasks,
            'total_predicted_tasks': total_predicted_tasks,
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
            
            logger.info(f"Logged step 2 prompt summary for {prompt_id}: F1={summary_data['mean_f1']:.4f}±{summary_data['std_f1']:.4f}")
        except Exception as e:
            logger.error(f"Failed to write step 2 prompt summary to CSV: {e}")
            logger.warning("Continuing evaluation despite logging failure")


def extract_ai_capabilities_from_final(final_json_str):
    """
    Extract AI capabilities from the Final column JSON.
    
    Args:
        final_json_str: JSON string from Final column
        
    Returns:
        List of AI capability strings
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
        
        # Collect ai_capability values
        capabilities = []
        for task in ai_tasks:
            if isinstance(task, dict) and 'ai_capability' in task:
                cap = task['ai_capability']
                if cap and isinstance(cap, str):
                    capabilities.append(cap.strip())
        
        return capabilities
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Failed to parse Final JSON: {e}")
        return []


def create_step2_dataset_from_training(train_file, output_file):
    """
    Create step 2 evaluation dataset from training data.
    
    Args:
        train_file: Path to training data CSV with Final column
        output_file: Path for output CSV
        
    Returns:
        DataFrame with AI capabilities for step 2 evaluation
    """
    logger.info(f"Loading training data: {train_file}")
    
    # Load training data
    try:
        df = pd.read_csv(train_file)
        logger.info(f"Loaded {len(df)} training records")
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return None
    
    # Check for Final column
    if 'Final' not in df.columns:
        logger.error("'Final' column not found in training data")
        logger.error(f"Available columns: {list(df.columns)}")
        return None
    
    # Extract all AI capabilities
    all_capabilities = []
    capability_metadata = []
    
    for idx, row in df.iterrows():
        capabilities = extract_ai_capabilities_from_final(row['Final'])
        
        for cap in capabilities:
            all_capabilities.append(cap)
            capability_metadata.append({
                'original_uid': row.get('uid', f'row_{idx}'),
                'company_name': row.get('company_name', 'Unknown'),
                'job_title': row.get('title', 'Unknown'),
                'ai_capability': cap
            })
    
    logger.info(f"Extracted {len(all_capabilities)} AI capabilities from {len(df)} training jobs")
    
    if not all_capabilities:
        logger.error("No AI capabilities found in training data")
        return None
    
    # Create dataset for step 2 evaluation
    step2_df = pd.DataFrame(capability_metadata)
    
    # Add empty columns for predictions and ground truth
    step2_df['predicted_separated_tasks'] = ''
    step2_df['ground_truth_separated_tasks'] = ''
    
    # Save to CSV
    step2_df.to_csv(output_file, index=False)
    logger.info(f"Created step 2 evaluation dataset: {output_file}")
    logger.info(f"Dataset contains {len(step2_df)} AI capabilities")
    
    # Print some examples
    logger.info("Example AI capabilities:")
    for i, cap in enumerate(all_capabilities[:5]):
        logger.info(f"{i+1}. {cap}")
    
    return step2_df


def run_step2_with_o3(input_file, output_file, model="o3-mini", reasoning_effort="medium", use_flex=False):
    """
    Run step 2 separation on the AI capabilities using O3.
    
    Args:
        input_file: CSV file with ai_capability column
        output_file: Output CSV file with predictions
        model: OpenAI model to use
        reasoning_effort: Reasoning effort level
        use_flex: Use flex processing for cost savings
        
    Returns:
        DataFrame with O3 predictions
    """
    logger.info(f"Running step 2 separation with {model} (reasoning effort: {reasoning_effort})")
    
    # Import the AITaskExtractor
    try:
        from stage_3_extract_ai_tasks import AITaskExtractor
    except ImportError as e:
        logger.error(f"Error importing AITaskExtractor: {e}")
        return None
    
    # Load the dataset
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} AI capabilities")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None
    
    # Initialize extractor
    try:
        extractor = AITaskExtractor(
            model_name=model, 
            reasoning_effort=reasoning_effort,
            use_flex=use_flex,
            verbosity="medium"
        )
        logger.info(f"Initialized {model} with {reasoning_effort} reasoning effort")
    except Exception as e:
        logger.error(f"Error initializing extractor: {e}")
        return None
    
    # Process each capability
    results = []
    errors = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            logger.info(f"Processing capability {idx + 1}/{len(df)}")
        
        ai_capability = row['ai_capability']
        
        try:
            # Run step 2 separation
            step2_result = extractor.step_2_separate_tasks(ai_capability)
            
            if step2_result:
                # Parse the result (plain text, one task per line)
                separated_tasks = step2_result.strip().split('\n')
                separated_tasks = [task.strip() for task in separated_tasks if task.strip()]
                
                # Store as JSON for the prediction column
                prediction_json = json.dumps(separated_tasks, ensure_ascii=False)
            else:
                prediction_json = json.dumps([])
                errors.append(f"Row {idx}: No result from step 2")
            
        except Exception as e:
            prediction_json = json.dumps([])
            errors.append(f"Row {idx}: Error - {e}")
        
        # Update the row
        row_data = row.copy()
        row_data['predicted_separated_tasks'] = prediction_json
        results.append(row_data)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    logger.info("Step 2 processing complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Total capabilities processed: {len(results_df)}")
    logger.info(f"Errors encountered: {len(errors)}")
    
    if errors:
        logger.info("Errors:")
        for error in errors[:10]:  # Show first 10 errors
            logger.info(f"  {error}")
        if len(errors) > 10:
            logger.info(f"  ... and {len(errors) - 10} more")
    
    # Show token usage
    logger.info("Token Usage:")
    logger.info(f"  Total tokens: {extractor.total_tokens:,}")
    logger.info(f"  Total cost: ${extractor.total_cost:.4f}")
    logger.info(f"  API calls: {extractor.api_calls}")
    
    return results_df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Step 3 Evaluation: Semantic Evaluation of AI Task Separation (Step 2)")
    
    # Mode selection
    parser.add_argument("--create-dataset", action="store_true",
                       help="Create step 2 evaluation dataset from training data")
    
    parser.add_argument("--train-file", type=str, 
                       default="Data/manual_coding_evaluation/train_data_scored_3062025_updated.csv",
                       help="Path to training data CSV with Final column (for --create-dataset)")
    
    # Evaluation mode
    parser.add_argument("--input-file", type=str,
                       help="Path to input file with AI capabilities and ground truth separated tasks")
    
    parser.add_argument("--capability-column", type=str, default="ai_capability",
                       help="Name of column containing AI capability descriptions")
    
    parser.add_argument("--ground-truth-column", type=str, default="ground_truth_separated_tasks",
                       help="Name of column containing ground truth separated tasks")
    
    parser.add_argument("--prediction-column", type=str, default="predicted_separated_tasks",
                       help="Name of column containing predictions")
    
    parser.add_argument("--prompt-id", type=str, default="step2_baseline_v1",
                       help="Identifier for the prompt being evaluated")
    
    parser.add_argument("--num-runs", type=int, default=5,
                       help="Number of evaluation runs (default: 5)")
    
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Similarity threshold for matching")
    
    parser.add_argument("--output-dir", type=str, default="Data",
                       help="Output directory for results")
    
    parser.add_argument("--notes", type=str, default="",
                       help="Optional notes for this evaluation run")
    
    # O3 step 2 execution
    parser.add_argument("--run-o3-step2", action="store_true",
                       help="Run O3 step 2 prompt to generate predictions (requires OpenAI API key)")
    
    parser.add_argument("--openai-model", type=str, default="o3-mini",
                       help="OpenAI model to use for step 2 prompt")
    
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"],
                       help="Reasoning effort for o-series models (default: medium)")
    
    parser.add_argument("--use-flex", action="store_true",
                       help="Use flex processing for cost savings (GPT-5, O3, O4-mini models only)")
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline for step 2."""
    args = parse_arguments()
    
    logger.info("Starting Step 3 Evaluation: Semantic Evaluation of AI Task Separation (Step 2)")
    
    # Mode 1: Create step 2 evaluation dataset from training data
    if args.create_dataset:
        from dotenv import load_dotenv
        load_dotenv('config.env')
        
        
        logger.info("="*60)
        logger.info("CREATE STEP 2 GROUND TRUTH DATASET")
        logger.info("="*60)
        
        # Step 1: Extract capabilities from training data
        logger.info("Step 1: Extracting AI capabilities from training data...")
        capabilities_file = os.path.join(args.output_dir, f"step2_capabilities_for_eval.csv")
        
        dataset = create_step2_dataset_from_training(args.train_file, capabilities_file)
        
        if dataset is None:
            logger.error("Failed to create dataset")
            return
        
        # Step 2: Run O3 step 2 separation if requested
        if args.run_o3_step2:
            logger.info(f"Step 2: Running {args.openai_model} step 2 separation...")
            
            results_file = os.path.join(args.output_dir, f"step2_with_o3_predictions.csv")
            
            try:
                results = run_step2_with_o3(
                    input_file=capabilities_file,
                    output_file=results_file, 
                    model=args.openai_model,
                    reasoning_effort=args.reasoning_effort,
                    use_flex=args.use_flex
                )
                
                if results is not None:
                    logger.info("✅ Step 2 dataset created successfully!")
                    logger.info(f"📁 Dataset with O3 predictions: {results_file}")
                    logger.info("")
                    logger.info("📝 Next steps:")
                    logger.info("1. Review the predicted_separated_tasks column")
                    logger.info("2. Fill in the ground_truth_separated_tasks column")
                    logger.info("3. Use the completed dataset for step 2 evaluation")
                else:
                    logger.error("❌ Failed to run O3 step 2")
                    logger.info(f"📁 Basic dataset (without predictions): {capabilities_file}")
                
            except Exception as e:
                logger.error(f"❌ Error running O3 step 2: {e}")
                logger.info(f"📁 Basic dataset (without predictions): {capabilities_file}")
        else:
            logger.info(f"✅ Capabilities dataset created: {capabilities_file}")
            logger.info("")
            logger.info("📝 To run O3 step 2 separation, use: --run-o3-step2")
        
        return
    
    # Mode 2: Evaluation mode (requires input file)
    if not args.input_file:
        logger.error("Either use --create-dataset to create a dataset, or provide --input-file for evaluation")
        return
    
    # Load data for evaluation
    logger.info(f"Loading input data: {args.input_file}")
    try:
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df)} AI capabilities for evaluation")
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return
    
    # Initialize step 2 executor if needed
    step2_executor = None
    if args.run_o3_step2:
        logger.info("Initializing step 2 prompt executor")
        try:
            step2_executor = Step2PromptExecutor(model=args.openai_model, reasoning_effort=args.reasoning_effort, use_flex=args.use_flex)
        except Exception as e:
            logger.error(f"Failed to initialize step 2 prompt executor: {e}")
            logger.error("Make sure you have set OPENAI_API_KEY in your environment or config.env")
            return
    
    # Generate predictions if needed
    current_df = df.copy()
    if args.run_o3_step2 and step2_executor:
        logger.info("Running step 2 prompt on dataset")
        current_df = step2_executor.process_dataset(current_df, args.capability_column)
        
        # Save dataset with predictions
        pred_file = os.path.join(args.output_dir, f"capabilities_with_step2_predictions_{args.prompt_id}.csv")
        current_df.to_csv(pred_file, index=False)
        logger.info(f"Saved dataset with step 2 predictions: {pred_file}")
    
    elif args.prediction_column not in current_df.columns:
        logger.error(f"Prediction column '{args.prediction_column}' not found in data")
        logger.error("Either use --run-o3-step2 to generate predictions, or ensure your data has the prediction column")
        logger.error(f"Available columns: {list(current_df.columns)}")
        return
    
    # Check for ground truth column
    if args.ground_truth_column not in current_df.columns:
        logger.error(f"Ground truth column '{args.ground_truth_column}' not found in data")
        logger.error("Make sure you have filled in the ground truth column")
        logger.error(f"Available columns: {list(current_df.columns)}")
        return
    
    # Initialize evaluator
    evaluator = Step2SemanticEvaluator(similarity_threshold=args.threshold)
    
    # Initialize performance tracker
    tracker = Step2PerformanceTracker(args.output_dir)
    
    # Get prompt text for logging if step2_executor is available
    prompt_text = ""
    if step2_executor:
        prompt_text = step2_executor.get_prompt_text()
    
    # Run multiple evaluations
    all_run_metrics = []
    
    for run_idx in range(1, args.num_runs + 1):
        logger.info(f"Starting evaluation run {run_idx}/{args.num_runs}")
        
        # Evaluate dataset
        metrics = evaluator.evaluate_dataset(current_df, args.prediction_column, args.ground_truth_column)
        all_run_metrics.append(metrics)
        
        # Log run result
        token_usage = step2_executor.get_token_usage_summary() if step2_executor else None
        
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
    final_token_usage = step2_executor.get_token_usage_summary() if step2_executor else None
    tracker.log_prompt_summary(args.prompt_id, all_run_metrics, evaluator.similarity_threshold, "test", prompt_text, final_token_usage, args.openai_model)
    
    # Print final summary
    print("\n" + "="*60)
    print("STEP 2 EVALUATION SUMMARY")
    print("="*60)
    print(f"Prompt ID: {args.prompt_id}")
    print(f"Runs: {args.num_runs}")  
    print(f"Threshold: {evaluator.similarity_threshold:.3f}")
    print("-" * 60)
    
    f1s = [m['macro_f1'] for m in all_run_metrics]
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    
    print(f"Dataset: {len(current_df)} AI capabilities")
    print(f"Mean F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    print("="*60)
    logger.info("Step 2 evaluation complete!")


if __name__ == "__main__":
    main()