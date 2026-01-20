#!/usr/bin/env python3
"""
Step 4 Evaluation: Semantic Evaluation of Final AI Application Filtering (Step 4)

This module evaluates how well the LLM prompt in step 4 performs final filtering
to retain only very specific AI applications and remove broad/generic ones.

Based on the same methodology as step 1, step 2, and step 3 evaluation.
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


class Step4PromptExecutor:
    """
    Executes the exact step 4 final filtering prompt from stage_3_extract_ai_tasks.py
    """
    
    def __init__(self, model_name="gpt-4.1-mini", reasoning_effort="low"):
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        
        # Load environment
        from dotenv import load_dotenv
        load_dotenv('config.env')
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Token tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        
        # OpenAI pricing (per 1M tokens)
        self.pricing = {
            "gpt-5": {"input": 1.25, "output": 10.00, "cached_input": 0.125},
            "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
            "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cached_input": 0.10},
            "gpt-4.1": {"input": 2.00, "output": 8.00, "cached_input": 0.50},
            "o3": {"input": 15.00, "output": 60.00, "cached_input": 7.50},
            "o3-mini": {"input": 0.50, "output": 2.00, "cached_input": 0.25},
        }
    
    def _make_api_call(self, system_prompt, user_prompt, max_tokens=200):
        """Make API call with proper model configuration"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Configure parameters based on model
        params = {
            "model": self.model_name,
            "messages": messages,
            "seed": 42,
            "response_format": {"type": "json_object"}
        }
        
        # O3 models only support default temperature (1), other models can use temperature=0
        if "o3" not in self.model_name:
            params["temperature"] = 0
            params["top_p"] = 1
        
        # Use max_completion_tokens for O3 models, max_tokens for others
        if "o3" in self.model_name:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
        
        # Add reasoning effort for supported models
        if ("gpt-5" in self.model_name and "mini" not in self.model_name) or "o3" in self.model_name:
            params["reasoning_effort"] = self.reasoning_effort
        
        response = self.client.chat.completions.create(**params)
        
        # Track usage
        if hasattr(response, 'usage'):
            tokens = response.usage.total_tokens
            self.total_tokens += tokens
            self.api_calls += 1
            
            # Calculate cost
            if self.model_name in self.pricing:
                input_cost = (response.usage.prompt_tokens / 1_000_000) * self.pricing[self.model_name]["input"]
                output_cost = (response.usage.completion_tokens / 1_000_000) * self.pricing[self.model_name]["output"]
                self.total_cost += input_cost + output_cost
        
        return response

    def execute_step4(self, step3_output: str) -> str:
        """Execute the exact step 4 final filtering prompt"""
        system_prompt = """The excerpt below describes how an artificial intelligence technology is being applied. Please determine if the application is very specific. If yes, please summarize the application (without outputing anything else). All references to any type of AI tool (e.g. natural language processing, machine learning, computer vision, generative AI, or any specific AI/ML algorithm) are redundant and should be stripped from the text. Otherwise, respond 'N/A'. Here are some examples:

-'Predictive Analytics' should be 'N/A' as it is very broad;
-'Data Visualization' should be 'N/A' as it is very broad;
-'AI-driven NFT Collection Visualization' should be kept as it is a very specific application.
-'Perform exploratory data analysis for invoice anomalies' should be 'invoice anomalies'
-'Provide self-service data access and custom visualization interfaces for the oceanic team' should be 'custom visualization interfaces for the oceanic team' as this is a specific application.

Please filter the following application and return your response as JSON in this format:
{"final_application": "your filtered text here or N/A"}"""

        try:
            response = self._make_api_call(system_prompt, step3_output)
            content = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                result = json.loads(content)
                return result.get("final_application", "")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {content}")
                return content
                
        except Exception as e:
            logger.error(f"Error in step 4 execution: {e}")
            return ""


class Step4SemanticEvaluator:
    """
    Evaluates step 4 final filtering using semantic similarity metrics
    """
    
    def __init__(self, model_name='BAAI/bge-large-en-v1.5'):
        """Initialize with BGE embedding model"""
        self.embedding_model = SentenceTransformer(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model.to(self.device)
        
        # Project paths
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "Data"
        self.eval_dir = self.data_dir / "evaluation"
        self.eval_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Step4SemanticEvaluator with {model_name} on {self.device}")

    def compute_step4_similarity(self, ground_truth: str, prediction: str) -> float:
        """
        Compute semantic similarity between ground truth and prediction.
        Special handling for N/A cases in final filtering.
        """
        # Normalize text
        gt_norm = ground_truth.strip() if ground_truth else ""
        pred_norm = prediction.strip() if prediction else ""
        
        # Handle N/A cases (both should be N/A or both should be specific applications)
        gt_is_na = (gt_norm.lower() in ['n/a', 'na', ''] or 
                   'n/a' in gt_norm.lower() or 
                   gt_norm.lower().startswith('n/a'))
        pred_is_na = (pred_norm.lower() in ['n/a', 'na', ''] or 
                     'n/a' in pred_norm.lower() or 
                     pred_norm.lower().startswith('n/a'))
        
        if gt_is_na and pred_is_na:
            return 1.0  # Both correctly identified as N/A (too broad)
        elif gt_is_na and not pred_is_na:
            return 0.0  # GT says N/A but prediction gave specific application
        elif not gt_is_na and pred_is_na:
            return 0.0  # GT has specific application but prediction said N/A
        else:
            # Both are specific applications - compute semantic similarity
            if not gt_norm or not pred_norm:
                return 0.0
            
            try:
                # Get embeddings
                gt_embedding = self.embedding_model.encode([gt_norm])
                pred_embedding = self.embedding_model.encode([pred_norm])
                
                # Compute cosine similarity
                similarity = np.dot(gt_embedding[0], pred_embedding[0]) / (
                    np.linalg.norm(gt_embedding[0]) * np.linalg.norm(pred_embedding[0])
                )
                
                return float(similarity)
            except Exception as e:
                logger.warning(f"Error computing similarity: {e}")
                return 0.0

    def create_step4_dataset_from_step3_results(self, step3_results_file: str, sample_size: int = 50) -> str:
        """
        Create step 4 evaluation dataset from step 3 results.
        Takes applications that passed step 3 filtering.
        """
        logger.info(f"Creating step 4 dataset from: {step3_results_file}")
        
        # Load step 3 results
        df = pd.read_csv(step3_results_file)
        logger.info(f"Loaded {len(df)} step 3 results")
        
        # Filter for applications that passed step 3 (not N/A)
        df_filtered = df[
            (df['step3_prediction'].notna()) & 
            (df['step3_prediction'].str.strip() != '') &
            (~df['step3_prediction'].str.lower().str.contains('n/a', na=False))
        ].copy()
        
        logger.info(f"Found {len(df_filtered)} applications that passed step 3 filtering")
        
        if len(df_filtered) == 0:
            raise ValueError("No applications passed step 3 filtering")
        
        # Sample for manual annotation
        if len(df_filtered) > sample_size:
            df_sample = df_filtered.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} applications for step 4 evaluation")
        else:
            df_sample = df_filtered.copy()
            logger.info(f"Using all {len(df_sample)} applications for step 4 evaluation")
        
        # Prepare step 4 input dataset
        step4_dataset = []
        for idx, row in df_sample.iterrows():
            step4_dataset.append({
                'step3_output': row['step3_prediction'],  # Input to step 4
                'original_ai_capability': row.get('original_ai_capability', ''),
                'separated_task': row.get('separated_task', ''),
                'original_uid': row.get('original_uid', ''),
                'company_name': row.get('company_name', ''),
                'job_title': row.get('job_title', ''),
                'step4_ground_truth': '',  # To be filled manually
            })
        
        # Save dataset for manual annotation
        output_file = self.eval_dir / f"step4_dataset_for_annotation.csv"
        
        step4_df = pd.DataFrame(step4_dataset)
        step4_df.to_csv(output_file, index=False)
        
        logger.info(f"Step 4 dataset saved to: {output_file}")
        logger.info(f"Please manually fill the 'step4_ground_truth' column with expected final applications")
        logger.info(f"Use 'N/A' for applications that are too broad/generic")
        
        return str(output_file)

    def run_step4_with_model(self, input_file: str, model_name: str = "gpt-4.1-mini", 
                            reasoning_effort: str = "low") -> str:
        """
        Run step 4 final filtering on dataset using specified model
        """
        logger.info(f"Running step 4 with model: {model_name}")
        
        # Load input dataset  
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} applications for step 4 processing")
        
        # Initialize prompt executor
        executor = Step4PromptExecutor(model_name=model_name, reasoning_effort=reasoning_effort)
        
        predictions = []
        for idx, row in df.iterrows():
            step3_output = row['step3_output']
            
            logger.info(f"Processing {idx+1}/{len(df)}: {step3_output[:100]}...")
            
            try:
                prediction = executor.execute_step4(step3_output)
                predictions.append(prediction)
                logger.info(f"Step 4 result: {prediction}")
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing item {idx}: {e}")
                predictions.append("")
        
        # Add predictions to dataframe
        df['step4_prediction'] = predictions
        
        # Save results
        output_file = input_file.replace('.csv', f'_with_step4_predictions_{model_name.replace("/", "_")}.csv')
        
        df.to_csv(output_file, index=False)
        
        logger.info(f"Step 4 predictions saved to: {output_file}")
        logger.info(f"Total API calls: {executor.api_calls}")
        logger.info(f"Total tokens: {executor.total_tokens:,}")
        logger.info(f"Total cost: ${executor.total_cost:.4f}")
        
        return output_file

    def evaluate_step4_performance(self, results_file: str, similarity_threshold: float = 0.7) -> Dict:
        """
        Evaluate step 4 performance using semantic similarity metrics
        """
        logger.info(f"Evaluating step 4 performance from: {results_file}")
        
        # Load results
        df = pd.read_csv(results_file)
        
        # Check required columns
        required_cols = ['step4_ground_truth', 'step4_prediction']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter valid rows
        df_valid = df[
            (df['step4_ground_truth'].notna()) & 
            (df['step4_ground_truth'].str.strip() != '') &
            (df['step4_prediction'].notna()) & 
            (df['step4_prediction'].str.strip() != '')
        ].copy()
        
        logger.info(f"Evaluating {len(df_valid)} valid predictions")
        
        if len(df_valid) == 0:
            logger.warning("No valid predictions to evaluate")
            return {}
        
        # Compute similarities
        similarities = []
        for idx, row in df_valid.iterrows():
            sim = self.compute_step4_similarity(
                row['step4_ground_truth'], 
                row['step4_prediction']
            )
            similarities.append(sim)
        
        df_valid['similarity'] = similarities
        
        # Calculate metrics
        mean_similarity = np.mean(similarities)
        median_similarity = np.median(similarities)
        
        # Threshold-based metrics
        high_similarity = np.sum(np.array(similarities) >= similarity_threshold)
        precision_at_threshold = high_similarity / len(similarities) if len(similarities) > 0 else 0
        
        # N/A vs Specific analysis
        df_valid['gt_is_na'] = df_valid['step4_ground_truth'].str.lower().str.contains('n/a', na=False)
        df_valid['pred_is_na'] = df_valid['step4_prediction'].str.lower().str.contains('n/a', na=False)
        
        na_accuracy = np.sum(df_valid['gt_is_na'] == df_valid['pred_is_na']) / len(df_valid)
        
        # Specific application similarity (exclude N/A cases)
        df_specific = df_valid[~df_valid['gt_is_na'] & ~df_valid['pred_is_na']]
        specific_similarity = np.mean(df_specific['similarity']) if len(df_specific) > 0 else None
        
        results = {
            'total_predictions': len(df_valid),
            'mean_similarity': mean_similarity,
            'median_similarity': median_similarity,
            'precision_at_threshold': precision_at_threshold,
            'similarity_threshold': similarity_threshold,
            'na_classification_accuracy': na_accuracy,
            'specific_application_similarity': specific_similarity,
            'n_specific_applications': len(df_specific),
            'n_na_ground_truth': np.sum(df_valid['gt_is_na']),
            'n_na_predictions': np.sum(df_valid['pred_is_na']),
        }
        
        # Save detailed results
        detailed_file = results_file.replace('.csv', f'_step4_evaluation.csv')
        df_valid.to_csv(detailed_file, index=False)
        
        # Log results
        logger.info("="*60)
        logger.info("STEP 4 EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Total predictions evaluated: {results['total_predictions']}")
        logger.info(f"Mean semantic similarity: {results['mean_similarity']:.4f}")
        logger.info(f"Median semantic similarity: {results['median_similarity']:.4f}")
        logger.info(f"Precision @ {similarity_threshold}: {results['precision_at_threshold']:.4f}")
        logger.info(f"N/A classification accuracy: {results['na_classification_accuracy']:.4f}")
        if specific_similarity is not None:
            logger.info(f"Specific application similarity: {results['specific_application_similarity']:.4f}")
        logger.info(f"Ground truth N/A: {results['n_na_ground_truth']}")
        logger.info(f"Predicted N/A: {results['n_na_predictions']}")
        logger.info(f"Specific applications: {results['n_specific_applications']}")
        logger.info(f"Detailed results saved to: {detailed_file}")
        
        return results


def main():
    """Command line interface for step 4 evaluation"""
    parser = argparse.ArgumentParser(description='Step 4 Final Filtering Evaluation')
    parser.add_argument('--action', choices=['create_dataset', 'run_model', 'evaluate'], 
                       required=True, help='Action to perform')
    parser.add_argument('--input_file', help='Input file path')
    parser.add_argument('--step3_results', help='Step 3 results file for dataset creation')
    parser.add_argument('--model', default='gpt-4.1-mini', help='Model name')
    parser.add_argument('--reasoning_effort', default='low', help='Reasoning effort level')
    parser.add_argument('--sample_size', type=int, default=50, help='Sample size for dataset creation')
    parser.add_argument('--similarity_threshold', type=float, default=0.7, help='Similarity threshold')
    
    args = parser.parse_args()
    
    evaluator = Step4SemanticEvaluator()
    
    if args.action == 'create_dataset':
        if not args.step3_results:
            print("--step3_results required for create_dataset action")
            return
        
        output_file = evaluator.create_step4_dataset_from_step3_results(
            args.step3_results, 
            args.sample_size
        )
        print(f"Dataset created: {output_file}")
    
    elif args.action == 'run_model':
        if not args.input_file:
            print("--input_file required for run_model action")
            return
        
        output_file = evaluator.run_step4_with_model(
            args.input_file,
            args.model,
            args.reasoning_effort
        )
        print(f"Model predictions saved: {output_file}")
    
    elif args.action == 'evaluate':
        if not args.input_file:
            print("--input_file required for evaluate action")
            return
        
        results = evaluator.evaluate_step4_performance(
            args.input_file,
            args.similarity_threshold
        )
        print("Evaluation completed")


if __name__ == "__main__":
    main()