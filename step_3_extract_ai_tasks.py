#!/usr/bin/env python3
"""
Extract AI Tasks from Job Advertisements - 3-Step LLM Prompting Pipeline

This script implements the 3-step prompting process described in the Hamptone methodology:
1. Extract AI applications from job ads
2. Filter and clean the applications  
3. Final filtering for specificity

Based on the Allardice/Kurer roadmap methodology.
"""

import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import openai
from tqdm import tqdm

class AITaskExtractor:
    """Extract AI tasks from job advertisements using 3-step LLM prompting"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "Data"
        self.llm_output_dir = self.data_dir / "llm_output"
        self.llm_output_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        load_dotenv('config.env')
        
        # Set up OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def load_deduplicated_data(self):
        """Load the latest deduplicated and translated AI jobs data (training set)"""
        # Find the latest deduplicated/translated train file (not test)
        pattern = "ai_development_deduplicated_translated_train_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            # Fallback to old pattern without train/test suffix
            pattern = "ai_development_deduplicated_translated_*.csv"
            files = [f for f in self.data_dir.glob(pattern) if 'test' not in f.name]
        
        if not files:
            raise FileNotFoundError(f"No training files found matching pattern: {pattern}")
        
        # Use the latest file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        print(f"Loading training data from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} job advertisements from training set")
        
        return df
    
    def load_test_data(self):
        """Load the latest deduplicated and translated test set data"""
        # Find the latest deduplicated/translated test file
        pattern = "ai_development_deduplicated_translated_test_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No test set files found matching pattern: {pattern}")
        
        # Use the latest file
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        print(f"Loading test data from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} job advertisements from test set")
        
        return df
    
    def load_data(self, dataset_type="train"):
        """Load either training or test data"""
        if dataset_type == "train":
            return self.load_deduplicated_data()
        elif dataset_type == "test":
            return self.load_test_data()
        else:
            raise ValueError(f"dataset_type must be 'train' or 'test', got '{dataset_type}'")
        
    def step_0_manual_coding_train_test(self):
        """ 
        For this, I use o3 to code the answers and then manually check them. This is the prompt I use:
        You are an occupational‑impact analyst.  
Your job: given a job advertisement, list every task that specific AI/ML application or ai/ml-enabled tools mentioned in the job description will perform or automate. 
⚠️ Do **not** include tasks that human software engineers will do while building, deploying, or maintaining the AI/ML system, but rather include what the systems or tools themselves do. 

Step‑by‑step instructions  
1. **Parse AI capabilities**  
   Extract every distinct function, feature, or workflow an AI/ML system in the advertizement is intended to deliver (e.g., “detect surface defects on steel coils,” “generate marketing copy,” “forecast spare‑parts demand”).  
2. **Deliver structured output**  
   Return a JSON array named ai_application_tasks, where each element has:
   {
     "task_id": sequential integer starting at 1,
     "ai_capability": "one to two sentences describing the AI capability",
     "brief_description": "one‑sentence clarification drawn from the job ad (in english)",
     "raw_excerpt": "verbatim excerpt that supports the use case, left in the ad’s original language",
     "explanation": "one to two sentence explanation for why this task was chosen"
   }
   """
        pass
    
    def step_1_extract_ai_applications(self, job_ads_text):
        """
        Step 1: Extract AI applications from job ads
        """
        system_prompt = """

        
        You are an occupational‑impact analyst.  
        Your job: given a job advertisement, list every task that specific AI/ML application or ai/ml-enabled tools mentioned in the job description will perform or automate. 
        ⚠️ Do **not** include tasks that human software engineers will do while building, deploying, or maintaining the AI/ML system, but rather include what the systems or tools themselves do. 

        Step‑by‑step instructions  
        1. **Parse AI capabilities**  
        Extract every distinct function, feature, or workflow an AI/ML system in the advertizement is intended to deliver (e.g., “detect surface defects on steel coils,” “generate marketing copy,” “forecast spare‑parts demand”).  
        2. **Deliver structured output**  
        Return a JSON array named ai_application_tasks, where each element has:
        {
            "task_id": sequential integer starting at 1,
            "ai_capability": "one to two sentences describing the AI capability",
            "brief_description": "one‑sentence clarification drawn from the job ad (in english)",
            "raw_excerpt": "verbatim excerpt that supports the use case, left in the ad’s original language",
            "explanation": "one to two sentence explanation for why this task was chosen"
        }

"""

        user_prompt = job_ads_text

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'"""\n{user_prompt}\n"""'}
                ],
                temperature=0,
                top_p=1,
                seed=42,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error in step 1: {e}")
            return None
    
    def step_2_separate_tasks(self, step1_output):
        """
        Step 2: Separate compound tasks into distinct work tasks
        """
        system_prompt = """Task
You’re given an excerpt describing how artificial-intelligence technology is being applied.

Goal Return every distinct work task **only**, one per line, preserving the original wording exactly, including all examples, clarifications, and non-essential info.
Do **not** add any other text—no headings, no “Output:” label, no bullets, no numbering, no blank lines.


Decision rule for multi-verb phrases
    • If all verbs share the same core object, deliverable, or outcome, keep them together in one clause.
    • If any verb produces a stand-alone object, deliverable, or outcome, put each in a separate clause.
   • When one verb phrase applies to **multiple nouns**, output **one clause per noun** that
      repeats the *entire* verb phrase. Do **not** create a clause for every verb–noun pairing.

Write each clause so that it:  
• Preserves all meaningful information from the original.                        

Examples
Input: "Develop predictive models and generate dashboards to guide hiring decisions."
Output:
Develop predictive models to guide hiring decisions.
Generate dashboards to guide hiring decisions.

Input: "Collect, clean, and analyze customer-support tickets."
Output:
Collect, clean, and analyze customer-support tickets.

Input: "Clean raw sensor data and build dashboards to monitor equipment status."
Output:
Clean raw sensor data to monitor equipment status.
Build dashboards to monitor equipment status.

Input: "Design, build, and test models to monitor equipment status."
Output:
Design, build, and test models to monitor equipment status.

Input: "Design, build, and test models and data pipelines to monitor equipment status."
Output:
Design, build, and test models to monitor equipment status.
Design, build, and test data pipelines to monitor equipment status."""


        user_prompt = step1_output

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0,
                top_p=1,
                seed=42
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error in step 2: {e}")
            return None
    
    def step_3_filter_applications(self, step2_output):
        """
        Step 3: Filter and clean applications
        """
        system_prompt = """The excerpt below describes how an artificial intelligence technology is being applied. Assume that it is already known that the excerpt refers to a use of artificial intelligence; the reader only wants to know the specific final application. Therefore, all references to any type of AI tool (e.g. natural language processing, machine learning, computer vision, generative AI, or any specific AI/ML algorithm) are redundant and should be stripped from the text. If the text only contains reference to an AI tool and without a clearly specified application, you should return 'N/A' when you filter the text.

For reference, here are a few examples of correctly applied filters:
-'AI tools are being used to measure text similarity in educational settings using NLP' should become 'Measure text similarity in educational settings'
-'Machine learning is being applied to perform tasks related to database analysis and firmware/software development for embedded environments' should become 'Perform tasks related to database analysis and firmware/software development for embedded environments'
-'AI-powered chatbots are being used to provide customers with quick solutions and answers using natural language processing capabilities.' should become 'Provide customers with quick solutions and answers.'
-'Analyzing customer reviews using NLP to understand customer needs and wants' should become 'Analyze customer reviews to understand customer needs and wants'
-'AI tool is being used to deploy computer vision model' should become 'N/A', because computer vision models themselves are an AI tool, and the exact use of computer vision is not specified.'

Please filter the following excerpt describing an AI application and return your response as JSON in this format:
{"filtered_application": "your filtered text here or N/A"}"""

        user_prompt = step2_output

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0,
                top_p=1,
                seed=42,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error in step 3: {e}")
            return None
    
    def step_4_final_filtering(self, step3_output):
        """
        Step 4: Final filtering for specificity
        """
        system_prompt = """The excerpt below describes how an artificial intelligence technology is being applied. Please determine if the application is very specific. If yes, please summarize the application (without outputing anything else). All references to any type of AI tool (e.g. natural language processing, machine learning, computer vision, generative AI, or any specific AI/ML algorithm) are redundant and should be stripped from the text. Otherwise, respond 'N/A'. Here are some examples:

-'Predictive Analytics' should be 'N/A' as it is very broad;
-'Data Visualization' should be 'N/A' as it is very broad;
-'AI-driven NFT Collection Visualization' should be kept as it is a very specific application.
-'Perform exploratory data analysis for invoice anomalies' should be 'invoice anomalies'
-'Provide self-service data access and custom visualization interfaces for the oceanic team' should be 'custom visualization interfaces for the oceanic team' as this is a specific application.

Please filter the following application and return your response as JSON in this format:
{"final_application": "your filtered text here or N/A"}"""

        user_prompt = step3_output

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0,
                top_p=1,
                seed=42,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error in step 4: {e}")
            return None
    
    
    def run_step1_only(self, dataset_type="train"):
        """Run only Step 1: Extract AI applications and save results"""
        print("="*60)
        print(f"STEP 1: AI APPLICATION EXTRACTION - {dataset_type.upper()} SET")
        print("="*60)
        
        # Load data
        df = self.load_data(dataset_type)
        
        all_results = []
        all_errors = []
        
        # Process each job for step 1
        for idx, job_row in tqdm(df.iterrows(), total=len(df), desc="Extracting AI applications"):
            job_ads_text = job_row['content_clean']
            job_uid = job_row['uid']
            
            # Step 1: Extract AI applications
            step1_result = self.step_1_extract_ai_applications(job_ads_text)
            if not step1_result:
                all_errors.append(f"Job {job_uid}: Step 1 returned None")
                continue
            
            # Try to parse the JSON response from step 1
            try:
                step1_json = json.loads(step1_result)
                # Handle both list format and single object format
                if isinstance(step1_json, list):
                    applications_list = step1_json
                elif isinstance(step1_json, dict):
                    if 'Key Application' in step1_json and 'Raw Excerpt' in step1_json:
                        applications_list = [step1_json]
                    elif 'applications' in step1_json:
                        applications_list = step1_json['applications']
                    else:
                        all_errors.append(f"Job {job_uid}: Step 1 object format not recognized: {step1_result[:200]}...")
                        continue
                else:
                    all_errors.append(f"Job {job_uid}: Step 1 result is not a valid format: {step1_result[:200]}...")
                    continue
            except json.JSONDecodeError as e:
                all_errors.append(f"Job {job_uid}: Step 1 JSON decode error: {e} - Content: {step1_result[:200]}...")
                continue
            
            # Store each application with job metadata
            for i, item in enumerate(applications_list):
                if isinstance(item, dict) and 'Key Application' in item:
                    all_results.append({
                        'key_application': item.get('Key Application', ''),
                        'raw_excerpt': item.get('Raw Excerpt', ''),
                        'job_title': job_row['title'],
                        'company_name': job_row['company_name'],
                        'job_uid': job_row['uid'],
                        'raw_text': job_row['content_clean']
                    })
            
            time.sleep(0.1)
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"step1_ai_applications_{dataset_type}_{timestamp}.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n📄 Step 1 results saved to: {output_path}")
        print(f"   - AI applications extracted: {len(results_df)}")
        print(f"   - Jobs processed: {len(df)}")
        print(f"   - Processing errors: {len(all_errors)}")
        
        if all_errors:
            print(f"\n⚠️  PROCESSING ERRORS ({len(all_errors)} total):")
            for error in all_errors[:10]:  # Show first 10 errors
                print(f"   - {error}")
            if len(all_errors) > 10:
                print(f"   ... and {len(all_errors) - 10} more errors")
        
        return str(output_path)
    
    def load_step1_results(self, dataset_type="train"):
        """Load the latest Step 1 results"""
        pattern = f"step1_ai_applications_{dataset_type}_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No Step 1 files found matching pattern: {pattern}")
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        print(f"Loading Step 1 data from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} AI applications from Step 1")
        
        return df
    
    def run_step2_only(self, dataset_type="train"):
        """Run only Step 2: Separate tasks and save results"""
        print("="*60)
        print(f"STEP 2: TASK SEPARATION - {dataset_type.upper()} SET")
        print("="*60)
        
        # Load Step 1 results
        df = self.load_step1_results(dataset_type)
        
        all_results = []
        all_errors = []
        
        # Process each application through step 2
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Separating tasks"):
            app_description = row['key_application']
            
            # Step 2: Separate tasks
            step2_result = self.step_2_separate_tasks(app_description)
            if not step2_result:
                all_errors.append(f"Row {idx} (Job {row['job_uid']}): Step 2 returned None")
                continue
            
            # Parse step 2 response (plain text)
            separated_tasks = step2_result.strip().split('\n')
            separated_tasks = [task.strip() for task in separated_tasks if task.strip()]
            
            if not separated_tasks:
                all_errors.append(f"Row {idx} (Job {row['job_uid']}): Step 2 returned no tasks")
                continue
            
            # Store each separated task with original metadata
            for task in separated_tasks:
                result_row = row.copy()
                result_row['step2_separated_task'] = task
                all_results.append(result_row)
            
            time.sleep(0.1)
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"step2_separated_tasks_{dataset_type}_{timestamp}.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n📄 Step 2 results saved to: {output_path}")
        print(f"   - Original applications: {len(df)}")
        print(f"   - Separated tasks: {len(results_df)}")
        print(f"   - Task expansion ratio: {len(results_df)/len(df):.2f}")
        print(f"   - Processing errors: {len(all_errors)}")
        
        return str(output_path)
    
    def load_step2_results(self, dataset_type="train"):
        """Load the latest Step 2 results"""
        pattern = f"step2_separated_tasks_{dataset_type}_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No Step 2 files found matching pattern: {pattern}")
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        print(f"Loading Step 2 data from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} separated tasks from Step 2")
        
        return df
    
    def run_step3_only(self, dataset_type="train"):
        """Run only Step 3: Filter applications and save results"""
        print("="*60)
        print(f"STEP 3: APPLICATION FILTERING - {dataset_type.upper()} SET")
        print("="*60)
        
        # Load Step 2 results
        df = self.load_step2_results(dataset_type)
        
        all_results = []
        all_errors = []
        
        # Process each separated task through step 3
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering applications"):
            task = row['step2_separated_task']
            
            # Step 3: Filter and clean
            step3_result = self.step_3_filter_applications(task)
            if not step3_result:
                all_errors.append(f"Row {idx} (Job {row['job_uid']}): Step 3 returned None")
                continue
            
            # Parse step 3 JSON response
            try:
                step3_json = json.loads(step3_result)
                filtered_app = step3_json.get('filtered_application', '')
                if not filtered_app or filtered_app == 'N/A':
                    continue  # Skip N/A results
            except json.JSONDecodeError as e:
                all_errors.append(f"Row {idx} (Job {row['job_uid']}): Step 3 JSON decode error: {e}")
                continue
            
            # Store with step 3 output
            result_row = row.copy()
            result_row['step3_output'] = filtered_app
            all_results.append(result_row)
            
            time.sleep(0.1)
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.data_dir / f"step3_filtered_tasks_{dataset_type}_{timestamp}.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"\n📄 Step 3 results saved to: {output_path}")
        print(f"   - Input tasks: {len(df)}")
        print(f"   - Filtered tasks: {len(results_df)}")
        print(f"   - Filtering success rate: {len(results_df)/len(df)*100:.1f}%")
        print(f"   - Processing errors: {len(all_errors)}")
        
        return str(output_path)

    def run_full_pipeline(self, dataset_type="train"):
        """Run the complete 3-step pipeline"""
        print("🚀 Starting complete AI task extraction pipeline...")
        
        step1_path = self.run_step1_only(dataset_type)
        step2_path = self.run_step2_only(dataset_type)
        step3_path = self.run_step3_only(dataset_type)
        
        print(f"\n{'='*60}")
        print("FULL PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Step 1 (Applications): {step1_path}")
        print(f"✅ Step 2 (Separated): {step2_path}")
        print(f"✅ Step 3 (Filtered): {step3_path}")
        
        return step3_path
    
    def load_step1_output(self):
        """Legacy function - use load_step1_results instead"""
        print("⚠️  This function is deprecated. Use load_step1_results('train') or load_step1_results('test') instead.")
        return self.load_step1_results("train")
    
    def process_step2_only(self):
        """Legacy function - use run_step2_only instead"""
        print("⚠️  This function is deprecated. Use run_step2_only('train') or run_step2_only('test') instead.")
        return self.run_step2_only("train")
    
    def run_extraction(self, dataset_type="train"):
        """Run the complete AI task extraction pipeline (legacy compatibility)"""
        try:
            output_path = self.run_full_pipeline(dataset_type)
            print(f"\n🎯 AI task extraction complete for {dataset_type} set!")
            print(f"Final results saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error during extraction: {e}")
            return None
    
    def run_both_datasets(self):
        """Run extraction on both training and test sets"""
        print("🚀 Starting AI task extraction for both train and test sets...")
        
        train_path = self.run_extraction("train")
        test_path = self.run_extraction("test")
        
        print(f"\n{'='*60}")
        print("BOTH DATASETS PROCESSED")
        print(f"{'='*60}")
        print(f"✅ Training set: {train_path}")
        print(f"✅ Test set: {test_path}")
        
        return train_path, test_path

def main():
    """Main function"""
    extractor = AITaskExtractor()
    output_path = extractor.run_extraction()
    
    if output_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Review extracted tasks for accuracy")
        print(f"2. Evaluate against manual coding")
        print(f"3. Iterate on prompts if needed")
        print(f"4. Proceed to evaluation stage")

def run_step1_train():
    """Run step 1 on training set"""
    extractor = AITaskExtractor()
    output_path = extractor.run_step1_only("train")
    print(f"\n📋 Next: Run step 2 with --step2-train")

def run_step1_test():
    """Run step 1 on test set"""
    extractor = AITaskExtractor()
    output_path = extractor.run_step1_only("test")
    print(f"\n📋 Next: Run step 2 with --step2-test")

def run_step2_train():
    """Run step 2 on training set"""
    extractor = AITaskExtractor()
    output_path = extractor.run_step2_only("train")
    print(f"\n📋 Next: Run step 3 with --step3-train")

def run_step2_test():
    """Run step 2 on test set"""
    extractor = AITaskExtractor()
    output_path = extractor.run_step2_only("test")
    print(f"\n📋 Next: Run step 3 with --step3-test")

def run_step3_train():
    """Run step 3 on training set"""
    extractor = AITaskExtractor()
    output_path = extractor.run_step3_only("train")
    print(f"\n✅ Final results: {output_path}")

def run_step3_test():
    """Run step 3 on test set"""
    extractor = AITaskExtractor()
    output_path = extractor.run_step3_only("test")
    print(f"\n✅ Final results: {output_path}")

def run_train_only():
    """Run extraction on training set only"""
    extractor = AITaskExtractor()
    output_path = extractor.run_extraction("train")
    
    if output_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Review extracted tasks for accuracy")
        print(f"2. Run on test set for validation")
        print(f"3. Compare train vs test performance")

def run_test_only():
    """Run extraction on test set only"""
    extractor = AITaskExtractor()
    output_path = extractor.run_extraction("test")
    
    if output_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Review extracted tasks for accuracy") 
        print(f"2. Compare with training set results")
        print(f"3. Evaluate for overfitting")

def run_both_sets():
    """Run extraction on both training and test sets"""
    extractor = AITaskExtractor()
    train_path, test_path = extractor.run_both_datasets()
    
    if train_path and test_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Compare training vs test set results")
        print(f"2. Check for overfitting indicators")
        print(f"3. Evaluate model generalization")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Individual steps
        if arg == "--step1-train":
            run_step1_train()
        elif arg == "--step1-test":
            run_step1_test()
        elif arg == "--step2-train":
            run_step2_train()
        elif arg == "--step2-test":
            run_step2_test()
        elif arg == "--step3-train":
            run_step3_train()
        elif arg == "--step3-test":
            run_step3_test()
        # Full pipelines
        elif arg == "--train":
            run_train_only()
        elif arg == "--test":
            run_test_only()
        elif arg == "--both":
            run_both_sets()
        else:
            print("Usage:")
            print("\n📋 Individual Steps:")
            print("  python3 step_3_extract_ai_tasks.py --step1-train    # Step 1: Extract applications (train)")
            print("  python3 step_3_extract_ai_tasks.py --step1-test     # Step 1: Extract applications (test)")
            print("  python3 step_3_extract_ai_tasks.py --step2-train    # Step 2: Separate tasks (train)")
            print("  python3 step_3_extract_ai_tasks.py --step2-test     # Step 2: Separate tasks (test)")
            print("  python3 step_3_extract_ai_tasks.py --step3-train    # Step 3: Filter applications (train)")
            print("  python3 step_3_extract_ai_tasks.py --step3-test     # Step 3: Filter applications (test)")
            print("\n🚀 Full Pipelines:")
            print("  python3 step_3_extract_ai_tasks.py --train          # Complete pipeline (train)")
            print("  python3 step_3_extract_ai_tasks.py --test           # Complete pipeline (test)")
            print("  python3 step_3_extract_ai_tasks.py --both           # Complete pipeline (both)")
    else:
        # Default behavior - show usage
        print("AI Task Extraction Pipeline - Please specify an option:")
        print("Run with --help or any invalid argument to see usage options")