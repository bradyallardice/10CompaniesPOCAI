#!/usr/bin/env python3
"""
Stage 3 Step 3 Tests: O*NET-style Task Filtering  
Tests final filtering with style validation
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path
import types
import pytest

sys.path.append(str(Path(__file__).parent.parent))
import stage_3_extract_ai_tasks as s3

def _refine_tasks(tasks_list):
    """Mock O*NET-style refinement logic"""
    if tasks_list == ["analyze data"]:
        return None  # Trigger style_validator_failed
    
    if any("TensorFlow" in task for task in tasks_list):
        return [
            "Train a supervised learning model using TensorFlow on labeled datasets.",
            "Deploy the trained model to an AWS SageMaker real-time endpoint.", 
            "Monitor, evaluate, and retrain models through an MLOps pipeline."
        ]
    else:
        return [
            "Fine-tune a BERT language model on domain-specific text.",
            "Orchestrate training and evaluation using Vertex AI Pipelines.",
            "Track runs and metrics with MLflow for reproducible deployments."
        ]

class _FakeStep3Client:
    """Mock client for Step 3 O*NET-style filtering"""
    
    def _process_request(self, text):
        """Process request and return refined tasks or style failure"""
        try:
            # Extract task list from the request
            if '"analyze data"' in text:
                return "STYLE_FAIL"  # Sentinel for style failure
            elif "TensorFlow" in text:
                refined = _refine_tasks(["train model using TensorFlow", "deploy model on SageMaker", "monitor pipeline runs"])
            elif "BERT" in text:
                refined = _refine_tasks(["fine-tune BERT", "use Vertex AI pipelines", "package and track with MLflow"])  
            else:
                return "STYLE_FAIL"
                
            return json.dumps(refined)
        except Exception:
            return "STYLE_FAIL"
    
    def create(self, **kwargs):
        """Mock both API endpoints"""
        payload = json.dumps(kwargs, default=str)
        content = self._process_request(payload)
        
        if "messages" in kwargs:
            # Chat completions API (traditional models)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content)
                )],
                usage=types.SimpleNamespace(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150
                )
            )
        else:
            # Responses API (GPT-5/O3 models)
            return types.SimpleNamespace(
                output_text=content,
                usage=types.SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150
                )
            )

class _FakeStep3Wrapper:
    def __init__(self):
        client = _FakeStep3Client()
        self.chat = types.SimpleNamespace(completions=client)
        self._responses = client
    
    @property
    def responses(self):
        return self._responses

class TestStage3Step3:
    """Test Step 3: O*NET-style filtering with style validation"""
    
    @pytest.fixture
    def fixtures_dir(self):
        return Path(__file__).parent / "fixtures"
    
    @pytest.fixture
    def golden_dir(self):
        return Path(__file__).parent / "golden"
    
    def test_step3_style_validation_mixed(self, fixtures_dir, golden_dir, tmp_path, monkeypatch):
        """Test Step 3 with valid tasks and style validation failures"""
        
        # Mock OpenAI client with style validation
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(
            OpenAI=lambda *a, **k: _FakeStep3Wrapper()
        ))
        
        input_file = fixtures_dir / "step3_input.csv"
        
        # Run Step 3
        result_path = s3.run_custom_file(
            str(input_file),
            "step2_output",
            "",
            "o3",
            False, False, "low", "medium",
            3  # step 3
        )
        
        # Step 3 stops pipeline when malformed responses are detected
        # result_path will be None when all responses fail style validation
        assert result_path is None, "Step 3 should return None when pipeline stops due to style validation failures"
        
        # Verify malformed file exists (Step 3 creates this and stops)
        malformed_path = Path(f"/Users/bradyallardice/Desktop/PhD/Projects/KurerAllardice2024/10CompaniesPOCAI/Data/custom_step3_step3_input_malformed.csv")
        assert malformed_path.exists(), "Step 3 should create malformed file when pipeline stops"
        
        malformed_df = pd.read_csv(malformed_path, dtype=str)
        assert len(malformed_df) >= 1, "Malformed file should contain problematic rows"
        
        # Clean up the malformed file for next test
        malformed_path.unlink(missing_ok=True)
    
    def test_step3_missing_step2_column_hard_error(self, tmp_path):
        """Test Step 3 hard error when step2_output column is missing"""
        
        bad_input = tmp_path / "bad_step3.csv"
        pd.DataFrame({
            "uid": [1],
            "title": ["Test"]
            # Missing step2_output column
        }).to_csv(bad_input, index=False)
        
        with pytest.raises((SystemExit, ValueError, KeyError)):
            s3.run_custom_file(
                str(bad_input),
                "step2_output",  # missing column
                "", "o3",
                False, False, "low", "medium", 3
            )
    
    def test_step3_all_style_failures_soft_exit(self, fixtures_dir, tmp_path, monkeypatch):
        """Test Step 3 when all tasks fail style validation - should exit 0"""
        
        class _AlwaysFailStyle:
            def __init__(self):
                self._responses = self
                self._chat = types.SimpleNamespace(completions=self)
            
            def create(self, **kwargs):
                content = "STYLE_FAIL"  # Always fail style validation
                if "messages" in kwargs:
                    return types.SimpleNamespace(
                        choices=[types.SimpleNamespace(
                            message=types.SimpleNamespace(content=content)
                        )],
                        usage=types.SimpleNamespace(
                            prompt_tokens=50,
                            completion_tokens=10,
                            total_tokens=60
                        )
                    )
                else:
                    return types.SimpleNamespace(
                        output_text=content,
                        usage=types.SimpleNamespace(
                            input_tokens=50,
                            output_tokens=10,
                            total_tokens=60
                        )
                    )
            
            @property
            def responses(self):
                return self
            
            @property
            def chat(self):
                return types.SimpleNamespace(completions=self)
        
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(
            OpenAI=lambda *a, **k: _AlwaysFailStyle()
        ))
        
        input_file = fixtures_dir / "step3_input.csv"
        
        # Step 3 should stop pipeline when all style validations fail
        result_path = s3.run_custom_file(
            str(input_file), "step2_output", "", "o3",
            False, False, "low", "medium", 3
        )
        
        # Step 3 returns None when pipeline stops due to style validation failures
        assert result_path is None, "Step 3 should return None when all responses fail style validation"
        
        # All rows should be in malformed file
        malformed_path = Path(f"/Users/bradyallardice/Desktop/PhD/Projects/KurerAllardice2024/10CompaniesPOCAI/Data/custom_step3_step3_input_malformed.csv")
        assert malformed_path.exists(), "Malformed file should be created when all responses fail style validation"
        malformed_df = pd.read_csv(malformed_path)
        assert len(malformed_df) == 3, "All 3 rows should be in malformed file"
        
        # Clean up
        malformed_path.unlink(missing_ok=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])