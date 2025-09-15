#!/usr/bin/env python3
"""
Stage 3 Step 2 Tests: Task Separation
Tests task separation with mixed valid/malformed inputs
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path
import types
import pytest
from unittest.mock import patch

sys.path.append(str(Path(__file__).parent.parent))
import stage_3_extract_ai_tasks as s3

class _FakeStep2Client:
    """Mock client that returns step 2 task separation results"""
    
    def __init__(self, good_mapping, bad_marker="not-json"):
        self.good_mapping = good_mapping
        self.bad_marker = bad_marker
    
    def _get_response(self, text):
        """Generate response based on input text content"""
        if self.bad_marker in text:
            return "not-json"
        
        # Check for recognizable content patterns
        for key, response in self.good_mapping.items():
            if key in text:
                return json.dumps(response)
        
        return "not-json"  # default to malformed
    
    def create(self, **kwargs):
        """Mock both chat.completions.create and responses.create"""
        payload = json.dumps(kwargs, default=str)
        content = self._get_response(payload)
        
        if "messages" in kwargs:
            # Chat API path (traditional models)
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
            # Responses API path (GPT-5/O3 models)
            return types.SimpleNamespace(
                output_text=content,
                usage=types.SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150
                )
            )

class _FakeClientWrapper:
    def __init__(self, good_mapping, bad_marker="not-json"):
        client = _FakeStep2Client(good_mapping, bad_marker)
        self.chat = types.SimpleNamespace(completions=client)
        self._responses = client
    
    @property
    def responses(self):
        return self._responses

class TestStage3Step2:
    """Test Step 2: Task separation with error handling"""
    
    @pytest.fixture
    def fixtures_dir(self):
        return Path(__file__).parent / "fixtures"
    
    @pytest.fixture
    def golden_dir(self):
        return Path(__file__).parent / "golden"
    
    def test_step2_mixed_soft_malformed(self, fixtures_dir, golden_dir, tmp_path, monkeypatch):
        """Test Step 2 with valid and malformed AI applications"""
        
        # Define mapping for fake responses
        good_mapping = {
            'model training with TensorFlow': [
                "train model using TensorFlow",
                "deploy model on SageMaker", 
                "monitor pipeline runs"
            ],
            'fine-tune BERT': [
                "fine-tune BERT",
                "use Vertex AI pipelines",
                "package and track with MLflow"
            ]
        }
        
        # Mock OpenAI client
        fake_client = _FakeClientWrapper(good_mapping)
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(
            OpenAI=lambda *a, **k: fake_client
        ))
        
        input_file = fixtures_dir / "step2_input.csv"
        
        # Run Step 2
        result_path = s3.run_custom_file(
            str(input_file),
            "ai_applications_raw",
            "",
            "gpt-5-mini",
            False, False, "medium", "medium", 
            2  # step 2
        )
        
        # Step 2 stops pipeline when malformed responses are detected
        # result_path will be None when all responses are malformed
        assert result_path is None, "Step 2 should return None when pipeline stops due to malformed responses"
        
        # Verify malformed file exists (Step 2 creates this and stops)
        malformed_path = Path(f"/Users/bradyallardice/Desktop/PhD/Projects/KurerAllardice2024/10CompaniesPOCAI/Data/custom_step2_step2_input_malformed.csv")
        assert malformed_path.exists(), "Step 2 should create malformed file when pipeline stops"
        
        malformed_df = pd.read_csv(malformed_path, dtype=str)
        assert len(malformed_df) >= 1, "Malformed file should contain problematic rows"
        
        # Clean up the malformed file for next test
        malformed_path.unlink(missing_ok=True)
    
    def test_step2_missing_input_column_hard_error(self, tmp_path):
        """Test Step 2 hard error when ai_applications_raw column missing"""
        
        bad_input = tmp_path / "bad_step2.csv"
        pd.DataFrame({
            "uid": [1],
            "title": ["Test"]
            # Missing ai_applications_raw column
        }).to_csv(bad_input, index=False)
        
        with pytest.raises((SystemExit, ValueError, KeyError)):
            s3.run_custom_file(
                str(bad_input),
                "ai_applications_raw",  # missing column
                "", "gpt-5-mini",
                False, False, "medium", "medium", 2
            )
    
    def test_step2_all_malformed_soft_exit(self, fixtures_dir, tmp_path, monkeypatch):
        """Test Step 2 when all inputs produce malformed JSON - should exit 0"""
        
        # Mock client that always returns malformed
        class _AlwaysBadStep2:
            def __init__(self):
                self._responses = self
                self._chat = types.SimpleNamespace(completions=self)
            
            def create(self, **kwargs):
                content = "completely-bad-json"
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
            OpenAI=lambda *a, **k: _AlwaysBadStep2()
        ))
        
        input_file = fixtures_dir / "step2_input.csv"
        
        # Step 2 should stop pipeline when all malformed
        result_path = s3.run_custom_file(
            str(input_file), "ai_applications_raw", "", "gpt-5-mini",
            False, False, "medium", "medium", 2
        )
        
        # Step 2 returns None when pipeline stops due to malformed responses
        assert result_path is None, "Step 2 should return None when all responses are malformed"
        
        # All rows should be in malformed file
        malformed_path = Path(f"/Users/bradyallardice/Desktop/PhD/Projects/KurerAllardice2024/10CompaniesPOCAI/Data/custom_step2_step2_input_malformed.csv")
        assert malformed_path.exists(), "Malformed file should be created when all responses fail"
        malformed_df = pd.read_csv(malformed_path)
        assert len(malformed_df) == 3, "All 3 rows should be in malformed file"
        
        # Clean up
        malformed_path.unlink(missing_ok=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])