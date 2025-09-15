#!/usr/bin/env python3
"""
Stage 3 Step 1 Tests: AI Application Extraction
Tests the complete pipeline with row-level error handling
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path
import types
import pytest
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import stage_3_extract_ai_tasks as s3

# ---- Fake OpenAI client for deterministic testing ----
class _FakeChoiceMsg:
    def __init__(self, content): 
        self.message = types.SimpleNamespace(content=content)

class _FakeChoice:
    def __init__(self, content): 
        self.choices = [types.SimpleNamespace(message=_FakeChoiceMsg(content))]

class _FakeResponsesObj:
    def create(self, **kwargs):
        # Simulate GPT-5/O3 responses.create() API
        prompt = json.dumps(kwargs, default=str)
        if "TensorFlow" in prompt and "SageMaker" in prompt:
            txt = '["model training with TensorFlow","deployment on SageMaker","mlops pipelines"]'
        elif "Vertex AI" in prompt and "MLflow" in prompt:
            txt = '["fine-tune BERT","use Vertex AI","deploy with MLflow"]'
        else:
            txt = ""  # Empty response to trigger malformed detection
        
        # GPT-5/O3 format with output_text and usage
        return types.SimpleNamespace(
            output_text=txt,
            usage=types.SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150
            )
        )

class _FakeChatCompletions:
    def create(self, **kwargs):
        # Simulate chat.completions API  
        messages = kwargs.get("messages", [])
        joined = " ".join(m.get("content","") for m in messages if isinstance(m, dict))
        
        if "TensorFlow" in joined and "SageMaker" in joined:
            content = '["model training with TensorFlow","deployment on SageMaker","mlops pipelines"]'
        elif "Vertex AI" in joined and "MLflow" in joined:
            content = '["fine-tune BERT","use Vertex AI","deploy with MLflow"]'
        else:
            content = ""  # Empty response to trigger malformed detection
            
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

class _FakeClient:
    def __init__(self, *a, **k):
        self._responses = _FakeResponsesObj()
        self._chat = types.SimpleNamespace(completions=_FakeChatCompletions())
    
    @property
    def responses(self): 
        return self._responses
    
    @property
    def chat(self): 
        return self._chat

class TestStage3Step1:
    """Test Step 1: AI application extraction with soft error handling"""
    
    @pytest.fixture
    def fixtures_dir(self):
        return Path(__file__).parent / "fixtures"
    
    @pytest.fixture  
    def golden_dir(self):
        return Path(__file__).parent / "golden"
        
    def test_step1_mixed_soft_malformed(self, fixtures_dir, golden_dir, tmp_path, monkeypatch):
        """Test Step 1 with mixed valid/malformed responses - soft error handling"""
        
        # Mock OpenAI client
        fake_openai = types.SimpleNamespace(OpenAI=_FakeClient)
        monkeypatch.setattr(s3, "openai", fake_openai, raising=True)
        
        # Input and expected output paths
        input_file = fixtures_dir / "step1_input.csv"
        output_file = tmp_path / "step1_output.csv"
        
        # Run Step 1 extraction
        result_path = s3.run_custom_file(
            str(input_file),
            "content_clean", 
            "",  # no suffix
            "o3",
            False,  # no batch
            False,  # no flex  
            "medium",  # reasoning effort
            "medium",  # verbosity
            1  # step
        )
        
        # Verify main output matches golden
        result_df = pd.read_csv(result_path, dtype=str)
        expected_df = pd.read_csv(golden_dir / "step1_output.csv", dtype=str)
        
        # Sort columns for comparison
        pd.testing.assert_frame_equal(
            result_df.sort_index(axis=1), 
            expected_df.sort_index(axis=1), 
            check_like=True
        )
        
        # Verify NO malformed file is created (empty responses are handled gracefully)
        malformed_path = Path(result_path).with_name(
            Path(result_path).stem + "_malformed.csv"
        )
        assert not malformed_path.exists(), "No malformed file should be created for empty responses"
        
    def test_step1_hard_error_missing_column(self, tmp_path):
        """Test Step 1 hard error when required column is missing"""
        
        # Create input with missing content_clean column
        bad_input = tmp_path / "bad_input.csv"
        pd.DataFrame({
            "uid": [1], 
            "title": ["Test"]
        }).to_csv(bad_input, index=False)
        
        # Should raise SystemExit or similar hard error
        with pytest.raises((SystemExit, ValueError, KeyError)):
            s3.run_custom_file(
                str(bad_input),
                "content_clean",  # missing column
                "",
                "o3", 
                False, False, "medium", "medium", 1
            )
            
    def test_step1_all_malformed_soft_exit(self, fixtures_dir, tmp_path, monkeypatch):
        """Test Step 1 when ALL rows are malformed - should still exit 0"""
        
        # Mock to return malformed JSON for everything
        class _AlwaysBadClient:
            def __init__(self, *a, **k):
                self._responses = self
                self._chat = types.SimpleNamespace(completions=self)
            
            def create(self, **kwargs):
                content = "completely-invalid-json-response"
                if "messages" in kwargs:
                    # Chat completions format
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
                    # Responses format (GPT-5/O3)
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
        
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(OpenAI=_AlwaysBadClient))
        
        input_file = fixtures_dir / "step1_input.csv"
        
        # Should complete successfully even with all malformed
        result_path = s3.run_custom_file(
            str(input_file), "content_clean", "", "o3", 
            False, False, "medium", "medium", 1
        )
        
        # Output file should exist
        assert Path(result_path).exists()
        
        # All rows should have the invalid response stored literally
        df = pd.read_csv(result_path, dtype=str)
        assert len(df) == 3
        
        # Verify all rows got the malformed content (Stage 3 stores it literally)
        for idx, row in df.iterrows():
            assert "completely-invalid-json-response" in str(row['ai_applications_raw'])
        
        # No malformed file should be created (Stage 3 handles this gracefully)
        malformed_path = Path(result_path).with_name(
            Path(result_path).stem + "_malformed.csv"
        )
        assert not malformed_path.exists(), "Stage 3 handles invalid responses gracefully - no malformed file expected"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])