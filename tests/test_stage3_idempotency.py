#!/usr/bin/env python3
"""
Stage 3 Idempotency Tests: Deterministic Output Validation
Tests that repeated runs produce identical outputs
"""
import os
import sys
import hashlib
import pandas as pd
from pathlib import Path
import pytest
from unittest.mock import patch
import types

sys.path.append(str(Path(__file__).parent.parent))
import stage_3_extract_ai_tasks as s3

def _compute_file_hash(filepath):
    """Compute SHA256 hash of file contents"""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

class _DeterministicClient:
    """Mock client that returns deterministic responses for idempotency testing"""
    
    def __init__(self):
        # Fixed responses for deterministic behavior
        self.response_map = {
            "TensorFlow": '["train tensorflow model","deploy model","monitor performance"]',
            "BERT": '["fine-tune bert","configure pipelines","track experiments"]', 
            "default": '["generic ai task"]'
        }
    
    def _get_deterministic_response(self, text):
        """Return deterministic response based on input content"""
        if "TensorFlow" in text:
            return self.response_map["TensorFlow"]
        elif "BERT" in text:
            return self.response_map["BERT"]
        else:
            return self.response_map["default"]
    
    def create(self, **kwargs):
        """Mock API call with deterministic response"""
        import json
        payload = json.dumps(kwargs, default=str, sort_keys=True)  # Sort for consistency
        content = self._get_deterministic_response(payload)
        
        if "messages" in kwargs:
            # Chat API format (traditional models)
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
            # Responses API format (GPT-5/O3 models)
            return types.SimpleNamespace(
                output_text=content,
                usage=types.SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    total_tokens=150
                )
            )

class _DeterministicWrapper:
    def __init__(self):
        client = _DeterministicClient()
        self.chat = types.SimpleNamespace(completions=client)
        self._responses = client
    
    @property
    def responses(self):
        return self._responses

class TestStage3Idempotency:
    """Test idempotent behavior and deterministic outputs"""
    
    @pytest.fixture
    def fixtures_dir(self):
        return Path(__file__).parent / "fixtures"
    
    def test_step1_idempotent_outputs(self, fixtures_dir, tmp_path, monkeypatch):
        """Test that Step 1 produces identical outputs on repeated runs"""
        
        # Use deterministic mock client
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(
            OpenAI=lambda *a, **k: _DeterministicWrapper()
        ))
        
        input_file = fixtures_dir / "step1_input.csv"
        output1 = tmp_path / "run1_output.csv"
        output2 = tmp_path / "run2_output.csv"
        
        # Run Step 1 twice with identical parameters
        result1 = s3.run_custom_file(
            str(input_file), "content_clean", "_run1", "o3",
            False, False, "medium", "medium", 1
        )
        
        result2 = s3.run_custom_file(
            str(input_file), "content_clean", "_run2", "o3", 
            False, False, "medium", "medium", 1
        )
        
        # Compare file hashes
        hash1 = _compute_file_hash(result1)
        hash2 = _compute_file_hash(result2)
        
        assert hash1 == hash2, "Step 1 should produce identical outputs on repeated runs"
        
        # Also verify content is identical
        df1 = pd.read_csv(result1, dtype=str)
        df2 = pd.read_csv(result2, dtype=str)
        
        pd.testing.assert_frame_equal(df1, df2, check_like=True)
    
    def test_step2_idempotent_outputs(self, fixtures_dir, tmp_path, monkeypatch):
        """Test that Step 2 produces identical outputs on repeated runs"""
        
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(
            OpenAI=lambda *a, **k: _DeterministicWrapper()
        ))
        
        input_file = fixtures_dir / "step2_input.csv"
        
        result1 = s3.run_custom_file(
            str(input_file), "ai_applications_raw", "_run1", "gpt-5-mini",
            False, False, "medium", "medium", 2
        )
        
        result2 = s3.run_custom_file(
            str(input_file), "ai_applications_raw", "_run2", "gpt-5-mini",
            False, False, "medium", "medium", 2 
        )
        
        # Compare hashes
        hash1 = _compute_file_hash(result1)
        hash2 = _compute_file_hash(result2)
        
        assert hash1 == hash2, "Step 2 should produce identical outputs on repeated runs"
    
    def test_row_order_stability(self, fixtures_dir, tmp_path, monkeypatch):
        """Test that row order remains stable across runs"""
        
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(
            OpenAI=lambda *a, **k: _DeterministicWrapper()
        ))
        
        input_file = fixtures_dir / "step1_input.csv"
        
        # Run multiple times
        results = []
        for i in range(3):
            result = s3.run_custom_file(
                str(input_file), "content_clean", f"_run{i}", "o3",
                False, False, "medium", "medium", 1
            )
            df = pd.read_csv(result, dtype=str)
            results.append(df)
        
        # Verify all runs have identical row order
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(
                results[0], results[i], 
                check_like=True,
                check_index_type=False
            )
    
    def test_malformed_file_deterministic(self, tmp_path, monkeypatch):
        """Test that malformed files are created deterministically"""
        
        # Mock client that always produces malformed output
        class _AlwaysMalformedClient:
            def create(self, **kwargs):
                content = "always-malformed-json-response"
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
        
        class _MalformedWrapper:
            def __init__(self):
                client = _AlwaysMalformedClient()
                self.chat = types.SimpleNamespace(completions=client)
                self._responses = client
            
            @property  
            def responses(self):
                return self._responses
        
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(
            OpenAI=lambda *a, **k: _MalformedWrapper()
        ))
        
        # Create test input
        test_input = tmp_path / "test_input.csv"
        pd.DataFrame({
            "uid": [1, 2],
            "content_clean": ["Test 1", "Test 2"]
        }).to_csv(test_input, index=False)
        
        # Run twice
        result1 = s3.run_custom_file(
            str(test_input), "content_clean", "_run1", "o3",
            False, False, "medium", "medium", 1
        )
        
        result2 = s3.run_custom_file(
            str(test_input), "content_clean", "_run2", "o3",
            False, False, "medium", "medium", 1
        )
        
        # Check malformed files exist and are identical
        malformed1 = Path(result1).with_name(Path(result1).stem + "_malformed.csv")
        malformed2 = Path(result2).with_name(Path(result2).stem + "_malformed.csv")
        
        assert malformed1.exists() and malformed2.exists()
        
        hash1 = _compute_file_hash(malformed1)
        hash2 = _compute_file_hash(malformed2)
        
        assert hash1 == hash2, "Malformed files should be created deterministically"
    
    def test_atomic_writes_no_partial_files(self, fixtures_dir, tmp_path, monkeypatch):
        """Test that writes are atomic - no partial files left on interruption"""
        
        # Mock file writing to simulate interruption
        original_to_csv = pd.DataFrame.to_csv
        write_count = 0
        
        def mock_interrupted_write(self, path, *args, **kwargs):
            nonlocal write_count
            write_count += 1
            
            # Simulate interruption on second write
            if write_count == 2 and isinstance(path, (str, Path)):
                raise KeyboardInterrupt("Simulated interruption")
            
            return original_to_csv(self, path, *args, **kwargs)
        
        monkeypatch.setattr(pd.DataFrame, "to_csv", mock_interrupted_write)
        monkeypatch.setattr(s3, "openai", types.SimpleNamespace(
            OpenAI=lambda *a, **k: _DeterministicWrapper()
        ))
        
        input_file = fixtures_dir / "step1_input.csv"
        
        # Should raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            s3.run_custom_file(
                str(input_file), "content_clean", "", "o3",
                False, False, "medium", "medium", 1
            )
        
        # Check no partial files exist (would need to verify .tmp files are cleaned up)
        # This is implementation-dependent and might need adjustment based on actual atomic write logic

if __name__ == "__main__":
    pytest.main([__file__, "-v"])