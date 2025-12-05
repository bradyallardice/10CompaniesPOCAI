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
import hashlib
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from collections import deque
import openai
from tqdm import tqdm

class AITaskExtractor:
    """Extract AI tasks from job advertisements using 3-step LLM prompting"""
    
    def __init__(self, model_name="gpt-4.1-mini", use_batch=False, use_flex=False, reasoning_effort="low", verbosity="medium",
                 max_attempts=2, max_new_calls=None, cost_budget=None, failure_rate_threshold=0.5, failure_window_size=10):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "Data"
        self.llm_output_dir = self.data_dir / "llm_output"
        self.llm_output_dir.mkdir(exist_ok=True)

        # Create cache directory
        self.cache_dir = self.data_dir / "llm_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Load environment variables
        load_dotenv('config.env')

        # Cost control and retry settings
        self.max_attempts = max_attempts
        self.max_new_calls = max_new_calls
        self.cost_budget = cost_budget
        self.failure_rate_threshold = failure_rate_threshold
        self.failure_window = deque(maxlen=failure_window_size)
        self.new_calls_count = 0
        
        # Set up OpenAI client and model configuration
        # Increase timeout for flex processing (up to 15 minutes as recommended)
        timeout = 900.0 if use_flex else 600.0  # 15 min for flex, 10 min for standard
        self.client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            timeout=timeout
        )
        self.model_name = model_name
        self.use_batch = use_batch
        self.use_flex = use_flex
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        
        # Initialize token tracking
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
        self.token_usage = []  # List to store individual API call details
        
        # OpenAI pricing structure (per 1M tokens, updated Jan 2025)
        self.pricing = {
            # GPT-5 Models (Latest Generation)
            "gpt-5": {
                "input": 1.25, "output": 10.00, "cached_input": 0.125
            },
            "gpt-5-mini": {
                "input": 0.25, "output": 2.00, "cached_input": 0.025
            },
            "gpt-5-nano": {
                "input": 0.05, "output": 0.40, "cached_input": 0.005
            },
            "gpt-5-chat-latest": {
                "input": 1.25, "output": 10.00, "cached_input": 0.125
            },
            
            # Latest GPT-4.1 Models
            "gpt-4.1": {
                "input": 2.00, "output": 8.00, "cached_input": 0.50
            },
            "gpt-4.1-2025-04-14": {
                "input": 2.00, "output": 8.00, "cached_input": 0.50
            },
            "gpt-4.1-mini": {
                "input": 0.40, "output": 1.60, "cached_input": 0.10
            },
            "gpt-4.1-mini-2025-04-14": {
                "input": 0.40, "output": 1.60, "cached_input": 0.10
            },
            "gpt-4.1-nano": {
                "input": 0.10, "output": 0.40, "cached_input": 0.025
            },
            "gpt-4.1-nano-2025-04-14": {
                "input": 0.10, "output": 0.40, "cached_input": 0.025
            },
            "gpt-4.5-preview": {
                "input": 75.00, "output": 150.00, "cached_input": 37.50
            },
            "gpt-4.5-preview-2025-02-27": {
                "input": 75.00, "output": 150.00, "cached_input": 37.50
            },
            
            # GPT-4o Models
            "gpt-4o": {
                "input": 2.50, "output": 10.00, "cached_input": 1.25
            },
            "gpt-4o-2024-08-06": {
                "input": 2.50, "output": 10.00, "cached_input": 1.25
            },
            "gpt-4o-audio-preview": {
                "input": 2.50, "output": 10.00
            },
            "gpt-4o-audio-preview-2024-12-17": {
                "input": 2.50, "output": 10.00
            },
            "gpt-4o-realtime-preview": {
                "input": 5.00, "output": 20.00, "cached_input": 2.50
            },
            "gpt-4o-realtime-preview-2025-06-03": {
                "input": 5.00, "output": 20.00, "cached_input": 2.50
            },
            "gpt-4o-mini": {
                "input": 0.15, "output": 0.60, "cached_input": 0.075
            },
            "gpt-4o-mini-2024-07-18": {
                "input": 0.15, "output": 0.60, "cached_input": 0.075
            },
            "gpt-4o-mini-audio-preview": {
                "input": 0.15, "output": 0.60
            },
            "gpt-4o-mini-audio-preview-2024-12-17": {
                "input": 0.15, "output": 0.60
            },
            "gpt-4o-mini-realtime-preview": {
                "input": 0.60, "output": 2.40, "cached_input": 0.30
            },
            "gpt-4o-mini-realtime-preview-2024-12-17": {
                "input": 0.60, "output": 2.40, "cached_input": 0.30
            },
            "gpt-4o-mini-search-preview": {
                "input": 0.15, "output": 0.60
            },
            "gpt-4o-mini-search-preview-2025-03-11": {
                "input": 0.15, "output": 0.60
            },
            "gpt-4o-search-preview": {
                "input": 2.50, "output": 10.00
            },
            "gpt-4o-search-preview-2025-03-11": {
                "input": 2.50, "output": 10.00
            },
            
            # o1 Series Models
            "o1": {
                "input": 15.00, "output": 60.00, "cached_input": 7.50
            },
            "o1-2024-12-17": {
                "input": 15.00, "output": 60.00, "cached_input": 7.50
            },
            "o1-pro": {
                "input": 150.00, "output": 600.00
            },
            "o1-pro-2025-03-19": {
                "input": 150.00, "output": 600.00
            },
            "o1-mini": {
                "input": 1.10, "output": 4.40, "cached_input": 0.55
            },
            "o1-mini-2024-09-12": {
                "input": 1.10, "output": 4.40, "cached_input": 0.55
            },
            
            # o3 Series Models
            "o3": {
                "input": 2.00, "output": 8.00, "cached_input": 0.50,
                "flex_input": 1.00, "flex_output": 4.00, "flex_cached_input": 0.25
            },
            "o3-2025-04-16": {
                "input": 2.00, "output": 8.00, "cached_input": 0.50,
                "flex_input": 1.00, "flex_output": 4.00, "flex_cached_input": 0.25
            },
            "o3-pro": {
                "input": 20.00, "output": 80.00
            },
            "o3-pro-2025-06-10": {
                "input": 20.00, "output": 80.00
            },
            "o3-deep-research": {
                "input": 10.00, "output": 40.00, "cached_input": 2.50
            },
            "o3-deep-research-2025-06-26": {
                "input": 10.00, "output": 40.00, "cached_input": 2.50
            },
            "o3-mini": {
                "input": 1.10, "output": 4.40, "cached_input": 0.55
            },
            "o3-mini-2025-01-31": {
                "input": 1.10, "output": 4.40, "cached_input": 0.55
            },
            
            # o4 Series Models
            "o4-mini": {
                "input": 1.10, "output": 4.40, "cached_input": 0.275,
                "flex_input": 0.55, "flex_output": 2.20, "flex_cached_input": 0.138
            },
            "o4-mini-2025-04-16": {
                "input": 1.10, "output": 4.40, "cached_input": 0.275,
                "flex_input": 0.55, "flex_output": 2.20, "flex_cached_input": 0.138
            },
            "o4-mini-deep-research": {
                "input": 2.00, "output": 8.00, "cached_input": 0.50
            },
            "o4-mini-deep-research-2025-06-26": {
                "input": 2.00, "output": 8.00, "cached_input": 0.50
            },
            
            # Other Models
            "codex-mini-latest": {
                "input": 1.50, "output": 6.00, "cached_input": 0.375
            },
            "computer-use-preview": {
                "input": 3.00, "output": 12.00
            },
            "computer-use-preview-2025-03-11": {
                "input": 3.00, "output": 12.00
            },
            "gpt-image-1": {
                "input": 5.00, "cached_input": 1.25
            },
            
            # Default fallback (using gpt-4o-mini pricing)
            "default": {
                "input": 0.15, "output": 0.60, "cached_input": 0.075
            }
        }
    
    def _get_model_pricing(self, model_name, use_batch=False, use_flex=False):
        """Get pricing for a specific model"""
        # Clean model name (remove version suffixes)
        clean_model = model_name.lower().strip()
        
        # Exact match first
        if clean_model in self.pricing:
            pricing = self.pricing[clean_model]
            if use_flex and "flex_input" in pricing:
                return pricing["flex_input"], pricing["flex_output"]
            elif use_batch and "cached_input" in pricing:
                return pricing["cached_input"], pricing["output"]
            else:
                return pricing["input"], pricing["output"]
        
        # Partial match for versioned models
        for model_key in self.pricing:
            if model_key in clean_model or clean_model.startswith(model_key):
                pricing = self.pricing[model_key]
                if use_flex and "flex_input" in pricing:
                    return pricing["flex_input"], pricing["flex_output"]
                elif use_batch and "cached_input" in pricing:
                    return pricing["cached_input"], pricing["output"]
                else:
                    return pricing["input"], pricing["output"]
        
        # Fallback to default pricing
        pricing = self.pricing["default"]
        if use_flex and "flex_input" in pricing:
            return pricing["flex_input"], pricing["flex_output"]
        elif use_batch and "cached_input" in pricing:
            return pricing["cached_input"], pricing["output"]
        else:
            return pricing["input"], pricing["output"]
    
    def _is_gpt5_model(self):
        """Check if the model uses the responses API (GPT-5 and O3 families)"""
        model_lower = self.model_name.lower()
        return model_lower.startswith('gpt-5') or model_lower.startswith('o3')
    
    def _supports_flex(self):
        """Check if the model supports flex processing"""
        model_lower = self.model_name.lower()
        # According to OpenAI docs, flex is available for GPT-5, O3, and O4-mini models
        return (model_lower.startswith('gpt-5') or 
                model_lower.startswith('o3') or 
                model_lower.startswith('o4-mini'))
    
    def _make_api_call_with_retry(self, api_call_func, max_retries=3):
        """
        Wrapper for API calls with retry logic for flex processing
        Handles 429 Resource Unavailable errors for flex processing
        """
        import time
        import random
        
        for attempt in range(max_retries + 1):
            try:
                return api_call_func()
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for flex-specific resource unavailable error
                if "resource unavailable" in error_str or "429" in error_str:
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        print(f"  Flex resource unavailable (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Final retry with standard processing
                        print(f"  Flex processing failed after {max_retries} retries, falling back to standard processing...")
                        if self.use_flex:
                            # Temporarily disable flex and retry once
                            original_flex = self.use_flex
                            self.use_flex = False
                            try:
                                result = api_call_func()
                                print("  ✅ Fallback to standard processing successful")
                                return result
                            finally:
                                self.use_flex = original_flex
                        raise e
                else:
                    # Non-flex related error, raise immediately
                    raise e
        
        return None  # Should not reach here
    
    def _get_model_params(self, max_tokens_value=500, verbosity="medium"):
        """Get the correct parameters for the model"""
        params = {}
        
        # GPT-5 models use new API structure
        if self._is_gpt5_model():
            # GPT-5 uses responses.create() with different parameter structure
            # No traditional parameters needed here as they're handled in the API call
            return params
        # o-series models (o3, o4-mini, etc.) use different parameters
        elif self.model_name.startswith('o'):
            # For o3 model: $2/1M input, $8/1M output
            # With ~1500 input tokens ($0.003) + 10,000 output tokens ($0.08) = ~$0.083 total
            # This keeps us well under the 10 cent per call limit
            params["max_completion_tokens"] = 10000  # Safe limit for 10 cent budget
            params["reasoning_effort"] = self.reasoning_effort  # low, medium, or high
            # o3 models don't support temperature=0, only default (1)
            params["temperature"] = 1
        else:
            params["max_tokens"] = max_tokens_value
            params["temperature"] = 0
        
        return params
    
    def _make_api_call_simple(self, system_prompt, user_prompt, max_tokens_value=500, 
                             temperature=0, top_p=1, seed=42, response_format=None):
        """Simplified API call using only chat completions endpoint for all models"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Simple parameters that work with all models
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens_value,
            "temperature": temperature,
            "top_p": top_p
        }
        
        # Add optional parameters
        if seed is not None:
            call_params["seed"] = seed
        if response_format is not None:
            call_params["response_format"] = response_format
        
        # Add reasoning effort for o-series models if specified
        if self.model_name.startswith('o') and hasattr(self, 'reasoning_effort'):
            call_params["reasoning_effort"] = self.reasoning_effort
        
        # Add flex processing if enabled for supported models
        if self.use_flex and self._supports_flex():
            call_params["service_tier"] = "flex"
            
        return self.client.chat.completions.create(**call_params)

    def _make_api_call(self, system_prompt, user_prompt, max_tokens_value=500, verbosity=None, 
                       temperature=None, top_p=None, seed=None, response_format=None):
        """Make API call using appropriate method based on model type"""
        
        # Use instance verbosity if not specified
        if verbosity is None:
            verbosity = self.verbosity
        
        # Define the actual API call as a lambda for retry wrapper
        if self._is_gpt5_model():
            # GPT-5/O3 use the responses.create() API
            combined_input = f"{system_prompt}\n\n{user_prompt}"
            
            def make_responses_call():
                call_params = {
                    "model": self.model_name,
                    "input": combined_input,
                    "reasoning": {
                        "effort": self.reasoning_effort
                    },
                    "text": {
                        "verbosity": verbosity or "medium"
                    }
                }
                
                # Add flex processing if enabled
                if self.use_flex and self._supports_flex():
                    call_params["service_tier"] = "flex"
                
                return self.client.responses.create(**call_params)
            
            # Use retry wrapper for flex processing
            if self.use_flex and self._supports_flex():
                return self._make_api_call_with_retry(make_responses_call)
            else:
                return make_responses_call()
        else:
            # Other models use simplified chat completions
            def make_chat_call():
                return self._make_api_call_simple(
                    system_prompt, user_prompt, max_tokens_value,
                    temperature or 0, top_p or 1, seed, response_format
                )
            
            # Use retry wrapper for flex processing
            if self.use_flex and self._supports_flex():
                return self._make_api_call_with_retry(make_chat_call)
            else:
                return make_chat_call()
        
        # ORIGINAL CODE (commented out temporarily):
        # if self._is_gpt5_model():
        #     # GPT-5 uses the new responses.create() API
        #     # Combine system and user prompts for GPT-5
        #     combined_input = f"{system_prompt}\n\n{user_prompt}"
        #     
        #     # Build GPT-5 specific parameters
        #     call_params = {
        #         "model": self.model_name,
        #         "input": combined_input,
        #         "reasoning": {
        #             "effort": self.reasoning_effort
        #         },
        #         "text": {
        #             "verbosity": verbosity
        #         }
        #     }
        #     
        #     # GPT-5 doesn't use traditional parameters like temperature, seed, etc.
        #     return self.client.responses.create(**call_params)
        #     
        # else:
        #     # Traditional models use chat.completions.create()
        #     messages = [
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": user_prompt}
        #     ]
        #     
        #     # Build traditional parameters
        #     call_params = {
        #         "model": self.model_name,
        #         "messages": messages,
        #         **self._get_model_params(max_tokens_value, verbosity)
        #     }
        #     
        #     # Add optional parameters if specified
        #     if temperature is not None:
        #         call_params["temperature"] = temperature
        #     if top_p is not None:
        #         call_params["top_p"] = top_p
        #     if seed is not None:
        #         call_params["seed"] = seed
        #     if response_format is not None:
        #         call_params["response_format"] = response_format
        #         
        #     return self.client.chat.completions.create(**call_params)
    
    def _extract_response_content(self, response):
        """Extract content from response based on model type"""
        if self._is_gpt5_model():
            # GPT-5/O3 responses have output_text attribute
            return response.output_text
        else:
            # Traditional models have choices[0].message.content
            return response.choices[0].message.content.strip()
        
        # ORIGINAL CODE (commented out temporarily):
        # if self._is_gpt5_model():
        #     # GPT-5 responses have output_text attribute
        #     return response.output_text
        # else:
        #     # Traditional models have choices[0].message.content
        #     return response.choices[0].message.content.strip()
    
    def _track_token_usage(self, response, model_name=None, use_batch=None, use_flex=None):
        """Track token usage and cost for API calls"""
        # Use instance variables as defaults
        model_name = model_name or self.model_name
        use_batch = use_batch if use_batch is not None else self.use_batch
        use_flex = use_flex if use_flex is not None else self.use_flex
        
        if hasattr(response, 'usage'):
            # Handle both chat completions and responses API usage formats
            if hasattr(response.usage, 'prompt_tokens'):
                # Chat completions API format
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
            else:
                # Responses API format (GPT-5, O3, etc.)
                prompt_tokens = response.usage.input_tokens
                completion_tokens = response.usage.output_tokens
                total_tokens = response.usage.total_tokens
            
            # Update totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            self.api_calls += 1
            
            # Get dynamic pricing
            input_price, output_price = self._get_model_pricing(model_name, use_batch, use_flex)
            
            # Calculate cost
            prompt_cost = (prompt_tokens / 1_000_000) * input_price
            completion_cost = (completion_tokens / 1_000_000) * output_price
            call_cost = prompt_cost + completion_cost
            
            self.total_cost += call_cost
            
            # Create processing indicator
            processing_type = ""
            if use_flex:
                processing_type = " (FLEX)"
            elif use_batch:
                processing_type = " (BATCH)"
            
            print(f"  API Call {self.api_calls}{processing_type}: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} tokens (${call_cost:.4f}) [{model_name}]")
            
            # Store individual call data for per-job tracking
            call_data = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'cost': call_cost,
                'model': model_name,
                'batch': use_batch,
                'flex': use_flex,
                'input_price_per_1m': input_price,
                'output_price_per_1m': output_price
            }
            self.token_usage.append(call_data)
            
            return call_data
        return None
    
    def print_token_summary(self):
        """Print summary of token usage and costs"""
        print(f"\n=== TOKEN USAGE SUMMARY ===")
        print(f"Total API calls: {self.api_calls}")
        print(f"Total prompt tokens: {self.total_prompt_tokens:,}")
        print(f"Total completion tokens: {self.total_completion_tokens:,}")
        print(f"Total tokens: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.4f}")
        if self.api_calls > 0:
            print(f"Average tokens per call: {self.total_tokens / self.api_calls:.1f}")
            print(f"Average cost per call: ${self.total_cost / self.api_calls:.4f}")
        print(f"==========================\n")

    # ===== CACHE SYSTEM =====

    def _get_cache_key(self, model, system_prompt, user_prompt, params):
        """Generate deterministic SHA256 cache key from API call parameters"""
        # Combine all parameters that affect the response
        cache_input = json.dumps({
            'model': model,
            'system': system_prompt,
            'user': user_prompt,
            'params': params
        }, sort_keys=True)
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _read_cache(self, cache_key, step_name):
        """Read from JSONL cache file for a specific step"""
        cache_file = self.cache_dir / f"step{step_name}.jsonl"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get('cache_key') == cache_key:
                            return entry.get('response')
        except Exception as e:
            print(f"  Warning: Cache read error: {e}")

        return None

    def _write_cache(self, cache_key, response, step_name):
        """Write to JSONL cache file for a specific step"""
        cache_file = self.cache_dir / f"step{step_name}.jsonl"

        try:
            entry = {
                'cache_key': cache_key,
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            with open(cache_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"  Warning: Cache write error: {e}")

    def _check_cost_guards(self):
        """Check if cost/call limits have been exceeded"""
        # Check max new calls
        if self.max_new_calls is not None and self.new_calls_count >= self.max_new_calls:
            raise RuntimeError(f"Maximum new API calls limit reached: {self.max_new_calls}")

        # Check cost budget
        if self.cost_budget is not None and self.total_cost >= self.cost_budget:
            raise RuntimeError(f"Cost budget exceeded: ${self.total_cost:.2f} >= ${self.cost_budget:.2f}")

        # Check failure rate
        if len(self.failure_window) == self.failure_window.maxlen:
            failure_rate = sum(self.failure_window) / len(self.failure_window)
            if failure_rate >= self.failure_rate_threshold:
                raise RuntimeError(
                    f"Failure rate too high: {failure_rate:.1%} >= {self.failure_rate_threshold:.1%} "
                    f"over last {len(self.failure_window)} calls"
                )

    # ===== JSON REPAIR SYSTEM =====

    def _fix_common_json_issues(self, response_text):
        """
        Attempt to repair common JSON formatting issues without calling the API.

        Args:
            response_text: Raw LLM response that may contain malformed JSON

        Returns:
            tuple: (repaired_json_string or None, was_repaired: bool)
        """
        if not response_text:
            return None, False

        original = response_text.strip()

        # Step 1: Strip code fences (```json ... ``` or ``` ... ```)
        if '```' in original:
            # Find content between code fences
            matches = re.findall(r'```(?:json)?\s*(.*?)\s*```', original, re.DOTALL)
            if matches:
                original = matches[0].strip()

        # Step 2: Extract JSON by bracket/brace counting
        # Find the first { or [
        start_brace = original.find('{')
        start_bracket = original.find('[')

        if start_brace == -1 and start_bracket == -1:
            return None, False

        # Determine which comes first
        if start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket):
            start_char = '{'
            end_char = '}'
            start_idx = start_brace
        else:
            start_char = '['
            end_char = ']'
            start_idx = start_bracket

        # Count braces/brackets to find matching closing
        count = 0
        end_idx = -1
        for i in range(start_idx, len(original)):
            if original[i] == start_char:
                count += 1
            elif original[i] == end_char:
                count -= 1
                if count == 0:
                    end_idx = i
                    break

        if end_idx != -1:
            extracted = original[start_idx:end_idx + 1]

            # Step 3: Try to parse the extracted JSON
            try:
                json.loads(extracted)
                return extracted, True
            except json.JSONDecodeError:
                # Try fixing common issues
                # Remove trailing commas
                fixed = re.sub(r',(\s*[}\]])', r'\1', extracted)
                try:
                    json.loads(fixed)
                    return fixed, True
                except json.JSONDecodeError:
                    pass

        # If all repairs failed, return None
        return None, False

    # ===== MALFORMED RESPONSE RECOVERY SYSTEM =====
    
    def _detect_malformed_response(self, response_text, step_name, expected_format="json"):
        """
        Detect if an LLM response is malformed and cannot be parsed.
        
        Args:
            response_text: Raw LLM response text
            step_name: Name of the step (for error context)
            expected_format: Expected response format ("json", "list", etc.)
            
        Returns:
            tuple: (is_malformed: bool, error_message: str or None)
        """
        if not response_text or not response_text.strip():
            return True, f"Empty response from {step_name}"
        
        if expected_format == "json":
            try:
                parsed = json.loads(response_text)
                
                # Additional validation based on step
                if step_name == "step1":
                    # Step 1 should have specific keys
                    if isinstance(parsed, dict):
                        if not ('Key Application' in parsed or 'applications' in parsed):
                            return True, f"Step 1 JSON missing expected keys: {list(parsed.keys())}"
                    elif isinstance(parsed, list):
                        if parsed and not isinstance(parsed[0], dict):
                            return True, f"Step 1 list contains non-dict items"
                        if parsed and not any(key in parsed[0] for key in ['Key Application', 'key_application']):
                            return True, f"Step 1 list items missing expected keys"
                    else:
                        return True, f"Step 1 unexpected JSON type: {type(parsed)}"
                        
                elif step_name == "step2":
                    # Step 2 should return plain text with separated tasks
                    return True, "Step 2 should return plain text, not JSON"
                    
                elif step_name == "step3":
                    # Step 3 should have 'ai_task' key
                    if isinstance(parsed, dict):
                        if 'ai_task' not in parsed:
                            return True, f"Step 3 JSON missing 'ai_task' key: {list(parsed.keys())}"
                    else:
                        return True, f"Step 3 should return dict with 'ai_task' key, got: {type(parsed)}"
                
                return False, None  # Valid JSON
                
            except json.JSONDecodeError as e:
                if step_name == "step2":
                    # Step 2 should NOT be JSON, so this is expected
                    pass
                else:
                    return True, f"JSON decode error in {step_name}: {str(e)}"
        
        elif expected_format == "text":
            # Step 2 plain text validation
            if step_name == "step2":
                # Check if response contains valid tasks (single task is acceptable)
                lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
                if not lines:
                    return True, "Step 2 response contains no separated tasks"
                # Single task is valid - some AI capabilities are atomic and don't need separation
                # Check for common malformed patterns
                if any(line.lower().startswith('sorry') or line.lower().startswith('i cannot') for line in lines):
                    return True, "Step 2 response contains refusal or error message"
                return False, None  # Valid text
        
        return False, None  # Default: assume valid if no format specified
    
    def _save_malformed_responses(self, malformed_data, step_name, dataset_type="train"):
        """
        Save malformed responses to a separate CSV with ledger tracking.
        Merges with existing malformed file, increments attempts, and removes resolved rows.

        Args:
            malformed_data: List of dictionaries with malformed response data
                           Expected keys: uid, error_message, raw_llm_response, etc.
            step_name: Name of the step (step1, step2, step3)
            dataset_type: Dataset type (train/test)
        """
        if not malformed_data:
            return None

        malformed_file = self.data_dir / f"{step_name}_malformed.csv"
        current_timestamp = datetime.now().isoformat()

        # Add ledger fields to new malformed data
        new_malformed_df = pd.DataFrame(malformed_data)

        # Initialize ledger fields for new entries
        if 'status' not in new_malformed_df.columns:
            new_malformed_df['status'] = 'pending'
        if 'attempts' not in new_malformed_df.columns:
            new_malformed_df['attempts'] = 1
        if 'last_error_type' not in new_malformed_df.columns:
            new_malformed_df['last_error_type'] = new_malformed_df.get('error_type', 'unknown')
        if 'last_error_message' not in new_malformed_df.columns:
            new_malformed_df['last_error_message'] = new_malformed_df.get('error_message', '')
        if 'last_attempt_at' not in new_malformed_df.columns:
            new_malformed_df['last_attempt_at'] = current_timestamp

        # Load existing malformed file if it exists
        if malformed_file.exists():
            existing_df = pd.read_csv(malformed_file)

            # Merge logic: increment attempts for UIDs that are still failing
            merged_rows = []
            new_uids = set(new_malformed_df['uid'])
            existing_uids = set(existing_df['uid'])

            # Process existing rows
            for _, row in existing_df.iterrows():
                uid = row['uid']
                if uid in new_uids:
                    # Still failing - increment attempts
                    new_row = new_malformed_df[new_malformed_df['uid'] == uid].iloc[0].to_dict()
                    attempts = row.get('attempts', 0) + 1
                    new_row['attempts'] = attempts

                    # Update status based on attempts
                    if attempts >= self.max_attempts:
                        new_row['status'] = 'needs_manual'
                    else:
                        new_row['status'] = 'failed'

                    new_row['last_attempt_at'] = current_timestamp
                    merged_rows.append(new_row)
                else:
                    # Not in new failures - this row was resolved, don't keep it
                    pass

            # Add completely new failures (not in existing)
            for _, row in new_malformed_df.iterrows():
                if row['uid'] not in existing_uids:
                    merged_rows.append(row.to_dict())

            final_df = pd.DataFrame(merged_rows)
        else:
            # No existing file, just use new data
            final_df = new_malformed_df

        # Save merged result
        final_df.to_csv(malformed_file, index=False)

        # Print summary
        needs_manual = len(final_df[final_df['status'] == 'needs_manual'])
        failed = len(final_df[final_df['status'] == 'failed'])
        pending = len(final_df[final_df['status'] == 'pending'])

        print(f"🚨 MALFORMED RESPONSES: {len(final_df)} total")
        print(f"   📊 Status breakdown: {pending} pending, {failed} failed, {needs_manual} needs_manual")
        print(f"   💾 Saved to: {malformed_file}")
        print(f"   📝 Columns: {list(final_df.columns)}")

        if needs_manual > 0:
            print(f"   ⚠️  {needs_manual} rows need manual intervention (>= {self.max_attempts} attempts)")

        return malformed_file
    
    def _detect_reprocessing_mode(self, step_name, dataset_type="train"):
        """
        Detect if we're in reprocessing mode (malformed file exists and main output exists).
        
        Args:
            step_name: Name of the step (step1, step2, step3)
            dataset_type: Dataset type (train/test)
            
        Returns:
            tuple: (is_reprocessing: bool, malformed_file_path: Path or None, main_output_path: Path or None)
        """
        malformed_file = self.data_dir / f"{step_name}_malformed.csv"
        
        # Find the main output file for this step
        patterns = {
            "step1": f"ai_applications_{dataset_type}_*.csv",
            "step2": f"ai_applications_separated_{dataset_type}_*.csv", 
            "step3": f"ai_applications_filtered_{dataset_type}_*.csv"
        }
        
        main_output_files = list(self.data_dir.glob(patterns[step_name]))
        main_output_path = sorted(main_output_files)[-1] if main_output_files else None
        
        is_reprocessing = malformed_file.exists() and main_output_path and main_output_path.exists()
        
        if is_reprocessing:
            print(f"🔄 REPROCESSING MODE detected for {step_name}")
            print(f"   Malformed file: {malformed_file}")
            print(f"   Main output file: {main_output_path}")
        
        return is_reprocessing, malformed_file if malformed_file.exists() else None, main_output_path
    
    def _merge_reprocessed_results(self, main_output_path, new_results_df, merge_key='uid'):
        """
        Merge reprocessed results with existing main output file.
        
        Args:
            main_output_path: Path to main output CSV file
            new_results_df: DataFrame with newly processed results
            merge_key: Column to merge on (default: 'uid')
            
        Returns:
            Path to updated main output file
        """
        if not main_output_path.exists():
            print(f"⚠️ Main output file not found: {main_output_path}")
            return None
        
        # Load existing results
        existing_df = pd.read_csv(main_output_path)
        print(f"📂 Loaded {len(existing_df)} existing results from {main_output_path}")
        
        # Remove any existing entries for the same UIDs (in case of re-reprocessing)
        new_uids = set(new_results_df[merge_key].unique())
        existing_df_filtered = existing_df[~existing_df[merge_key].isin(new_uids)]
        
        # Combine with new results
        combined_df = pd.concat([existing_df_filtered, new_results_df], ignore_index=True)
        
        # Save back to main output file
        combined_df.to_csv(main_output_path, index=False)
        
        print(f"✅ Merged {len(new_results_df)} new results with {len(existing_df_filtered)} existing results")
        print(f"📊 Total results: {len(combined_df)} in {main_output_path}")
        
        return main_output_path
    
    def _check_pipeline_continuation(self, step_name, malformed_file_path):
        """
        Check if pipeline can continue or must stop due to malformed responses.
        
        Args:
            step_name: Name of the step
            malformed_file_path: Path to malformed responses file (None if no malformed responses)
            
        Returns:
            bool: True if pipeline can continue, False if must stop
        """
        if malformed_file_path and malformed_file_path.exists():
            # Check if malformed file has any rows
            try:
                malformed_df = pd.read_csv(malformed_file_path)
                if len(malformed_df) > 0:
                    print(f"\n{'='*60}")
                    print(f"🛑 PIPELINE STOPPED AT {step_name.upper()}")
                    print(f"{'='*60}")
                    print(f"🚨 {len(malformed_df)} responses could not be parsed after all repair attempts")
                    print(f"📁 Malformed responses saved to: {malformed_file_path}")
                    print(f"🔧 Please fix the malformed responses and rerun {step_name}")
                    print(f"💡 The system will automatically detect reprocessing and merge results")
                    print(f"{'='*60}\n")
                    return False
            except Exception as e:
                print(f"⚠️ Error checking malformed file: {e}")
                return False
        
        return True  # Can continue
        
    def load_deduplicated_data(self):
        """Load the latest deduplicated and translated AI jobs data (training set)"""
        # First try training companies pattern (current approach)
        pattern = "ai_development_training_companies_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            # Fallback to deduplicated/translated train file pattern
            pattern = "ai_development_deduplicated_translated_train_*.csv"
            files = list(self.data_dir.glob(pattern))
        
        if not files:
            # Fallback to old pattern without train/test suffix
            pattern = "ai_development_deduplicated_translated_*.csv"
            files = [f for f in self.data_dir.glob(pattern) if 'test' not in f.name]
        
        if not files:
            raise FileNotFoundError(f"No training files found matching any pattern")
        
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
    
    def load_dev_data(self):
        """Load dev data split"""
        dev_file = self.data_dir / "dev_data_split.csv"
        
        if not dev_file.exists():
            raise FileNotFoundError(f"Dev data file not found: {dev_file}")
        
        df = pd.read_csv(dev_file)
        print(f"Loaded {len(df)} job advertisements from dev set")
        
        return df
    
    def load_data(self, dataset_type="train"):
        """Load training, dev, or test data"""
        if dataset_type == "train":
            return self.load_deduplicated_data()
        elif dataset_type == "test":
            return self.load_test_data()
        elif dataset_type == "dev":
            return self.load_dev_data()
        else:
            raise ValueError(f"dataset_type must be 'train', 'dev', or 'test', got '{dataset_type}'")
    
    def load_custom_file(self, file_path):
        """Load data from a custom CSV file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Custom input file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} job advertisements from: {file_path}")
        
        return df
    
    def step_1_extract_ai_applications_v2(self, job_ads_text):
        """
        Step 1: Extract AI applications with improved few-shot examples focusing on completeness
        """
        system_prompt = """You are an occupational‑impact analyst.  
        Your job: given a job advertisement, list every task that specific AI/ML application or ai/ml-enabled tools mentioned in the job description will perform or automate. 
        ⚠️ Do **not** include tasks that human software engineers will do while building, deploying, or maintaining the AI/ML system, but rather include what the systems or tools themselves do. 


        Step‑by‑step instructions  
        1. **Parse AI capabilities**  
        Extract every distinct function, feature, or workflow an AI/ML system in the advertizement is intended to deliver (e.g., "detect surface defects on steel coils," "generate marketing copy," "forecast spare‑parts demand").  
        2. **Deliver structured output**  
        Return a JSON array named ai_application_tasks, where each element has:
        {
            "task_id": sequential integer starting at 1,
            "ai_capability": "one to two sentences describing the AI capability",
            "brief_description": "one‑sentence clarification drawn from the job ad (in english)",
            "raw_excerpt": "verbatim excerpt that supports the use case, left in the ad's original language",
        }"""

        user_prompt = job_ads_text

        try:
            response = self._make_api_call(
                system_prompt=system_prompt,
                user_prompt=f'"""\n{user_prompt}\n"""',
                max_tokens_value=500,
                verbosity="medium",
                temperature=0,
                top_p=1,
                seed=42,
                response_format={"type": "json_object"}
            )
            
            # Track token usage
            self._track_token_usage(response)
            
            return self._extract_response_content(response)
        
        except Exception as e:
            print(f"Error in step 1 v2: {e}")
            return None

    def step_1_extract_ai_applications_ensemble(self, job_ads_text, num_models=5, temperature=0.5):
        """
        Step 1: Extract AI applications using ensemble of LLMs with majority voting
        
        Args:
            job_ads_text: The job advertisement text
            num_models: Number of LLM runs in the ensemble (default: 5)
            temperature: Temperature for sampling diversity (default: 0.5)
        
        Returns:
            JSON string with consensus AI capabilities
        """
        system_prompt = """ 
        You are an occupational‑impact analyst.  
        Your job: given a job advertisement, list every task that specific AI/ML application or ai/ml-enabled tools mentioned in the job description will perform or automate. 
        ⚠️ Do **not** include tasks that human software engineers will do while building, deploying, or maintaining the AI/ML system, but rather include what the systems or tools themselves do. 

        GOOD EXAMPLES (different types, varying specificity levels):

        1. "Deliver machine-learning capabilities within customer solutions to support innovative business ideas."
        (Somewhat broad but has specific purpose - to support innovative business ideas)

        2. "Automate ingestion, cleansing, transformation, enrichment and delivery of data via Databricks pipelines"
        (Mentions what the AI/ML capabilities in Databricks do - specific data processing functions)

        3. "Artificial-intelligence algorithms that automate or optimise machine-control and other digital dentistry processes to improve or automate CAD/CAM machine operations or decision-making steps."
        (Very specific - mentions the operations the AI/ML is used for, even when job ad doesn't clearly state it)

        4. "Automatically generate actionable strategic insights for leadership based on statistical and machine-learning analysis."
        (Simple extraction showing clear AI/ML capability and its business purpose)

        BAD EXAMPLES (non-AI/ML tasks that should NOT be extracted):
        ❌ "Create presentations via PowerPoint to present to key stakeholders"
        ❌ "Coordinate meetings with development teams using project management software"
        ❌ "Write technical documentation and user manuals"
        ❌ "Conduct code reviews and manage software deployments"
        ❌ "Analyze business requirements and create system specifications"


        Step‑by‑step instructions  
        1. **Parse AI capabilities**  
        Extract every distinct function, feature, or workflow an AI/ML system in the advertizement is intended to deliver (e.g., "detect surface defects on steel coils," "generate marketing copy," "forecast spare‑parts demand").  
        2. **Deliver structured output**  
        Return a JSON array named ai_application_tasks, where each element has:
        {
            "task_id": sequential integer starting at 1,
            "ai_capability": "one to two sentences describing the AI capability",
            "brief_description": "one‑sentence clarification drawn from the job ad (in english)",
            "raw_excerpt": "verbatim excerpt that supports the use case, left in the ad's original language",
            "explanation": "one to two sentence explanation for why this task was chosen"
        }
   """

        user_prompt = job_ads_text
        
        # Collect responses from ensemble
        ensemble_responses = []
        ensemble_capabilities = []
        
        print(f"Running ensemble of {num_models} models with temperature {temperature}...")
        
        for i in range(num_models):
            try:
                print(f"  Model {i+1}/{num_models}...")
                response = self._make_api_call(
                    system_prompt=system_prompt,
                    user_prompt=f"Job Advertisement:\n\n{user_prompt}",
                    max_tokens_value=500,
                    verbosity="medium",
                    temperature=temperature  # Higher temperature for diversity
                )
                
                # Track token usage
                self._track_token_usage(response)
                
                response_content = self._extract_response_content(response)
                ensemble_responses.append(response_content)
                
                # Parse the JSON response to extract capabilities
                try:
                    import json
                    import re
                    
                    # Clean response content - remove markdown code blocks
                    clean_content = response_content.strip()
                    if clean_content.startswith('```json'):
                        clean_content = clean_content[7:]  # Remove ```json
                    if clean_content.endswith('```'):
                        clean_content = clean_content[:-3]  # Remove ```
                    clean_content = clean_content.strip()
                    
                    # Try to extract JSON from response (sometimes there's extra text)
                    json_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
                    if json_match:
                        json_text = json_match.group()
                        # Try to fix common JSON issues
                        json_text = self._fix_common_json_issues(json_text)
                        parsed = json.loads(json_text)
                    else:
                        # Try to fix common JSON issues in the clean content
                        clean_content = self._fix_common_json_issues(clean_content)
                        parsed = json.loads(clean_content)
                    
                    if "ai_application_tasks" in parsed:
                        capabilities = [task["ai_capability"] for task in parsed["ai_application_tasks"]]
                        ensemble_capabilities.append(capabilities)
                        print(f"    Successfully extracted {len(capabilities)} capabilities from response {i+1}")
                    else:
                        print(f"    Warning: No ai_application_tasks found in response {i+1}")
                        ensemble_capabilities.append([])
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    print(f"    Warning: Could not parse JSON from response {i+1}: {e}")
                    # Try to extract capabilities using text parsing as fallback
                    capabilities = self._fallback_capability_extraction(response_content)
                    ensemble_capabilities.append(capabilities)
                    if capabilities:
                        print(f"    Fallback extraction found {len(capabilities)} capabilities")
                
                # Small delay between requests
                import time
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    Error in ensemble model {i+1}: {e}")
                ensemble_responses.append("")
                ensemble_capabilities.append([])
        
        # Perform majority voting on capabilities (use 2/5 threshold instead of 3/5 for more flexibility)
        vote_threshold = max(2, num_models // 3)  # At least 2, or 1/3 of models
        consensus_capabilities = self._majority_vote_capabilities(ensemble_capabilities, threshold=vote_threshold)
        
        # Format consensus result in the expected JSON format
        consensus_tasks = []
        for i, capability in enumerate(consensus_capabilities):
            consensus_tasks.append({
                "task_id": i + 1,
                "ai_capability": capability,
                "brief_description": "Consensus capability from ensemble voting",
                "raw_excerpt": "Multiple sources", 
                "explanation": f"Selected by majority vote from {num_models} models"
            })
        
        consensus_result = {
            "ai_application_tasks": consensus_tasks
        }
        
        print(f"Ensemble complete: {len(consensus_capabilities)} consensus capabilities from {num_models} models")
        
        return consensus_result
    
    def _fix_common_json_issues(self, json_text):
        """Fix common JSON formatting issues that cause parsing failures"""
        import re
        
        # Remove trailing commas before closing brackets/braces
        json_text = re.sub(r',(\s*[\]}])', r'\1', json_text)
        
        # Fix incomplete JSON by finding the last complete object/array
        # Count braces and brackets to find where JSON likely ends
        brace_count = 0
        bracket_count = 0
        last_complete_pos = 0
        
        for i, char in enumerate(json_text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_complete_pos = i + 1
            elif char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
        
        # If JSON appears incomplete, truncate to last complete position
        if brace_count > 0 and last_complete_pos > 0:
            json_text = json_text[:last_complete_pos]
        
        return json_text
    
    def _fallback_capability_extraction(self, response_content):
        """
        Fallback method to extract capabilities when JSON parsing fails
        
        Args:
            response_content: Raw text response from the model
            
        Returns:
            List of capability strings
        """
        # Simple text extraction as fallback - look for bullet points or numbered items
        capabilities = []
        lines = response_content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for lines that start with bullets, numbers, or dashes
            if line and (line.startswith('•') or line.startswith('-') or 
                        line.startswith('*') or (len(line) > 3 and line[0].isdigit() and line[1:3] == '. ')):
                # Clean up the capability text
                capability = line.lstrip('•-*0123456789. ').strip()
                if capability and len(capability) > 10:  # Only keep substantial text
                    capabilities.append(capability)
        
        return capabilities
    
    def _majority_vote_capabilities(self, ensemble_capabilities, threshold=3):
        """
        Perform majority voting on extracted capabilities using semantic similarity
        
        Args:
            ensemble_capabilities: List of lists, each containing capabilities from one model
            threshold: Minimum number of models that must agree (default: 3 for 5-model ensemble)
        
        Returns:
            List of consensus capabilities
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Flatten all capabilities from all models
        all_capabilities = []
        for model_caps in ensemble_capabilities:
            all_capabilities.extend(model_caps)
        
        if not all_capabilities:
            return []
        
        # Load embedding model for semantic similarity
        try:
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            embeddings = model.encode(all_capabilities)
        except Exception as e:
            print(f"Warning: Could not load embedding model, using exact string matching: {e}")
            return self._majority_vote_exact_match(ensemble_capabilities, threshold)
        
        # Group similar capabilities using clustering
        similarity_matrix = cosine_similarity(embeddings)
        similarity_threshold = 0.75  # Capabilities with >75% similarity are considered the same
        
        # Simple clustering: find groups of similar capabilities
        visited = set()
        capability_groups = []
        
        for i, cap in enumerate(all_capabilities):
            if i in visited:
                continue
                
            # Find all capabilities similar to this one
            similar_indices = []
            for j in range(len(all_capabilities)):
                if similarity_matrix[i][j] >= similarity_threshold:
                    similar_indices.append(j)
                    visited.add(j)
            
            if similar_indices:
                group_capabilities = [all_capabilities[idx] for idx in similar_indices]
                capability_groups.append(group_capabilities)
        
        # Apply majority voting: keep groups with at least 'threshold' members
        consensus_capabilities = []
        for group in capability_groups:
            if len(group) >= threshold:
                # Use the most common capability in the group (or first one if tied)
                from collections import Counter
                capability_counts = Counter(group)
                most_common_capability = capability_counts.most_common(1)[0][0]
                consensus_capabilities.append(most_common_capability)
        
        print(f"Majority voting: {len(capability_groups)} groups found, {len(consensus_capabilities)} passed threshold of {threshold}")
        
        return consensus_capabilities
    
    def _majority_vote_exact_match(self, ensemble_capabilities, threshold=3):
        """
        Fallback majority voting using exact string matching
        """
        from collections import Counter
        
        # Count occurrences of each capability across all models
        all_capabilities = []
        for model_caps in ensemble_capabilities:
            all_capabilities.extend(model_caps)
        
        capability_counts = Counter(all_capabilities)
        
        # Keep capabilities that appear in at least 'threshold' models
        consensus_capabilities = []
        for capability, count in capability_counts.items():
            if count >= threshold:
                consensus_capabilities.append(capability)
        
        print(f"Exact match voting: {len(consensus_capabilities)} capabilities passed threshold of {threshold}")
        
        return consensus_capabilities

    def step_0_manual_coding_train_test(self):
        """ 
        For this, I use o3 to code the answers and then manually check them. This is the prompt I use:
        You are an occupational‑impact analyst.  
Your job: given a job advertisement, list every task that specific AI/ML application or ai/ml-enabled tools mentioned in the job description will perform or automate. 
⚠️ Do **not** include tasks that human software engineers will do while building, deploying, or maintaining the AI/ML system, but rather include what the systems or tools themselves do. 



GOOD EXAMPLES (different types, varying specificity levels):

1. "Deliver machine-learning capabilities within customer solutions to support innovative business ideas."
   (Somewhat broad but has specific purpose - to support innovative business ideas)

2. "Automate ingestion, cleansing, transformation, enrichment and delivery of data via Databricks pipelines"
   (Mentions what the AI/ML capabilities in Databricks do - specific data processing functions)

3. "Artificial-intelligence algorithms that automate or optimise machine-control and other digital dentistry processes to improve or automate CAD/CAM machine operations or decision-making steps."
   (Very specific - mentions the operations the AI/ML is used for, even when job ad doesn't clearly state it)

4. "Automatically generate actionable strategic insights for leadership based on statistical and machine-learning analysis."
   (Simple extraction showing clear AI/ML capability and its business purpose)

BAD EXAMPLES (non-AI/ML tasks that should NOT be extracted):
❌ "Create presentations via PowerPoint to present to key stakeholders"
❌ "Coordinate meetings with development teams using project management software"
❌ "Write technical documentation and user manuals"
❌ "Conduct code reviews and manage software deployments"
❌ "Analyze business requirements and create system specifications"


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
    
    def step_1_extract_ai_applications(self, job_ads_text, use_cache=True):
        """
        Step 1: Extract AI applications from job ads with caching and cost guards
        """
        system_prompt = """

        
        You are an occupational‑impact analyst.  
        Your job: Extract every specific, technical AI/ML capability that will be performed or automated by AI systems mentioned in the job advertisement. 
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

        # Build params dict for cache key
        params = {
            "max_tokens": 500,
            "verbosity": "medium",
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "response_format": {"type": "json_object"}
        }

        # Check cache first
        cache_key = self._get_cache_key(self.model_name, system_prompt, f'"""\n{user_prompt}\n"""', params)
        if use_cache:
            cached_response = self._read_cache(cache_key, "1")
            if cached_response:
                return cached_response

        try:
            # Check cost guards before making call
            self._check_cost_guards()

            response = self._make_api_call(
                system_prompt=system_prompt,
                user_prompt=f'"""\n{user_prompt}\n"""',
                max_tokens_value=500,
                verbosity="medium",
                temperature=0,
                top_p=1,
                seed=42,
                response_format={"type": "json_object"}
            )

            # Track token usage
            self._track_token_usage(response)
            self.new_calls_count += 1
            self.failure_window.append(0)  # Success

            result = self._extract_response_content(response)

            # Write to cache
            if use_cache:
                self._write_cache(cache_key, result, "1")

            return result

        except Exception as e:
            self.failure_window.append(1)  # Failure
            print(f"Error in step 1: {e}")
            return None

    def step_2_separate_tasks(self, step1_output, use_cache=True):
        """
        Step 2: Separate compound tasks into distinct work tasks
        """
        system_prompt = """Role. You are a surgical extractor. Work everything out in your hidden scratchpad, but output only the final clauses.

Task. Given an excerpt that describes how AI technology is being applied, return every distinct work task only, one per line, preserving the original wording (keep any examples, parentheticals, clarifications, punctuation, and casing).

Output contract (hard):

Newline-separated clauses, no bullets, numbers, headings, labels, commentary, or blank lines.

No paraphrasing or normalization; copy the task text exactly as it appears, trimming only leading/trailing spaces.

De-duplicate exact or case-insensitive duplicates.

Keep the order of appearance.

If no tasks are present, output nothing (empty output).

What counts as a task. A task is a verb phrase describing an action that produces a concrete object, deliverable, or outcome (e.g., “build dashboards,” “detect defects in images”). Ignore meta-text like “Responsibilities include:” or headers.

Conjunction rules (decision rule):

Multi-verb, same object/outcome → one clause.
“Collect, clean, and analyze customer-support tickets.” ➜ one clause.

Multi-verb, different objects/outcomes → split.
“Clean raw sensor data and build dashboards to monitor equipment status.” ➜ two clauses.

One verb phrase + multiple nouns → one clause per noun, repeat the full verb phrase.
Don’t cross-multiply verbs×nouns.

Treat slashes (“/”) as coordinators: if they point to distinct objects, split; if synonyms/variants of the same object, keep.

Semicolons/colons/lists denote potential splits; apply the rules above after segmentation.

Edge handling:

Keep parentheticals and e.g./i.e. content inside the clause.

Keep qualifiers like time/frequency/purpose if attached to the task (“to guide hiring decisions,” “weekly”).

Keep model/tool/technique names when they’re part of the described work (“fine-tune BERT models,” “monitor with Grafana”).

Few-shot examples (additions to yours):

Input: “Preprocess and analyze images and audio for defect detection.”
Output:
Preprocess and analyze images for defect detection.
Preprocess and analyze audio for defect detection.

Input: “Generate alerts and update incident tickets when anomalies are detected.”
Output:
Generate alerts when anomalies are detected.
Update incident tickets when anomalies are detected.

Input: “Rank leads (e.g., SMB, mid-market, enterprise) using a propensity model.”
Output:
Rank leads (e.g., SMB, mid-market, enterprise) using a propensity model.

"""


        user_prompt = step1_output

        # Build params dict for cache key
        params = {
            "max_tokens": 300,
            "verbosity": "medium",
            "temperature": 0,
            "top_p": 1,
            "seed": 42
        }

        # Check cache first
        cache_key = self._get_cache_key(self.model_name, system_prompt, user_prompt, params)
        if use_cache:
            cached_response = self._read_cache(cache_key, "2")
            if cached_response:
                return cached_response

        try:
            # Check cost guards before making call
            self._check_cost_guards()

            response = self._make_api_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens_value=300,
                verbosity="medium",
                temperature=0,
                top_p=1,
                seed=42
            )

            # Track token usage
            self._track_token_usage(response)
            self.new_calls_count += 1
            self.failure_window.append(0)  # Success

            result = self._extract_response_content(response)

            # Write to cache
            if use_cache:
                self._write_cache(cache_key, result, "2")

            return result

        except Exception as e:
            self.failure_window.append(1)  # Failure
            print(f"Error in step 2: {e}")
            return None

    def step_3_filter_applications(self, step2_output, use_cache=True):
        """
        Step 3: Filter and clean applications
        """
        system_prompt = """You are given a description of how AI/ML is applied in a job.

Rewrite it as the **single autonomous deliverable the AI system produces**, phrased like an O*NET task.

**Only two rules**
1) Focus on the **specific output the system produces automatically** (deliverable), not what humans do.
2) **Avoid generic verbs** like “analyze”, “process”, or “generate insights”; instead use a concrete capability verb (e.g., detect, classify, forecast, segment, extract, rank, score, recommend, optimize, summarize, translate).

**Output (JSON)**
{"ai_task":"<one sentence or N/A>"}"""
        user_prompt = step2_output

        # Build params dict for cache key
        params = {
            "max_tokens": 300,
            "verbosity": "medium",
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "response_format": {"type": "json_object"}
        }

        # Check cache first
        cache_key = self._get_cache_key(self.model_name, system_prompt, user_prompt, params)
        if use_cache:
            cached_response = self._read_cache(cache_key, "3")
            if cached_response:
                return cached_response

        try:
            # Check cost guards before making call
            self._check_cost_guards()

            response = self._make_api_call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens_value=300,
                verbosity="medium",
                temperature=0,
                top_p=1,
                seed=42,
                response_format={"type": "json_object"}
            )

            # Track token usage
            self._track_token_usage(response)
            self.new_calls_count += 1
            self.failure_window.append(0)  # Success

            result = self._extract_response_content(response)

            # Write to cache
            if use_cache:
                self._write_cache(cache_key, result, "3")

            return result

        except Exception as e:
            self.failure_window.append(1)  # Failure
            print(f"Error in step 3: {e}")
            return None

#     def step_4_final_filtering(self, step3_output):
#         """
#         Step 4: Final filtering for specificity
#         """
#         system_prompt = """The excerpt below describes how an artificial intelligence technology is being applied. Please determine if the application is very specific. If yes, please summarize the application (without outputing anything else). All references to any type of AI tool (e.g. natural language processing, machine learning, computer vision, generative AI, or any specific AI/ML algorithm) are redundant and should be stripped from the text. Otherwise, respond 'N/A'. Here are some examples:

# -'Predictive Analytics' should be 'N/A' as it is very broad;
# -'Data Visualization' should be 'N/A' as it is very broad;
# -'AI-driven NFT Collection Visualization' should be kept as it is a very specific application.
# -'Perform exploratory data analysis for invoice anomalies' should be 'invoice anomalies'
# -'Provide self-service data access and custom visualization interfaces for the oceanic team' should be 'custom visualization interfaces for the oceanic team' as this is a specific application.

# Please filter the following application and return your response as JSON in this format:
# {"final_application": "your filtered text here or N/A"}"""

#         user_prompt = step3_output

#         try:
#             response = self._make_api_call(
#                 system_prompt=system_prompt,
#                 user_prompt=user_prompt,
#                 max_tokens_value=200,
#                 verbosity="medium",
#                 temperature=0,
#                 top_p=1,
#                 seed=42,
#                 response_format={"type": "json_object"}
#             )
            
#             # Track token usage
#             self._track_token_usage(response)
            
#             return self._extract_response_content(response)
        
#         except Exception as e:
#             print(f"Error in step 4: {e}")
#             return None
    
    
    def run_step1_only(self, dataset_type="train"):
        """Run only Step 1: Extract AI applications and save results with malformed recovery system"""
        print("="*60)
        print(f"STEP 1: AI APPLICATION EXTRACTION - {dataset_type.upper()} SET")
        print("="*60)
        
        # Check for reprocessing mode
        is_reprocessing, malformed_file, main_output_path = self._detect_reprocessing_mode("step1", dataset_type)
        
        if is_reprocessing:
            # Load malformed file for reprocessing
            malformed_df = pd.read_csv(malformed_file)

            # Filter: only process rows that are pending/failed and haven't exceeded max attempts
            if 'status' in malformed_df.columns and 'attempts' in malformed_df.columns:
                processable = malformed_df[
                    (malformed_df['status'].isin(['pending', 'failed'])) &
                    (malformed_df['attempts'] < self.max_attempts)
                ]
                print(f"🔄 Reprocessing {len(processable)} / {len(malformed_df)} malformed responses")
                print(f"   Filtered out {len(malformed_df) - len(processable)} rows (needs_manual or max_attempts reached)")
                df = processable
            else:
                # Legacy file without ledger fields - process all
                print(f"🔄 Reprocessing {len(malformed_df)} malformed responses (legacy format)")
                df = malformed_df
        else:
            # Load original data
            df = self.load_data(dataset_type)
        
        all_results = []
        malformed_responses = []
        
        # Process each job for step 1
        for idx, job_row in tqdm(df.iterrows(), total=len(df), desc="Extracting AI applications"):
            job_ads_text = job_row['content_clean']
            job_uid = job_row['uid']
            
            # Step 1: Extract AI applications
            step1_result = self.step_1_extract_ai_applications(job_ads_text)
            if not step1_result:
                malformed_responses.append({
                    'uid': job_uid,
                    'content_clean': job_ads_text,
                    'company_name': job_row.get('company_name', ''),
                    'raw_llm_response': '',
                    'error_message': 'Step 1 returned None',
                    'step': 'step1'
                })
                continue
            
            # Detect malformed responses using the new system
            is_malformed, error_message = self._detect_malformed_response(step1_result, "step1", "json")

            if is_malformed:
                # Try automatic JSON repair before giving up
                repaired_json, was_repaired = self._fix_common_json_issues(step1_result)
                if was_repaired and repaired_json:
                    # Validate repaired JSON
                    is_still_malformed, _ = self._detect_malformed_response(repaired_json, "step1", "json")
                    if not is_still_malformed:
                        # Repair successful! Use repaired version
                        step1_result = repaired_json
                        is_malformed = False
                        print(f"  ✅ Auto-repaired JSON for UID {job_uid}")
                    else:
                        # Repair didn't fix the issue
                        error_type = 'json_parse' if 'JSON' in error_message or 'json' in error_message else 'validation'
                        malformed_responses.append({
                            'uid': job_uid,
                            'content_clean': job_ads_text,
                            'company_name': job_row.get('company_name', ''),
                            'raw_llm_response': step1_result,
                            'error_message': error_message,
                            'error_type': error_type,
                            'step': 'step1'
                        })
                        continue
                else:
                    # Could not repair
                    error_type = 'json_parse' if 'JSON' in error_message or 'json' in error_message else 'validation'
                    malformed_responses.append({
                        'uid': job_uid,
                        'content_clean': job_ads_text,
                        'company_name': job_row.get('company_name', ''),
                        'raw_llm_response': step1_result,
                        'error_message': error_message,
                        'error_type': error_type,
                        'step': 'step1'
                    })
                    continue
            
            # Parse valid JSON response
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
                        # This should have been caught by malformed detection, but as backup
                        malformed_responses.append({
                            'uid': job_uid,
                            'content_clean': job_ads_text,
                            'company_name': job_row.get('company_name', ''),
                            'raw_llm_response': step1_result,
                            'error_message': f"Step 1 object format not recognized: {list(step1_json.keys())}",
                            'step': 'step1'
                        })
                        continue
                else:
                    malformed_responses.append({
                        'uid': job_uid,
                        'content_clean': job_ads_text,
                        'company_name': job_row.get('company_name', ''),
                        'raw_llm_response': step1_result,
                        'error_message': f"Step 1 result is not a valid format: {type(step1_json)}",
                        'step': 'step1'
                    })
                    continue
            except json.JSONDecodeError as e:
                # This should have been caught by malformed detection, but as backup
                malformed_responses.append({
                    'uid': job_uid,
                    'content_clean': job_ads_text,
                    'company_name': job_row.get('company_name', ''),
                    'raw_llm_response': step1_result,
                    'error_message': f"JSON decode error: {str(e)}",
                    'step': 'step1'
                })
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
        
        # Save malformed responses if any
        malformed_file_path = None
        if malformed_responses:
            malformed_file_path = self._save_malformed_responses(malformed_responses, "step1", dataset_type)
        
        # Handle results saving based on processing mode
        if is_reprocessing and all_results:
            # Merge with existing results
            results_df = pd.DataFrame(all_results)
            self._merge_reprocessed_results(main_output_path, results_df, 'uid')
            output_path = main_output_path
        else:
            # Save new results
            if all_results:
                results_df = pd.DataFrame(all_results)
                output_path = self.data_dir / f"ai_applications_{dataset_type}.csv"
                results_df.to_csv(output_path, index=False, encoding='utf-8')
            else:
                output_path = None
        
        # Print summary
        print(f"\n📄 Step 1 COMPLETE:")
        if output_path:
            print(f"   - Results saved to: {output_path}")
            print(f"   - AI applications extracted: {len(all_results)}")
        print(f"   - Jobs processed: {len(df)}")
        print(f"   - Malformed responses: {len(malformed_responses)}")
        
        # Check if pipeline can continue
        can_continue = self._check_pipeline_continuation("step1", malformed_file_path)
        
        if not can_continue:
            return None  # Pipeline stopped
        
        # Clean up malformed file if reprocessing was successful
        if is_reprocessing and not malformed_responses and malformed_file:
            try:
                malformed_file.unlink()  # Delete the malformed file
                print(f"✅ Cleaned up resolved malformed file: {malformed_file}")
            except Exception as e:
                print(f"⚠️ Could not clean up malformed file: {e}")
        
        return str(output_path) if output_path else None
    
    def load_step1_results(self, dataset_type="train"):
        """Load the latest Step 1 results"""
        pattern = f"ai_applications_{dataset_type}_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No Step 1 files found matching pattern: {pattern}")
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        print(f"Loading Step 1 data from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} AI applications from Step 1")
        
        return df
    
    def run_step2_only(self, dataset_type="train"):
        """Run only Step 2: Separate tasks and save results with malformed recovery system"""
        print("="*60)
        print(f"STEP 2: TASK SEPARATION - {dataset_type.upper()} SET")
        print("="*60)
        
        # Check for reprocessing mode
        is_reprocessing, malformed_file, main_output_path = self._detect_reprocessing_mode("step2", dataset_type)
        
        if is_reprocessing:
            # Load malformed file for reprocessing
            malformed_df = pd.read_csv(malformed_file)
            print(f"🔄 Reprocessing {len(malformed_df)} malformed responses from {malformed_file}")
            df = malformed_df
            # For reprocessing, we need the key_application from the malformed file
        else:
            # Load Step 1 results
            df = self.load_step1_results(dataset_type)
        
        all_results = []
        malformed_responses = []
        
        # Process each application through step 2
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Separating tasks"):
            app_description = row['key_application']
            row_uid = row.get('uid', row.get('job_uid', f'row_{idx}'))
            
            # Step 2: Separate tasks
            step2_result = self.step_2_separate_tasks(app_description)
            if not step2_result:
                malformed_responses.append({
                    'uid': row_uid,
                    'key_application': app_description,
                    'job_title': row.get('job_title', ''),
                    'company_name': row.get('company_name', ''),
                    'raw_llm_response': '',
                    'error_message': 'Step 2 returned None',
                    'step': 'step2'
                })
                continue
            
            # Detect malformed responses using the new system
            is_malformed, error_message = self._detect_malformed_response(step2_result, "step2", "text")
            
            if is_malformed:
                # Add to malformed responses for manual fixing
                malformed_responses.append({
                    'uid': row_uid,
                    'key_application': app_description,
                    'job_title': row.get('job_title', ''),
                    'company_name': row.get('company_name', ''),
                    'raw_llm_response': step2_result,
                    'error_message': error_message,
                    'step': 'step2'
                })
                continue
            
            # Parse step 2 response (plain text)
            separated_tasks = step2_result.strip().split('\n')
            separated_tasks = [task.strip() for task in separated_tasks if task.strip()]
            
            if not separated_tasks:
                # This should have been caught by malformed detection, but as backup
                malformed_responses.append({
                    'uid': row_uid,
                    'key_application': app_description,
                    'job_title': row.get('job_title', ''),
                    'company_name': row.get('company_name', ''),
                    'raw_llm_response': step2_result,
                    'error_message': 'Step 2 returned no parseable tasks',
                    'step': 'step2'
                })
                continue
            
            # Store each separated task with original metadata
            for task in separated_tasks:
                result_row = row.copy()
                result_row['step2_separated_task'] = task
                all_results.append(result_row)
            
            time.sleep(0.1)
        
        # Save malformed responses if any
        malformed_file_path = None
        if malformed_responses:
            malformed_file_path = self._save_malformed_responses(malformed_responses, "step2", dataset_type)
        
        # Handle results saving based on processing mode
        if is_reprocessing and all_results:
            # Merge with existing results
            results_df = pd.DataFrame(all_results)
            self._merge_reprocessed_results(main_output_path, results_df, 'uid')
            output_path = main_output_path
        else:
            # Save new results
            if all_results:
                results_df = pd.DataFrame(all_results)
                output_path = self.data_dir / f"ai_applications_separated_{dataset_type}.csv"
                results_df.to_csv(output_path, index=False, encoding='utf-8')
            else:
                output_path = None
        
        # Print summary
        print(f"\n📄 Step 2 COMPLETE:")
        if output_path:
            print(f"   - Results saved to: {output_path}")
            print(f"   - Separated tasks: {len(all_results)}")
        print(f"   - Original applications: {len(df)}")
        if all_results and len(df) > 0:
            print(f"   - Task expansion ratio: {len(all_results)/len(df):.2f}")
        print(f"   - Malformed responses: {len(malformed_responses)}")
        
        # Check if pipeline can continue
        can_continue = self._check_pipeline_continuation("step2", malformed_file_path)
        
        if not can_continue:
            return None  # Pipeline stopped
        
        # Clean up malformed file if reprocessing was successful
        if is_reprocessing and not malformed_responses and malformed_file:
            try:
                malformed_file.unlink()  # Delete the malformed file
                print(f"✅ Cleaned up resolved malformed file: {malformed_file}")
            except Exception as e:
                print(f"⚠️ Could not clean up malformed file: {e}")
        
        return str(output_path) if output_path else None
    
    def load_step2_results(self, dataset_type="train"):
        """Load the latest Step 2 results"""
        pattern = f"ai_applications_separated_{dataset_type}_*.csv"
        files = list(self.data_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No Step 2 files found matching pattern: {pattern}")
        
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        print(f"Loading Step 2 data from: {latest_file.name}")
        
        df = pd.read_csv(latest_file)
        print(f"Loaded {len(df)} separated tasks from Step 2")
        
        return df
    
    def run_step3_only(self, dataset_type="train"):
        """Run only Step 3: Filter applications and save results with malformed recovery system"""
        print("="*60)
        print(f"STEP 3: APPLICATION FILTERING - {dataset_type.upper()} SET")
        print("="*60)
        
        # Check for reprocessing mode
        is_reprocessing, malformed_file, main_output_path = self._detect_reprocessing_mode("step3", dataset_type)
        
        if is_reprocessing:
            # Load malformed file for reprocessing
            malformed_df = pd.read_csv(malformed_file)
            print(f"🔄 Reprocessing {len(malformed_df)} malformed responses from {malformed_file}")
            df = malformed_df
        else:
            # Load Step 2 results
            df = self.load_step2_results(dataset_type)
        
        all_results = []
        malformed_responses = []
        
        # Process each separated task through step 3
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering applications"):
            task = row['step2_separated_task']
            row_uid = row.get('uid', row.get('job_uid', f'row_{idx}'))
            
            # Step 3: Filter and clean
            step3_result = self.step_3_filter_applications(task)
            if not step3_result:
                malformed_responses.append({
                    'uid': row_uid,
                    'step2_separated_task': task,
                    'job_title': row.get('job_title', ''),
                    'company_name': row.get('company_name', ''),
                    'raw_llm_response': '',
                    'error_message': 'Step 3 returned None',
                    'step': 'step3'
                })
                continue
            
            # Detect malformed responses using the new system
            is_malformed, error_message = self._detect_malformed_response(step3_result, "step3", "json")
            
            if is_malformed:
                # Add to malformed responses for manual fixing
                malformed_responses.append({
                    'uid': row_uid,
                    'step2_separated_task': task,
                    'job_title': row.get('job_title', ''),
                    'company_name': row.get('company_name', ''),
                    'raw_llm_response': step3_result,
                    'error_message': error_message,
                    'step': 'step3'
                })
                continue
            
            # Parse step 3 JSON response
            try:
                step3_json = json.loads(step3_result)
                ai_task = step3_json.get('ai_task', '')
                if not ai_task or ai_task == 'N/A':
                    continue  # Skip N/A results (this is expected, not malformed)
            except json.JSONDecodeError as e:
                # This should have been caught by malformed detection, but as backup
                malformed_responses.append({
                    'uid': row_uid,
                    'step2_separated_task': task,
                    'job_title': row.get('job_title', ''),
                    'company_name': row.get('company_name', ''),
                    'raw_llm_response': step3_result,
                    'error_message': f"JSON decode error: {str(e)}",
                    'step': 'step3'
                })
                continue
            
            # Store with step 3 output
            result_row = row.copy()
            result_row['step3_output'] = ai_task
            all_results.append(result_row)
            
            time.sleep(0.1)
        
        # Save malformed responses if any
        malformed_file_path = None
        if malformed_responses:
            malformed_file_path = self._save_malformed_responses(malformed_responses, "step3", dataset_type)
        
        # Handle results saving based on processing mode
        if is_reprocessing and all_results:
            # Merge with existing results
            results_df = pd.DataFrame(all_results)
            self._merge_reprocessed_results(main_output_path, results_df, 'uid')
            output_path = main_output_path
        else:
            # Save new results
            if all_results:
                results_df = pd.DataFrame(all_results)
                output_path = self.data_dir / f"ai_applications_filtered_{dataset_type}.csv"
                results_df.to_csv(output_path, index=False, encoding='utf-8')
            else:
                output_path = None
        
        # Print summary
        print(f"\n📄 Step 3 COMPLETE:")
        if output_path:
            print(f"   - Results saved to: {output_path}")
            print(f"   - Filtered tasks: {len(all_results)}")
        print(f"   - Input tasks: {len(df)}")
        if all_results and len(df) > 0:
            print(f"   - Filter acceptance rate: {len(all_results)/len(df):.2%}")
        print(f"   - Malformed responses: {len(malformed_responses)}")
        
        # Check if pipeline can continue
        can_continue = self._check_pipeline_continuation("step3", malformed_file_path)
        
        if not can_continue:
            return None  # Pipeline stopped
        
        # Clean up malformed file if reprocessing was successful
        if is_reprocessing and not malformed_responses and malformed_file:
            try:
                malformed_file.unlink()  # Delete the malformed file
                print(f"✅ Cleaned up resolved malformed file: {malformed_file}")
            except Exception as e:
                print(f"⚠️ Could not clean up malformed file: {e}")
        
        return str(output_path) if output_path else None
    
    def run_ensemble_only(self, dataset_type="train"):
        """Run only the ensemble method on specified dataset"""
        print(f"\n🔄 Running ensemble method on {dataset_type} dataset...")
        
        # Initialize token usage tracking if not already done
        if not hasattr(self, 'token_usage'):
            self.token_usage = []
        
        # Load data
        df = self.load_data(dataset_type)
        
        # Create output filename
        output_file = self.data_dir / f"ai_applications_ensemble_{dataset_type}.csv"
        
        # Process all job ads with ensemble
        all_results = []
        
        for idx, row in df.iterrows():
            print(f"\n📋 Processing job {idx+1}/{len(df)}: {row.get('job_title', 'N/A')}")
            
            # Use ensemble method for step 1
            job_text = row.get('translated_text', row.get('job_description', ''))
            if not job_text:
                print(f"    ⚠️ No job text found for job {idx+1}")
                continue
            
            # Track token usage for this job
            initial_token_count = len(self.token_usage)
            initial_total_tokens = sum(call['prompt_tokens'] + call['completion_tokens'] for call in self.token_usage)
            initial_total_cost = sum(call['cost'] for call in self.token_usage)
            
            applications_result = self.step_1_extract_ai_applications_ensemble(job_text)
            
            # Calculate token usage for this job
            final_token_count = len(self.token_usage)
            final_total_tokens = sum(call['prompt_tokens'] + call['completion_tokens'] for call in self.token_usage)
            final_total_cost = sum(call['cost'] for call in self.token_usage)
            
            job_total_tokens = final_total_tokens - initial_total_tokens
            job_total_cost = final_total_cost - initial_total_cost
            job_api_calls = final_token_count - initial_token_count
            
            # Extract the task list from the result
            applications = applications_result.get('ai_application_tasks', []) if isinstance(applications_result, dict) else []
            
            # Add job metadata and token usage to each application
            for app in applications:
                app.update({
                    'job_id': row.get('job_id', f'job_{idx}'),
                    'job_title': row.get('job_title', 'N/A'),
                    'company_name': row.get('company_name', 'N/A'),
                    'method': 'ensemble',
                    'job_total_tokens': job_total_tokens,
                    'job_total_cost': round(job_total_cost, 6),
                    'job_api_calls': job_api_calls
                })
                all_results.append(app)
            
            # If no applications found, still record the job with token usage
            if not applications:
                all_results.append({
                    'job_id': row.get('job_id', f'job_{idx}'),
                    'job_title': row.get('job_title', 'N/A'),
                    'company_name': row.get('company_name', 'N/A'),
                    'method': 'ensemble',
                    'task_id': 0,
                    'ai_capability': 'No AI applications found',
                    'brief_description': 'No AI applications were identified in this job',
                    'raw_excerpt': '',
                    'explanation': 'Ensemble method found no consensus AI applications',
                    'job_total_tokens': job_total_tokens,
                    'job_total_cost': round(job_total_cost, 6),
                    'job_api_calls': job_api_calls
                })
        
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df.to_csv(output_file, index=False)
            print(f"\n✅ Ensemble processing completed. Results saved to: {output_file}")
        else:
            print(f"\n⚠️ No applications extracted from {dataset_type} dataset")
            return None
        
        return output_file

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

    def run_custom_file_step1(self, input_file, content_column="content_clean", output_suffix=""):
        """Run step 1 extraction on a custom input file"""
        print(f"🚀 Starting AI task extraction on custom file: {input_file}")
        print("Env OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
        print(f"Using content column: {content_column}")
        
        # Load custom data
        df = self.load_custom_file(input_file)
        
        # Check if content column exists
        if content_column not in df.columns:
            raise ValueError(f"Content column '{content_column}' not found in file. Available columns: {list(df.columns)}")
        
        all_results = []
        all_errors = []
        
        # Process each job advertisement
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing jobs"):
            try:
                content = str(row[content_column]) if pd.notna(row[content_column]) else ""
                if not content.strip():
                    print(f"Warning: Empty content for job {index+1}")
                    all_results.append([])
                    continue
                
                # Extract AI applications
                applications = self.step_1_extract_ai_applications_v2(content)
                all_results.append(applications)
                
            except Exception as e:
                print(f"Error processing job {index+1}: {str(e)}")
                all_errors.append({"job_index": index, "error": str(e)})
                all_results.append([])
        
        # Add results to dataframe
        df['ai_applications_raw'] = [json.dumps(apps) if apps else "[]" for apps in all_results]
        df['num_applications'] = [len(apps) if apps else 0 for apps in all_results]
        
        # Create output filename
        input_name = Path(input_file).stem
        suffix = f"_{output_suffix}" if output_suffix else ""
        output_file = self.llm_output_dir / f"{input_name}_step1_extracted{suffix}.csv"
        
        # Save results
        df.to_csv(output_file, index=False)
        
        print(f"\n{'='*60}")
        print("CUSTOM FILE STEP 1 COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Processed {len(df)} job advertisements")
        print(f"✅ Total applications extracted: {sum(len(apps) for apps in all_results)}")
        print(f"✅ Results saved to: {output_file}")
        
        if all_errors:
            print(f"⚠️  {len(all_errors)} errors encountered")
        
        # Print token usage summary
        self.print_token_summary()
        
        return output_file

    def run_custom_file_step2(self, input_file, content_column="step1_output", output_suffix=""):
        """Run step 2 (task separation) on a custom input file with malformed handling"""
        print(f"🚀 Starting Step 2: Task Separation on custom file: {input_file}")
        print(f"Using content column: {content_column}")
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Check for reprocessing mode
        malformed_file = self.data_dir / f"custom_step2_{base_name}_malformed.csv"
        is_reprocessing = malformed_file.exists()
        
        if is_reprocessing:
            print(f"🔄 Reprocessing mode detected - loading malformed responses from: {malformed_file}")
            df = pd.read_csv(malformed_file)
        else:
            # Load custom data
            df = self.load_custom_file(input_file)
            
            # Check if content column exists
            if content_column not in df.columns:
                raise ValueError(f"Content column '{content_column}' not found in file. Available columns: {list(df.columns)}")
        
        all_results = []
        malformed_responses = []
        
        # Process each job and explode into individual AI capability rows, then task rows
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing jobs"):
            try:
                content = str(row[content_column]) if pd.notna(row[content_column]) else ""
                row_uid = row.get('uid', f'row_{index}')
                
                if not content.strip():
                    malformed_responses.append({
                        'uid': row_uid,
                        content_column: content,
                        'job_title': row.get('title', ''),
                        'company_name': row.get('company_name', ''),
                        'raw_llm_response': '',
                        'error_message': 'Empty content',
                        'step': 'custom_step2'
                    })
                    continue
                
                # Step 2: Parse JSON and explode into individual AI capability rows
                try:
                    # Parse the ai_applications_raw JSON (handle double-encoding)
                    import json
                    if content.startswith('"'):
                        # Double-encoded JSON - decode twice
                        json_string = json.loads(content)
                        step1_data = json.loads(json_string)
                    else:
                        # Single-encoded JSON
                        step1_data = json.loads(content)
                    
                    # Extract AI capabilities
                    if 'ai_application_tasks' not in step1_data:
                        malformed_responses.append({
                            'uid': row_uid,
                            content_column: content,
                            'job_title': row.get('title', ''),
                            'company_name': row.get('company_name', ''),
                            'raw_llm_response': '',
                            'error_message': 'Missing ai_application_tasks in JSON',
                            'step': 'custom_step2'
                        })
                        continue
                    
                    ai_capabilities = step1_data['ai_application_tasks']
                    if not ai_capabilities or len(ai_capabilities) == 0:
                        # No AI applications - skip this job (expected behavior)
                        continue
                    
                    # Process each AI capability individually and create individual task rows
                    for capability_idx, ai_app in enumerate(ai_capabilities):
                        ai_capability = ai_app.get('ai_capability', '')
                        if not ai_capability.strip():
                            continue
                        
                        # Send each capability to Step 2 LLM for task separation
                        separated_tasks_raw = self.step_2_separate_tasks(ai_capability)
                        
                        if not separated_tasks_raw:
                            malformed_responses.append({
                                'uid': f"{row_uid}_cap{capability_idx+1}",
                                'ai_capability': ai_capability,
                                'job_title': row.get('title', ''),
                                'company_name': row.get('company_name', ''),
                                'raw_llm_response': '',
                                'error_message': f'Step 2 returned None for capability: {ai_capability[:50]}...',
                                'step': 'custom_step2'
                            })
                            continue
                        
                        # Detect malformed responses
                        is_malformed, error_message = self._detect_malformed_response(separated_tasks_raw, "step2", "text")
                        
                        if is_malformed:
                            malformed_responses.append({
                                'uid': f"{row_uid}_cap{capability_idx+1}",
                                'ai_capability': ai_capability,
                                'job_title': row.get('title', ''),
                                'company_name': row.get('company_name', ''),
                                'raw_llm_response': separated_tasks_raw,
                                'error_message': f'Malformed response for capability: {error_message}',
                                'step': 'custom_step2'
                            })
                            continue
                        
                        # Parse successful response and create individual task rows
                        separated_tasks = [task.strip() for task in separated_tasks_raw.split('\n') if task.strip()]
                        
                        if not separated_tasks:
                            malformed_responses.append({
                                'uid': f"{row_uid}_cap{capability_idx+1}",
                                'ai_capability': ai_capability,
                                'job_title': row.get('title', ''),
                                'company_name': row.get('company_name', ''),
                                'raw_llm_response': separated_tasks_raw,
                                'error_message': 'No parseable tasks found in response',
                                'step': 'custom_step2'
                            })
                            continue
                        
                        # Create one row per separated task
                        for task_idx, separated_task in enumerate(separated_tasks):
                            task_row = row.copy()
                            task_row['uid'] = f"{row_uid}_cap{capability_idx+1}_task{task_idx+1}"
                            task_row['ai_capability'] = ai_capability  # The original AI capability
                            task_row['step2_output'] = separated_task  # Individual separated task
                            task_row['capability_index'] = capability_idx + 1
                            task_row['task_index'] = task_idx + 1
                            task_row['original_job_uid'] = row_uid  # Keep reference to original job
                            all_results.append(task_row)
                    
                except json.JSONDecodeError as e:
                    malformed_responses.append({
                        'uid': row_uid,
                        content_column: content,
                        'job_title': row.get('title', ''),
                        'company_name': row.get('company_name', ''),
                        'raw_llm_response': '',
                        'error_message': f'JSON parsing error: {str(e)}',
                        'step': 'custom_step2'
                    })
                    continue
                    
            except Exception as e:
                error_msg = f"Error processing job {index+1}: {str(e)}"
                print(error_msg)
                malformed_responses.append({
                    'uid': row.get('uid', f'row_{index}'),
                    content_column: row.get(content_column, ''),
                    'job_title': row.get('title', ''),
                    'company_name': row.get('company_name', ''),
                    'raw_llm_response': '',
                    'error_message': error_msg,
                    'step': 'custom_step2'
                })
        
        # Save malformed responses if any
        malformed_file_path = None
        if malformed_responses:
            malformed_file_path = self._save_malformed_responses(malformed_responses, f"custom_step2_{base_name}", "custom")
        
        # Save successful results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_suffix = f"_step2_{output_suffix}" if output_suffix else "_step2"
        output_file = os.path.join(self.data_dir, "llm_output", f"{base_name}{output_suffix}_{timestamp}.csv")
        
        if all_results:
            df_output = pd.DataFrame(all_results)
            df_output.to_csv(output_file, index=False)
        else:
            output_file = None
        
        print(f"{'='*60}")
        print("CUSTOM FILE STEP 2 COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Processed {len(df)} job advertisements")
        print(f"✅ Individual task rows created: {len(all_results)}")
        
        # Calculate expansion statistics
        if all_results:
            original_jobs = len(set(row.get('original_job_uid', row.get('uid', '')) for row in all_results))
            unique_capabilities = len(set((row.get('original_job_uid', ''), row.get('ai_capability', '')) for row in all_results))
            print(f"✅ Original jobs with AI: {original_jobs}")
            print(f"✅ Total AI capabilities: {unique_capabilities}")
            print(f"✅ Task expansion ratio: {len(all_results)/original_jobs:.1f} tasks per job")
        if output_file:
            print(f"✅ Results saved to: {output_file}")
        print(f"⚠️  Malformed responses: {len(malformed_responses)}")
        
        if malformed_file_path:
            print(f"⚠️  Malformed responses saved to: {malformed_file_path}")
            print(f"   Fix the responses and rerun the same command to reprocess")
        
        # Check pipeline continuation
        can_continue = self._check_pipeline_continuation("custom_step2", malformed_file_path)
        
        # Clean up malformed file if reprocessing was successful
        if is_reprocessing and not malformed_responses and malformed_file.exists():
            try:
                malformed_file.unlink()
                print(f"✅ Cleaned up resolved malformed file: {malformed_file}")
            except Exception as e:
                print(f"⚠️ Could not clean up malformed file: {e}")
        
        return output_file

    def run_custom_file_step3(self, input_file, content_column="step2_output", output_suffix=""):
        """Run step 3 (final filtering) on a custom input file with malformed handling"""
        print(f"🚀 Starting Step 3: Final Filtering on custom file: {input_file}")
        print(f"Using content column: {content_column}")
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Check for reprocessing mode
        malformed_file = self.data_dir / f"custom_step3_{base_name}_malformed.csv"
        is_reprocessing = malformed_file.exists()
        
        if is_reprocessing:
            print(f"🔄 Reprocessing mode detected - loading malformed responses from: {malformed_file}")
            df = pd.read_csv(malformed_file)
        else:
            # Load custom data
            df = self.load_custom_file(input_file)
            
            # Check if content column exists
            if content_column not in df.columns:
                raise ValueError(f"Content column '{content_column}' not found in file. Available columns: {list(df.columns)}")
        
        all_results = []
        malformed_responses = []
        
        # Process each row through step 3
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing jobs"):
            try:
                content = str(row[content_column]) if pd.notna(row[content_column]) else ""
                row_uid = row.get('uid', f'row_{index}')
                
                if not content.strip():
                    malformed_responses.append({
                        'uid': row_uid,
                        content_column: content,
                        'job_title': row.get('title', ''),
                        'company_name': row.get('company_name', ''),
                        'raw_llm_response': '',
                        'error_message': 'Empty content',
                        'step': 'custom_step3'
                    })
                    continue
                
                # Split content by lines and process each task
                tasks = [task.strip() for task in content.split('\n') if task.strip()]
                filtered_tasks = []
                task_errors = []
                
                for task in tasks:
                    if task:
                        # Step 3: Filter each task
                        filtered_result = self.step_3_filter_applications(task)
                        
                        if not filtered_result:
                            task_errors.append(f"Step 3 returned None for task: {task[:50]}...")
                            continue
                        
                        # Detect malformed JSON responses
                        is_malformed, error_message = self._detect_malformed_response(filtered_result, "step3", "json")
                        
                        if is_malformed:
                            task_errors.append(f"Malformed response for task '{task[:50]}...': {error_message}")
                            continue
                        
                        # Parse JSON response
                        try:
                            import json
                            parsed = json.loads(filtered_result)
                            ai_task = parsed.get('ai_task', '')
                            if ai_task and ai_task != 'N/A':
                                filtered_tasks.append(ai_task)
                        except json.JSONDecodeError as e:
                            task_errors.append(f"JSON parsing failed for task '{task[:50]}...': {str(e)}")
                
                # Check if we have any successful tasks
                if not filtered_tasks and task_errors:
                    malformed_responses.append({
                        'uid': row_uid,
                        content_column: content,
                        'job_title': row.get('title', ''),
                        'company_name': row.get('company_name', ''),
                        'raw_llm_response': str(task_errors),
                        'error_message': f"All {len(tasks)} tasks failed filtering: {'; '.join(task_errors[:3])}",
                        'step': 'custom_step3'
                    })
                    continue
                
                # Add successful result
                result_row = row.copy()
                result_row['step3_output'] = '\n'.join(filtered_tasks)
                result_row['num_final_tasks'] = len(filtered_tasks)
                all_results.append(result_row)
                    
            except Exception as e:
                error_msg = f"Error processing job {index+1}: {str(e)}"
                print(error_msg)
                malformed_responses.append({
                    'uid': row.get('uid', f'row_{index}'),
                    content_column: row.get(content_column, ''),
                    'job_title': row.get('title', ''),
                    'company_name': row.get('company_name', ''),
                    'raw_llm_response': '',
                    'error_message': error_msg,
                    'step': 'custom_step3'
                })
        
        # Save malformed responses if any
        malformed_file_path = None
        if malformed_responses:
            malformed_file_path = self._save_malformed_responses(malformed_responses, f"custom_step3_{base_name}", "custom")
        
        # Save successful results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_suffix = f"_step3_{output_suffix}" if output_suffix else "_step3"
        output_file = os.path.join(self.data_dir, "llm_output", f"{base_name}{output_suffix}_{timestamp}.csv")
        
        if all_results:
            df_output = pd.DataFrame(all_results)
            df_output.to_csv(output_file, index=False)
        else:
            output_file = None
        
        print(f"{'='*60}")
        print("CUSTOM FILE STEP 3 COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Processed {len(df)} job advertisements")
        print(f"✅ Successful results: {len(all_results)}")
        print(f"✅ Total final tasks: {sum(row.get('num_final_tasks', 0) for row in all_results)}")
        if output_file:
            print(f"✅ Results saved to: {output_file}")
        print(f"⚠️  Malformed responses: {len(malformed_responses)}")
        
        if malformed_file_path:
            print(f"⚠️  Malformed responses saved to: {malformed_file_path}")
            print(f"   Fix the responses and rerun the same command to reprocess")
        
        # Check pipeline continuation
        can_continue = self._check_pipeline_continuation("custom_step3", malformed_file_path)
        
        # Clean up malformed file if reprocessing was successful
        if is_reprocessing and not malformed_responses and malformed_file.exists():
            try:
                malformed_file.unlink()
                print(f"✅ Cleaned up resolved malformed file: {malformed_file}")
            except Exception as e:
                print(f"⚠️ Could not clean up malformed file: {e}")
        
        return output_file

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

def run_step1_train(model_name="gpt-4.1-mini", use_batch=False, use_flex=False, reasoning_effort="low", verbosity="medium"):
    """Run step 1 on training set"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex, reasoning_effort, verbosity)
    output_path = extractor.run_step1_only("train")
    extractor.print_token_summary()
    print(f"\n📋 Next: Run step 2 with --step2-train")

def run_step1_test():
    """Run step 1 on test set"""
    extractor = AITaskExtractor()
    output_path = extractor.run_step1_only("test")
    print(f"\n📋 Next: Run step 2 with --step2-test")

def run_step2_train(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run step 2 on training set"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_step2_only("train")
    extractor.print_token_summary()
    print(f"\n📋 Next: Run step 3 with --step3-train")

def run_step2_test(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run step 2 on test set"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_step2_only("test")
    extractor.print_token_summary()
    print(f"\n📋 Next: Run step 3 with --step3-test")

def run_step3_train(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run step 3 on training set"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_step3_only("train")
    extractor.print_token_summary()
    print(f"\n✅ Final results: {output_path}")

def run_step3_test(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run step 3 on test set"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_step3_only("test")
    extractor.print_token_summary()
    print(f"\n✅ Final results: {output_path}")

def run_train_only(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run extraction on training set only"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_extraction("train")
    extractor.print_token_summary()
    
    if output_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Review extracted tasks for accuracy")
        print(f"2. Run on test set for validation")
        print(f"3. Compare train vs test performance")

def run_test_only(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run extraction on test set only"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_extraction("test")
    extractor.print_token_summary()
    
    if output_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Review extracted tasks for accuracy") 
        print(f"2. Compare with training set results")
        print(f"3. Evaluate for overfitting")

def run_both_sets(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run extraction on both training and test sets"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    train_path, test_path = extractor.run_both_datasets()
    extractor.print_token_summary()
    
    if train_path and test_path:
        print(f"\n📋 Next Steps:")
        print(f"1. Compare training vs test set results")
        print(f"2. Check for overfitting indicators")
        print(f"3. Evaluate model generalization")

def run_ensemble_train(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run ensemble method on training set"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_ensemble_only("train")
    extractor.print_token_summary()
    print(f"\n✅ Ensemble results: {output_path}")

def run_ensemble_dev(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run ensemble method on dev set"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_ensemble_only("dev")
    extractor.print_token_summary()
    print(f"\n✅ Ensemble results: {output_path}")

def run_ensemble_test(model_name="gpt-4.1-mini", use_batch=False, use_flex=False):
    """Run ensemble method on test set"""  
    extractor = AITaskExtractor(model_name, use_batch, use_flex)
    output_path = extractor.run_ensemble_only("test")
    extractor.print_token_summary()
    print(f"\n✅ Ensemble results: {output_path}")

def run_custom_file(input_file, content_column="content_clean", output_suffix="", model_name="gpt-4.1-mini", use_batch=False, use_flex=False, reasoning_effort="low", verbosity="medium", step=1, max_attempts=2, max_new_calls=None, cost_budget=None):
    """Run specified step extraction on a custom input file"""
    extractor = AITaskExtractor(model_name, use_batch, use_flex, reasoning_effort, verbosity, max_attempts, max_new_calls, cost_budget)
    
    if step == 1:
        output_path = extractor.run_custom_file_step1(input_file, content_column, output_suffix)
    elif step == 2:
        output_path = extractor.run_custom_file_step2(input_file, content_column, output_suffix)
    elif step == 3:
        output_path = extractor.run_custom_file_step3(input_file, content_column, output_suffix)
    else:
        raise ValueError(f"Invalid step: {step}. Must be 1, 2, or 3.")
    
    extractor.print_token_summary()
    print(f"\n✅ Custom file step {step} processing complete: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Task Extraction Pipeline')
    parser.add_argument('action', nargs='?', help='Action to perform')
    parser.add_argument('--model', default='gpt-4.1-mini', help='OpenAI model to use (default: gpt-4.1-mini)')
    parser.add_argument('--batch', action='store_true', help='Use batch processing (cached input pricing)')
    parser.add_argument('--flex', action='store_true', help='Use flex processing (reduced pricing)')
    parser.add_argument('--reasoning-effort', default='low', choices=['minimal', 'low', 'medium', 'high'], 
                       help='Reasoning effort for o-series and GPT-5 models (default: low)')
    parser.add_argument('--verbosity', default='medium', choices=['low', 'medium', 'high'],
                       help='Text verbosity for GPT-5 models (default: medium)')
    parser.add_argument('--input-file', type=str, help='Path to custom input CSV file')
    parser.add_argument('--content-column', type=str, default='content_clean', help='Column name containing job advertisement content')
    parser.add_argument('--output-suffix', type=str, default='', help='Suffix for output filename')
    parser.add_argument('--step', type=int, choices=[1, 2, 3], default=1, help='Which step to run (1=extract apps, 2=separate tasks, 3=filter tasks)')

    # Cost control and retry arguments
    parser.add_argument('--max-attempts', type=int, default=2, help='Maximum retry attempts per malformed row (default: 2)')
    parser.add_argument('--max-new-calls', type=int, help='Hard limit on new API calls per run (prevents runaway costs)')
    parser.add_argument('--cost-budget', type=float, help='Maximum total cost budget in dollars (aborts if exceeded)')
    parser.add_argument('--failure-rate-threshold', type=float, default=0.5, help='Abort if failure rate exceeds this (default: 0.5)')
    parser.add_argument('--failure-window-size', type=int, default=10, help='Window size for failure rate calculation (default: 10)')

    # Reprocessing and targeting arguments
    parser.add_argument('--retry-failed', action='store_true', help='Use malformed CSV as input and retry only pending/failed rows')
    parser.add_argument('--only-uids', type=str, help='Process only specific UIDs (comma-separated or path to CSV file)')
    parser.add_argument('--batch-size', type=int, help='Process in batches of this size (for memory control)')
    parser.add_argument('--flush-every', type=int, default=100, help='Write results to disk every N rows (default: 100)')

    # Dry run and debugging
    parser.add_argument('--no-api', action='store_true', help='Dry run: attempt local JSON repair only, no API calls')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache reads/writes for this run')

    # Handle command-line actions that start with --
    args = parser.parse_args()
    
    # Extract model configuration
    model_name = args.model
    use_batch = args.batch
    use_flex = args.flex
    reasoning_effort = getattr(args, 'reasoning_effort', 'low')
    verbosity = getattr(args, 'verbosity', 'medium')
    
    # Ensure only one processing type is selected
    if use_batch and use_flex:
        print("Error: Cannot use both --batch and --flex options simultaneously")
        sys.exit(1)
    
    # Print model configuration
    processing_type = ""
    if use_flex:
        if AITaskExtractor(model_name)._supports_flex():
            processing_type = " (FLEX processing - ~50% cost savings)"
        else:
            processing_type = " (FLEX not supported for this model)"
            print("⚠️ Warning: Flex processing is only available for GPT-5, O3, and O4-mini models")
            use_flex = False
    elif use_batch:
        processing_type = " (BATCH processing)"
    
    print(f"Using model: {model_name}{processing_type}")
    
    if use_flex:
        print("💰 Flex processing enabled: Lower costs, longer response times, automatic fallback to standard processing")
    
    action = args.action
    
    if action:
        # Individual steps
        if action == "--step1-train":
            run_step1_train(model_name, use_batch, use_flex, reasoning_effort, verbosity)
        elif action == "--step1-test":
            run_step1_test(model_name, use_batch, use_flex, reasoning_effort, verbosity)
        elif action == "--step2-train":
            run_step2_train(model_name, use_batch, use_flex, reasoning_effort, verbosity)
        elif action == "--step2-test":
            run_step2_test(model_name, use_batch, use_flex, reasoning_effort, verbosity)
        elif action == "--step3-train":
            run_step3_train(model_name, use_batch, use_flex, reasoning_effort, verbosity)
        elif action == "--step3-test":
            run_step3_test(model_name, use_batch, use_flex, reasoning_effort, verbosity)
        # Full pipelines
        elif action == "--train":
            run_train_only(model_name, use_batch, use_flex)
        elif action == "--test":
            run_test_only(model_name, use_batch, use_flex)
        elif action == "--both":
            run_both_sets(model_name, use_batch, use_flex)
        elif action == "--ensemble-train" or action == "ensemble-train":
            run_ensemble_train(model_name, use_batch, use_flex)
        elif action == "--ensemble-dev" or action == "ensemble-dev":
            run_ensemble_dev(model_name, use_batch, use_flex)
        elif action == "--ensemble-test" or action == "ensemble-test":
            run_ensemble_test(model_name, use_batch, use_flex)
        elif action == "--custom" or action == "custom":
            if not args.input_file:
                print("Error: --input-file is required for custom file processing")
                sys.exit(1)
            run_custom_file(
                args.input_file,
                args.content_column,
                args.output_suffix,
                model_name,
                use_batch,
                use_flex,
                reasoning_effort,
                verbosity,
                args.step,
                args.max_attempts,
                args.max_new_calls,
                args.cost_budget
            )
        else:
            print("Usage:")
            print("\n📋 Individual Steps:")
            print("  python3 stage_3_extract_ai_tasks.py --step1-train    # Step 1: Extract applications (train)")
            print("  python3 stage_3_extract_ai_tasks.py --step1-test     # Step 1: Extract applications (test)")
            print("  python3 stage_3_extract_ai_tasks.py --step2-train    # Step 2: Separate tasks (train)")
            print("  python3 stage_3_extract_ai_tasks.py --step2-test     # Step 2: Separate tasks (test)")
            print("  python3 stage_3_extract_ai_tasks.py --step3-train    # Step 3: Filter applications (train)")
            print("  python3 stage_3_extract_ai_tasks.py --step3-test     # Step 3: Filter applications (test)")
            print("\n🚀 Full Pipelines:")
            print("  python3 stage_3_extract_ai_tasks.py --train          # Complete pipeline (train)")
            print("  python3 stage_3_extract_ai_tasks.py --test           # Complete pipeline (test)")
            print("  python3 stage_3_extract_ai_tasks.py --both           # Complete pipeline (both)")
            print("\n🎯 Ensemble Methods:")
            print("  python3 stage_3_extract_ai_tasks.py --ensemble-train # Ensemble method (train)")
            print("  python3 stage_3_extract_ai_tasks.py --ensemble-dev   # Ensemble method (dev)")
            print("  python3 stage_3_extract_ai_tasks.py --ensemble-test  # Ensemble method (test)")
            print("\n📁 Custom File Processing:")
            print("  python3 stage_3_extract_ai_tasks.py custom --input-file FILE.csv --step 1 --model o3 --reasoning-effort medium")
            print("  python3 stage_3_extract_ai_tasks.py custom --input-file FILE.csv --step 2 --content-column ai_applications_raw --model gpt-5-mini")
            print("  python3 stage_3_extract_ai_tasks.py custom --input-file FILE.csv --step 3 --content-column step2_output --model o3 --reasoning-effort low")
            print("\n💰 Cost Control:")
            print("  --max-attempts N          # Cap retries per row (default: 2)")
            print("  --max-new-calls N         # Hard limit on new API calls")
            print("  --cost-budget DOLLARS     # Abort if budget exceeded")
            print("\n🔄 Reprocessing:")
            print("  --retry-failed            # Process malformed CSV only")
            print("  --only-uids 123,456       # Target specific UIDs")
            print("  --no-api                  # Dry run: local repair only, no API calls")
            print("  --no-cache                # Disable cache for this run")
            print("\n📝 Example with cost controls:")
            print("  python3 stage_3_extract_ai_tasks.py custom --input-file FILE.csv --step 1 --max-new-calls 200 --cost-budget 10")
    else:
        # Check if input file was provided without action - assume custom processing
        if args.input_file:
            print("No action specified but input file provided - using custom file processing")
            run_custom_file(args.input_file, args.content_column, args.output_suffix, model_name, use_batch, use_flex, reasoning_effort, verbosity, args.step)
        else:
            # Default behavior - show usage
            print("AI Task Extraction Pipeline - Please specify an option:")
            print("Run with --help or any invalid argument to see usage options")