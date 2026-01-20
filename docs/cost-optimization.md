# Cost Optimization Features

## Overview

The pipeline implements multiple cost optimization strategies to minimize API expenses while maintaining model quality and performance.

---

## Flex Processing

### What It Is
Flex processing offers **50% cost reduction** on supported OpenAI models by allowing up to 15-minute processing windows instead of the standard timeout.

### Supported Models
- GPT-5 models
- O3 models
- O4-mini models

### How It Works
- Standard timeout: 10 minutes (600 seconds)
- Flex timeout: 15 minutes (900 seconds)
- OpenAI prioritizes flex requests during off-peak periods
- Cost reduction applies automatically when requests run during optimal windows

### Usage in Code
```python
use_flex=True  # Enable flex processing in AITaskExtractor
```

### When to Use
- Large batch processing (>1000 items)
- Complex reasoning tasks (O3 with medium/high reasoning)
- Non-time-critical processing
- Sufficient time budget available (up to 15 minutes per batch)

### When NOT to Use
- Real-time or time-sensitive operations
- Small batches (<100 items)
- Simple tasks that complete quickly

---

## Intelligent Retries

### Exponential Backoff
- Automatic retry on rate-limit errors (HTTP 429)
- Exponential backoff with jitter: 2^attempt + random(0,1)
- Prevents thundering herd problem

### Implementation
- Built into OpenAI SDK configuration
- Automatic retry up to specified `max_attempts`
- Logs retry attempts for monitoring

### Example Configuration
```python
max_attempts=2  # Retry failed requests once
```

---

## Model Selection Strategy

### Cost Hierarchy (per 1M tokens, Jan 2025)

**Cheapest Options:**
- GPT-5-nano: $0.05 input, $0.40 output
- GPT-5-mini: $0.25 input, $2.00 output

**Mid-Range:**
- GPT-4.1-mini: $0.40 input, $1.60 output

**High-Performance (Reasoning):**
- GPT-5: $1.25 input, $10.00 output (flex: $0.625 input)
- O3: Variable based on reasoning effort

### Recommended Usage by Stage

**Stage 3 - Step 1 (AI Application Extraction)**
- **Recommended**: O3 with medium reasoning
- **Rationale**: Requires semantic understanding of AI concepts
- **Cost**: ~$15-25 per 100 items (flex processing)
- **Quality**: F1 = 0.7817 on training set

**Stage 3 - Step 2 (Task Separation)**
- **Recommended**: GPT-5-mini
- **Rationale**: Simpler task, cost-effective
- **Cost**: ~$3-5 per 100 items
- **Quality**: F1 = 0.8442 (best of all steps!)

**Stage 3 - Step 3 (Final Filtering)**
- **Recommended**: O3 with low reasoning
- **Rationale**: Validation task, lower reasoning needed
- **Cost**: ~$8-12 per 100 items
- **Quality**: F1 = 0.9194 with JSON fix

### Fallback Strategy
- Primary model fails → automatically fall back to standard processing (no flex)
- Quality preserved, cost slightly higher
- Ensures robustness over perfect cost optimization

---

## Batched Operations

### Memory Efficiency
- **Date-based batching** for Stage 0: 6-month chunks
- **Row-based batching** for Stage 3: Configurable batch size
- **Checkpoint batching** for Stage 4: Process similarity matrices in chunks

### Benefits
- Process large datasets without memory overflow
- Monitor progress with batch-level granularity
- Resume from checkpoints on failure

### Example: Stage 0
```bash
python3 stage_0_get_job_ads.py --extract-keywords-batched
# Processes 6-month date chunks from PostgreSQL
# Each chunk < 2GB memory
```

### Example: Stage 3
```bash
python3 stage_3_extract_ai_tasks.py custom \
  --input-file Data/ai_development_deduplicated_custom.csv \
  --step 1 \
  --batch-size 100  # Process 100 rows per API call
```

---

## Cost Control Mechanisms

### Budget Caps
```python
cost_budget=None  # Optional: Stop if cumulative cost exceeds limit
max_new_calls=None  # Optional: Limit number of API calls
```

### Token Tracking
- Track tokens per API call
- Aggregate token usage by step
- Monitor cost in real-time
- Save token_usage list for analysis

### Failure Monitoring
- Track failure rate in rolling window
- Default window: last 10 attempts
- Default threshold: 50% failure rate
- Stop processing if threshold exceeded (prevents runaway costs)

### Example Usage
```python
extractor = AITaskExtractor(
    model_name="gpt-5-mini",
    use_flex=True,
    max_attempts=2,
    cost_budget=50.00,  # Stop if exceeds $50
    max_new_calls=1000,  # Stop after 1000 calls
    failure_rate_threshold=0.5,
    failure_window_size=10
)
```

---

## Pricing Reference (Updated Jan 2025)

### GPT-5 Models
- **gpt-5**: $1.25 input, $10.00 output
- **gpt-5-mini**: $0.25 input, $2.00 output (5x cheaper than gpt-5)
- **gpt-5-nano**: $0.05 input, $0.40 output (25x cheaper than gpt-5)
- **gpt-5-chat-latest**: $1.25 input, $10.00 output

### GPT-4.1 Models
- **gpt-4.1**: $2.00 input, $8.00 output
- **gpt-4.1-mini**: $0.40 input, $1.60 output

### Legacy Models (Cached)
- Cached input tokens: 10% of standard input cost
- 5-minute context window (most recent 128K tokens)
- Enables cost reduction on repeated analyses

---

## Cost Estimation Examples

### Stage 3 Full Pipeline (56,942 jobs)

**Assumption**: Average job description = 300 tokens

**Step 1 (O3, medium reasoning + flex)**
- Input tokens: 56,942 × 300 × 1.1 = ~18.8M
- Output tokens: ~5M
- Cost: (18.8M × $0.625) + (5M × $5.00) = $13,750 + $25,000 = **~$38,750** (flex savings: 50%)

**Step 2 (GPT-5-mini)**
- Input tokens: ~18.8M
- Output tokens: ~3M
- Cost: (18.8M × $0.25) + (3M × $2.00) = $4,700 + $6,000 = **~$10,700**

**Step 3 (O3, low reasoning + flex)**
- Input tokens: ~18.8M
- Output tokens: ~4M
- Cost: (18.8M × $0.3125) + (4M × $3.00) = $5,875 + $12,000 = **~$17,875** (flex savings: 50%)

**Total for full pipeline: ~$67,325** (with flex processing on reasoning models)

---

## Best Practices for Cost Optimization

1. **Use flex processing** for large batches and reasoning-heavy tasks
2. **Start with GPT-5-mini** for simple extraction tasks
3. **Use O3 only when needed** for complex semantic understanding
4. **Test on small samples first** (e.g., 100 rows) before full runs
5. **Monitor token usage** in real-time to catch unexpected patterns
6. **Set budget caps** and failure thresholds to prevent runaway costs
7. **Reuse embeddings** where possible (Stage 4 checkpoint system)
8. **Profile before scaling** - understand token consumption on representative data

---

## Monitoring and Analysis

### Token Usage Tracking
- AITaskExtractor maintains `token_usage` list
- Each item: `{model, tokens_in, tokens_out, cost, timestamp}`
- Export for analysis: `df_tokens = pd.DataFrame(extractor.token_usage)`

### Cost Analysis
```python
total_cost = extractor.total_cost
api_calls = extractor.api_calls
avg_cost_per_call = total_cost / api_calls
tokens_per_call = extractor.total_tokens / api_calls
```

### Debugging High Costs
- Check if flex processing is enabled
- Verify model selection matches recommendation
- Look for unexpected output token counts (may indicate malformed responses)
- Check for high retry rates (may indicate rate limiting)
