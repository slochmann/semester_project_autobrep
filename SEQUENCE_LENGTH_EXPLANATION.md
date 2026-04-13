# Why Training Doesn't Error Despite Exceeded Sequence Lengths

## The Problem
The `verify_parquet.py` script reports many samples as having exceeded sequence length (> 3000 tokens), but when you run `train_lora_ar.py`, it completes without errors. This seems contradictory, but it actually reveals how AutoBrep handles sequences.

## How AutoBrep Actually Works

### Sequence Truncation (Not Filtering)
Instead of rejecting samples that exceed `max_seq`, AutoBrep **automatically truncates** them during data loading:

**Location**: `core/src/autobrep/data/abc_data.py` in the `map_func` method

```python
# If the sequence is too long, randomly sample a context window
if len(full_seq) > self.hparams.max_seq:
    start_idx = np.random.randint(0, len(full_seq) - self.hparams.max_seq + 1)
    full_seq = full_seq[start_idx:start_idx + self.hparams.max_seq]
```

**What this means**:
- AutoBrep randomly selects a `max_seq`-length window from the full sequence
- This is **not an error condition** — it's by design
- No exceptions are raised
- **But you lose geometry information** from parts of the model that get cut off

### Default max_seq Value
From `core/src/autobrep/models/autoregressive.py`:
```python
max_seq: int = 2500  # Default value
```

The verification script was checking against 3000, but AutoBrep defaults to **2500 tokens**.

## Why This Is a Problem for Your LoRA Fine-Tuning

When your verification script flags a sample with estimated `seq_len = 250 faces × 10 + 500 edges × 9 + 15 ≈ 6500 tokens`:

1. **At training time**: AutoBrep randomly samples a 2500-token window from this 6500-token sequence
2. **Probability each epoch**:
   - The same sample might see different geometry each time (random truncation point)
   - Some epochs might miss critical topology information
3. **For 100 pipe flange samples**:
   - Limited data diversity is made worse by random truncation
   - Model sees inconsistent geometry across training
   - Poor generalization to full geometry at inference time

## Updated Verification Script

The script has been updated to reflect this behavior:
- Flags are now **warnings** (⚠️) not errors
- Message clearly states: "will be truncated during training; loses topology info"
- Helps you understand which samples will experience information loss

## Recommendations for Your Training

1. **Understand your data**:
   - Run `python verify_parquet.py --parquet <path> --n_samples 100` to see how many samples get truncated
   - If many exceed 2500 tokens, you're losing topology information

2. **If many samples exceed 2500 tokens**:
   - Consider pre-processing to simplify geometry (reduce faces/edges)
   - Or accept information loss as part of the fine-tuning trade-off
   - Monitor validation loss to see if truncation hurts generalization

3. **For inference**:
   - When you generate new designs with the LoRA model, they won't be constrained by this
   - But training data seen in partially-truncated form may limit capability

## Key Constants

| Parameter | Value | Source |
|-----------|-------|--------|
| `max_seq` | 2500 | `autoregressive.py` default |
| Tokens/face | 10 | 6 (bbox) + 4 (geometry) |
| Tokens/edge | 9 | 6 (bbox) + 2 (geometry) + 1 (topology) |
| Overhead | ~15 | Special tokens (start/end, BFT markers) |

## Formula for Sequence Length Estimation

```
seq_len = (num_faces × 10) + (num_edges × 9) + 15
```

For example:
- 100 faces, 180 edges: `100×10 + 180×9 + 15 = 1000 + 1620 + 15 = 2635 tokens` → **will be truncated**
- 80 faces, 120 edges: `80×10 + 120×9 + 15 = 800 + 1080 + 15 = 1895 tokens` → **safe**
