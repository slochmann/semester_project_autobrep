# AutoBrep Model Output Format Analysis

## 1. What Does the Forward Method of AutoBrepModel Return?

**Location:** [`cloned_project/AutoBrep/core/src/autobrep/models/autoregressive.py`](cloned_project/AutoBrep/core/src/autobrep/models/autoregressive.py#L440-L448)

### Training Mode (common_step):
**Returns:** A single **scalar loss** value (torch.Tensor)

```python
def common_step(self, batch):
    # ... token processing ...
    loss = self.cad_gpt(
        updated_token,
        cond_mask=loss_mask,
        attn_mask=None,
    )
    return loss  # Returns scalar loss
```

The AutoBrepModel does NOT have an explicit `forward()` method in the class definition. Instead, it:
- Inherits from `BrepBase` (PyTorch Lightning module)
- Uses `common_step()` during training which calls the underlying `self.cad_gpt` (XTransformer)
- The PyTorch Lightning framework calls `training_step()` which internally calls `common_step()`

---

## 2. How is ar_decoder Used/Called?

**Location:** [`cloned_project/AutoBrep/core/src/autobrep/network.py`](cloned_project/AutoBrep/core/src/autobrep/network.py#L1285-1300)

### Architecture:
```python
self.ar_decoder = AutoregressiveWrapper(
    TransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq,
        l2norm_embed=True,
        use_abs_pos_emb=False,
        emb_dropout=0.1,
        attn_layers=attn_layers,  # XTransformerDecoder
    ),
    ignore_index=-1,  # Ignores tokens with value -1
)
```

### Training Usage:
The `ar_decoder` is called with `return_outputs=True` to get logits:

```python
_, (logits, _) = self.ar_decoder(
    x,                    # Input tokens
    return_outputs=True,  # Returns (loss, (logits, _))
    attn_mask=attn_mask,
)
```

### Inference Usage:
```python
def generate(self, prompt, temperature, threshold):
    return self.cad_gpt.ar_decoder.generate(
        prompts=prompt,
        seq_len=self.hparams.max_seq,
        eos_token=MMTokenIndex.EOS.value,
        temperature=temperature,
        filter_logits_fn=partial(top_p, thres=threshold),
        cache_kv=True,
    )
```

---

## 3. Does ar_decoder Compute Loss Internally?

**Yes, partially.**

The XTransformer class (wrapper around ar_decoder) **computes the loss internally** by calling the ar_decoder and then using `F.cross_entropy()` on the output logits:

```python
class XTransformer(nn.Module):
    def forward(self, x, cond_mask=None, attn_mask=None):
        """forward pass"""
        target = x[:, 1:]  # Shift targets

        # Call ar_decoder to get logits
        _, (logits, _) = self.ar_decoder(
            x,
            return_outputs=True,
            attn_mask=attn_mask,
        )

        # Apply loss computation based on mask
        if cond_mask is not None:
            cond_mask = cond_mask[:, :-1]
            _logits_ = logits[~cond_mask]  # True = ignore
            _target_ = target[~cond_mask]
        else:
            _logits_ = logits.reshape(-1, logits.size(-1))
            _target_ = target.reshape(-1)

        loss = F.cross_entropy(_logits_, _target_, ignore_index=-1)
        return loss  # Returns loss, not logits
```

**Key behaviors:**
- The `ar_decoder.ignore_index=-1` tells it to ignore padding tokens
- Cross-entropy loss is computed on **teacher-forced targets** (shifted input)
- Loss can be masked using `cond_mask` to ignore conditional tokens
- Returns **scalar loss** only (not logits or predictions)

---

## 4. Expected Output Format When Passing input_ids Through model_lora

**Location:** [`cloned_project/AutoBrep/core/src/autobrep/models/autoregressive.py`](cloned_project/AutoBrep/core/src/autobrep/models/autoregressive.py#L440-L448)

### Training/Common Step:

**Input:**
```python
batch = {
    "seq": torch.Tensor,           # Shape: (batch_size, seq_len)
    "face_ncs": torch.Tensor,      # Shape: (batch_size, num_faces, ...)
    "edge_ncs": torch.Tensor,      # Shape: (batch_size, num_edges, ...)
}
```

**Processing:**
1. Tokens are encoded with FSQ codes
2. Updated tokens are padded to `max_seq` length
3. Loss masks are created to mark conditional vs. unconditional tokens
4. Attention masks are created to allow/prevent attending to certain positions

**Output:**
```python
loss: torch.Tensor  # Scalar loss value (shape: ())
```

### Inference/Generate:

**Input:**
```python
prompt: torch.Tensor  # Shape: (batch_size, prompt_len)
temperature: float   # Sampling temperature
threshold: float     # Top-p threshold
```

**Processing:**
The `ar_decoder.generate()` method:
- Takes prompt tokens
- Autoregresses to generate tokens one-at-one
- Uses top-p filtering with given threshold
- Returns tokens up to `max_seq` or until EOS token

**Output:**
```python
samples: torch.Tensor  # Shape: (batch_size, seq_len) - generated token ids
```

---

## Summary Table

| Aspect | Value |
|--------|-------|
| **Training Forward Output** | Scalar loss (cross-entropy) |
| **ar_decoder Type** | `AutoregressiveWrapper(TransformerWrapper(...))` |
| **ar_decoder Computes Loss** | Yes (internally via cross_entropy) |
| **Loss Computation** | Teacher-forced on shifted targets with masking |
| **Inference Output** | Token sequences (shape: batch_size × seq_len) |
| **Padding Token** | -1 (ignored in loss) |
| **Max Sequence Length** | 2500 (default) |
| **Vocabulary Size** | face_z_pad + surf_codebook_size + edge_codebook_size |

---

## Key Implementation Details

1. **AutoBrepModel** is a PyTorch Lightning module that:
   - Does NOT expose a raw `forward()` method
   - Uses `common_step()` for training which handles loss computation
   - Uses `generate()` for inference autoregression

2. **XTransformer** is the actual transformer wrapper that:
   - Wraps `ar_decoder` (AutoregressiveWrapper)
   - DOES compute loss internally in its `forward()` method
   - Returns loss directly (not logits)

3. **Loss Masking:**
   - Conditional tokens (first 4 tokens: BOS, complexity flags) are ignored
   - Padding tokens (-1) are automatically ignored
   - Supports level-based masking for unconditioned generation dropout

4. **For LoRA Fine-tuning:**
   - The model expects the same input format as training
   - Loss computation remains the same
   - Only the XTransformer weights are typically fine-tuned (not VAEs)
