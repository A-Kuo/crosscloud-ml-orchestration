# Skill: Entropy Debugging & Analysis

## When to Use

Use this skill when investigating unexpected routing behaviour, debugging entropy values, or analysing whether the attention entropy signal is performing as expected.

## Context

The routing signal `H_route` is the mean Shannon entropy across attention heads in the final N layers of the DistilBERT probe model. The value lives on [0, log(T)] where T is the sequence length. In practice for DistilBERT with typical inputs:
- **H < 1.5**: very confident (focused attention, clear pattern match)
- **H ~ 1.5-2.5**: moderate uncertainty (typical range for most text)
- **H > 2.5**: high uncertainty (diffuse attention, ambiguous or out-of-distribution)

## Diagnostic Steps

### 1. Inspect Per-Head Entropy Breakdown

When H_route seems wrong, the per-head breakdown reveals whether one pathological head is skewing the aggregate:

```python
from router.entropy import AttentionEntropyProbe

probe = AttentionEntropyProbe(n_final_layers=2)
result = probe.compute("your problematic input text here")

print(f"H_route: {result.h_route:.4f}")
print(f"Per-head entropies (layers × heads):")
for layer_idx, row in enumerate(result.per_head_entropies):
    for head_idx, val in enumerate(row):
        print(f"  Layer {layer_idx} Head {head_idx}: {val:.4f}")
```

A single head near `log(T)` while others are low suggests that head is attending uniformly (common for separator tokens).

### 2. Compare Against Calibration Curve

Check where the input falls on the isotonic regression curve:

```python
from router.threshold import ThresholdCalibrator, CalibrationResult

result = CalibrationResult.load("path/to/calibration.json")
calibrator = ThresholdCalibrator(risk_tolerance=0.15)
# Reload the iso model from saved data, or refit:
# calibrator.fit(entropy_scores, error_labels)

print(f"tau: {result.tau:.4f}")
print(f"P(error | H={h:.4f}): {calibrator.predict_error_prob(h):.4f}")
```

### 3. Token-Level Attention Inspection

For deeper debugging, extract raw attention tensors and visualise which tokens receive attention:

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased", output_attentions=True)
model.eval()

enc = tokenizer("your text", return_tensors="pt")
with torch.no_grad():
    out = model(**enc, output_attentions=True)

tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
attn_last = out.attentions[-1].squeeze(0)  # (heads, T, T)
# attn_last[head_idx][query_pos] gives the attention distribution for that query
```

### 4. Drift Signal Verification

If the Airflow entropy_audit task is flagging drift but it seems spurious, verify the KL divergence manually:

```python
import numpy as np

def kl_divergence(p_samples, q_samples, n_bins=50):
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    eps = 1e-10
    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)
    p_hist = np.clip(p_hist, eps, None); p_hist /= p_hist.sum()
    q_hist = np.clip(q_hist, eps, None); q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))
```

Increase `n_bins` if distributions are narrow (reduces histogram binning noise). Decrease if sample sizes are small (avoids empty bins).

## Known Pitfalls

- **[CLS] and [SEP] tokens** often have near-uniform attention, inflating H_route. The current implementation includes these tokens in the average. If this is problematic, mask them out before averaging.
- **Very short inputs** (< 5 tokens) produce unreliable entropy because the softmax distribution has very few elements.
- **The probe model and the served model may disagree.** DistilBERT attention entropy does not directly reflect the ABSA or hallucination model's internal state. Entropy is a proxy signal.
