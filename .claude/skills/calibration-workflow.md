# Skill: Threshold Calibration Workflow

## When to Use

Use this skill when recalibrating the routing threshold tau, evaluating calibration quality, or troubleshooting cases where the Airflow `threshold_recalibrate` task produces unexpected tau values.

## Background

Tau is derived via **isotonic regression** (pool-adjacent-violators algorithm) on (entropy, error) pairs. Unlike Platt scaling (logistic), isotonic regression is non-parametric — it finds the optimal monotone step function without assuming a functional form. This matters because the entropy-error relationship is often non-linear and non-sigmoidal.

Key properties:
- The fitted function is guaranteed **monotone non-decreasing**
- Tau is the **first entropy value** where `P(error|H) >= epsilon` (risk tolerance)
- If `P(error)` never reaches epsilon, tau is set to the max observed entropy (everything routes fast-path)

## Full Manual Calibration

```python
import numpy as np
from router.threshold import ThresholdCalibrator

# 1. Load your (entropy, error) pairs from BigQuery or sample data
#    In practice: query routing_events where is_error is not null
entropy_scores = np.array([...])  # H_route values
error_labels = np.array([...])    # 1 = model was wrong, 0 = correct

# 2. Choose risk tolerance (epsilon)
#    0.10 = aggressive escalation (more requests to SageMaker)
#    0.15 = balanced (spec default)
#    0.25 = permissive (most requests stay on fast path)
calibrator = ThresholdCalibrator(risk_tolerance=0.15)

# 3. Fit and get result
result = calibrator.fit(
    entropy_scores,
    error_labels,
    previous_tau_history=[1.85, 1.92, 2.01],  # from prior runs
)

print(f"New tau:  {result.tau:.4f}")
print(f"AUROC:   {result.auroc:.4f}")
print(f"Samples: {result.n_samples}")

# 4. Verify monotonicity visually
test_grid = np.linspace(entropy_scores.min(), entropy_scores.max(), 50)
for h in test_grid:
    p = calibrator.predict_error_prob(h)
    marker = " <-- tau" if abs(h - result.tau) < 0.05 else ""
    print(f"  H={h:.3f}  P(error)={p:.4f}{marker}")

# 5. Save
result.save("calibration_result.json")
```

## Evaluating Calibration Quality

### AUROC

AUROC > 0.65 means entropy is a useful error predictor for routing. AUROC < 0.55 means entropy is near random for this data — investigate whether the error labels are meaningful or the probe model is inappropriate.

### Calibration Plot

```python
import numpy as np

# Bin entropy scores and compute empirical error rate per bin
n_bins = 10
bins = np.linspace(entropy_scores.min(), entropy_scores.max(), n_bins + 1)
for i in range(n_bins):
    mask = (entropy_scores >= bins[i]) & (entropy_scores < bins[i+1])
    if mask.sum() == 0:
        continue
    empirical = error_labels[mask].mean()
    predicted = calibrator.predict_error_prob((bins[i] + bins[i+1]) / 2)
    print(f"  Bin [{bins[i]:.2f}, {bins[i+1]:.2f}): n={mask.sum():4d}  "
          f"empirical={empirical:.3f}  predicted={predicted:.3f}")
```

Large gaps between empirical and predicted error rates suggest the isotonic model is overfit (too few samples) or the entropy-error relationship has shifted.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Tau at maximum entropy (everything fast-path) | P(error) never reaches epsilon | Lower risk_tolerance or check if error labels are all zeros |
| Tau very low (everything escalated) | Error rate is high across all entropy values | Check if error labels are correct; consider raising epsilon |
| AUROC near 0.5 | Entropy does not predict errors for this data distribution | Investigate probe model choice; consider using more final layers |
| Tau oscillating between runs | Small sample size or distribution shift between runs | Increase calibration window; smooth tau via exponential moving average |

## Pushing Tau to Production

After calibration, tau must be written to the parameter store so the live router picks it up:

- **AWS SSM:** `aws ssm put-parameter --name /crosscloud/tau --value "1.95" --type String --overwrite`
- **GCP Secret Manager:** `gcloud secrets versions add crosscloud-tau --data-file=-`
- **Local mock:** Written to `/tmp/current_tau.json`

The router's `InferenceRouter.update_tau(new_tau)` method can also be called at runtime (e.g., via an admin endpoint).
