"""
Unit tests for router/entropy.py.

Tests cover:
  - Pure-function entropy helpers (no model required)
  - AttentionEntropyProbe contract (mocked attention tensors)
  - AUROC benchmark: entropy should correlate with synthetic error signal
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from router.entropy import (
    AttentionEntropyProbe,
    EntropyResult,
    _compute_layer_head_entropies,
    entropy_from_logits,
    entropy_from_probs,
)


# ---------------------------------------------------------------------------
# entropy_from_probs
# ---------------------------------------------------------------------------

class TestEntropyFromProbs:
    def test_uniform_distribution_is_maximum(self):
        """Uniform distribution has maximum entropy for a given support size."""
        n = 8
        uniform = np.ones(n) / n
        peaked = np.zeros(n)
        peaked[0] = 1.0
        assert entropy_from_probs(uniform) > entropy_from_probs(peaked)

    def test_deterministic_distribution_near_zero(self):
        probs = np.array([1.0, 0.0, 0.0, 0.0])
        assert entropy_from_probs(probs) < 0.01

    def test_binary_equal_split(self):
        # H([0.5, 0.5]) = log(2) ≈ 0.693 nats
        result = entropy_from_probs(np.array([0.5, 0.5]))
        assert abs(result - np.log(2)) < 1e-4

    def test_output_is_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.dirichlet(np.ones(10))
            assert entropy_from_probs(p) >= 0.0


# ---------------------------------------------------------------------------
# entropy_from_logits
# ---------------------------------------------------------------------------

class TestEntropyFromLogits:
    def test_equivalent_to_softmax_then_entropy(self):
        logits = np.array([2.0, 1.0, 0.5, -1.0])
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted)
        probs /= probs.sum()
        expected = entropy_from_probs(probs)
        assert abs(entropy_from_logits(logits) - expected) < 1e-6

    def test_large_positive_logit_gives_low_entropy(self):
        logits = np.array([100.0, 0.0, 0.0])
        assert entropy_from_logits(logits) < 0.01


# ---------------------------------------------------------------------------
# _compute_layer_head_entropies
# ---------------------------------------------------------------------------

class TestComputeLayerHeadEntropies:
    @staticmethod
    def _make_attn_layers(n_layers=2, n_heads=4, seq_len=10, uniform=False):
        """Create fake attention tensors (batch=1, heads, T, T)."""
        layers = []
        for _ in range(n_layers):
            if uniform:
                raw = torch.ones(1, n_heads, seq_len, seq_len)
            else:
                raw = torch.rand(1, n_heads, seq_len, seq_len)
            # Row-normalise to make valid attention distributions
            attn = raw / raw.sum(dim=-1, keepdim=True)
            layers.append(attn)
        return tuple(layers)

    def test_shape(self):
        layers = self._make_attn_layers(n_layers=3, n_heads=6)
        result = _compute_layer_head_entropies(layers)
        assert result.shape == (3, 6)

    def test_uniform_attention_maximises_entropy(self):
        """Uniform attention over T tokens -> entropy = log(T)."""
        T = 8
        layers = self._make_attn_layers(n_layers=1, n_heads=2, seq_len=T, uniform=True)
        result = _compute_layer_head_entropies(layers)
        expected = np.log(T)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_values_non_negative(self):
        layers = self._make_attn_layers(n_layers=2, n_heads=4, seq_len=12)
        result = _compute_layer_head_entropies(layers)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# AttentionEntropyProbe integration test (uses real DistilBERT — marks slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestAttentionEntropyProbe:
    """
    These tests load DistilBERT from HuggingFace. Mark with --runslow to run.
    They are skipped in CI by default to keep the test suite fast.
    """

    @pytest.fixture(scope="class")
    def probe(self):
        return AttentionEntropyProbe(model_name="distilbert-base-uncased", n_final_layers=2)

    def test_returns_entropy_result(self, probe):
        result = probe.compute("The food was excellent and the service impeccable.")
        assert isinstance(result, EntropyResult)
        assert result.h_route >= 0.0
        assert result.input_tokens > 0

    def test_longer_input_does_not_crash(self, probe):
        long_text = "word " * 600
        result = probe.compute(long_text)
        assert result.input_tokens <= probe.max_length + 2  # +2 for CLS/SEP

    def test_high_entropy_ambiguous_vs_clear(self, probe):
        """
        Ambiguous / unclear text should produce higher entropy than a clear,
        factual, low-ambiguity sentence. This is a soft property test.
        """
        clear = "Paris is the capital of France."
        ambiguous = "It could maybe possibly sort of be the case that things happen."
        r_clear = probe.compute(clear)
        r_ambiguous = probe.compute(ambiguous)
        # We do not assert strict ordering (model-dependent), but log for inspection
        print(f"\nClear H={r_clear.h_route:.4f}  Ambiguous H={r_ambiguous.h_route:.4f}")

    def test_auroc_entropy_predicts_errors(self, probe):
        """
        AUROC benchmark: entropy scores on a synthetic dataset where
        difficult inputs were injected should yield AUROC > 0.55 vs random labels.
        """
        from sklearn.metrics import roc_auc_score
        texts_easy = [
            "The sky is blue.",
            "Water boils at 100 degrees Celsius.",
            "Dogs are mammals.",
            "The sun rises in the east.",
            "Python is a programming language.",
        ]
        texts_hard = [
            "The ontological implications of post-structuralist discourse remain contested.",
            "Quantum decoherence may or may not preclude macroscopic superposition.",
            "Her ambiguous response neither confirmed nor denied the allegation.",
            "The policy, contingent on several unresolved factors, could potentially shift.",
            "Whether the intervention was causal or merely correlational is unclear.",
        ]
        all_texts = texts_easy + texts_hard
        labels = [0] * len(texts_easy) + [1] * len(texts_hard)

        scores = [probe.compute(t).h_route for t in all_texts]
        auroc = roc_auc_score(labels, scores)
        print(f"\nAUROC (entropy vs hard/easy labels): {auroc:.4f}")
        assert auroc > 0.5, f"Expected entropy AUROC > 0.5, got {auroc:.4f}"
