"""
Attention entropy computation module.

Computes the Shannon entropy of transformer attention distributions extracted
from a lightweight probe model (DistilBERT by default). The resulting scalar
H_route is used as the routing signal in InferenceRouter.

Math reference (from spec):
    Per-head entropy:    H(alpha_h_i) = -sum_j alpha_h_i_j * log(alpha_h_i_j)
    Aggregated signal:   H_route = (1 / L*H) * sum over layers l, heads h, query i of H(alpha)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel

from .types import EntropyResult

logger = logging.getLogger(__name__)

_EPS = 1e-12  # numerical stability floor to avoid log(0)


class AttentionEntropyProbe:
    """
    Lightweight probe that runs a transformer forward pass with attention
    output enabled and returns the aggregated Shannon entropy of attention
    distributions across all heads in the specified final N layers.

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Defaults to 'distilbert-base-uncased'.
        DistilBERT is chosen as the probe for speed; it has 6 layers and
        12 heads per layer.
    n_final_layers:
        Number of final transformer layers to include in the entropy
        aggregation. Using only the final layers captures high-level semantic
        uncertainty while keeping computation cheap.
    device:
        'cpu', 'cuda', or 'auto' (picks CUDA if available).
    max_length:
        Maximum tokenization length. Longer inputs are truncated.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        n_final_layers: int = 2,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.n_final_layers = n_final_layers
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info("Loading probe model %s on %s", model_name, self.device)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModel.from_pretrained(
            model_name, output_attentions=True
        ).to(self.device)
        self.model.eval()

        # Cache layer count for slicing
        self._n_layers: int = self.model.config.num_hidden_layers

    @torch.no_grad()
    def compute(self, text: str) -> EntropyResult:
        """
        Run a forward pass on `text` and return an EntropyResult.

        The core loop over attention tensors:
            attentions: tuple of (batch=1, heads, seq, seq) per layer
        We take the final `n_final_layers` layers, iterate over heads,
        and for each head compute the entropy of each query row, then
        average across queries and heads.
        """
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        # outputs.attentions: tuple[Tensor(1, H, T, T)] of length L
        attentions: tuple = outputs.attentions
        n_tokens = input_ids.shape[1]

        target_layers = attentions[-self.n_final_layers :]
        per_head_entropies = _compute_layer_head_entropies(target_layers)

        h_route = float(per_head_entropies.mean())

        return EntropyResult(
            h_route=h_route,
            per_head_entropies=per_head_entropies,
            input_tokens=n_tokens,
            model_name=self.model_name,
        )

    @torch.no_grad()
    def compute_batch(self, texts: List[str]) -> List[EntropyResult]:
        """
        Compute entropy for multiple inputs in one transformer forward pass.
        """
        if not texts:
            return []
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        attentions: tuple = outputs.attentions
        target_layers = attentions[-self.n_final_layers :]
        batch_size = input_ids.shape[0]
        results: List[EntropyResult] = []
        for i in range(batch_size):
            per_sample_layers = tuple(layer[i : i + 1] for layer in target_layers)
            per_head_entropies = _compute_layer_head_entropies(per_sample_layers)
            n_tokens = int(attention_mask[i].sum().item())
            results.append(
                EntropyResult(
                    h_route=float(per_head_entropies.mean()),
                    per_head_entropies=per_head_entropies,
                    input_tokens=n_tokens,
                    model_name=self.model_name,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Pure-function helpers (importable without an AttentionEntropyProbe instance)
# ---------------------------------------------------------------------------

def _compute_layer_head_entropies(
    attention_layers: tuple,
) -> np.ndarray:
    """
    Given a tuple of attention tensors from the final N layers, compute the
    mean per-head entropy averaged over query positions.

    Parameters
    ----------
    attention_layers:
        Tuple of Tensors, each shape (batch=1, n_heads, seq_len, seq_len).
        Values are softmax-normalised attention weights in [0, 1].

    Returns
    -------
    np.ndarray of shape (n_layers, n_heads) containing mean entropy per head.
    """
    results = []
    for layer_attn in attention_layers:
        # layer_attn: (1, H, T, T) — squeeze batch dimension
        attn = layer_attn.squeeze(0).cpu().numpy()  # (H, T, T)
        n_heads, T, _ = attn.shape
        head_entropies = []
        for h in range(n_heads):
            # attn[h] shape: (T, T), each row is a distribution over keys
            head_h = attn[h]  # (T, T)
            head_h = np.clip(head_h, _EPS, 1.0)
            # Shannon entropy per query position, then mean across positions
            row_entropy = -np.sum(head_h * np.log(head_h), axis=-1)  # (T,)
            head_entropies.append(float(row_entropy.mean()))
        results.append(head_entropies)
    return np.array(results)  # (n_layers, n_heads)


def entropy_from_logits(logits: np.ndarray) -> float:
    """
    Compute Shannon entropy from raw logits (applies softmax internally).

    Useful for computing entropy on model output distributions rather than
    attention weights. Returns entropy in nats.
    """
    logits = np.asarray(logits, dtype=np.float64)
    # numerically stable softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    probs = np.clip(probs, _EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def entropy_from_probs(probs: np.ndarray) -> float:
    """
    Compute Shannon entropy from a probability vector. Values are clipped
    to [eps, 1] before log to avoid -inf.
    """
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, _EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))
