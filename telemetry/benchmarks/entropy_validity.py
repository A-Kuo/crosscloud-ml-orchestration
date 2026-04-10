"""
Benchmark entropy against baseline uncertainty signals.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from router.entropy import AttentionEntropyProbe, entropy_from_logits
from inference.models import run_absa


@dataclass
class BenchmarkResult:
    entropy_auroc: float
    token_count_auroc: float
    logit_entropy_auroc: float
    confidence_auroc: float
    pseudo_perplexity_auroc: float


def _simulate_dataset(seed: int = 42, n: int = 60) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    easy = [f"Clear positive review number {i}" for i in range(n // 2)]
    hard = [f"Ambiguous contradictory statement number {i}" for i in range(n // 2)]
    texts = easy + hard
    labels = np.array([0] * len(easy) + [1] * len(hard), dtype=int)
    rng.shuffle(labels)
    return texts, labels


def run_benchmark(seed: int = 42) -> BenchmarkResult:
    texts, labels = _simulate_dataset(seed=seed)
    probe = AttentionEntropyProbe()

    entropies = []
    token_counts = []
    logit_entropies = []
    confidences = []
    pseudo_perplexities = []

    rng = np.random.default_rng(seed)
    for text in texts:
        e = probe.compute(text)
        entropies.append(e.h_route)
        token_counts.append(e.input_tokens)

        # Synthetic logit distribution baseline.
        logits = rng.normal(0, 1, size=5)
        logit_entropies.append(entropy_from_logits(logits))

        absa = run_absa(text)
        confidences.append(1.0 - float(absa["score"]))

        # Lightweight pseudo perplexity proxy from token count.
        pseudo_perplexities.append(np.log1p(e.input_tokens))

    return BenchmarkResult(
        entropy_auroc=float(roc_auc_score(labels, entropies)),
        token_count_auroc=float(roc_auc_score(labels, token_counts)),
        logit_entropy_auroc=float(roc_auc_score(labels, logit_entropies)),
        confidence_auroc=float(roc_auc_score(labels, confidences)),
        pseudo_perplexity_auroc=float(roc_auc_score(labels, pseudo_perplexities)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entropy validity benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_benchmark(seed=args.seed)
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
