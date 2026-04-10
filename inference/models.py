"""
Model loading and inference helpers for both cloud targets.

GCP Cloud Run endpoint: BERT-based Aspect-Based Sentiment Analysis (ABSA).
AWS SageMaker endpoint: Fine-tuned hallucination detection scorer.

In this portfolio implementation both models are thin wrappers around
HuggingFace pipelines that can be run locally. The Dockerfile packages them
so the same image can be deployed to Cloud Run (GCP) or pulled by SageMaker.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from transformers import Pipeline

logger = logging.getLogger(__name__)

ModelTarget = Literal["absa", "hallucination"]


@lru_cache(maxsize=2)
def load_model(target: ModelTarget) -> Any:
    """
    Load and cache a HuggingFace pipeline for the given target.

    absa          -> nlptown/bert-base-multilingual-uncased-sentiment
                     (stands in for a full ABSA fine-tune; swap with your
                      fine-tuned checkpoint)
    hallucination -> vectara/hallucination_evaluation_model
                     (a cross-encoder scoring factual consistency)
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1

    if target == "absa":
        logger.info("Loading ABSA model (GCP fast-path stand-in)")
        return pipeline(
            "text-classification",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=device,
            truncation=True,
        )

    if target == "hallucination":
        logger.info("Loading hallucination detection model (AWS heavy-path stand-in)")
        return pipeline(
            "text-classification",
            model="vectara/hallucination_evaluation_model",
            device=device,
            truncation=True,
        )

    raise ValueError(f"Unknown target: {target}")


def run_absa(text: str) -> dict:
    """
    Run the ABSA pipeline on `text` and return a normalised result dict.

    Returns
    -------
    {
        "label": str,   e.g. "4 stars"
        "score": float, confidence in [0,1]
        "target": "absa"
    }
    """
    pipe = load_model("absa")
    raw = pipe(text)[0]
    return {
        "label": raw["label"],
        "score": round(raw["score"], 4),
        "target": "absa",
    }


def run_hallucination_scorer(text: str, hypothesis: Optional[str] = None) -> dict:
    """
    Run the hallucination detection scorer.

    If `hypothesis` is provided the model scores (text, hypothesis) as a
    factual consistency pair. Otherwise it scores `text` alone.

    Returns
    -------
    {
        "label": str,   "consistent" or "hallucinated"
        "score": float, factual consistency score in [0,1]
        "target": "hallucination"
    }
    """
    pipe = load_model("hallucination")
    input_text = f"{text} [SEP] {hypothesis}" if hypothesis else text
    raw = pipe(input_text)[0]
    return {
        "label": raw["label"],
        "score": round(raw["score"], 4),
        "target": "hallucination",
    }
