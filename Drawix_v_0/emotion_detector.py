"""
emotion_detector.py — Text Emotion Classification Module

This module is responsible for analyzing input text and classifying it into
one of several emotional categories. It supports two providers:

  1. HuggingFace (default) — Uses a pretrained DistilBERT model fine-tuned
     on the dair-ai/emotion dataset. Runs locally, no API key needed for
     basic usage.

  2. Ollama — Falls back to a locally running LLM via the Ollama API.
     Useful when you don't want to download the HuggingFace model or
     prefer using a larger language model.

Data Flow:
    text (str)  →  detect_emotion()  →  EmotionResult(label, score, intensity)

The EmotionResult is then consumed by voice_modulator.py to determine
vocal parameters.
"""

import logging
from typing import Optional

from config import (
    EMOTION_MODEL_NAME,
    CONFIDENCE_THRESHOLD,
    DEFAULT_EMOTION,
    EmotionResult,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Lazy-loaded HuggingFace pipeline (singleton)
# ──────────────────────────────────────────────
_hf_pipeline = None


def _get_hf_pipeline():
    """
    Lazily initialise the HuggingFace emotion classification pipeline.
    The model is downloaded on first use and cached locally by the
    transformers library (~/.cache/huggingface/).
    """
    global _hf_pipeline
    if _hf_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading HuggingFace model: {EMOTION_MODEL_NAME}")
            _hf_pipeline = hf_pipeline(
                "text-classification",
                model=EMOTION_MODEL_NAME,
                top_k=None,           # Return scores for ALL labels
                truncation=True,
            )
            logger.info("HuggingFace model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise RuntimeError(
                f"Could not load HuggingFace emotion model '{EMOTION_MODEL_NAME}'. "
                f"Make sure 'transformers' and 'torch' are installed.\n"
                f"Error: {e}"
            )
    return _hf_pipeline


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def detect_emotion(text: str, provider: str = "huggingface") -> EmotionResult:
    """
    Detect the dominant emotion in the given text.

    Args:
        text:     The input text to analyse.
        provider: "huggingface" (default) or "ollama".

    Returns:
        EmotionResult with the detected emotion label, confidence score,
        normalised intensity, and a dict of all emotion scores.
    """
    if not text or not text.strip():
        return EmotionResult(
            label=DEFAULT_EMOTION,
            score=0.0,
            intensity=0.0,
            all_scores={},
        )

    text = text.strip()

    if provider == "ollama":
        return _detect_with_ollama(text)
    else:
        return _detect_with_huggingface(text)


def _detect_with_huggingface(text: str) -> EmotionResult:
    """
    Classify emotion using the HuggingFace pretrained model.

    The model returns a list of dicts sorted by score, e.g.:
        [{"label": "joy", "score": 0.98}, {"label": "sadness", "score": 0.01}, ...]

    We take the top label, compute intensity from the score, and return.
    """
    pipe = _get_hf_pipeline()

    try:
        results = pipe(text)

        # `top_k=None` returns a list of lists — we want the first (and only) inner list
        if isinstance(results[0], list):
            scores_list = results[0]
        else:
            scores_list = results

        # Build a dict of all scores
        all_scores = {item["label"]: round(item["score"], 4) for item in scores_list}

        # Sort to find the dominant emotion
        sorted_scores = sorted(scores_list, key=lambda x: x["score"], reverse=True)
        top = sorted_scores[0]

        label = top["label"].lower()
        score = top["score"]

        # If confidence is below threshold, treat as neutral
        if score < CONFIDENCE_THRESHOLD:
            label = DEFAULT_EMOTION
            intensity = 0.0
        else:
            # Normalise intensity: map score from [threshold, 1.0] → [0.0, 1.0]
            intensity = min(1.0, (score - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD))

        logger.info(f"HuggingFace detected: {label} (score={score:.4f}, intensity={intensity:.2f})")

        return EmotionResult(
            label=label,
            score=round(score, 4),
            intensity=round(intensity, 4),
            all_scores=all_scores,
        )

    except Exception as e:
        logger.error(f"HuggingFace emotion detection failed: {e}")
        return EmotionResult(
            label=DEFAULT_EMOTION,
            score=0.0,
            intensity=0.0,
            all_scores={},
        )


def _detect_with_ollama(text: str) -> EmotionResult:
    """
    Classify emotion using the Ollama API (delegates to ollama_client.py).
    """
    from ollama_client import classify_emotion_with_ollama
    return classify_emotion_with_ollama(text)
