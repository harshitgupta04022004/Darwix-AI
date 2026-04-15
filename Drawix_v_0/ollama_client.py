"""
ollama_client.py — Ollama API Integration for Emotion Detection

This module provides an alternative emotion detection backend using a
locally running Ollama LLM server. It sends a structured prompt to the
LLM asking it to classify the emotion of the input text and return
a JSON response.

Data Flow:
    text (str)
      → structured prompt
      → POST /api/generate (Ollama)
      → parse JSON response
      → EmotionResult

Configuration:
    - OLLAMA_API_URL: Base URL of the Ollama server (default: http://localhost:11434)
    - OLLAMA_MODEL:   Model name to use (default: llama3)
    
    Set these via environment variables or in a .env file.

Connection to other modules:
    - Called by emotion_detector.py when provider="ollama"
    - Returns the same EmotionResult dataclass used throughout the system
"""

import json
import logging
import re
from typing import Optional

import requests

from config import (
    OLLAMA_API_URL,
    OLLAMA_MODEL,
    SUPPORTED_EMOTIONS,
    DEFAULT_EMOTION,
    CONFIDENCE_THRESHOLD,
    EmotionResult,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Prompt Template
# ──────────────────────────────────────────────

EMOTION_PROMPT_TEMPLATE = """You are an emotion classification assistant.

Analyze the following text and classify its primary emotion into EXACTLY ONE of these categories:
{emotions}

Also rate your confidence in the classification from 0.0 to 1.0.

Text to analyze: "{text}"

You MUST respond with ONLY a valid JSON object in this exact format, nothing else:
{{"emotion": "<emotion_label>", "confidence": <float_between_0_and_1>}}

Response:"""


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def classify_emotion_with_ollama(text: str) -> EmotionResult:
    """
    Send text to the Ollama API for emotion classification.
    
    Args:
        text: The input text to classify.

    Returns:
        EmotionResult with the detected emotion, confidence, and intensity.
    """
    prompt = EMOTION_PROMPT_TEMPLATE.format(
        emotions=", ".join(SUPPORTED_EMOTIONS),
        text=text,
    )

    try:
        response = _call_ollama_api(prompt)
        return _parse_ollama_response(response)
    except Exception as e:
        logger.error(f"Ollama emotion classification failed: {e}")
        return EmotionResult(
            label=DEFAULT_EMOTION,
            score=0.0,
            intensity=0.0,
            all_scores={},
        )


def _call_ollama_api(prompt: str) -> str:
    """
    Make a POST request to the Ollama /api/generate endpoint.
    
    Returns the full generated text from the LLM.
    """
    url = f"{OLLAMA_API_URL}/api/generate"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,            # Get the full response at once
        "options": {
            "temperature": 0.1,     # Low temperature for deterministic output
            "num_predict": 100,     # We only need a short JSON response
        },
    }

    logger.info(f"Calling Ollama API at {url} with model '{OLLAMA_MODEL}'")

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_API_URL}. "
            f"Make sure Ollama is running: 'ollama serve'"
        )
    except requests.Timeout:
        raise RuntimeError("Ollama API request timed out after 30 seconds.")
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama API returned an error: {e}")


def _parse_ollama_response(response_text: str) -> EmotionResult:
    """
    Parse the JSON response from the Ollama LLM.

    The LLM is instructed to return: {"emotion": "...", "confidence": 0.X}
    We extract the JSON, validate the emotion label, and compute intensity.
    """
    # Try to extract JSON from the response (LLMs sometimes add extra text)
    json_match = re.search(r'\{[^}]+\}', response_text)
    if not json_match:
        logger.warning(f"Could not find JSON in Ollama response: {response_text[:200]}")
        return EmotionResult(
            label=DEFAULT_EMOTION,
            score=0.0,
            intensity=0.0,
            all_scores={},
        )

    try:
        parsed = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from Ollama response: {e}")
        return EmotionResult(
            label=DEFAULT_EMOTION,
            score=0.0,
            intensity=0.0,
            all_scores={},
        )

    emotion = parsed.get("emotion", DEFAULT_EMOTION).lower().strip()
    confidence = float(parsed.get("confidence", 0.0))

    # Validate emotion label
    if emotion not in SUPPORTED_EMOTIONS:
        logger.warning(f"Ollama returned unsupported emotion '{emotion}', defaulting to neutral")
        emotion = DEFAULT_EMOTION

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    # Compute intensity
    if confidence < CONFIDENCE_THRESHOLD:
        emotion = DEFAULT_EMOTION
        intensity = 0.0
    else:
        intensity = min(1.0, (confidence - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD))

    logger.info(f"Ollama detected: {emotion} (confidence={confidence:.4f}, intensity={intensity:.2f})")

    return EmotionResult(
        label=emotion,
        score=round(confidence, 4),
        intensity=round(intensity, 4),
        all_scores={emotion: round(confidence, 4)},  # Ollama only gives us the top emotion
    )
