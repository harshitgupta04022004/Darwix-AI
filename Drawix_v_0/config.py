"""
config.py — Central Configuration for The Empathy Engine

This module holds all configurable parameters, API keys, default values,
and the emotion-to-voice mapping table. Every other module imports from here
to ensure a single source of truth.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()


# ──────────────────────────────────────────────
#  API Keys & External Service Configuration
# ──────────────────────────────────────────────

HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY", None)
OLLAMA_API_URL: str = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")


# ──────────────────────────────────────────────
#  Emotion Detection Configuration
# ──────────────────────────────────────────────

# HuggingFace model for emotion classification
# This model classifies text into: joy, sadness, anger, fear, surprise, love
EMOTION_MODEL_NAME: str = "bhadresh-savani/distilbert-base-uncased-emotion"

# Supported emotion labels (aligned with the HuggingFace model output)
SUPPORTED_EMOTIONS: list = ["joy", "sadness", "anger", "fear", "surprise", "love"]

# Default emotion when detection fails
DEFAULT_EMOTION: str = "neutral"

# Minimum confidence threshold — below this, we classify as neutral
CONFIDENCE_THRESHOLD: float = 0.35


# ──────────────────────────────────────────────
#  Data Classes
# ──────────────────────────────────────────────

@dataclass
class EmotionResult:
    """Result of emotion detection."""
    label: str              # e.g. "joy", "sadness", "anger"
    score: float            # Raw confidence score (0.0 – 1.0)
    intensity: float        # Normalized intensity level (0.0 – 1.0)
    all_scores: Dict[str, float] = field(default_factory=dict)  # All emotion scores


@dataclass
class VocalParameters:
    """Vocal modulation parameters to apply to TTS output."""
    rate_factor: float      # Speed multiplier (1.0 = normal)
    pitch_semitones: float  # Pitch shift in semitones (0 = no change)
    volume_db: float        # Volume adjustment in dB (0 = no change)
    emotion: str = ""       # The emotion that produced these params
    intensity: float = 0.0  # The intensity that produced these params


# ──────────────────────────────────────────────
#  Emotion → Voice Mapping Table
# ──────────────────────────────────────────────
#
#  Each emotion defines a MIN and MAX range for each vocal parameter.
#  The actual value is interpolated using the emotion's intensity:
#
#    actual_value = min_val + (max_val - min_val) * intensity
#
#  This gives us "intensity scaling" — a slightly happy text won't
#  sound as exaggerated as an extremely happy one.

EMOTION_VOICE_MAP: Dict[str, Dict[str, tuple]] = {
    "joy": {
        "rate_factor":      (1.10, 1.30),    # Speak faster when happy
        "pitch_semitones":  (2.0,  4.0),     # Higher pitch
        "volume_db":        (2.0,  6.0),     # Louder
    },
    "sadness": {
        "rate_factor":      (0.88, 0.78),    # Speak slower when sad (mild → extreme)
        "pitch_semitones":  (-2.0, -4.0),    # Lower pitch (mild → extreme)
        "volume_db":        (-3.0, -6.0),    # Quieter (mild → extreme)
    },
    "anger": {
        "rate_factor":      (1.08, 1.25),    # Faster, aggressive
        "pitch_semitones":  (1.0,  3.0),     # Slightly higher pitch
        "volume_db":        (4.0,  8.0),     # Much louder
    },
    "fear": {
        "rate_factor":      (1.15, 1.35),    # Fast, panicked
        "pitch_semitones":  (2.0,  5.0),     # Higher pitch (trembling)
        "volume_db":        (-1.0, -4.0),    # Quieter/whispering (mild → extreme)
    },
    "surprise": {
        "rate_factor":      (1.10, 1.30),    # Faster
        "pitch_semitones":  (3.0,  6.0),     # Much higher pitch
        "volume_db":        (2.0,  5.0),     # Louder
    },
    "love": {
        "rate_factor":      (0.95, 0.88),    # Slower, gentle (mild → extreme)
        "pitch_semitones":  (0.5,  2.0),     # Slightly higher
        "volume_db":        (-0.5, -2.0),    # Softer (mild → extreme)
    },
    "neutral": {
        "rate_factor":      (1.0,  1.0),     # No change
        "pitch_semitones":  (0.0,  0.0),     # No change
        "volume_db":        (0.0,  0.0),     # No change
    },
}


# ──────────────────────────────────────────────
#  TTS & Audio Configuration
# ──────────────────────────────────────────────

# Language for gTTS
TTS_LANGUAGE: str = "en"

# Output directory for generated audio files
OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
#  Web Server Configuration
# ──────────────────────────────────────────────

SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8000
