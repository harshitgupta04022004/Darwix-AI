"""
voice_modulator.py — Emotion → Vocal Parameter Mapping Module

This module takes an EmotionResult (from emotion_detector.py) and produces
a VocalParameters object that describes how the TTS audio should be modified.

Key Concept — Intensity Scaling:
    The mapping is NOT a flat lookup. Each emotion defines a MIN and MAX
    range for rate, pitch, and volume. The actual value is linearly
    interpolated based on the emotion's *intensity* (derived from confidence):

        actual = min_val + (max_val - min_val) × intensity

    This means:
        "This is good."              → slight pitch increase
        "This is the best news ever!" → dramatic pitch + rate increase

Data Flow:
    EmotionResult  →  get_vocal_parameters()  →  VocalParameters

Connection to other modules:
    - Receives EmotionResult from emotion_detector.py
    - Produces VocalParameters consumed by tts_engine.py → audio_processor.py
    - All mapping data is defined in config.py (EMOTION_VOICE_MAP)
"""

import logging

from config import (
    EMOTION_VOICE_MAP,
    DEFAULT_EMOTION,
    EmotionResult,
    VocalParameters,
)

logger = logging.getLogger(__name__)


def get_vocal_parameters(emotion_result: EmotionResult) -> VocalParameters:
    """
    Convert an EmotionResult into concrete vocal modulation parameters.

    The function looks up the emotion in EMOTION_VOICE_MAP and interpolates
    each parameter (rate, pitch, volume) between its min and max values
    using the emotion's intensity.

    Args:
        emotion_result: The classified emotion with label, score, and intensity.

    Returns:
        VocalParameters with rate_factor, pitch_semitones, and volume_db.
    """
    label = emotion_result.label.lower()
    intensity = emotion_result.intensity

    # Fall back to neutral if emotion is not in our map
    if label not in EMOTION_VOICE_MAP:
        logger.warning(f"Unknown emotion '{label}', defaulting to neutral mapping.")
        label = DEFAULT_EMOTION
        intensity = 0.0

    mapping = EMOTION_VOICE_MAP[label]

    # Interpolate each parameter: min + (max - min) * intensity
    rate_factor = _interpolate(mapping["rate_factor"], intensity)
    pitch_semitones = _interpolate(mapping["pitch_semitones"], intensity)
    volume_db = _interpolate(mapping["volume_db"], intensity)

    params = VocalParameters(
        rate_factor=round(rate_factor, 3),
        pitch_semitones=round(pitch_semitones, 2),
        volume_db=round(volume_db, 2),
        emotion=label,
        intensity=round(intensity, 4),
    )

    logger.info(
        f"Vocal parameters for '{label}' (intensity={intensity:.2f}): "
        f"rate={params.rate_factor}, pitch={params.pitch_semitones}st, "
        f"volume={params.volume_db}dB"
    )

    return params


def _interpolate(value_range: tuple, intensity: float) -> float:
    """
    Linearly interpolate between (mild, extreme) values using intensity [0, 1].

    The tuples in EMOTION_VOICE_MAP are ordered as (mild_value, extreme_value):
        intensity=0.0  →  mild_value    (close to neutral)
        intensity=1.0  →  extreme_value (maximum emotional deviation)

    Example — joy rate (1.10, 1.30):
        intensity=0.3  →  1.10 + (1.30 - 1.10) × 0.3 = 1.16x
        intensity=0.9  →  1.10 + (1.30 - 1.10) × 0.9 = 1.28x

    Example — sadness rate (0.88, 0.78):
        intensity=0.3  →  0.88 + (0.78 - 0.88) × 0.3 = 0.85x
        intensity=0.9  →  0.88 + (0.78 - 0.88) × 0.9 = 0.79x
    """
    mild_val, extreme_val = value_range
    return mild_val + (extreme_val - mild_val) * intensity


def describe_parameters(params: VocalParameters) -> str:
    """
    Generate a human-readable description of the vocal parameters.
    Useful for CLI output and logging.
    """
    descriptions = []

    # Rate
    if params.rate_factor > 1.05:
        descriptions.append(f"Speaking {(params.rate_factor - 1) * 100:.0f}% faster")
    elif params.rate_factor < 0.95:
        descriptions.append(f"Speaking {(1 - params.rate_factor) * 100:.0f}% slower")
    else:
        descriptions.append("Normal speaking speed")

    # Pitch
    if params.pitch_semitones > 0.5:
        descriptions.append(f"Pitch raised by {params.pitch_semitones:.1f} semitones")
    elif params.pitch_semitones < -0.5:
        descriptions.append(f"Pitch lowered by {abs(params.pitch_semitones):.1f} semitones")
    else:
        descriptions.append("Normal pitch")

    # Volume
    if params.volume_db > 1.0:
        descriptions.append(f"Volume increased by {params.volume_db:.1f} dB")
    elif params.volume_db < -1.0:
        descriptions.append(f"Volume decreased by {abs(params.volume_db):.1f} dB")
    else:
        descriptions.append("Normal volume")

    return " | ".join(descriptions)
